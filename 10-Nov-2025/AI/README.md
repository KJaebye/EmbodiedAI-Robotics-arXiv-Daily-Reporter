# Reasoning Is All You Need for Urban Planning AI 

**Title (ZH)**: 你需要的只有推理：面向城市规划的AI 

**Authors**: Sijie Yang, Jiatong Li, Filip Biljecki  

**Link**: [PDF](https://arxiv.org/pdf/2511.05375)  

**Abstract**: AI has proven highly successful at urban planning analysis -- learning patterns from data to predict future conditions. The next frontier is AI-assisted decision-making: agents that recommend sites, allocate resources, and evaluate trade-offs while reasoning transparently about constraints and stakeholder values. Recent breakthroughs in reasoning AI -- CoT prompting, ReAct, and multi-agent collaboration frameworks -- now make this vision achievable.
This position paper presents the Agentic Urban Planning AI Framework for reasoning-capable planning agents that integrates three cognitive layers (Perception, Foundation, Reasoning) with six logic components (Analysis, Generation, Verification, Evaluation, Collaboration, Decision) through a multi-agents collaboration framework. We demonstrate why planning decisions require explicit reasoning capabilities that are value-based (applying normative principles), rule-grounded (guaranteeing constraint satisfaction), and explainable (generating transparent justifications) -- requirements that statistical learning alone cannot fulfill. We compare reasoning agents with statistical learning, present a comprehensive architecture with benchmark evaluation metrics, and outline critical research challenges. This framework shows how AI agents can augment human planners by systematically exploring solution spaces, verifying regulatory compliance, and deliberating over trade-offs transparently -- not replacing human judgment but amplifying it with computational reasoning capabilities. 

**Abstract (ZH)**: 基于推理的智能城市规划AI框架：具备推理能力的规划代理集成认知层与逻辑组件并通过多代理协作框架实现 

---
# Cleaning Maintenance Logs with LLM Agents for Improved Predictive Maintenance 

**Title (ZH)**: 使用LLM代理清理维护日志以改进预测性维护 

**Authors**: Valeriu Dimidov, Faisal Hawlader, Sasan Jafarnejad, Raphaël Frank  

**Link**: [PDF](https://arxiv.org/pdf/2511.05311)  

**Abstract**: Economic constraints, limited availability of datasets for reproducibility and shortages of specialized expertise have long been recognized as key challenges to the adoption and advancement of predictive maintenance (PdM) in the automotive sector. Recent progress in large language models (LLMs) presents an opportunity to overcome these barriers and speed up the transition of PdM from research to industrial practice. Under these conditions, we explore the potential of LLM-based agents to support PdM cleaning pipelines. Specifically, we focus on maintenance logs, a critical data source for training well-performing machine learning (ML) models, but one often affected by errors such as typos, missing fields, near-duplicate entries, and incorrect dates. We evaluate LLM agents on cleaning tasks involving six distinct types of noise. Our findings show that LLMs are effective at handling generic cleaning tasks and offer a promising foundation for future industrial applications. While domain-specific errors remain challenging, these results highlight the potential for further improvements through specialized training and enhanced agentic capabilities. 

**Abstract (ZH)**: 经济约束、可用数据集有限以及专用专家短缺长期被视为阻碍汽车领域预测性维护（PdM）采纳和发展的关键挑战。近年来，大型语言模型（LLMs）的进步为克服这些障碍并加速PdM从研究向工业实践的过渡提供了机遇。在这些条件下，我们探讨了基于LLM的代理在支持PdM清洗管道方面的潜力。具体而言，我们专注于维护日志，这是训练高性能机器学习（ML）模型的关键数据源，但这些日志常常受到拼写错误、缺失字段、近似重复条目和错误日期等错误的影响。我们评估了LLM代理在涉及六种不同类型的噪声的清洗任务中的表现。研究结果表明，LLM在处理通用清洗任务方面具有有效性，并为未来的工业应用奠定了有前途的基础。尽管领域特定的错误仍然具有挑战性，但这些结果突显了通过专门训练和增强代理能力以进一步改进的潜力。 

---
# Autonomous generation of different courses of action in mechanized combat operations 

**Title (ZH)**: 自主生成机械化战斗行动的不同方案 

**Authors**: Johan Schubert, Patrik Hansen, Pontus Hörling, Ronnie Johansson  

**Link**: [PDF](https://arxiv.org/pdf/2511.05182)  

**Abstract**: In this paper, we propose a methodology designed to support decision-making during the execution phase of military ground combat operations, with a focus on one's actions. This methodology generates and evaluates recommendations for various courses of action for a mechanized battalion, commencing with an initial set assessed by their anticipated outcomes. It systematically produces thousands of individual action alternatives, followed by evaluations aimed at identifying alternative courses of action with superior outcomes. These alternatives are appraised in light of the opponent's status and actions, considering unit composition, force ratios, types of offense and defense, and anticipated advance rates. Field manuals evaluate battle outcomes and advancement rates. The processes of generation and evaluation work concurrently, yielding a variety of alternative courses of action. This approach facilitates the management of new course generation based on previously evaluated actions. As the combat unfolds and conditions evolve, revised courses of action are formulated for the decision-maker within a sequential decision-making framework. 

**Abstract (ZH)**: 本文提出了一种方法论，旨在支持军事地面作战执行阶段的决策制定，重点关注个体行动。该方法论生成并评估针对机械化营的各种行动方案建议，始于由预期结果评估的一组初始方案。它系统地生成数千个个体行动选项，随后进行评估以识别具有更好结果的替代行动方案。这些替代方案会根据敌方状况和行动、单位编组、力量比例、进攻与防御类型以及预期推进速度来进行评估。战术手册评估战斗结果和推进速度。生成和评估的过程同时进行，产生多种替代行动方案。该方法有助于基于先前评估的行动来管理新的行动生成。随着战斗的展开和条件的变化，决策者在其 sequential 决策框架内制定修订后的行动方案。 

---
# ORCHID: Orchestrated Retrieval-Augmented Classification with Human-in-the-Loop Intelligent Decision-Making for High-Risk Property 

**Title (ZH)**: ORCHID: 组合检索增强分类与人类在环智能决策机制以应对高风险财产评估 

**Authors**: Maria Mahbub, Vanessa Lama, Sanjay Das, Brian Starks, Christopher Polchek, Saffell Silvers, Lauren Deck, Prasanna Balaprakash, Tirthankar Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2511.04956)  

**Abstract**: High-Risk Property (HRP) classification is critical at U.S. Department of Energy (DOE) sites, where inventories include sensitive and often dual-use equipment. Compliance must track evolving rules designated by various export control policies to make transparent and auditable decisions. Traditional expert-only workflows are time-consuming, backlog-prone, and struggle to keep pace with shifting regulatory boundaries. We demo ORCHID, a modular agentic system for HRP classification that pairs retrieval-augmented generation (RAG) with human oversight to produce policy-based outputs that can be audited. Small cooperating agents, retrieval, description refiner, classifier, validator, and feedback logger, coordinate via agent-to-agent messaging and invoke tools through the Model Context Protocol (MCP) for model-agnostic on-premise operation. The interface follows an Item to Evidence to Decision loop with step-by-step reasoning, on-policy citations, and append-only audit bundles (run-cards, prompts, evidence). In preliminary tests on real HRP cases, ORCHID improves accuracy and traceability over a non-agentic baseline while deferring uncertain items to Subject Matter Experts (SMEs). The demonstration shows single item submission, grounded citations, SME feedback capture, and exportable audit artifacts, illustrating a practical path to trustworthy LLM assistance in sensitive DOE compliance workflows. 

**Abstract (ZH)**: HRP分类对于美国能源部（DOE）站点至关重要，其中库存包括敏感且 often 双重用途 的设备。合规性必须遵循各种出口控制政策指定的 evolving 规则，以实现透明和可审计的决策。传统由专家独自操作的工作流程耗时、容易积压，并且难以跟上不断变化的监管界限。我们演示了ORCHID，这是一种模块化的代理系统，结合了检索增强生成（RAG）和人的监督，以生成可审核的基于政策的输出。小规模合作代理，检索、描述润色者、分类器、验证器和反馈记录器，通过代理间消息通信协调，并通过模型上下文协议（MCP）调用工具以实现模型无关的本地操作。界面遵循项目到证据到决策的循环，包括逐步推理、基于政策的引用和追加审计捆绑（运行卡、提示、证据）。初步测试结果显示，ORCHID在真实HRP案例中提高了准确性和可追溯性，并将不确定的项目提交给专家。演示展示了单个项目提交、基于参考的引文、专家反馈捕获以及可导出的审计产物，展示了在敏感DOE合规流程中实现可信的LLM辅助的实际路径。 

---
# Real-Time Reasoning Agents in Evolving Environments 

**Title (ZH)**: 实时推理代理在演变环境中 

**Authors**: Yule Wen, Yixin Ye, Yanzhe Zhang, Diyi Yang, Hao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04898)  

**Abstract**: Agents in the real world must make not only logical but also timely judgments. This requires continuous awareness of the dynamic environment: hazards emerge, opportunities arise, and other agents act, while the agent's reasoning is still unfolding. Despite advances in language model reasoning, existing approaches fail to account for this dynamic nature. We introduce real-time reasoning as a new problem formulation for agents in evolving environments and build Real-Time Reasoning Gym to demonstrate it. We study two paradigms for deploying language models in agents: (1) reactive agents, which employ language models with bounded reasoning computation for rapid responses, and (2) planning agents, which allow extended reasoning computation for complex problems. Our experiments show that even state-of-the-art models struggle with making logical and timely judgments in either paradigm. To address this limitation, we propose AgileThinker, which simultaneously engages both reasoning paradigms. AgileThinker consistently outperforms agents engaging only one reasoning paradigm as the task difficulty and time pressure rise, effectively balancing reasoning depth and response latency. Our work establishes real-time reasoning as a critical testbed for developing practical agents and provides a foundation for research in temporally constrained AI systems, highlighting a path toward real-time capable agents. 

**Abstract (ZH)**: 实时推理：动态环境中的及时判断与逻辑推理 

---
# DMA: Online RAG Alignment with Human Feedback 

**Title (ZH)**: DMA：具有人类反馈的在线RAG对齐 

**Authors**: Yu Bai, Yukai Miao, Dawei Wang, Li Chen, Fei Long, Rundi Zhai, Dan Li, Yanyu Ren, Tianfeng Liu, Hongtao Xie, Ce Yang, Xuhui Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.04880)  

**Abstract**: Retrieval-augmented generation (RAG) systems often rely on static retrieval, limiting adaptation to evolving intent and content drift. We introduce Dynamic Memory Alignment (DMA), an online learning framework that systematically incorporates multi-granularity human feedback to align ranking in interactive settings. DMA organizes document-, list-, and response-level signals into a coherent learning pipeline: supervised training for pointwise and listwise rankers, policy optimization driven by response-level preferences, and knowledge distillation into a lightweight scorer for low-latency serving. Throughout this paper, memory refers to the model's working memory, which is the entire context visible to the LLM for In-Context Learning.
We adopt a dual-track evaluation protocol mirroring deployment: (i) large-scale online A/B ablations to isolate the utility of each feedback source, and (ii) few-shot offline tests on knowledge-intensive benchmarks. Online, a multi-month industrial deployment further shows substantial improvements in human engagement. Offline, DMA preserves competitive foundational retrieval while yielding notable gains on conversational QA (TriviaQA, HotpotQA). Taken together, these results position DMA as a principled approach to feedback-driven, real-time adaptation in RAG without sacrificing baseline capability. 

**Abstract (ZH)**: 动态内存对齐（DMA）：一种在线学习框架，用于交互式设置中的多粒度人类反馈集成 

---
# Epistemic Reject Option Prediction 

**Title (ZH)**: 知识拒绝选项预测 

**Authors**: Vojtech Franc, Jakub Paplham  

**Link**: [PDF](https://arxiv.org/pdf/2511.04855)  

**Abstract**: In high-stakes applications, predictive models must not only produce accurate predictions but also quantify and communicate their uncertainty. Reject-option prediction addresses this by allowing the model to abstain when prediction uncertainty is high. Traditional reject-option approaches focus solely on aleatoric uncertainty, an assumption valid only when large training data makes the epistemic uncertainty negligible. However, in many practical scenarios, limited data makes this assumption unrealistic. This paper introduces the epistemic reject-option predictor, which abstains in regions of high epistemic uncertainty caused by insufficient data. Building on Bayesian learning, we redefine the optimal predictor as the one that minimizes expected regret -- the performance gap between the learned model and the Bayes-optimal predictor with full knowledge of the data distribution. The model abstains when the regret for a given input exceeds a specified rejection cost. To our knowledge, this is the first principled framework that enables learning predictors capable of identifying inputs for which the training data is insufficient to make reliable decisions. 

**Abstract (ZH)**: 在高风险应用中，预测模型不仅要生成准确的预测，还必须量化和交流其不确定性。弃权预测通过允许模型在预测不确定性高时 abstain 来解决这一问题。传统弃权方法仅侧重于Aleatoric不确定性，这一假设仅在大量训练数据使Epistemic不确定性可忽略不计时才成立。然而，在许多实际场景中，有限的数据使这一假设不现实。本文引入了Epistemic弃权预测器，在由于数据不足导致Epistemic不确定性高的区域 abstain。基于贝叶斯学习，我们将最优预测器定义为能最小化期望后悔（即所学模型与完全了解数据分布的Bayes-最优预测器之间的性能差距）的预测器。当给定输入的后悔超过指定的弃权成本时，模型会 abstain。据我们所知，这是首个能够学习出能够识别训练数据不足的输入的预测器的原理性框架。 

---
# A hybrid solution approach for the Integrated Healthcare Timetabling Competition 2024 

**Title (ZH)**: 集成医疗时间表竞赛2024的混合解决方案方法 

**Authors**: Daniela Guericke, Rolf van der Hulst, Asal Karimpour, Ieke Schrader, Matthias Walter  

**Link**: [PDF](https://arxiv.org/pdf/2511.04685)  

**Abstract**: We report about the algorithm, implementation and results submitted to the Integrated Healthcare Timetabling Competition 2024 by Team Twente, which scored third in the competition. Our approach combines mixed-integer programming, constraint programming and simulated annealing in a 3-phase solution approach based on decomposition into subproblems. Next to describing our approach and describing our design decisions, we share our insights and, for the first time, lower bounds on the optimal solution values for the benchmark instances. We finally highlight open problems for which we think that addressing them could improve our approach even further. 

**Abstract (ZH)**: 我们报告了Twente团队提交给2024年综合健康 care排程竞赛的算法、实现及结果，该团队在竞赛中获得第三名。我们的方法基于子问题分解，采用三阶段解决方案，结合了混合整数规划、约束编程和模拟退火算法。除了描述我们的方法和设计决策外，我们还分享了我们的见解，并首次提供了基准实例的最优解值下界。最后，我们强调了一些我们认为进一步优化我们方法的问题。 

---
# TimeSearch-R: Adaptive Temporal Search for Long-Form Video Understanding via Self-Verification Reinforcement Learning 

**Title (ZH)**: TimeSearch-R：自验证强化学习驱动的长视频适应性 temporal 搜索理解 

**Authors**: Junwen Pan, Qizhe Zhang, Rui Zhang, Ming Lu, Xin Wan, Yuan Zhang, Chang Liu, Qi She  

**Link**: [PDF](https://arxiv.org/pdf/2511.05489)  

**Abstract**: Temporal search aims to identify a minimal set of relevant frames from tens of thousands based on a given query, serving as a foundation for accurate long-form video understanding. Existing works attempt to progressively narrow the search space. However, these approaches typically rely on a hand-crafted search process, lacking end-to-end optimization for learning optimal search strategies. In this paper, we propose TimeSearch-R, which reformulates temporal search as interleaved text-video thinking, seamlessly integrating searching video clips into the reasoning process through reinforcement learning (RL). However, applying RL training methods, such as Group Relative Policy Optimization (GRPO), to video reasoning can result in unsupervised intermediate search decisions. This leads to insufficient exploration of the video content and inconsistent logical reasoning. To address these issues, we introduce GRPO with Completeness Self-Verification (GRPO-CSV), which gathers searched video frames from the interleaved reasoning process and utilizes the same policy model to verify the adequacy of searched frames, thereby improving the completeness of video reasoning. Additionally, we construct datasets specifically designed for the SFT cold-start and RL training of GRPO-CSV, filtering out samples with weak temporal dependencies to enhance task difficulty and improve temporal search capabilities. Extensive experiments demonstrate that TimeSearch-R achieves significant improvements on temporal search benchmarks such as Haystack-LVBench and Haystack-Ego4D, as well as long-form video understanding benchmarks like VideoMME and MLVU. Notably, TimeSearch-R establishes a new state-of-the-art on LongVideoBench with 4.1% improvement over the base model Qwen2.5-VL and 2.0% over the advanced video reasoning model Video-R1. Our code is available at this https URL. 

**Abstract (ZH)**: 基于时间的搜索旨在根据给定的查询从成千上万的帧中识别出一个相关的最小帧集，为准确理解长视频奠定基础。现有工作试图逐步缩小搜索空间。然而，这些方法通常依赖于手工设计的搜索过程，缺乏端到端优化以学习最优的搜索策略。本文提出了TimeSearch-R，将其时间搜索重新表述为交替的文本-视频思考过程，并通过强化学习（RL）无缝地将视频片段的搜索过程融入推理过程。然而，将如Group Relative Policy Optimization (GRPO)等RL训练方法应用于视频推理可能会导致未监督的中间搜索决策，这将导致对视频内容探索不足和逻辑推理不一致。为此，我们引入了带有完整性自我验证的GRPO (GRPO-CSV)，它是通过强化学习交替推理过程中收集搜索到的视频帧，并利用相同的策略模型验证所搜索帧的充分性，从而提高视频推理的完整性。此外，我们构建了专门用于SFT冷启动和GRPO-CSV的RL训练的数据集，过滤掉时间依赖性较弱的样本，以增强任务难度并改进时间搜索能力。大量实验表明，TimeSearch-R在Haystack-LVBench、Haystack-Ego4D等时间搜索基准测试和VideoMME、MLVU等长视频理解基准测试中取得了显著改进。值得注意的是，TimeSearch-R在LongVideoBench上建立了新的state-of-the-art，分别比基线模型Qwen2.5-VL提高了4.1%，比先进的视频推理模型Video-R1提高了2.0%。代码已发布在https://。 

---
# DGTN: Graph-Enhanced Transformer with Diffusive Attention Gating Mechanism for Enzyme DDG Prediction 

**Title (ZH)**: DGTN：增强图的变换器与扩散注意力门控机制在酶DDG预测中的应用 

**Authors**: Abigail Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05483)  

**Abstract**: Predicting the effect of amino acid mutations on enzyme thermodynamic stability (DDG) is fundamental to protein engineering and drug design. While recent deep learning approaches have shown promise, they often process sequence and structure information independently, failing to capture the intricate coupling between local structural geometry and global sequential patterns. We present DGTN (Diffused Graph-Transformer Network), a novel architecture that co-learns graph neural network (GNN) weights for structural priors and transformer attention through a diffusion mechanism. Our key innovation is a bidirectional diffusion process where: (1) GNN-derived structural embeddings guide transformer attention via learnable diffusion kernels, and (2) transformer representations refine GNN message passing through attention-modulated graph updates. We provide rigorous mathematical analysis showing this co-learning scheme achieves provably better approximation bounds than independent processing. On ProTherm and SKEMPI benchmarks, DGTN achieves state-of-the-art performance (Pearson Rho = 0.87, RMSE = 1.21 kcal/mol), with 6.2% improvement over best baselines. Ablation studies confirm the diffusion mechanism contributes 4.8 points to correlation. Our theoretical analysis proves the diffused attention converges to optimal structure-sequence coupling, with convergence rate O(1/sqrt(T) ) where T is diffusion steps. This work establishes a principled framework for integrating heterogeneous protein representations through learnable diffusion. 

**Abstract (ZH)**: 预测氨基酸突变对酶热力学稳定性（DDG）的影响是蛋白质工程和药物设计中的基础。虽然近期的深度学习方法显示出前景，但它们通常独立处理序列和结构信息，未能捕捉局部结构几何与全局序列模式之间的复杂耦合。我们提出了一种新颖的架构 DGTN（扩散图变换网络），该架构通过扩散机制同时学习图神经网络（GNN）的结构先验权重和变压器注意机制。我们的核心创新在于双向扩散过程：（1）GNN 获取的结构嵌入通过可学习的扩散核指导变压器注意；（2）变压器表示通过注意调制的图更新改进 GNN 的消息传递。我们进行了严格的数学分析，证明这种联合学习方案比独立处理方案具有更好的逼近界。在 ProTherm 和 SKEMPI 基准测试中，DGTN 达到了最先进的性能（皮尔森相关系数 = 0.87，均方根误差 = 1.21 kcal/mol），比最佳基线提高了 6.2%。消融研究证实，扩散机制增加了 4.8 个百分点的相关性。我们的理论分析证明扩散注意会收敛到最优的结构-序列耦合，收敛速率为 O(1/sqrt(T))，其中 T 是扩散步数。本工作建立了通过可学习扩散整合异构蛋白质表示的原理框架。 

---
# On Flow Matching KL Divergence 

**Title (ZH)**: 流匹配KL散度 

**Authors**: Maojiang Su, Jerry Yao-Chieh Hu, Sophia Pi, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05480)  

**Abstract**: We derive a deterministic, non-asymptotic upper bound on the Kullback-Leibler (KL) divergence of the flow-matching distribution approximation. In particular, if the $L_2$ flow-matching loss is bounded by $\epsilon^2 > 0$, then the KL divergence between the true data distribution and the estimated distribution is bounded by $A_1 \epsilon + A_2 \epsilon^2$. Here, the constants $A_1$ and $A_2$ depend only on the regularities of the data and velocity fields. Consequently, this bound implies statistical convergence rates of Flow Matching Transformers under the Total Variation (TV) distance. We show that, flow matching achieves nearly minimax-optimal efficiency in estimating smooth distributions. Our results make the statistical efficiency of flow matching comparable to that of diffusion models under the TV distance. Numerical studies on synthetic and learned velocities corroborate our theory. 

**Abstract (ZH)**: 我们推导出了流动匹配分布逼近的Kullback-Leibler（KL）散度的一个确定性、非渐近上界。特别地，如果$L_2$流动匹配损失受限于$\epsilon^2 > 0$，那么真实数据分布与估计分布之间的KL散度受限于$A_1 \epsilon + A_2 \epsilon^2$。这里，常数$A_1$和$A_2$仅取决于数据和速度场的正则性。因此，这一界意味着流动匹配变压器在Total Variation（TV）距离下的统计收敛速率。我们证明流动匹配在估计光滑分布时几乎达到最小最大最优效率。我们的结果使得流动匹配在TV距离下的统计效率与扩散模型相当。合成和学习的速度数值研究证实了我们的理论。 

---
# AI Literacy Assessment Revisited: A Task-Oriented Approach Aligned with Real-world Occupations 

**Title (ZH)**: AI素养评估重访：面向实际职业的任务导向方法 

**Authors**: Christopher Bogart, Aparna Warrier, Arav Agarwal, Ross Higashi, Yufan Zhang, Jesse Flot, Jaromir Savelka, Heather Burte, Majd Sakr  

**Link**: [PDF](https://arxiv.org/pdf/2511.05475)  

**Abstract**: As artificial intelligence (AI) systems become ubiquitous in professional contexts, there is an urgent need to equip workers, often with backgrounds outside of STEM, with the skills to use these tools effectively as well as responsibly, that is, to be AI literate. However, prevailing definitions and therefore assessments of AI literacy often emphasize foundational technical knowledge, such as programming, mathematics, and statistics, over practical knowledge such as interpreting model outputs, selecting tools, or identifying ethical concerns. This leaves a noticeable gap in assessing someone's AI literacy for real-world job use. We propose a work-task-oriented assessment model for AI literacy which is grounded in the competencies required for effective use of AI tools in professional settings. We describe the development of a novel AI literacy assessment instrument, and accompanying formative assessments, in the context of a US Navy robotics training program. The program included training in robotics and AI literacy, as well as a competition with practical tasks and a multiple choice scenario task meant to simulate use of AI in a job setting. We found that, as a measure of applied AI literacy, the competition's scenario task outperformed the tests we adopted from past research or developed ourselves. We argue that when training people for AI-related work, educators should consider evaluating them with instruments that emphasize highly contextualized practical skills rather than abstract technical knowledge, especially when preparing workers without technical backgrounds for AI-integrated roles. 

**Abstract (ZH)**: 随着人工智能（AI）系统在专业领域中的普及，迫切需要为往往背景非STEM领域的工作者提供技能培训，使他们能够有效地且负责任地使用这些工具，即具备AI素养。然而，现有对AI素养的定义和评估往往侧重于基础的技术知识，如编程、数学和统计，而忽视了诸如解释模型输出、选择工具或识别伦理问题等实践知识。这在评估实际工作场景中的AI素养时留下了明显的不足。我们提出了一种基于工作任务的AI素养评估模型，该模型立足于在专业环境中有效使用AI工具所需的技能。我们在美国海军机器人训练项目中开发了一种新颖的AI素养评估工具及其配套的形成性评估。该项目包括机器人和AI素养的培训，以及包含实际任务和模拟工作场景选择题的任务竞赛。我们发现，在衡量应用AI素养方面，竞赛中的情景任务优于我们从以往研究中采用或自己开发的测试工具。我们认为，在培训从事AI相关工作的人员时，教育者应考虑使用强调高度具体实践技能的评估工具，尤其是在为缺乏技术背景的人员准备AI集成角色时。 

---
# SWE-Compass: Towards Unified Evaluation of Agentic Coding Abilities for Large Language Models 

**Title (ZH)**: SWE-Compass: 向统一评估大型语言模型自主编码能力的方向努力 

**Authors**: Jingxuan Xu, Ken Deng, Weihao Li, Songwei Yu, Huaixi Tang, Haoyang Huang, Zhiyi Lai, Zizheng Zhan, Yanan Wu, Chenchen Zhang, Kepeng Lei, Yifan Yao, Xinping Lei, Wenqiang Zhu, Zongxian Feng, Han Li, Junqi Xiong, Dailin Li, Zuchen Gao, Kun Wu, Wen Xiang, Ziqi Zhan, Yuanxing Zhang, Wuxuan Gong, Ziyuan Gao, Guanxiang Wang, Yirong Xue, Xiaojiang Zhang, Jinghui Wang, Huiming Wang, Wenhao Zhuang, Zhaoxiang Zhang, Yuqun Zhang, Haotian Zhang, Bin Chen, Jiaheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05459)  

**Abstract**: Evaluating large language models (LLMs) for software engineering has been limited by narrow task coverage, language bias, and insufficient alignment with real-world developer workflows. Existing benchmarks often focus on algorithmic problems or Python-centric bug fixing, leaving critical dimensions of software engineering underexplored. To address these gaps, we introduce SWE-Compass1, a comprehensive benchmark that unifies heterogeneous code-related evaluations into a structured and production-aligned framework. SWE-Compass spans 8 task types, 8 programming scenarios, and 10 programming languages, with 2000 high-quality instances curated from authentic GitHub pull requests and refined through systematic filtering and validation. We benchmark ten state-of-the-art LLMs under two agentic frameworks, SWE-Agent and Claude Code, revealing a clear hierarchy of difficulty across task types, languages, and scenarios. Moreover, by aligning evaluation with real-world developer practices, SWE-Compass provides a rigorous and reproducible foundation for diagnosing and advancing agentic coding capabilities in large language models. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）在软件工程中的应用受限于任务覆盖范围狭窄、语言偏差以及与实际开发者工作流的不充分对齐。现有基准测试往往集中于算法问题或以Python为主的 bug 修复，导致软件工程的关键维度被忽略。为解决这些问题，我们引入了 SWE-Compass1，这是一种全面的基准测试，将异构代码相关评估统一到一个结构化且与生产环境对齐的框架中。SWE-Compass覆盖了8种任务类型、8种编程场景和10种编程语言，包括2000个高质量实例，这些实例来源于真实的GitHub拉取请求，并经过系统筛选和验证。我们使用SWE-Agent和Claude Code两种代理框架对10种最先进的LLM进行了基准测试，揭示了任务类型、语言和场景的清晰难度等级。此外，通过将评估与实际开发者实践对齐，SWE-Compass为诊断和提升大型语言模型的代理编码能力提供了严谨且可重复的基础。 

---
# Self-adaptive weighting and sampling for physics-informed neural networks 

**Title (ZH)**: 自适应加权和采样对于物理导向神经网络 

**Authors**: Wenqian Chen, Amanda Howard, Panos Stinis  

**Link**: [PDF](https://arxiv.org/pdf/2511.05452)  

**Abstract**: Physics-informed deep learning has emerged as a promising framework for solving partial differential equations (PDEs). Nevertheless, training these models on complex problems remains challenging, often leading to limited accuracy and efficiency. In this work, we introduce a hybrid adaptive sampling and weighting method to enhance the performance of physics-informed neural networks (PINNs). The adaptive sampling component identifies training points in regions where the solution exhibits rapid variation, while the adaptive weighting component balances the convergence rate across training points. Numerical experiments show that applying only adaptive sampling or only adaptive weighting is insufficient to consistently achieve accurate predictions, particularly when training points are scarce. Since each method emphasizes different aspects of the solution, their effectiveness is problem dependent. By combining both strategies, the proposed framework consistently improves prediction accuracy and training efficiency, offering a more robust approach for solving PDEs with PINNs. 

**Abstract (ZH)**: 基于物理的知识驱动深度学习已成为解决偏微分方程(PDEs)的有前景框架。然而，训练这些模型在复杂问题上仍具有挑战性，通常导致精度和效率受限。在本文中，我们提出了一种混合自适应采样和加权方法以提高物理知情神经网络(PINNs)的性能。自适应采样组件在解决方案快速变化的区域识别训练点，而自适应加权组件在训练点之间平衡收敛率。数值实验表明，仅使用自适应采样或仅使用自适应加权不足以在训练点稀缺时始终实现准确的预测。由于每种方法强调解决方案的不同方面，它们的有效性取决于问题。通过结合这两种策略，所提出的方法能够一致地提高预测精度和训练效率，为使用PINNs求解PDEs提供更稳健的方法。 

---
# APP: Accelerated Path Patching with Task-Specific Pruning 

**Title (ZH)**: APP: 加速路径修补与任务特定剪枝 

**Authors**: Frauke Andersen, William Rudman, Ruochen Zhang, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2511.05442)  

**Abstract**: Circuit discovery is a key step in many mechanistic interpretability pipelines. Current methods, such as Path Patching, are computationally expensive and have limited in-depth circuit analysis for smaller models. In this study, we propose Accelerated Path Patching (APP), a hybrid approach leveraging our novel contrastive attention head pruning method to drastically reduce the search space of circuit discovery methods. Our Contrastive-FLAP pruning algorithm uses techniques from causal mediation analysis to assign higher pruning scores to task-specific attention heads, leading to higher performing sparse models compared to traditional pruning techniques. Although Contrastive-FLAP is successful at preserving task-specific heads that existing pruning algorithms remove at low sparsity ratios, the circuits found by Contrastive-FLAP alone are too large to satisfy the minimality constraint required in circuit analysis. APP first applies Contrastive-FLAP to reduce the search space on required for circuit discovery algorithms by, on average, 56\%. Next, APP, applies traditional Path Patching on the remaining attention heads, leading to a speed up of 59.63\%-93.27\% compared to Path Patching applied to the dense model. Despite the substantial computational saving that APP provides, circuits obtained from APP exhibit substantial overlap and similar performance to previously established Path Patching circuits 

**Abstract (ZH)**: 加速路径修补：一种用于电路发现的速度与准确性的平衡方法 

---
# "I Like That You Have to Poke Around": Instructors on How Experiential Approaches to AI Literacy Spark Inquiry and Critical Thinking 

**Title (ZH)**: “我喜欢你需要探索一下”：关于体验式AI素养教学激发探究与批判性思维的教学体验 

**Authors**: Aparna Maya Warrier, Arav Agarwal, Jaromir Savelka, Christopher Bogart, Heather Burte  

**Link**: [PDF](https://arxiv.org/pdf/2511.05430)  

**Abstract**: As artificial intelligence (AI) increasingly shapes decision-making across domains, there is a growing need to support AI literacy among learners beyond computer science. However, many current approaches rely on programming-heavy tools or abstract lecture-based content, limiting accessibility for non-STEM audiences. This paper presents findings from a study of AI User, a modular, web-based curriculum that teaches core AI concepts through interactive, no-code projects grounded in real-world scenarios. The curriculum includes eight projects; this study focuses on instructor feedback on Projects 5-8, which address applied topics such as natural language processing, computer vision, decision support, and responsible AI. Fifteen community college instructors participated in structured focus groups, completing the projects as learners and providing feedback through individual reflection and group discussion. Using thematic analysis, we examined how instructors evaluated the design, instructional value, and classroom applicability of these experiential activities. Findings highlight instructors' appreciation for exploratory tasks, role-based simulations, and real-world relevance, while also surfacing design trade-offs around cognitive load, guidance, and adaptability for diverse learners. This work extends prior research on AI literacy by centering instructor perspectives on teaching complex AI topics without code. It offers actionable insights for designing inclusive, experiential AI learning resources that scale across disciplines and learner backgrounds. 

**Abstract (ZH)**: 人工智能用户：一种模块化的基于交互式项目的人工智能课程设计与应用研究 

---
# ProDER: A Continual Learning Approach for Fault Prediction in Evolving Smart Grids 

**Title (ZH)**: ProDER： evolving智能电网中故障预测的持续学习方法 

**Authors**: Emad Efatinasab, Nahal Azadi, Davide Dalle Pezze, Gian Antonio Susto, Chuadhry Mujeeb Ahmed, Mirco Rampazzo  

**Link**: [PDF](https://arxiv.org/pdf/2511.05420)  

**Abstract**: As smart grids evolve to meet growing energy demands and modern operational challenges, the ability to accurately predict faults becomes increasingly critical. However, existing AI-based fault prediction models struggle to ensure reliability in evolving environments where they are required to adapt to new fault types and operational zones. In this paper, we propose a continual learning (CL) framework in the smart grid context to evolve the model together with the environment. We design four realistic evaluation scenarios grounded in class-incremental and domain-incremental learning to emulate evolving grid conditions. We further introduce Prototype-based Dark Experience Replay (ProDER), a unified replay-based approach that integrates prototype-based feature regularization, logit distillation, and a prototype-guided replay memory. ProDER achieves the best performance among tested CL techniques, with only a 0.045 accuracy drop for fault type prediction and 0.015 for fault zone prediction. These results demonstrate the practicality of CL for scalable, real-world fault prediction in smart grids. 

**Abstract (ZH)**: 基于智能电网的持续学习框架在故障预测中的应用 

---
# Multi-modal Loop Closure Detection with Foundation Models in Severely Unstructured Environments 

**Title (ZH)**: 严重无序环境中基于多模态基础模型的回环检测 

**Authors**: Laura Alejandra Encinar Gonzalez, John Folkesson, Rudolph Triebel, Riccardo Giubilato  

**Link**: [PDF](https://arxiv.org/pdf/2511.05404)  

**Abstract**: Robust loop closure detection is a critical component of Simultaneous Localization and Mapping (SLAM) algorithms in GNSS-denied environments, such as in the context of planetary exploration. In these settings, visual place recognition often fails due to aliasing and weak textures, while LiDAR-based methods suffer from sparsity and ambiguity. This paper presents MPRF, a multimodal pipeline that leverages transformer-based foundation models for both vision and LiDAR modalities to achieve robust loop closure in severely unstructured environments. Unlike prior work limited to retrieval, MPRF integrates a two-stage visual retrieval strategy with explicit 6-DoF pose estimation, combining DINOv2 features with SALAD aggregation for efficient candidate screening and SONATA-based LiDAR descriptors for geometric verification. Experiments on the S3LI dataset and S3LI Vulcano dataset show that MPRF outperforms state-of-the-art retrieval methods in precision while enhancing pose estimation robustness in low-texture regions. By providing interpretable correspondences suitable for SLAM back-ends, MPRF achieves a favorable trade-off between accuracy, efficiency, and reliability, demonstrating the potential of foundation models to unify place recognition and pose estimation. Code and models will be released at this http URL. 

**Abstract (ZH)**: GNSS受限环境下鲁棒环回闭合检测在行星探测中同步定位与建图算法中的关键作用：多模态管道MPRF及其应用 

---
# Robust Neural Audio Fingerprinting using Music Foundation Models 

**Title (ZH)**: 鲁棒的音乐基础模型驱动的音频指纹识别 

**Authors**: Shubhr Singh, Kiran Bhat, Xavier Riley, Benjamin Resnick, John Thickstun, Walter De Brouwer  

**Link**: [PDF](https://arxiv.org/pdf/2511.05399)  

**Abstract**: The proliferation of distorted, compressed, and manipulated music on modern media platforms like TikTok motivates the development of more robust audio fingerprinting techniques to identify the sources of musical recordings. In this paper, we develop and evaluate new neural audio fingerprinting techniques with the aim of improving their robustness. We make two contributions to neural fingerprinting methodology: (1) we use a pretrained music foundation model as the backbone of the neural architecture and (2) we expand the use of data augmentation to train fingerprinting models under a wide variety of audio manipulations, including time streching, pitch modulation, compression, and filtering. We systematically evaluate our methods in comparison to two state-of-the-art neural fingerprinting models: NAFP and GraFPrint. Results show that fingerprints extracted with music foundation models (e.g., MuQ, MERT) consistently outperform models trained from scratch or pretrained on non-musical audio. Segment-level evaluation further reveals their capability to accurately localize fingerprint matches, an important practical feature for catalog management. 

**Abstract (ZH)**: 现代媒体平台如TikTok上泛滥的扭曲、压缩和篡改音乐促使发展更为 robust 的音频指纹技术以识别音乐录音的来源。本文开发并评估了新的神经音频指纹技术，旨在提高其 robustness。我们在神经指纹方法论中做出了两项贡献：（1）使用预训练的音乐基础模型作为神经架构的骨干，并（2）扩展数据增强的使用，以在各种音频操作（包括时间拉伸、音调调制、压缩和滤波）下训练指纹模型。我们将方法系统性地与两种最先进的神经指纹模型（NAFP和GraFPrint）进行了比较评估。结果表明，使用音乐基础模型提取的指纹（如MuQ、MERT）始终优于从头训练或基于非音乐音频预训练的模型。段落级别评估还显示了其准确定位指纹匹配的能力，这是目录管理中的一个重要实用特征。 

---
# Sample Complexity of Distributionally Robust Off-Dynamics Reinforcement Learning with Online Interaction 

**Title (ZH)**: 分布鲁棒离动力强化学习的在线交互样本复杂性 

**Authors**: Yiting He, Zhishuai Liu, Weixin Wang, Pan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05396)  

**Abstract**: Off-dynamics reinforcement learning (RL), where training and deployment transition dynamics are different, can be formulated as learning in a robust Markov decision process (RMDP) where uncertainties in transition dynamics are imposed. Existing literature mostly assumes access to generative models allowing arbitrary state-action queries or pre-collected datasets with a good state coverage of the deployment environment, bypassing the challenge of exploration. In this work, we study a more realistic and challenging setting where the agent is limited to online interaction with the training environment. To capture the intrinsic difficulty of exploration in online RMDPs, we introduce the supremal visitation ratio, a novel quantity that measures the mismatch between the training dynamics and the deployment dynamics. We show that if this ratio is unbounded, online learning becomes exponentially hard. We propose the first computationally efficient algorithm that achieves sublinear regret in online RMDPs with $f$-divergence based transition uncertainties. We also establish matching regret lower bounds, demonstrating that our algorithm achieves optimal dependence on both the supremal visitation ratio and the number of interaction episodes. Finally, we validate our theoretical results through comprehensive numerical experiments. 

**Abstract (ZH)**: 离线动力学强化学习(RL)中的鲁棒马尔可夫决策过程(RMDP)：探索挑战与在线高效算法 

---
# AI Assisted AR Assembly: Object Recognition and Computer Vision for Augmented Reality Assisted Assembly 

**Title (ZH)**: AI辅助AR装配：物体识别与增强现实装配中的计算机视觉 

**Authors**: Alexander Htet Kyaw, Haotian Ma, Sasa Zivkovic, Jenny Sabin  

**Link**: [PDF](https://arxiv.org/pdf/2511.05394)  

**Abstract**: We present an AI-assisted Augmented Reality assembly workflow that uses deep learning-based object recognition to identify different assembly components and display step-by-step instructions. For each assembly step, the system displays a bounding box around the corresponding components in the physical space, and where the component should be placed. By connecting assembly instructions with the real-time location of relevant components, the system eliminates the need for manual searching, sorting, or labeling of different components before each assembly. To demonstrate the feasibility of using object recognition for AR-assisted assembly, we highlight a case study involving the assembly of LEGO sculptures. 

**Abstract (ZH)**: 基于深度学习的物体识别的AI辅助增强现实装配工作流：以LEGO雕塑装配案例研究为例 

---
# TeaRAG: A Token-Efficient Agentic Retrieval-Augmented Generation Framework 

**Title (ZH)**: TeaRAG: 一种高效的代理检索增强生成框架 

**Authors**: Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen, Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05385)  

**Abstract**: Retrieval-Augmented Generation (RAG) utilizes external knowledge to augment Large Language Models' (LLMs) reliability. For flexibility, agentic RAG employs autonomous, multi-round retrieval and reasoning to resolve queries. Although recent agentic RAG has improved via reinforcement learning, they often incur substantial token overhead from search and reasoning processes. This trade-off prioritizes accuracy over efficiency. To address this issue, this work proposes TeaRAG, a token-efficient agentic RAG framework capable of compressing both retrieval content and reasoning steps. 1) First, the retrieved content is compressed by augmenting chunk-based semantic retrieval with a graph retrieval using concise triplets. A knowledge association graph is then built from semantic similarity and co-occurrence. Finally, Personalized PageRank is leveraged to highlight key knowledge within this graph, reducing the number of tokens per retrieval. 2) Besides, to reduce reasoning steps, Iterative Process-aware Direct Preference Optimization (IP-DPO) is proposed. Specifically, our reward function evaluates the knowledge sufficiency by a knowledge matching mechanism, while penalizing excessive reasoning steps. This design can produce high-quality preference-pair datasets, supporting iterative DPO to improve reasoning conciseness. Across six datasets, TeaRAG improves the average Exact Match by 4% and 2% while reducing output tokens by 61% and 59% on Llama3-8B-Instruct and Qwen2.5-14B-Instruct, respectively. Code is available at this https URL. 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)利用外部知识增强大型语言模型的可靠性。为了提高灵活性，代理RAG采用自主多轮检索和推理来解决问题。尽管最近的代理RAG通过强化学习有所改进，但在检索和推理过程中通常会产生大量的token开销，这种权衡优先考虑准确性而牺牲了效率。为了解决这一问题，本工作提出了TeaRAG，这是一种token高效的代理RAG框架，能够同时压缩检索内容和推理步骤。1) 首先，通过将基于块的语义检索与简洁三元组的图检索相结合来压缩检索内容。然后，根据语义相似性和共现性构建知识关联图。最后，利用个性化PageRank突出显示图中的关键知识，从而减少每次检索的token数量。2) 此外，为了减少推理步骤，提出了迭代过程感知直接偏好优化（IP-DPO）。具体来说，我们的奖励函数通过知识匹配机制评估知识充分性，并对过多的推理步骤进行惩罚。这种设计能够生成高质量的偏好对数据集，支持迭代DPO提高推理的简洁性。在六个数据集上，TeaRAG分别将Llama3-8B-Instruct和Qwen2.5-14B-Instruct的精确匹配均值提高了4%和2%，同时分别减少了61%和59%的输出token。代码可通过此网站获得。 

---
# AI Literacy for Community Colleges: Instructors' Perspectives on Scenario-Based and Interactive Approaches to Teaching AI 

**Title (ZH)**: 社区学院的AI素养培养：基于场景和互动教学方法的教员视角 

**Authors**: Aparna Maya Warrier, Arav Agarwal, Jaromir Savelka, Christopher A Bogart, Heather Burte  

**Link**: [PDF](https://arxiv.org/pdf/2511.05363)  

**Abstract**: This research category full paper investigates how community college instructors evaluate interactive, no-code AI literacy resources designed for non-STEM learners. As artificial intelligence becomes increasingly integrated into everyday technologies, AI literacy - the ability to evaluate AI systems, communicate with them, and understand their broader impacts - has emerged as a critical skill across disciplines. Yet effective, scalable approaches for teaching these concepts in higher education remain limited, particularly for students outside STEM fields.
To address this gap, we developed AI User, an interactive online curriculum that introduces core AI concepts through scenario - based activities set in real - world contexts. This study presents findings from four focus groups with instructors who engaged with AI User materials and participated in structured feedback activities. Thematic analysis revealed that instructors valued exploratory tasks that simulated real - world AI use cases and fostered experimentation, while also identifying challenges related to scaffolding, accessibility, and multi-modal support. A ranking task for instructional support materials showed a strong preference for interactive demonstrations over traditional educational materials like conceptual guides or lecture slides.
These findings offer insights into instructor perspectives on making AI concepts more accessible and relevant for broad learner audiences. They also inform the design of AI literacy tools that align with diverse teaching contexts and support critical engagement with AI in higher education. 

**Abstract (ZH)**: 本研究类论文探讨了社区学院教师如何评估为非 STEM 学习者设计的互动式无代码 AI 文盲资源。随着人工智能逐渐融入日常技术中，AI 文盲——即评估 AI 系统、与之交流以及理解其更广泛影响的能力——已成为跨学科的关键技能。然而，高等教育中有效、可扩展的方法仍有限，特别是在 STEM 领域以外的学生中。

为解决这一缺口，我们开发了 AI 用户，一种互动式的在线课程，通过基于实际场景的活动介绍了核心 AI 概念。本研究呈现了四次以教师为中心的焦点小组的研究成果，这些教师接触了 AI 用户的材料并参与了结构化的反馈活动。主题分析显示，教师重视模拟真实世界 AI 使用案例的探索性任务，并促进了实验，同时也指出了关于支架、无障碍性和多模态支持的相关挑战。对于教学支持材料的排序任务表明，师生更偏好互动演示而非传统的教育材料如概念指南或讲义幻灯片。

这些发现为如何使 AI 概念对于更广泛的学习者群体更具可访问性和相关性提供了教师视角的见解。它们还为设计与多样化的教学情境相契合的 AI 文盲工具提供了指导，并支持在高等教育中对 AI 的批判性参与。 

---
# A multimodal multiplex of the mental lexicon for multilingual individuals 

**Title (ZH)**: 多模态多重语境中的心理词典构建研究：面向多语言个体 

**Authors**: Maria Huynh, Wilder C. Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2511.05361)  

**Abstract**: Historically, bilingualism was often perceived as an additional cognitive load that could hinder linguistic and intellectual development. However, over the last three decades, this view has changed considerably. Numerous studies have aimed to model and understand the architecture of the bilingual word recognition system Dijkstra and van Heuven (2002), investigating how parallel activation operates in the brain and how one language influences another Kroll et al. (2015). Increasingly, evidence suggests that multilinguals, individuals who speak three or more languages, can perform better than monolinguals in various linguistic and cognitive tasks, such as learning an additional language Abu-Rabia and Sanitsky (2010). This research proposal focuses on the study of the mental lexicon and how it may be structured in individuals who speak multiple languages. Building on the work of Stella et al. (2018), who investigated explosive learning in humans using a multiplex model of the mental lexicon, and the Bilingual Interactive Activation (BIA+) framework proposed by Dijkstra and van Heuven (2002), the present study applies the same multilayer network principles introduced by Kivela et al. (2014). Our experimental design extends previous research by incorporating multimodality into the multiplex model, introducing an additional layer that connects visual inputs to their corresponding lexical representations across the multilingual layers of the mental lexicon. In this research, we aim to explore how a heritage language influences the acquisition of another language. Specifically, we ask: Does the presence of visual input in a translation task influence participants' proficiency and accuracy compared to text-only conditions? 

**Abstract (ZH)**: 历史上传统认为双语会增加认知负担，从而阻碍语言和智力发展。然而，在过去的三十年里，这一观点已经发生了显著变化。众多研究致力于模式化和理解双语词汇识别系统的架构Dijkstra和van Heuven（2002），探究并行激活在大脑中的运作方式以及一种语言如何影响另一种语言Kroll等（2015）。越来越多的证据表明，掌握三种或以上语言的多语者在各种语言和认知任务上表现优于单一语言者，例如学习额外的语言Abu-Rabia和Sanitsky（2010）。本研究提案专注于多语言者心理词典的研究及其可能的结构。基于Stella等（2018）利用多复层模型研究人类爆炸性学习的工作，以及Dijkstra和van Heuven（2002）提出的双语交互激活（BIA+）框架，并借鉴Kivela等（2014）提出的多层网络原理，本研究将多模态引入复层模型中，增加了一层将视觉输入连接到多语言心理词典相应词汇表征的机制。本研究旨在探讨母语如何影响另一种语言的习得。具体而言，我们的问题是：翻译任务中的视觉输入是否会增加参与者的表现水平和准确度，与仅使用文本的条件相比？ 

---
# Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders 

**Title (ZH)**: 基于噪声增强自编码器的音乐表示感知对齐 

**Authors**: Mathias Rose Bjare, Giorgia Cantisani, Marco Pasini, Stefan Lattner, Gerhard Widmer  

**Link**: [PDF](https://arxiv.org/pdf/2511.05350)  

**Abstract**: We argue that training autoencoders to reconstruct inputs from noised versions of their encodings, when combined with perceptual losses, yields encodings that are structured according to a perceptual hierarchy. We demonstrate the emergence of this hierarchical structure by showing that, after training an audio autoencoder in this manner, perceptually salient information is captured in coarser representation structures than with conventional training. Furthermore, we show that such perceptual hierarchies improve latent diffusion decoding in the context of estimating surprisal in music pitches and predicting EEG-brain responses to music listening. Pretrained weights are available on this http URL. 

**Abstract (ZH)**: 我们将训练自编码器从噪声版本的编码重建输入与感知损失结合使用的过程论证为生成根据感知层次结构组织的编码。通过训练一种音频自编码器来展示这种层次结构的 emergence，我们表明，在这种方式下训练后，感知上显著的信息在较粗的表示结构中被捕获，这与常规训练相比有所不同。此外，我们还展示了这种感知层次结构在音乐音高似然估计和预测音乐聆听后的EEG-脑响应方面通过潜在扩散解码的改进效果。预训练权重可在以下网址获取：。 

---
# What Are the Facts? Automated Extraction of Court-Established Facts from Criminal-Court Opinions 

**Title (ZH)**: 法院确立的事实是什么？自动提取刑事法院判决中确立的事实 

**Authors**: Klára Bendová, Tomáš Knap, Jan Černý, Vojtěch Pour, Jaromir Savelka, Ivana Kvapilíková, Jakub Drápal  

**Link**: [PDF](https://arxiv.org/pdf/2511.05320)  

**Abstract**: Criminal justice administrative data contain only a limited amount of information about the committed offense. However, there is an unused source of extensive information in continental European courts' decisions: descriptions of criminal behaviors in verdicts by which offenders are found guilty. In this paper, we study the feasibility of extracting these descriptions from publicly available court decisions from Slovakia. We use two different approaches for retrieval: regular expressions and large language models (LLMs). Our baseline was a simple method employing regular expressions to identify typical words occurring before and after the description. The advanced regular expression approach further focused on "sparing" and its normalization (insertion of spaces between individual letters), typical for delineating the description. The LLM approach involved prompting the Gemini Flash 2.0 model to extract the descriptions using predefined instructions. Although the baseline identified descriptions in only 40.5% of verdicts, both methods significantly outperformed it, achieving 97% with advanced regular expressions and 98.75% with LLMs, and 99.5% when combined. Evaluation by law students showed that both advanced methods matched human annotations in about 90% of cases, compared to just 34.5% for the baseline. LLMs fully matched human-labeled descriptions in 91.75% of instances, and a combination of advanced regular expressions with LLMs reached 92%. 

**Abstract (ZH)**: 大陆欧洲法院判决中关于犯罪行为描述的提取：基于正则表达式和大型语言模型的方法研究 

---
# Rethinking Metrics and Diffusion Architecture for 3D Point Cloud Generation 

**Title (ZH)**: 重思用于3D点云生成的度量标准和扩散架构 

**Authors**: Matteo Bastico, David Ryckelynck, Laurent Corté, Yannick Tillier, Etienne Decencière  

**Link**: [PDF](https://arxiv.org/pdf/2511.05308)  

**Abstract**: As 3D point clouds become a cornerstone of modern technology, the need for sophisticated generative models and reliable evaluation metrics has grown exponentially. In this work, we first expose that some commonly used metrics for evaluating generated point clouds, particularly those based on Chamfer Distance (CD), lack robustness against defects and fail to capture geometric fidelity and local shape consistency when used as quality indicators. We further show that introducing samples alignment prior to distance calculation and replacing CD with Density-Aware Chamfer Distance (DCD) are simple yet essential steps to ensure the consistency and robustness of point cloud generative model evaluation metrics. While existing metrics primarily focus on directly comparing 3D Euclidean coordinates, we present a novel metric, named Surface Normal Concordance (SNC), which approximates surface similarity by comparing estimated point normals. This new metric, when combined with traditional ones, provides a more comprehensive evaluation of the quality of generated samples. Finally, leveraging recent advancements in transformer-based models for point cloud analysis, such as serialized patch attention , we propose a new architecture for generating high-fidelity 3D structures, the Diffusion Point Transformer. We perform extensive experiments and comparisons on the ShapeNet dataset, showing that our model outperforms previous solutions, particularly in terms of quality of generated point clouds, achieving new state-of-the-art. Code available at this https URL. 

**Abstract (ZH)**: 随着3D点云成为现代技术的基石，对复杂的生成模型和可靠的评估指标的需求呈指数级增长。在本文中，我们首先揭示了某些常用用于评估生成点云的指标，特别是基于Chamfer距离（CD）的指标，在面对缺陷时缺乏鲁棒性，并且在作为质量指标时无法捕捉几何保真度和局部形状一致性。我们进一步表明，在进行距离计算之前引入样本对齐以及用密度感知Chamfer距离（DCD）取代CD是确保点云生成模型评估指标一致性和鲁棒性的简单但至关重要的步骤。虽然现有的指标主要侧重于直接比较3D欧几里得坐标，我们提出了一种新的指标，名为表面法线一致性（SNC），它通过比较估计的点法线来近似表面相似性。该新指标与传统指标结合使用，为生成样本的质量提供了更为全面的评估。最后，利用基于变换器的点云分析的最新进展，如序贯块注意力，我们提出了一种生成高保真3D结构的新架构，即扩散点变换器（Diffusion Point Transformer）。我们在Shapenet数据集上进行了广泛的实验和比较，结果显示我们的模型在生成点云的质量方面优于之前的解决方案，达到了新的最先进水平。代码可在如下链接获取：this https URL。 

---
# LiveStar: Live Streaming Assistant for Real-World Online Video Understanding 

**Title (ZH)**: LiveStar: 实时直播助手 for 实际应用场景中的在线视频理解 

**Authors**: Zhenyu Yang, Kairui Zhang, Yuhang Hu, Bing Wang, Shengsheng Qian, Bin Wen, Fan Yang, Tingting Gao, Weiming Dong, Changsheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05299)  

**Abstract**: Despite significant progress in Video Large Language Models (Video-LLMs) for offline video understanding, existing online Video-LLMs typically struggle to simultaneously process continuous frame-by-frame inputs and determine optimal response timing, often compromising real-time responsiveness and narrative coherence. To address these limitations, we introduce LiveStar, a pioneering live streaming assistant that achieves always-on proactive responses through adaptive streaming decoding. Specifically, LiveStar incorporates: (1) a training strategy enabling incremental video-language alignment for variable-length video streams, preserving temporal consistency across dynamically evolving frame sequences; (2) a response-silence decoding framework that determines optimal proactive response timing via a single forward pass verification; (3) memory-aware acceleration via peak-end memory compression for online inference on 10+ minute videos, combined with streaming key-value cache to achieve 1.53x faster inference. We also construct an OmniStar dataset, a comprehensive dataset for training and benchmarking that encompasses 15 diverse real-world scenarios and 5 evaluation tasks for online video understanding. Extensive experiments across three benchmarks demonstrate LiveStar's state-of-the-art performance, achieving an average 19.5% improvement in semantic correctness with 18.1% reduced timing difference compared to existing online Video-LLMs, while improving FPS by 12.0% across all five OmniStar tasks. Our model and dataset can be accessed at this https URL. 

**Abstract (ZH)**: 尽管在离线视频理解方面取得了显著进展，现有的在线Video Large Language Models（Video-LLMs）通常难以同时处理连续的逐帧输入并确定最优响应时机，常常牺牲实时响应性和叙事连贯性。为了应对这些限制，我们引入了LiveStar，这是一种创新的直播助手，通过自适应流解码实现始终如一的主动响应。具体来说，LiveStar融合了：（1）一种训练策略，实现变长视频流的增量视频语言对齐，保持动态演变帧序列间的时序一致性；（2）一种响应静默解码框架，通过单次前向验证确定最优主动响应时机；（3）一种基于峰值尾部记忆压缩的内存感知加速技术，结合流式键值缓存，实现10分钟以上视频的在线推理加速1.53倍。我们还构建了一个全方位的OmniStar数据集，这是一个包含15种多样的真实场景和5项评估任务的全面数据集，用于在线视频理解的训练和基准测试。在三个基准上的广泛实验表明，LiveStar实现了最先进的性能，与现有的在线Video-LLMs相比，语义准确性平均提高19.5%，响应时间差减少18.1%，同时在所有五个OmniStar任务中的FPS提高12.0%。我们的模型和数据集可以在以下链接访问。 

---
# DeepEyesV2: Toward Agentic Multimodal Model 

**Title (ZH)**: DeepEyesV2: 向自主多模态模型迈进 

**Authors**: Jack Hong, Chenxiao Zhao, ChengLin Zhu, Weiheng Lu, Guohai Xu, Xing Yu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05271)  

**Abstract**: Agentic multimodal models should not only comprehend text and images, but also actively invoke external tools, such as code execution environments and web search, and integrate these operations into reasoning. In this work, we introduce DeepEyesV2 and explore how to build an agentic multimodal model from the perspectives of data construction, training methods, and model evaluation. We observe that direct reinforcement learning alone fails to induce robust tool-use behavior. This phenomenon motivates a two-stage training pipeline: a cold-start stage to establish tool-use patterns, and reinforcement learning stage to further refine tool invocation. We curate a diverse, moderately challenging training dataset, specifically including examples where tool use is beneficial. We further introduce RealX-Bench, a comprehensive benchmark designed to evaluate real-world multimodal reasoning, which inherently requires the integration of multiple capabilities, including perception, search, and reasoning. We evaluate DeepEyesV2 on RealX-Bench and other representative benchmarks, demonstrating its effectiveness across real-world understanding, mathematical reasoning, and search-intensive tasks. Moreover, DeepEyesV2 exhibits task-adaptive tool invocation, tending to use image operations for perception tasks and numerical computations for reasoning tasks. Reinforcement learning further enables complex tool combinations and allows model to selectively invoke tools based on context. We hope our study can provide guidance for community in developing agentic multimodal models. 

**Abstract (ZH)**: 代理多模态模型不仅应理解文本和图像，还应主动调用外部工具，如代码执行环境和网络搜索，并将这些操作整合到推理中。在本文中，我们介绍了DeepEyesV2，并从数据构建、训练方法和模型评估的角度探讨了如何构建代理多模态模型。我们观察到单独的强化学习直接调用工具行为效果不佳。这一现象促使我们提出两阶段的训练pipeline：冷启动阶段建立工具使用模式，强化学习阶段进一步细化工具调用。我们构建了一个多样化的、具有适度挑战性的训练数据集，特别包括了工具使用有益的例子。我们进一步引入了RealX-Bench，这是一个全面的基准测试，旨在评估多模态推理能力，涵盖了感知、搜索和推理等多个能力的综合应用。在RealX-Bench和其它代表性基准上评估DeepEyesV2，展示了其在现实理解、数学推理和检索密集任务中的有效性。此外，DeepEyesV2表现出任务适应性的工具调用，倾向于在感知任务中使用图像操作，在推理任务中使用数值计算。强化学习进一步促进了复杂工具组合，并使模型能够根据上下文选择性地调用工具。我们希望我们的研究能为代理多模态模型的开发提供指导。 

---
# TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems 

**Title (ZH)**: TAMAS：多智能体大型语言模型系统的对抗风险benchmark研究 

**Authors**: Ishan Kavathekar, Hemang Jain, Ameya Rathod, Ponnurangam Kumaraguru, Tanuja Ganu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05269)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems. 

**Abstract (ZH)**: 威胁和多agents系统中的攻击：多agents语言模型系统（TAMAS） 

---
# Integrating Score-Based Diffusion Models with Machine Learning-Enhanced Localization for Advanced Data Assimilation in Geological Carbon Storage 

**Title (ZH)**: 基于机器学习增强定位的评分基于扩散模型在地质碳储层高级数据同化中的集成应用 

**Authors**: Gabriel Serrão Seabra, Nikolaj T. Mücke, Vinicius Luiz Santos Silva, Alexandre A. Emerick, Denis Voskov, Femke Vossepoel  

**Link**: [PDF](https://arxiv.org/pdf/2511.05266)  

**Abstract**: Accurate characterization of subsurface heterogeneity is important for the safe and effective implementation of geological carbon storage (GCS) projects. This paper explores how machine learning methods can enhance data assimilation for GCS with a framework that integrates score-based diffusion models with machine learning-enhanced localization in channelized reservoirs during CO$_2$ injection. We employ a machine learning-enhanced localization framework that uses large ensembles ($N_s = 5000$) with permeabilities generated by the diffusion model and states computed by simple ML algorithms to improve covariance estimation for the Ensemble Smoother with Multiple Data Assimilation (ESMDA). We apply ML algorithms to a prior ensemble of channelized permeability fields, generated with the geostatistical model FLUVSIM. Our approach is applied on a CO$_2$ injection scenario simulated using the Delft Advanced Research Terra Simulator (DARTS). Our ML-based localization maintains significantly more ensemble variance than when localization is not applied, while achieving comparable data-matching quality. This framework has practical implications for GCS projects, helping improve the reliability of uncertainty quantification for risk assessment. 

**Abstract (ZH)**: 准确表征地下异质性对于地质碳储存(GCS)项目的安全和有效实施至关重要。本文探讨了如何通过将基于分数扩散模型的机器学习方法集成到计算通道化油藏在CO$_2$注入期间的机器学习增强定位框架中，以提高数据同化的精度。我们利用由扩散模型生成的渗透率和简单机器学习算法计算的状态的大规模集合（$N_s = 5000$）来改进成群平滑法与多重数据同化（ESMDA）的协方差估计。我们将机器学习算法应用于由地质统计模型FLUVSIM生成的先验通道化渗透率场。该方法应用于使用Delft高级研究地球模拟器（DARTS）模拟的CO$_2$注入场景。基于机器学习的定位保持了显著更多的集合变异性，同时实现了相当的数据匹配质量。该框架对GCS项目具有实际意义，有助于提高风险评估中不确定性量化结果的可靠性。 

---
# An End-to-End Deep Reinforcement Learning Approach for Solving the Traveling Salesman Problem with Drones 

**Title (ZH)**: 基于无人机的旅行商问题的端到端深度强化学习方法 

**Authors**: Taihelong Zeng, Yun Lin, Yuhe Shi, Yan Li, Zhiqing Wei, Xuanru Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.05265)  

**Abstract**: The emergence of truck-drone collaborative systems in last-mile logistics has positioned the Traveling Salesman Problem with Drones (TSP-D) as a pivotal extension of classical routing optimization, where synchronized vehicle coordination promises substantial operational efficiency and reduced environmental impact, yet introduces NP-hard combinatorial complexity beyond the reach of conventional optimization paradigms. Deep reinforcement learning offers a theoretically grounded framework to address TSP-D's inherent challenges through self-supervised policy learning and adaptive decision-making. This study proposes a hierarchical Actor-Critic deep reinforcement learning framework for solving the TSP-D problem. The architecture consists of two primary components: a Transformer-inspired encoder and an efficient Minimal Gated Unit decoder. The encoder incorporates a novel, optimized k-nearest neighbors sparse attention mechanism specifically for focusing on relevant spatial relationships, further enhanced by the integration of global node features. The Minimal Gated Unit decoder processes these encoded representations to efficiently generate solution sequences. The entire framework operates within an asynchronous advantage actor-critic paradigm. Experimental results show that, on benchmark TSP-D instances of various scales (N=10 to 100), the proposed model can obtain competitive or even superior solutions in shorter average computation times compared to high-performance heuristic algorithms and existing reinforcement learning methods. Moreover, compared to advanced reinforcement learning algorithm benchmarks, the proposed framework significantly reduces the total training time required while achieving superior final performance, highlighting its notable advantage in training efficiency. 

**Abstract (ZH)**: 基于卡车-无人机协作系统的最后一公里物流中，飞行员问题与无人机（TSP-D）成为经典路径优化问题的重要扩展，同步车辆协调有望显著提高操作效率并减少环境影响，但同时也引入了超越传统优化范式的NP难组合复杂性。深度强化学习提供了一种理论上可靠的方法来解决TSP-D固有的挑战，通过自监督策略学习和适应性决策。本研究提出了一种分层演员-评论家深度强化学习框架来解决TSP-D问题。该框架由两个主要组成部分构成：受变压器启发的编码器和高效的最小门控单元解码器。编码器嵌入了一种新型的优化k最近邻稀疏注意力机制，专门用于聚焦于相关空间关系，并通过融合全局节点特征得以增强。最小门控单元解码器处理这些编码表示，以高效生成解决方案序列。整个框架运行于异步优势演员-评论家模式下。实验结果表明，针对不同规模（N=10至100）的标准TSP-D实例，所提出模型能够在较短时间内获得竞争力甚至更优的解决方案，相较于高性能启发式算法和现有强化学习方法。此外，与先进的强化学习算法基准相比，所提出的框架显著减少了所需的总训练时间，同时实现了更优异的最终性能，凸显了其在训练效率方面的显著优势。 

---
# OregairuChar: A Benchmark Dataset for Character Appearance Frequency Analysis in My Teen Romantic Comedy SNAFU 

**Title (ZH)**: OregairuChar：一部少女浪漫喜剧《SNAFU》中角色 Appearance 频率分析基准数据集 

**Authors**: Qi Sun, Dingju Zhou, Lina Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05263)  

**Abstract**: The analysis of character appearance frequency is essential for understanding narrative structure, character prominence, and story progression in anime. In this work, we introduce OregairuChar, a benchmark dataset designed for appearance frequency analysis in the anime series My Teen Romantic Comedy SNAFU. The dataset comprises 1600 manually selected frames from the third season, annotated with 2860 bounding boxes across 11 main characters. OregairuChar captures diverse visual challenges, including occlusion, pose variation, and inter-character similarity, providing a realistic basis for appearance-based studies. To enable quantitative research, we benchmark several object detection models on the dataset and leverage their predictions for fine-grained, episode-level analysis of character presence over time. This approach reveals patterns of character prominence and their evolution within the narrative. By emphasizing appearance frequency, OregairuChar serves as a valuable resource for exploring computational narrative dynamics and character-centric storytelling in stylized media. 

**Abstract (ZH)**: 动漫《我的困境青春喜剧》中角色出场频率分析对于理解叙事结构、角色突出性和故事情节进展至关重要。本文介绍OregairuChar，这是一个用于分析《我的困境青春喜剧》动漫系列角色出场频率的基准数据集。该数据集包含第三季中1600个手动选择的帧，标注了11个主要角色共计2860个边界框。OregairuChar捕捉了多样化的视觉挑战，包括遮挡、姿势变化和角色之间的相似性，为基于外观的研究提供了现实基础。为了支持定量研究，我们在该数据集上 benchmark 了多个物体检测模型，并利用其预测结果进行精细的时间进程级别的角色存在分析。该方法揭示了角色突出性模式及其在叙述中的演变。通过强调出场频率，OregairuChar 成为探索计算叙事动力学和角色中心叙事在风格化媒体中的宝贵资源。 

---
# A Gate-Based Quantum Genetic Algorithm for Real-Valued Global Optimization 

**Title (ZH)**: 基于门操作的量子遗传算法用于实值全局优化 

**Authors**: Leandro C. Souza, Laurent E. Dardenne, Renato Portugal  

**Link**: [PDF](https://arxiv.org/pdf/2511.05254)  

**Abstract**: We propose a gate-based Quantum Genetic Algorithm (QGA) for real-valued global optimization. In this model, individuals are represented by quantum circuits whose measurement outcomes are decoded into real-valued vectors through binary discretization. Evolutionary operators act directly on circuit structures, allowing mutation and crossover to explore the space of gate-based encodings. Both fixed-depth and variable-depth variants are introduced, enabling either uniform circuit complexity or adaptive structural evolution. Fitness is evaluated through quantum sampling, using the mean decoded output of measurement outcomes as the argument of the objective function. To isolate the impact of quantum resources, we compare gate sets with and without the Hadamard gate, showing that superposition consistently improves convergence and robustness across benchmark functions such as the Rastrigin function. Furthermore, we demonstrate that introducing pairwise inter-individual entanglement in the population accelerates early convergence, revealing that quantum correlations among individuals provide an additional optimization advantage. Together, these results show that both superposition and entanglement enhance the search dynamics of evolutionary quantum algorithms, establishing gate-based QGAs as a promising framework for quantum-enhanced global optimization. 

**Abstract (ZH)**: 基于门的量子遗传算法（QGA）用于实值全局优化 

---
# Accurate online action and gesture recognition system using detectors and Deep SPD Siamese Networks 

**Title (ZH)**: 基于检测器和Deep SPD Siamese网络的准确在线动作和手势识别系统 

**Authors**: Mohamed Sanim Akremi, Rim Slama, Hedi Tabia  

**Link**: [PDF](https://arxiv.org/pdf/2511.05250)  

**Abstract**: Online continuous motion recognition is a hot topic of research since it is more practical in real life application cases. Recently, Skeleton-based approaches have become increasingly popular, demonstrating the power of using such 3D temporal data. However, most of these works have focused on segment-based recognition and are not suitable for the online scenarios. In this paper, we propose an online recognition system for skeleton sequence streaming composed from two main components: a detector and a classifier, which use a Semi-Positive Definite (SPD) matrix representation and a Siamese network. The powerful statistical representations for the skeletal data given by the SPD matrices and the learning of their semantic similarity by the Siamese network enable the detector to predict time intervals of the motions throughout an unsegmented sequence. In addition, they ensure the classifier capability to recognize the motion in each predicted interval. The proposed detector is flexible and able to identify the kinetic state continuously. We conduct extensive experiments on both hand gesture and body action recognition benchmarks to prove the accuracy of our online recognition system which in most cases outperforms state-of-the-art performances. 

**Abstract (ZH)**: 在线连续动作识别是研究的热点话题，因为它在实际应用场景中更具实用性。近年来，基于骨架的方法越来越受欢迎，展示了利用这种3D时序数据的强大力量。然而，大多数这些工作主要集中在基于段的识别上，并不适用于在线场景。本文提出了一种基于骨架序列流的数据在线识别系统，由两个主要组件组成：一个检测器和一个分类器，它们使用半正定矩阵（SPD矩阵）表示和Siamese网络。由半正定矩阵提供的强大统计表示和Siamese网络学习的语义相似性，使检测器能够预测未分段序列中的动作时间间隔。此外，这保证了分类器能够在每个预测间隔内识别动作。所提出的检测器灵活且能够连续识别运动状态。我们在手部手势和人体动作识别基准数据集上进行了广泛实验，以证明我们提出的在线识别系统的准确性，在大多数情况下超越了现有的最先进的性能。 

---
# 4D3R: Motion-Aware Neural Reconstruction and Rendering of Dynamic Scenes from Monocular Videos 

**Title (ZH)**: 4D3R：基于单目视频的动态场景运动感知神经重建与渲染 

**Authors**: Mengqi Guo, Bo Xu, Yanyan Li, Gim Hee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05229)  

**Abstract**: Novel view synthesis from monocular videos of dynamic scenes with unknown camera poses remains a fundamental challenge in computer vision and graphics. While recent advances in 3D representations such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have shown promising results for static scenes, they struggle with dynamic content and typically rely on pre-computed camera poses. We present 4D3R, a pose-free dynamic neural rendering framework that decouples static and dynamic components through a two-stage approach. Our method first leverages 3D foundational models for initial pose and geometry estimation, followed by motion-aware refinement. 4D3R introduces two key technical innovations: (1) a motion-aware bundle adjustment (MA-BA) module that combines transformer-based learned priors with SAM2 for robust dynamic object segmentation, enabling more accurate camera pose refinement; and (2) an efficient Motion-Aware Gaussian Splatting (MA-GS) representation that uses control points with a deformation field MLP and linear blend skinning to model dynamic motion, significantly reducing computational cost while maintaining high-quality reconstruction. Extensive experiments on real-world dynamic datasets demonstrate that our approach achieves up to 1.8dB PSNR improvement over state-of-the-art methods, particularly in challenging scenarios with large dynamic objects, while reducing computational requirements by 5x compared to previous dynamic scene representations. 

**Abstract (ZH)**: 单目动态场景中具有未知摄像机姿态的新型视角合成仍然是计算机视觉和图形学中的一个基本挑战。我们提出了4D3R，这是一种无需摄像机姿态的动态神经渲染框架，通过两阶段方法解耦静态和动态组件。该方法首先利用3D基础模型进行初始姿态和几何估计，然后进行运动感知细化。4D3R引入了两项关键技术创新：(1) 运动感知束调整（MA-BA）模块，结合基于变压器的学习先验与SAM2，实现稳健的动态物体分割，从而提高摄像机姿态优化的准确性；(2) 高效的运动感知高斯点聚合（MA-GS）表示，利用带有变形场MLP和线性混合皮肤的控制点，来建模动态运动，显著降低计算成本同时保持高质量的重建。在真实世界的动态数据集上的大量实验表明，与现有方法相比，我们的方法在复杂场景下特别是大型动态物体中实现了高达1.8dB PSNR的性能提升，并且计算需求降低了5倍。 

---
# No One-Model-Fits-All: Uncovering Spatio-Temporal Forecasting Trade-offs with Graph Neural Networks and Foundation Models 

**Title (ZH)**: 没有一模 fitting 皆适用：基于图神经网络和基础模型揭示时空Forecasting权衡 

**Authors**: Ragini Gupta, Naman Raina, Bo Chen, Li Chen, Claudiu Danilov, Josh Eckhardt, Keyshla Bernard, Klara Nahrstedt  

**Link**: [PDF](https://arxiv.org/pdf/2511.05179)  

**Abstract**: Modern IoT deployments for environmental sensing produce high volume spatiotemporal data to support downstream tasks such as forecasting, typically powered by machine learning models. While existing filtering and strategic deployment techniques optimize collected data volume at the edge, they overlook how variations in sampling frequencies and spatial coverage affect downstream model performance. In many forecasting models, incorporating data from additional sensors denoise predictions by providing broader spatial contexts. This interplay between sampling frequency, spatial coverage and different forecasting model architectures remain underexplored. This work presents a systematic study of forecasting models - classical models (VAR), neural networks (GRU, Transformer), spatio-temporal graph neural networks (STGNNs), and time series foundation models (TSFMs: Chronos Moirai, TimesFM) under varying spatial sensor nodes density and sampling intervals using real-world temperature data in a wireless sensor network. Our results show that STGNNs are effective when sensor deployments are sparse and sampling rate is moderate, leveraging spatial correlations via encoded graph structure to compensate for limited coverage. In contrast, TSFMs perform competitively at high frequencies but degrade when spatial coverage from neighboring sensors is reduced. Crucially, the multivariate TSFM Moirai outperforms all models by natively learning cross-sensor dependencies. These findings offer actionable insights for building efficient forecasting pipelines in spatio-temporal systems. All code for model configurations, training, dataset, and logs are open-sourced for reproducibility: this https URL 

**Abstract (ZH)**: 现代物联网部署在环境感知中的时空数据分析及其在不同采样频率和空间覆盖下的预测模型研究 

---
# Model Merging Improves Zero-Shot Generalization in Bioacoustic Foundation Models 

**Title (ZH)**: 模型合并提高生物声学基础模型的零样本泛化能力 

**Authors**: Davide Marincione, Donato Crisostomi, Roberto Dessi, Emanuele Rodolà, Emanuele Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2511.05171)  

**Abstract**: Foundation models capable of generalizing across species and tasks represent a promising new frontier in bioacoustics, with NatureLM being one of the most prominent examples. While its domain-specific fine-tuning yields strong performance on bioacoustic benchmarks, we observe that it also introduces trade-offs in instruction-following flexibility. For instance, NatureLM achieves high accuracy when prompted for either the common or scientific name individually, but its accuracy drops significantly when both are requested in a single prompt. We address this by applying a simple model merging strategy that interpolates NatureLM with its base language model, recovering instruction-following capabilities with minimal loss of domain expertise. Finally, we show that the merged model exhibits markedly stronger zero-shot generalization, achieving over a 200% relative improvement and setting a new state-of-the-art in closed-set zero-shot classification of unseen species. 

**Abstract (ZH)**: 跨物种和任务通用的基础模型代表了生物声学的一个有前景的新前沿，NatureLM 是其中最具代表性的例子。虽然其领域特定调优在生物声学基准测试中表现出色，但我们观察到它在指令遵循灵活性上也引入了权衡。例如，NatureLM 在分别要求普通名称或科学名称时可以达到高准确性，但在同一提示中同时请求两者时，其准确性会显著下降。我们通过一种简单的模型合并策略，将 NatureLM 与其基础语言模型进行插值，以最小程度的领域专业知识损失恢复了指令遵循能力。最后，我们展示了合并后的模型表现出显著更强的零样本泛化能力，实现了超过 200% 的相对改进，并在封闭集零样本分类未见物种方面达到了新的技术水平。 

---
# Generating Software Architecture Description from Source Code using Reverse Engineering and Large Language Model 

**Title (ZH)**: 使用逆向工程和大型语言模型从源代码生成软件架构描述 

**Authors**: Ahmad Hatahet, Christoph Knieke, Andreas Rausch  

**Link**: [PDF](https://arxiv.org/pdf/2511.05165)  

**Abstract**: Software Architecture Descriptions (SADs) are essential for managing the inherent complexity of modern software systems. They enable high-level architectural reasoning, guide design decisions, and facilitate effective communication among diverse stakeholders. However, in practice, SADs are often missing, outdated, or poorly aligned with the system's actual implementation. Consequently, developers are compelled to derive architectural insights directly from source code-a time-intensive process that increases cognitive load, slows new developer onboarding, and contributes to the gradual degradation of clarity over the system's lifetime. To address these issues, we propose a semi-automated generation of SADs from source code by integrating reverse engineering (RE) techniques with a Large Language Model (LLM). Our approach recovers both static and behavioral architectural views by extracting a comprehensive component diagram, filtering architecturally significant elements (core components) via prompt engineering, and generating state machine diagrams to model component behavior based on underlying code logic with few-shots prompting. This resulting views representation offer a scalable and maintainable alternative to traditional manual architectural documentation. This methodology, demonstrated using C++ examples, highlights the potent capability of LLMs to: 1) abstract the component diagram, thereby reducing the reliance on human expert involvement, and 2) accurately represent complex software behaviors, especially when enriched with domain-specific knowledge through few-shot prompting. These findings suggest a viable path toward significantly reducing manual effort while enhancing system understanding and long-term maintainability. 

**Abstract (ZH)**: 软件架构描述（SADs）对于管理现代软件系统的固有复杂性至关重要。它们 enabling 高级架构推理，指导设计决策，并促进多方利益相关者之间的有效沟通。然而，在实践中，SADs 往往缺失、过时或与系统的实际实现 poorly 对齐。因此，开发人员不得不直接从源代码中推导出架构见解——这一耗时的过程增加了认知负担，减慢了新开发者的入职速度，并导致系统在其生命周期中逐渐失去清晰度。为了解决这些问题，我们提出了一种半自动化的SADs 生成方法，通过将反向工程（RE）技术与大规模语言模型（LLM）集成来从源代码中生成SADs。该方法通过提取全面的组件图、利用提示工程筛选出架构上重要的元素（核心组件）以及基于底层代码逻辑生成状态机图来建模组件行为，从而恢复静态和行为架构视图。这些生成的视图表示为传统手动架构文档提供了可扩展且可维护的替代方案。该方法通过C++示例展示，突显了LLM的强大能力：1) 抽象组件图，从而减少对人工专家的依赖，2) 准确表示复杂软件行为，特别是在通过少样本提示增强领域特定知识时。这些发现表明了一条减轻手动努力、增强系统理解和长期可维护性的可行途径。 

---
# SmartSecChain-SDN: A Blockchain-Integrated Intelligent Framework for Secure and Efficient Software-Defined Networks 

**Title (ZH)**: SmartSecChain-SDN：一种集成区块链的安全高效软件定义网络智能框架 

**Authors**: Azhar Hussain Mozumder, M. John Basha, Chayapathi A. R  

**Link**: [PDF](https://arxiv.org/pdf/2511.05156)  

**Abstract**: With more and more existing networks being transformed to Software-Defined Networking (SDN), they need to be more secure and demand smarter ways of traffic control. This work, SmartSecChain-SDN, is a platform that combines machine learning based intrusion detection, blockchain-based storage of logs, and application-awareness-based priority in SDN networks. To detect network intrusions in a real-time, precision and low-false positives setup, the framework utilizes the application of advanced machine learning algorithms, namely Random Forest, XGBoost, CatBoost, and CNN-BiLSTM. SmartSecChain-SDN is based on the Hyperledger Fabric, which is a permissioned blockchain technology, to provide secure, scalable, and privacy-preserving storage and, thus, guarantee that the Intrusion Detection System (IDS) records cannot be altered and can be analyzed comprehensively. The system also has Quality of Service (QoS) rules and traffic shaping based on applications, which enables prioritization of critical services, such as VoIP, video conferencing, and business applications, as well as de-prioritization of non-essential traffic, such as downloads and updates. Mininet can simulate real-time SDN scenarios because it is used to prototype whole architectures. It is also compatible with controllers OpenDaylight and Ryu. It has tested the framework using the InSDN dataset and proved that it can identify different kinds of cyberattacks and handle bandwidth allocation efficiently under circumstances of resource constraints. SmartSecChain-SDN comprehensively addresses SDN system protection, securing and enhancing. The proposed study offers an innovative, extensible way to improve cybersecurity, regulatory compliance, and the administration of next-generation programmable networks. 

**Abstract (ZH)**: SmartSecChain-SDN：一种结合机器学习入侵检测、基于区块链的日志存储和应用感知优先级的SDN平台 

---
# From Linear Probing to Joint-Weighted Token Hierarchy: A Foundation Model Bridging Global and Cellular Representations in Biomarker Detection 

**Title (ZH)**: 从线性探查到联合加权令牌层次结构：一种连接全局和细胞表示的生物标志物检测基础模型 

**Authors**: Jingsong Liu, Han Li, Nassir Navab, Peter J. Schüffler  

**Link**: [PDF](https://arxiv.org/pdf/2511.05150)  

**Abstract**: AI-based biomarkers can infer molecular features directly from hematoxylin & eosin (H&E) slides, yet most pathology foundation models (PFMs) rely on global patch-level embeddings and overlook cell-level morphology. We present a PFM model, JWTH (Joint-Weighted Token Hierarchy), which integrates large-scale self-supervised pretraining with cell-centric post-tuning and attention pooling to fuse local and global tokens. Across four tasks involving four biomarkers and eight cohorts, JWTH achieves up to 8.3% higher balanced accuracy and 1.2% average improvement over prior PFMs, advancing interpretable and robust AI-based biomarker detection in digital pathology. 

**Abstract (ZH)**: 基于AI的生物标志物可以从苏木精和伊红（H&E）切片中直接推断出分子特征，但大多数病理基础模型（PFMs）依赖于全局patches级别的嵌入而忽视了细胞水平的形态特征。我们提出了一种PFM模型，JWTH（联合加权token层次结构），该模型结合了大规模自我监督预训练与以细胞为中心的后调整和注意力池化，以融合局部和全局token。在涉及四种生物标志物和八个队列的四项任务中，JWTH在平衡准确率上最高可提高8.3%，平均提高1.2%，从而推进了数字病理学中可解释和稳健的AI生物标志物检测。 

---
# DL101 Neural Network Outputs and Loss Functions 

**Title (ZH)**: DL101 神经网络输出与损失函数 

**Authors**: Fernando Berzal  

**Link**: [PDF](https://arxiv.org/pdf/2511.05131)  

**Abstract**: The loss function used to train a neural network is strongly connected to its output layer from a statistical point of view. This technical report analyzes common activation functions for a neural network output layer, like linear, sigmoid, ReLU, and softmax, detailing their mathematical properties and their appropriate use cases. A strong statistical justification exists for the selection of the suitable loss function for training a deep learning model. This report connects common loss functions such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and various Cross-Entropy losses to the statistical principle of Maximum Likelihood Estimation (MLE). Choosing a specific loss function is equivalent to assuming a specific probability distribution for the model output, highlighting the link between these functions and the Generalized Linear Models (GLMs) that underlie network output layers. Additional scenarios of practical interest are also considered, such as alternative output encodings, constrained outputs, and distributions with heavy tails. 

**Abstract (ZH)**: 使用的损失函数从统计角度来看与神经网络的输出层有密切联系。本技术报告分析了常见的神经网络输出层激活函数，如线性、Sigmoid、ReLU和Softmax，详细介绍了它们的数学属性及其适用场景。选择了合适的损失函数进行深度学习模型训练具有较强的统计学依据。本报告将常见的损失函数，如均方误差（MSE）、均绝对误差（MAE）和各种交叉熵损失，与最大似然估计（MLE）的基本原理联系起来。选择特定的损失函数相当于假设模型输出的概率分布，强调这些函数与构成网络输出层的基础广义线性模型（GLMs）之间的联系。此外，还考虑了实际应用中的其他场景，如替代输出编码、受限输出以及具有厚尾的分布。 

---
# Deep learning models are vulnerable, but adversarial examples are even more vulnerable 

**Title (ZH)**: 深度学习模型易受攻击，但对抗样本更加脆弱 

**Authors**: Jun Li, Yanwei Xu, Keran Li, Xiaoli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05073)  

**Abstract**: Understanding intrinsic differences between adversarial examples and clean samples is key to enhancing DNN robustness and detection against adversarial attacks. This study first empirically finds that image-based adversarial examples are notably sensitive to occlusion. Controlled experiments on CIFAR-10 used nine canonical attacks (e.g., FGSM, PGD) to generate adversarial examples, paired with original samples for evaluation. We introduce Sliding Mask Confidence Entropy (SMCE) to quantify model confidence fluctuation under occlusion. Using 1800+ test images, SMCE calculations supported by Mask Entropy Field Maps and statistical distributions show adversarial examples have significantly higher confidence volatility under occlusion than originals. Based on this, we propose Sliding Window Mask-based Adversarial Example Detection (SWM-AED), which avoids catastrophic overfitting of conventional adversarial training. Evaluations across classifiers and attacks on CIFAR-10 demonstrate robust performance, with accuracy over 62% in most cases and up to 96.5%. 

**Abstract (ZH)**: 基于图像的对抗样本在遮挡下明显比干净样本更敏感，理解这一点对于增强DNN的鲁棒性和对抗攻击检测至关重要。本研究首先经验性地发现基于图像的对抗样本在遮挡下尤为敏感。通过在CIFAR-10上进行受控实验，使用九种经典的攻击方法（如FGSM、PGD）生成对抗样本，并与原始样本进行配对评估。我们引入滑动掩码置信熵（SMCE）来量化在遮挡下的模型置信度波动。利用超过1800张测试图像，支持掩码熵场图和统计分布的SMCE计算显示，对抗样本在遮挡下的置信度波动显著高于原始样本。基于此，我们提出滑动窗口掩码基于的对抗样本检测（SWM-AED），该方法避免了常规对抗训练的灾难性过拟合。在CIFAR-10上的分类器和攻击评估中，显示出稳健的性能，大多数情况下准确率超过62%，最高可达96.5%。 

---
# No Pose Estimation? No Problem: Pose-Agnostic and Instance-Aware Test-Time Adaptation for Monocular Depth Estimation 

**Title (ZH)**: 无需姿态估计？不成问题：单目深度估计的姿势无关且实例感知的测试时适应 

**Authors**: Mingyu Sung, Hyeonmin Choe, Il-Min Kim, Sangseok Yun, Jae Mo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05055)  

**Abstract**: Monocular depth estimation (MDE), inferring pixel-level depths in single RGB images from a monocular camera, plays a crucial and pivotal role in a variety of AI applications demanding a three-dimensional (3D) topographical scene. In the real-world scenarios, MDE models often need to be deployed in environments with different conditions from those for training. Test-time (domain) adaptation (TTA) is one of the compelling and practical approaches to address the issue. Although there have been notable advancements in TTA for MDE, particularly in a self-supervised manner, existing methods are still ineffective and problematic when applied to diverse and dynamic environments. To break through this challenge, we propose a novel and high-performing TTA framework for MDE, named PITTA. Our approach incorporates two key innovative strategies: (i) pose-agnostic TTA paradigm for MDE and (ii) instance-aware image masking. Specifically, PITTA enables highly effective TTA on a pretrained MDE network in a pose-agnostic manner without resorting to any camera pose information. Besides, our instance-aware masking strategy extracts instance-wise masks for dynamic objects (e.g., vehicles, pedestrians, etc.) from a segmentation mask produced by a pretrained panoptic segmentation network, by removing static objects including background components. To further boost performance, we also present a simple yet effective edge extraction methodology for the input image (i.e., a single monocular image) and depth map. Extensive experimental evaluations on DrivingStereo and Waymo datasets with varying environmental conditions demonstrate that our proposed framework, PITTA, surpasses the existing state-of-the-art techniques with remarkable performance improvements in MDE during TTA. 

**Abstract (ZH)**: 单目深度估计的测试时域适应框架：PITTA 

---
# Accelerating HDC-CNN Hybrid Models Using Custom Instructions on RISC-V GPUs 

**Title (ZH)**: 在RISC-V GPU上使用自定义指令加速HDC-CNN混合模型 

**Authors**: Wakuto Matsumi, Riaz-Ul-Haque Mian  

**Link**: [PDF](https://arxiv.org/pdf/2511.05053)  

**Abstract**: Machine learning based on neural networks has advanced rapidly, but the high energy consumption required for training and inference remains a major challenge. Hyperdimensional Computing (HDC) offers a lightweight, brain-inspired alternative that enables high parallelism but often suffers from lower accuracy on complex visual tasks. To overcome this, hybrid accelerators combining HDC and Convolutional Neural Networks (CNNs) have been proposed, though their adoption is limited by poor generalizability and programmability. The rise of open-source RISC-V architectures has created new opportunities for domain-specific GPU design. Unlike traditional proprietary GPUs, emerging RISC-V-based GPUs provide flexible, programmable platforms suitable for custom computation models such as HDC. In this study, we design and implement custom GPU instructions optimized for HDC operations, enabling efficient processing for hybrid HDC-CNN workloads. Experimental results using four types of custom HDC instructions show a performance improvement of up to 56.2 times in microbenchmark tests, demonstrating the potential of RISC-V GPUs for energy-efficient, high-performance computing. 

**Abstract (ZH)**: 基于神经网络的机器学习快速发展，但其训练和推理所需的高能耗仍然是一个重大挑战。超维度计算(HDC)提供了一种轻量级、启发式的替代方案，能够实现高并行度，但在复杂视觉任务上往往准确率较低。为克服这一问题，结合HDC和卷积神经网络(CNN)的混合加速器已被提出，不过由于其泛化能力和编程性较差，其应用受限。开源RISC-V架构的兴起为专用GPU设计提供了新机会。不同于传统的专有GPU，新兴的RISC-V基GPU提供了灵活可编程的平台，适合用于定制计算模型如HDC。在本研究中，我们为HDC操作设计并实现了定制GPU指令，以实现混合HDC-CNN工作负载的高效处理。实验结果表明，在微基准测试中，使用四种定制HDC指令可实现高达56.2倍的性能提升，展示了RISC-V GPU在高效能计算中的潜力。 

---
# UA-Code-Bench: A Competitive Programming Benchmark for Evaluating LLM Code Generation in Ukrainian 

**Title (ZH)**: UA-Code-Bench: 用于评估乌克兰语编程代码生成的LLM基准 

**Authors**: Mykyta Syromiatnikov, Victoria Ruvinskaya  

**Link**: [PDF](https://arxiv.org/pdf/2511.05040)  

**Abstract**: Evaluating the real capabilities of large language models in low-resource languages still represents a challenge, as many existing benchmarks focus on widespread tasks translated from English or evaluate only simple language understanding. This paper introduces UA-Code-Bench, a new open-source benchmark established for a thorough evaluation of language models' code generation and competitive programming problem-solving abilities in Ukrainian. The benchmark comprises 500 problems from the Eolymp platform, evenly distributed across five complexity levels from very easy to very hard. A diverse set of 13 leading proprietary and open-source models, generating Python solutions based on a one-shot prompt, was evaluated via the dedicated Eolymp environment against hidden tests, ensuring code correctness. The obtained results reveal that even top-performing models, such as OpenAI o3 and GPT-5, solve only half of the problems, highlighting the challenge of code generation in low-resource natural language. Furthermore, this research presents a comprehensive analysis of performance across various difficulty levels, as well as an assessment of solution uniqueness and computational efficiency, measured by both elapsed time and memory consumption of the generated solutions. In conclusion, this work demonstrates the value of competitive programming benchmarks in evaluating large language models, especially in underrepresented languages. It also paves the way for future research on multilingual code generation and reasoning-enhanced models. The benchmark, data parsing, preparation, code generation, and evaluation scripts are available at this https URL. 

**Abstract (ZH)**: 评估大型语言模型在低资源语言中的实际能力仍然是一项挑战，因为许多现有基准聚焦于从英语翻译过来的广泛任务或仅评估简单的语言理解能力。本文介绍了UA-Code-Bench，一个新的开源基准，旨在全面评估语言模型在乌克兰语中的代码生成和竞赛编程问题解决能力。该基准包括500个来自Eolymp平台的问题，按复杂度均匀分布，从非常简单到非常困难。通过专用的Eolymp环境，在隐藏测试条件下评估了13种不同的领先专有和开源模型，确保代码正确性。获得的结果表明，即使是性能最佳的模型，如OpenAI o3和GPT-5，也只能解决一半的问题，突显了在低资源自然语言中的代码生成挑战。此外，本文还对不同难度级别的性能进行了全面分析，评估了解决方案的唯一性和计算效率，通过生成解决方案所花费的时间和内存消耗进行衡量。总之，这项工作证明了竞赛编程基准在评估大型语言模型方面的价值，特别是在代表性不足的语言中。此外，它也为多语言代码生成和增强推理模型的未来研究开辟了道路。基准、数据解析、准备、代码生成和评估脚本可在以下链接获取。 

---
# PECL: A Heterogeneous Parallel Multi-Domain Network for Radar-Based Human Activity Recognition 

**Title (ZH)**: 基于雷达的人体活动识别异构并行多域网络：PECL 

**Authors**: Jiuqi Yan, Chendong Xu, Dongyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.05039)  

**Abstract**: Radar systems are increasingly favored for medical applications because they provide non-intrusive monitoring with high privacy and robustness to lighting conditions. However, existing research typically relies on single-domain radar signals and overlooks the temporal dependencies inherent in human activity, which complicates the classification of similar actions. To address this issue, we designed the Parallel-EfficientNet-CBAM-LSTM (PECL) network to process data in three complementary domains: Range-Time, Doppler-Time, and Range-Doppler. PECL combines a channel-spatial attention module and temporal units to capture more features and dynamic dependencies during action sequences, improving both accuracy and robustness. The experimental results show that PECL achieves an accuracy of 96.16% on the same dataset, outperforming existing methods by at least 4.78%. PECL also performs best in distinguishing between easily confused actions. Despite its strong performance, PECL maintains moderate model complexity, with 23.42M parameters and 1324.82M FLOPs. Its parameter-efficient design further reduces computational cost. 

**Abstract (ZH)**: 雷达系统因其提供非侵入性监测、高隐私性和对光照条件的 robustness，在医疗应用中越来越受欢迎。然而，现有研究通常依赖于单域雷达信号，忽视了人类活动内在的时间依赖性，这使得相似动作的分类变得复杂。为解决这一问题，我们设计了 Parallel-EfficientNet-CBAM-LSTM (PECL) 网络以处理三个互补域的数据：Range-Time、Doppler-Time 和 Range-Doppler。PECL 结合了通道-空间注意力模块和时间单元，在动作序列中捕获更多的特征和动态依赖性，从而提高准确性和 robustness。实验结果显示，PECL 在同一数据集上的准确率达到 96.16%，比现有方法至少高出 4.78%。此外，PECL 在区分容易混淆的动作方面表现最佳。尽管具有强大的性能，PECL 的模型复杂度适中，参数量为 23.42M，FLOPs 为 1324.82M。其参数高效的架构进一步降低了计算成本。 

---
# Dynamic Residual Encoding with Slide-Level Contrastive Learning for End-to-End Whole Slide Image Representation 

**Title (ZH)**: 滑块级对比学习引导的动态残差编码用于端到端全视野图像表示 

**Authors**: Jing Jin, Xu Liu, Te Gao, Zhihong Shi, Yixiong Liang, Ruiqing Zheng, Hulin Kuang, Min Zeng, Shichao Kan  

**Link**: [PDF](https://arxiv.org/pdf/2511.05034)  

**Abstract**: Whole Slide Image (WSI) representation is critical for cancer subtyping, cancer recognition and mutation this http URL an end-to-end WSI representation model poses significant challenges, as a standard gigapixel slide can contain tens of thousands of image tiles, making it difficult to compute gradients of all tiles in a single mini-batch due to current GPU limitations. To address this challenge, we propose a method of dynamic residual encoding with slide-level contrastive learning (DRE-SLCL) for end-to-end WSI representation. Our approach utilizes a memory bank to store the features of tiles across all WSIs in the dataset. During training, a mini-batch usually contains multiple WSIs. For each WSI in the batch, a subset of tiles is randomly sampled and their features are computed using a tile encoder. Then, additional tile features from the same WSI are selected from the memory bank. The representation of each individual WSI is generated using a residual encoding technique that incorporates both the sampled features and those retrieved from the memory bank. Finally, the slide-level contrastive loss is computed based on the representations and histopathology reports ofthe WSIs within the mini-batch. Experiments conducted over cancer subtyping, cancer recognition, and mutation prediction tasks proved the effectiveness of the proposed DRE-SLCL method. 

**Abstract (ZH)**: Whole Slide Image (WSI) 表征对于癌症亚型划分、癌症识别和突变分析至关重要。由于标准的吉格拉像素切片中包含成千上万的图像块，当前GPU限制使得在一个小批量中计算所有块的梯度变得困难。为了解决这一挑战，我们提出了一种基于切片级对比学习的动态残差编码方法（DRE-SLCL），以实现端到端的WSI表征。我们的方法利用记忆库存储数据集中所有WSI图像块的特征。在训练过程中，一个批次通常包含多个WSI。对于批次中的每个WSI，随机采样一组图像块并使用图像块编码器计算它们的特征，然后从记忆库中选择相同WSI的额外图像块特征。每个WSI的表示使用残差编码技术生成，该技术结合了采样特征和从记忆库检索的特征。最后，基于批次中WSI的表示和组织病理报告计算切片级对比损失。在癌症亚型划分、癌症识别和突变预测任务上的实验证明了提出的方法DRE-SLCL的有效性。 

---
# OvA-LP: A Simple and Efficient Framework for Federated Learning on Non-IID Data 

**Title (ZH)**: OvA-LP：一种处理非同态数据联邦学习的简单高效框架 

**Authors**: Dongjin Park, Hasung Yeo, Joon-Woo Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.05028)  

**Abstract**: Federated fine-tuning (FFT) adapts foundation models to decentralized data but remains fragile under heterogeneous client distributions due to local drift, i.e., client-level update divergences that induce systematic bias and amplified variance in the global model. Existing aggregation and personalization methods largely correct drift post hoc, which proves brittle under extreme non-IID conditions. We introduce OvA-LP, a minimalist framework that is, to our knowledge, the first explicitly designed to suppress drift at its source within the PEFT-based FFT paradigm. OvA-LP combines linear probing on a frozen encoder with a one-vs-all head and a simple two-stage procedure, preserving pretrained feature geometry and decoupling logits to prevent the mechanisms that amplify drift. On CIFAR-100 with 100 clients, averaged over shard-1, shard-2, and Bernoulli-Dirichlet partitions, OvA-LP retains 95.9% of its IID accuracy, whereas state-of-the-art FFT baselines retain only 10.1% (PFPT) and 34.5% (FFT-MoE) under the same conditions. OvA-LP further maintains resilience under both symmetric and asymmetric label noise. In addition, precomputing encoder features makes per-round cost nearly independent of encoder size. Together, these results demonstrate that OvA-LP provides a principled and efficient basis for robust FFT under heterogeneity. 

**Abstract (ZH)**: 面向异构客户端分布的联邦微调去偏移框架OvA-LP 

---
# 8bit-GPT: Exploring Human-AI Interaction on Obsolete Macintosh Operating Systems 

**Title (ZH)**: 8位-GPT：探索过时Macintosh操作系统上的人工智能交互 

**Authors**: Hala Sheta  

**Link**: [PDF](https://arxiv.org/pdf/2511.05025)  

**Abstract**: The proliferation of assistive chatbots offering efficient, personalized communication has driven widespread over-reliance on them for decision-making, information-seeking and everyday tasks. This dependence was found to have adverse consequences on information retention as well as lead to superficial emotional attachment. As such, this work introduces 8bit-GPT; a language model simulated on a legacy Macintosh Operating System, to evoke reflection on the nature of Human-AI interaction and the consequences of anthropomorphic rhetoric. Drawing on reflective design principles such as slow-technology and counterfunctionality, this work aims to foreground the presence of chatbots as a tool by defamiliarizing the interface and prioritizing inefficient interaction, creating a friction between the familiar and not. 

**Abstract (ZH)**: 基于8bit-GPT的语言模型模拟在经典Macintosh操作系统上探讨人机交互的本质及其拟人化 rhetoric 的后果 

---
# Pluralistic Behavior Suite: Stress-Testing Multi-Turn Adherence to Custom Behavioral Policies 

**Title (ZH)**: 多元行为套件：多重轮次压力测试自定义行为政策的合规性 

**Authors**: Prasoon Varshney, Makesh Narsimhan Sreedhar, Liwei Jiang, Traian Rebedea, Christopher Parisien  

**Link**: [PDF](https://arxiv.org/pdf/2511.05018)  

**Abstract**: Large language models (LLMs) are typically aligned to a universal set of safety and usage principles intended for broad public acceptability. Yet, real-world applications of LLMs often take place within organizational ecosystems shaped by distinctive corporate policies, regulatory requirements, use cases, brand guidelines, and ethical commitments. This reality highlights the need for rigorous and comprehensive evaluation of LLMs with pluralistic alignment goals, an alignment paradigm that emphasizes adaptability to diverse user values and needs. In this work, we present PLURALISTIC BEHAVIOR SUITE (PBSUITE), a dynamic evaluation suite designed to systematically assess LLMs' capacity to adhere to pluralistic alignment specifications in multi-turn, interactive conversations. PBSUITE consists of (1) a diverse dataset of 300 realistic LLM behavioral policies, grounded in 30 industries; and (2) a dynamic evaluation framework for stress-testing model compliance with custom behavioral specifications under adversarial conditions. Using PBSUITE, We find that leading open- and closed-source LLMs maintain robust adherence to behavioral policies in single-turn settings (less than 4% failure rates), but their compliance weakens substantially in multi-turn adversarial interactions (up to 84% failure rates). These findings highlight that existing model alignment and safety moderation methods fall short in coherently enforcing pluralistic behavioral policies in real-world LLM interactions. Our work contributes both the dataset and analytical framework to support future research toward robust and context-aware pluralistic alignment techniques. 

**Abstract (ZH)**: PLURALISTIC BEHAVIOR SUITE: A DYNAMIC EVALUATION FRAMEWORK FOR SYSTEMATIC ASSESSMENT OF LLMs' ADHERENCE TO PLURALISTIC ALIGNMENT SPECIFICATIONS 

---
# Multi-agent Coordination via Flow Matching 

**Title (ZH)**: 基于流匹配的多agent协调 

**Authors**: Dongsu Lee, Daehee Lee, Amy Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.05005)  

**Abstract**: This work presents MAC-Flow, a simple yet expressive framework for multi-agent coordination. We argue that requirements of effective coordination are twofold: (i) a rich representation of the diverse joint behaviors present in offline data and (ii) the ability to act efficiently in real time. However, prior approaches often sacrifice one for the other, i.e., denoising diffusion-based solutions capture complex coordination but are computationally slow, while Gaussian policy-based solutions are fast but brittle in handling multi-agent interaction. MAC-Flow addresses this trade-off by first learning a flow-based representation of joint behaviors, and then distilling it into decentralized one-step policies that preserve coordination while enabling fast execution. Across four different benchmarks, including $12$ environments and $34$ datasets, MAC-Flow alleviates the trade-off between performance and computational cost, specifically achieving about $\boldsymbol{\times14.5}$ faster inference compared to diffusion-based MARL methods, while maintaining good performance. At the same time, its inference speed is similar to that of prior Gaussian policy-based offline multi-agent reinforcement learning (MARL) methods. 

**Abstract (ZH)**: MAC-Flow：一种简单高效的多agent协调框架 

---
# Query Generation Pipeline with Enhanced Answerability Assessment for Financial Information Retrieval 

**Title (ZH)**: 带有增强答案可靠性评估的金融信息检索查询生成管道 

**Authors**: Hyunkyu Kim, Yeeun Yoo, Youngjun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2511.05000)  

**Abstract**: As financial applications of large language models (LLMs) gain attention, accurate Information Retrieval (IR) remains crucial for reliable AI services. However, existing benchmarks fail to capture the complex and domain-specific information needs of real-world banking scenarios. Building domain-specific IR benchmarks is costly and constrained by legal restrictions on using real customer data. To address these challenges, we propose a systematic methodology for constructing domain-specific IR benchmarks through LLM-based query generation. As a concrete implementation of this methodology, our pipeline combines single and multi-document query generation with an enhanced and reasoning-augmented answerability assessment method, achieving stronger alignment with human judgments than prior approaches. Using this methodology, we construct KoBankIR, comprising 815 queries derived from 204 official banking documents. Our experiments show that existing retrieval models struggle with the complex multi-document queries in KoBankIR, demonstrating the value of our systematic approach for domain-specific benchmark construction and underscoring the need for improved retrieval techniques in financial domains. 

**Abstract (ZH)**: 随着大规模语言模型在金融领域的应用引起关注，准确的信息检索（IR）对于可靠的AI服务仍然至关重要。然而，现有的基准测试未能捕捉到真实银行业场景中复杂和领域特定的信息需求。构建领域特定的IR基准测试成本高昂，并受限于使用真实客户数据的法律限制。为应对这些挑战，我们提出了一种通过基于大语言模型的查询生成来系统地构建领域特定IR基准测试的方法。作为该方法的具体实现，我们的流水线结合了单文档和多文档查询生成，并采用增强和推理增强的答案可评估性评估方法，比先前的方法更接近人类判断。通过这种方法，我们构建了KoBankIR，包含来自204份官方银行业文件的815个查询。我们的实验表明，现有的检索模型难以处理KoBankIR中的复杂多文档查询，这强调了在金融领域构建领域特定基准测试和改进检索技术的重要性。 

---
# BiPETE: A Bi-Positional Embedding Transformer Encoder for Risk Assessment of Alcohol and Substance Use Disorder with Electronic Health Records 

**Title (ZH)**: BiPETE: 一种双位置嵌入变换器编码器，用于电子健康记录中酒精和物质使用障碍风险评估 

**Authors**: Daniel S. Lee, Mayra S. Haedo-Cruz, Chen Jiang, Oshin Miranda, LiRong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04998)  

**Abstract**: Transformer-based deep learning models have shown promise for disease risk prediction using electronic health records(EHRs), but modeling temporal dependencies remains a key challenge due to irregular visit intervals and lack of uniform structure. We propose a Bi-Positional Embedding Transformer Encoder or BiPETE for single-disease prediction, which integrates rotary positional embeddings to encode relative visit timing and sinusoidal embeddings to preserve visit order. Without relying on large-scale pretraining, BiPETE is trained on EHR data from two mental health cohorts-depressive disorder and post-traumatic stress disorder (PTSD)-to predict the risk of alcohol and substance use disorders (ASUD). BiPETE outperforms baseline models, improving the area under the precision-recall curve (AUPRC) by 34% and 50% in the depression and PTSD cohorts, respectively. An ablation study further confirms the effectiveness of the dual positional encoding strategy. We apply the Integrated Gradients method to interpret model predictions, identifying key clinical features associated with ASUD risk and protection, such as abnormal inflammatory, hematologic, and metabolic markers, as well as specific medications and comorbidities. Overall, these key clinical features identified by the attribution methods contribute to a deeper understanding of the risk assessment process and offer valuable clues for mitigating potential risks. In summary, our study presents a practical and interpretable framework for disease risk prediction using EHR data, which can achieve strong performance. 

**Abstract (ZH)**: 基于Transformer的深度学习模型在利用电子健康记录（EHR）进行疾病风险预测方面显示出潜力，但由于就诊间隔不规则和缺乏统一结构，建模时间依赖关系仍然是一个关键挑战。我们提出了一种双位置嵌入变换器编码器（BiPETE），用于单一疾病的预测，该模型结合旋转位置嵌入编码相对就诊时间，并采用正弦嵌入保留就诊顺序。在不需要大规模预训练的情况下，BiPETE 在两个心理健康队列（抑郁障碍和创伤后应激障碍）的EHR数据上进行训练，以预测酒精和物质使用障碍（ASUD）的风险。BiPETE 在抑郁障碍和创伤后应激障碍队列中分别将精确召回曲线下的面积（AUPRC）提高了34%和50%，基线模型的表现更优。消融研究进一步证实了双重位置编码策略的有效性。我们应用集成梯度方法解释模型预测，识别与ASUD风险和保护相关的关键临床特征，例如异常的炎症、血液学和代谢标志物，以及特定药物和共病情况。总体而言，这些通过归因方法识别的关键临床特征有助于更深刻地理解风险评估过程，并为缓解潜在风险提供了宝贵的线索。总之，我们的研究提出了一种实用且可解释的框架，用于利用EHR数据进行疾病风险预测，能够实现较强的表现。 

---
# Enhancing Public Speaking Skills in Engineering Students Through AI 

**Title (ZH)**: 通过AI提升工程学生公众演讲技能 

**Authors**: Amol Harsh, Brainerd Prince, Siddharth Siddharth, Deepan Raj Prabakar Muthirayan, Kabir S Bhalla, Esraaj Sarkar Gupta, Siddharth Sahu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04995)  

**Abstract**: This research-to-practice full paper was inspired by the persistent challenge in effective communication among engineering students. Public speaking is a necessary skill for future engineers as they have to communicate technical knowledge with diverse stakeholders. While universities offer courses or workshops, they are unable to offer sustained and personalized training to students. Providing comprehensive feedback on both verbal and non-verbal aspects of public speaking is time-intensive, making consistent and individualized assessment impractical. This study integrates research on verbal and non-verbal cues in public speaking to develop an AI-driven assessment model for engineering students. Our approach combines speech analysis, computer vision, and sentiment detection into a multi-modal AI system that provides assessment and feedback. The model evaluates (1) verbal communication (pitch, loudness, pacing, intonation), (2) non-verbal communication (facial expressions, gestures, posture), and (3) expressive coherence, a novel integration ensuring alignment between speech and body language. Unlike previous systems that assess these aspects separately, our model fuses multiple modalities to deliver personalized, scalable feedback. Preliminary testing demonstrated that our AI-generated feedback was moderately aligned with expert evaluations. Among the state-of-the-art AI models evaluated, all of which were Large Language Models (LLMs), including Gemini and OpenAI models, Gemini Pro emerged as the best-performing, showing the strongest agreement with human annotators. By eliminating reliance on human evaluators, this AI-driven public speaking trainer enables repeated practice, helping students naturally align their speech with body language and emotion, crucial for impactful and professional communication. 

**Abstract (ZH)**: 基于研究到实践的公共演讲评估模型构建：面向工程学生的多模态AI系统 

---
# Learning Fourier shapes to probe the geometric world of deep neural networks 

**Title (ZH)**: 学习傅里叶形状以探究深度神经网络的几何世界 

**Authors**: Jian Wang, Yixing Yong, Haixia Bi, Lijun He, Fan Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.04970)  

**Abstract**: While both shape and texture are fundamental to visual recognition, research on deep neural networks (DNNs) has predominantly focused on the latter, leaving their geometric understanding poorly probed. Here, we show: first, that optimized shapes can act as potent semantic carriers, generating high-confidence classifications from inputs defined purely by their geometry; second, that they are high-fidelity interpretability tools that precisely isolate a model's salient regions; and third, that they constitute a new, generalizable adversarial paradigm capable of deceiving downstream visual tasks. This is achieved through an end-to-end differentiable framework that unifies a powerful Fourier series to parameterize arbitrary shapes, a winding number-based mapping to translate them into the pixel grid required by DNNs, and signal energy constraints that enhance optimization efficiency while ensuring physically plausible shapes. Our work provides a versatile framework for probing the geometric world of DNNs and opens new frontiers for challenging and understanding machine perception. 

**Abstract (ZH)**: 尽管形状和纹理都是视觉识别的基础，深度神经网络（DNNs）的研究主要集中在纹理上，而对形状的几何理解探究不足。在这里，我们展示：首先，优化后的形状可以作为强大的语义载体，仅通过其几何定义即可生成高置信度的分类结果；其次，形状是高保真的可解释性工具，能够精确隔离模型的关键区域；第三，它们构成一种新的、可泛化的对抗范式，能够迷惑下游视觉任务。这是通过一个端到端可微框架实现的，该框架集成了强大的傅里叶级数来参数化任意形状、基于环绕数的映射将形状转换为DNNs所需的像素网格，并通过信号能量约束提高优化效率同时确保物理上合理的形状。我们的工作提供了一种灵活的框架来探究DNNs的几何世界，并为挑战和理解机器感知开辟了新的前沿。 

---
# Pattern-Aware Diffusion Synthesis of fMRI/dMRI with Tissue and Microstructural Refinement 

**Title (ZH)**: 具有组织和微观结构细化的模式感知扩散合成fMRI/dMRI 

**Authors**: Xiongri Shen, Jiaqi Wang, Yi Zhong, Zhenxi Song, Leilei Zhao, Yichen Wei, Lingyan Liang, Shuqiang Wang, Baiying Lei, Demao Deng, Zhiguo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04963)  

**Abstract**: Magnetic resonance imaging (MRI), especially functional MRI (fMRI) and diffusion MRI (dMRI), is essential for studying neurodegenerative diseases. However, missing modalities pose a major barrier to their clinical use. Although GAN- and diffusion model-based approaches have shown some promise in modality completion, they remain limited in fMRI-dMRI synthesis due to (1) significant BOLD vs. diffusion-weighted signal differences between fMRI and dMRI in time/gradient axis, and (2) inadequate integration of disease-related neuroanatomical patterns during generation. To address these challenges, we propose PDS, introducing two key innovations: (1) a pattern-aware dual-modal 3D diffusion framework for cross-modality learning, and (2) a tissue refinement network integrated with a efficient microstructure refinement to maintain structural fidelity and fine details. Evaluated on OASIS-3, ADNI, and in-house datasets, our method achieves state-of-the-art results, with PSNR/SSIM scores of 29.83 dB/90.84\% for fMRI synthesis (+1.54 dB/+4.12\% over baselines) and 30.00 dB/77.55\% for dMRI synthesis (+1.02 dB/+2.2\%). In clinical validation, the synthesized data show strong diagnostic performance, achieving 67.92\%/66.02\%/64.15\% accuracy (NC vs. MCI vs. AD) in hybrid real-synthetic experiments. Code is available in \href{this https URL}{PDS GitHub Repository} 

**Abstract (ZH)**: 磁共振成像（MRI），尤其是功能性MRI（fMRI）和弥散MRI（dMRI），对于研究神经退行性疾病至关重要。然而，缺失的模态是其临床应用的主要障碍。尽管基于生成对抗网络（GAN）和扩散模型的方法在模态完成方面显示了一定的潜力，但在fMRI-dMRI合成方面仍然受到限制，原因包括（1）fMRI和dMRI在时间/梯度轴上的显著BOLD与扩散加权信号差异，以及（2）生成过程中不充分集成与疾病相关的神经解剖模式。为了解决这些挑战，我们提出了PDS，并引入了两项关键创新：（1）一种模式感知的双模态3D扩散框架，用于跨模态学习；（2）结合高效微结构细化的组织精炼网络，以保持结构保真度和精细细节。在OASIS-3、ADNI和内部数据集上的评估显示，我们的方法取得了最先进的成果，fMRI合成的PSNR/SSIM得分为29.83 dB/90.84%（比基线分别提高1.54 dB/4.12%），dMRI合成的得分为30.00 dB/77.55%（比基线分别提高1.02 dB/2.2%）。在临床验证中，合成数据显示出强大的诊断性能，在混合真实-合成实验中分别达到67.92%/66.02%/64.15%的准确率（正常老年人与轻度认知障碍 vs. 老年痴呆症）。代码可在PDS GitHub Repository获取。 

---
# Too Good to be Bad: On the Failure of LLMs to Role-Play Villains 

**Title (ZH)**: Too Good to be Bad: On the Failure of LLMs to Role-Play Villains 

**Authors**: Zihao Yi, Qingxuan Jiang, Ruotian Ma, Xingyu Chen, Qu Yang, Mengru Wang, Fanghua Ye, Ying Shen, Zhaopeng Tu, Xiaolong Li, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2511.04962)  

**Abstract**: Large Language Models (LLMs) are increasingly tasked with creative generation, including the simulation of fictional characters. However, their ability to portray non-prosocial, antagonistic personas remains largely unexamined. We hypothesize that the safety alignment of modern LLMs creates a fundamental conflict with the task of authentically role-playing morally ambiguous or villainous characters. To investigate this, we introduce the Moral RolePlay benchmark, a new dataset featuring a four-level moral alignment scale and a balanced test set for rigorous evaluation. We task state-of-the-art LLMs with role-playing characters from moral paragons to pure villains. Our large-scale evaluation reveals a consistent, monotonic decline in role-playing fidelity as character morality decreases. We find that models struggle most with traits directly antithetical to safety principles, such as ``Deceitful'' and ``Manipulative'', often substituting nuanced malevolence with superficial aggression. Furthermore, we demonstrate that general chatbot proficiency is a poor predictor of villain role-playing ability, with highly safety-aligned models performing particularly poorly. Our work provides the first systematic evidence of this critical limitation, highlighting a key tension between model safety and creative fidelity. Our benchmark and findings pave the way for developing more nuanced, context-aware alignment methods. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在创造性生成中的应用日益增多，包括模拟虚构角色。然而，它们表现非亲社会、敌对人格的能力仍鲜有研究。我们推测，现代LLMs的安全对齐在其诚实地扮演道德模糊或反派角色的任务上造成了根本矛盾。为了研究这一现象，我们引入了道德角色扮演基准（Moral RolePlay benchmark），该基准包括一个四级道德对齐尺度和一个平衡的测试集，以进行严格的评估。我们要求当前最先进的LLMs扮演从道德典范到纯粹反派的各种角色。大规模评估表明，随着角色道德性的降低，角色扮演的准确性呈一致且单调地下降。我们发现，模型在与安全原则直接对立的特质上最难处理，如“诡诈”和“ manipulative”，经常用表面的侵略性替代复杂的恶意。此外，我们证明了通用聊天机器人技能并不能很好地预测反派角色扮演能力，而高度安全对齐的模型在这方面表现尤其差。我们的工作提供了这一关键限制的首个系统性证据，高亮了模型安全与创造性准确之间的重要张力。我们的基准和发现为进一步发展更为细腻、情境感知的对齐方法铺平了道路。 

---
# DeepForgeSeal: Latent Space-Driven Semi-Fragile Watermarking for Deepfake Detection Using Multi-Agent Adversarial Reinforcement Learning 

**Title (ZH)**: DeepForgeSeal: 基于潜空间的半脆弱深度伪造检测水印技术及其多智能体对抗强化学习方法 

**Authors**: Tharindu Fernando, Clinton Fookes, Sridha Sridharan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04949)  

**Abstract**: Rapid advances in generative AI have led to increasingly realistic deepfakes, posing growing challenges for law enforcement and public trust. Existing passive deepfake detectors struggle to keep pace, largely due to their dependence on specific forgery artifacts, which limits their ability to generalize to new deepfake types. Proactive deepfake detection using watermarks has emerged to address the challenge of identifying high-quality synthetic media. However, these methods often struggle to balance robustness against benign distortions with sensitivity to malicious tampering. This paper introduces a novel deep learning framework that harnesses high-dimensional latent space representations and the Multi-Agent Adversarial Reinforcement Learning (MAARL) paradigm to develop a robust and adaptive watermarking approach. Specifically, we develop a learnable watermark embedder that operates in the latent space, capturing high-level image semantics, while offering precise control over message encoding and extraction. The MAARL paradigm empowers the learnable watermarking agent to pursue an optimal balance between robustness and fragility by interacting with a dynamic curriculum of benign and malicious image manipulations simulated by an adversarial attacker agent. Comprehensive evaluations on the CelebA and CelebA-HQ benchmarks reveal that our method consistently outperforms state-of-the-art approaches, achieving improvements of over 4.5% on CelebA and more than 5.3% on CelebA-HQ under challenging manipulation scenarios. 

**Abstract (ZH)**: 快速发展的生成式AI导致了越来越逼真的deepfake，给执法和公众信任带来了日益增加的挑战。现有的被动deepfake检测器难以跟上这一进展，主要原因是它们依赖于特定的伪造特征，这限制了它们对新类型deepfake的泛化能力。为了应对高質量合成媒体的识别挑战，主动式deepfake检测方法利用水印技术 emerges to address the challenge of identifying high-quality synthetic media. 然而，这些方法往往难以在鲁棒性和对恶意篡改的敏感性之间找到平衡。本文介绍了一种新颖的深度学习框架，该框架利用高维潜空间表示和多智能体对抗强化学习（MAARL）范式来开发出一种鲁棒性和适应性强的水印方法。具体来说，我们开发了一种在潜空间中操作的学习型水印嵌入器，该嵌入器捕捉高层图像语义，同时对信息编码和提取提供精确控制。MAARL范式使学习型水印智能体能够通过与由对抗攻击智能体模拟的动态课程中良性与恶意图像篡改的交互，追求鲁棒性和脆弱性的最佳平衡。在CelebA和CelebA-HQ基准上的全面评估表明，我们的方法在具有挑战性的篡改场景中始终优于现有方法，在CelebA上实现了超过4.5%的性能提升，在CelebA-HQ上超过了5.3%。 

---
# A benchmark multimodal oro-dental dataset for large vision-language models 

**Title (ZH)**: 面向大规模视觉-语言模型的基准多模态口腔数据集 

**Authors**: Haoxin Lv, Ijazul Haq, Jin Du, Jiaxin Ma, Binnian Zhu, Xiaobing Dang, Chaoan Liang, Ruxu Du, Yingjie Zhang, Muhammad Saqib  

**Link**: [PDF](https://arxiv.org/pdf/2511.04948)  

**Abstract**: The advancement of artificial intelligence in oral healthcare relies on the availability of large-scale multimodal datasets that capture the complexity of clinical practice. In this paper, we present a comprehensive multimodal dataset, comprising 8775 dental checkups from 4800 patients collected over eight years (2018-2025), with patients ranging from 10 to 90 years of age. The dataset includes 50000 intraoral images, 8056 radiographs, and detailed textual records, including diagnoses, treatment plans, and follow-up notes. The data were collected under standard ethical guidelines and annotated for benchmarking. To demonstrate its utility, we fine-tuned state-of-the-art large vision-language models, Qwen-VL 3B and 7B, and evaluated them on two tasks: classification of six oro-dental anomalies and generation of complete diagnostic reports from multimodal inputs. We compared the fine-tuned models with their base counterparts and GPT-4o. The fine-tuned models achieved substantial gains over these baselines, validating the dataset and underscoring its effectiveness in advancing AI-driven oro-dental healthcare solutions. The dataset is publicly available, providing an essential resource for future research in AI dentistry. 

**Abstract (ZH)**: 人工智能在口腔健康管理中的进步依赖于大规模多模态数据集的可用性，以捕捉临床实践的复杂性。本文介绍了一个全面的多模态数据集，包含从2018年至2025年八年间4800名患者累积的8775次牙科检查，患者年龄从10岁至90岁不等。数据集包括50000张口内图像、8056张牙片以及详细的文本记录，包括诊断、治疗计划和随访笔记。数据收集遵循标准的伦理准则并用于基准标注。为了展示其应用价值，我们对最新的大型视觉-语言模型Qwen-VL 3B和7B进行了微调，并在两个任务上进行了评估：六种口腔异常的分类和从多模态输入生成完整的诊断报告。微调后的模型与基线模型和GPT-4o进行了比较，结果显示微调模型取得了显著提升，验证了该数据集的有效性，突显了其在推动基于AI的口腔健康管理解决方案方面的成效。该数据集公开可用，为未来AI牙科研究提供了重要资源。 

---
# Search Is Not Retrieval: Decoupling Semantic Matching from Contextual Assembly in RAG 

**Title (ZH)**: 搜索不同于检索：在RAG中解耦语义匹配与上下文组装 

**Authors**: Harshit Nainwani, Hediyeh Baban  

**Link**: [PDF](https://arxiv.org/pdf/2511.04939)  

**Abstract**: Retrieval systems are essential to contemporary AI pipelines, although most confuse two separate processes: finding relevant information and giving enough context for reasoning. We introduce the Search-Is-Not-Retrieve (SINR) framework, a dual-layer architecture that distinguishes between fine-grained search representations and coarse-grained retrieval contexts. SINR enhances the composability, scalability, and context fidelity of retrieval systems by directly connecting small, semantically accurate search chunks to larger, contextually complete retrieve chunks, all without incurring extra processing costs. This design changes retrieval from a passive step to an active one, making the system architecture more like how people process information. We discuss the SINR framework's conceptual foundation, formal structure, implementation issues, and qualitative outcomes. This provides a practical foundation for the next generation of AI systems that use retrieval. 

**Abstract (ZH)**: 检索系统是当代AI管道中的关键组件，尽管大多数系统混淆了两个单独的过程：找到相关信息和提供足够的上下文进行推理。我们提出了Search-Is-Not-Retrieve (SINR)框架，这是一种双层架构，区分了细粒度的搜索表示和粗粒度的检索上下文。SINR通过直接将小的、语义准确的搜索片段连接到较大的、上下文完整的检索片段来增强检索系统的可组合性、可扩展性和上下文保真度，而不增加额外的处理成本。这种设计将检索从被动步骤转变为积极步骤，使系统架构更加类似于人类处理信息的方式。我们讨论了SINR框架的概念基础、形式结构、实施问题和定性结果，为下一代使用检索的AI系统提供了实用的基础。 

---
# BudgetMem: Learning Selective Memory Policies for Cost-Efficient Long-Context Processing in Language Models 

**Title (ZH)**: BudgetMem：学习成本效益的选择性内存策略以实现语言模型中高效长上下文处理 

**Authors**: Chandra Vamsi Krishna Alla, Harish Naidu Gaddam, Manohar Kommi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04919)  

**Abstract**: Large Language Models (LLMs) face significant computational and memory constraints when processing long contexts, despite growing demand for applications requiring reasoning over extensive documents, multi-session dialogues, and book length texts. While recent advances have extended context windows to 100K-1M tokens, such approaches incur prohibitive costs for resource constrained deployments. We propose BudgetMem, a novel memory augmented architecture that learns what to remember rather than remembering everything. Our system combines selective memory policies with feature based salience scoring (entity density, TF-IDF, discourse markers, position bias) to decide which information merits storage under strict budget constraints. Unlike existing retrieval augmented generation (RAG) systems that store all chunks, BudgetMem employs learned gating mechanisms coupled with BM25 sparse retrieval for efficient information access. Through comprehensive experiments on 700 question answer pairs across short (237 tokens) and long (5K-10K tokens) documents with Llama-3.2-3B-Instruct, we demonstrate that BudgetMem achieves remarkable results on long documents: only 1.0% F1 score degradation while saving 72.4% memory compared to baseline RAG. We validate our approach through budget sensitivity analysis (testing 7 budget ratios), naive baseline comparisons, and document length analysis, showing that BudgetMem's benefits increase with document length. Our work provides a practical pathway for deploying capable long context systems on modest hardware, democratizing access to advanced language understanding capabilities. 

**Abstract (ZH)**: BudgetMem：在严格内存预算下处理长上下文的新型记忆增强架构 

---
# MERaLiON-SER: Robust Speech Emotion Recognition Model for English and SEA Languages 

**Title (ZH)**: MERaLiON-SER：针对英语和东南亚语言的稳健语音情感识别模型 

**Authors**: Hardik B. Sailor, Aw Ai Ti, Chen Fang Yih Nancy, Chiu Ying Lay, Ding Yang, He Yingxu, Jiang Ridong, Li Jingtao, Liao Jingyi, Liu Zhuohan, Lu Yanfeng, Ma Yi, Manas Gupta, Muhammad Huzaifah Bin Md Shahrin, Nabilah Binte Md Johan, Nattadaporn Lertcheva, Pan Chunlei, Pham Minh Duc, Siti Maryam Binte Ahmad Subaidi, Siti Umairah Binte Mohammad Salleh, Sun Shuo, Tarun Kumar Vangani, Wang Qiongqiong, Won Cheng Yi Lewis, Wong Heng Meng Jeremy, Wu Jinyang, Zhang Huayun, Zhang Longyin, Zou Xunlong  

**Link**: [PDF](https://arxiv.org/pdf/2511.04914)  

**Abstract**: We present MERaLiON-SER, a robust speech emotion recognition model de- signed for English and Southeast Asian languages. The model is trained using a hybrid objective combining weighted categorical cross-entropy and Concordance Correlation Coefficient (CCC) losses for joint discrete and dimensional emotion modelling. This dual approach enables the model to capture both the distinct categories of emotion (like happy or angry) and the fine-grained, such as arousal (intensity), valence (positivity/negativity), and dominance (sense of control), lead- ing to a more comprehensive and robust representation of human affect. Extensive evaluations across multilingual Singaporean languages (English, Chinese, Malay, and Tamil ) and other public benchmarks show that MERaLiON-SER consistently surpasses both open-source speech encoders and large Audio-LLMs. These results underscore the importance of specialised speech-only models for accurate paralin- guistic understanding and cross-lingual generalisation. Furthermore, the proposed framework provides a foundation for integrating emotion-aware perception into future agentic audio systems, enabling more empathetic and contextually adaptive multimodal reasoning. 

**Abstract (ZH)**: MERaLiON-SER：一种适用于英语和东南亚语言的稳健语音情感识别模型 

---
# A Dual Perspective on Decision-Focused Learning: Scalable Training via Dual-Guided Surrogates 

**Title (ZH)**: 决策导向学习的双重视角：通过双重引导替代目标实现可扩展训练 

**Authors**: Paula Rodriguez-Diaz, Kirk Bansak Elisabeth Paulson  

**Link**: [PDF](https://arxiv.org/pdf/2511.04909)  

**Abstract**: Many real-world decisions are made under uncertainty by solving optimization problems using predicted quantities. This predict-then-optimize paradigm has motivated decision-focused learning, which trains models with awareness of how the optimizer uses predictions, improving the performance of downstream decisions. Despite its promise, scaling is challenging: state-of-the-art methods either differentiate through a solver or rely on task-specific surrogates, both of which require frequent and expensive calls to an optimizer, often a combinatorial one. In this paper, we leverage dual variables from the downstream problem to shape learning and introduce Dual-Guided Loss (DGL), a simple, scalable objective that preserves decision alignment while reducing solver dependence. We construct DGL specifically for combinatorial selection problems with natural one-of-many constraints, such as matching, knapsack, and shortest path. Our approach (a) decouples optimization from gradient updates by solving the downstream problem only periodically; (b) between refreshes, trains on dual-adjusted targets using simple differentiable surrogate losses; and (c) as refreshes become less frequent, drives training cost toward standard supervised learning while retaining strong decision alignment. We prove that DGL has asymptotically diminishing decision regret, analyze runtime complexity, and show on two problem classes that DGL matches or exceeds state-of-the-art DFL methods while using far fewer solver calls and substantially less training time. Code is available at this https URL. 

**Abstract (ZH)**: 面向双变量的优化学习：一种简单的可扩展目标（Dual-Guided Loss for Decision-Focused Learning） 

---
# You Need Reasoning to Learn Reasoning: The Limitations of Label-Free RL in Weak Base Models 

**Title (ZH)**: 你需要通过推理来学习推理：弱基模型中无标签RL的局限性 

**Authors**: Shuvendu Roy, Hossein Hajimirsadeghi, Mengyao Zhai, Golnoosh Samei  

**Link**: [PDF](https://arxiv.org/pdf/2511.04902)  

**Abstract**: Recent advances in large language models have demonstrated the promise of unsupervised reinforcement learning (RL) methods for enhancing reasoning capabilities without external supervision. However, the generalizability of these label-free RL approaches to smaller base models with limited reasoning capabilities remains unexplored. In this work, we systematically investigate the performance of label-free RL methods across different model sizes and reasoning strengths, from 0.5B to 7B parameters. Our empirical analysis reveals critical limitations: label-free RL is highly dependent on the base model's pre-existing reasoning capability, with performance often degrading below baseline levels for weaker models. We find that smaller models fail to generate sufficiently long or diverse chain-of-thought reasoning to enable effective self-reflection, and that training data difficulty plays a crucial role in determining success. To address these challenges, we propose a simple yet effective method for label-free RL that utilizes curriculum learning to progressively introduce harder problems during training and mask no-majority rollouts during training. Additionally, we introduce a data curation pipeline to generate samples with predefined difficulty. Our approach demonstrates consistent improvements across all model sizes and reasoning capabilities, providing a path toward more robust unsupervised RL that can bootstrap reasoning abilities in resource-constrained models. We make our code available at this https URL 

**Abstract (ZH)**: 近期大型语言模型的发展展示了无监督强化学习方法在增强推理能力方面的潜力，无需外部监督。然而，这些无标签的强化学习方法对于小型基础模型的泛化能力，这些基础模型具有有限的推理能力，尚未被探索。在这项工作中，我们系统地研究了无标签RL方法在不同模型规模和推理能力下的性能，从0.5B到7B参数。我们的实证分析揭示了关键的局限性：无标签RL高度依赖于基础模型预先存在的推理能力，较弱模型的表现通常会低于基线水平。我们发现，较小的模型无法生成足够长或多样化的推理链，以实现有效的自我反思，而训练数据难度在决定成功方面发挥着关键作用。为了应对这些挑战，我们提出了一种简单而有效的方法来增强无标签RL，利用课程学习在训练过程中逐步引入更难的问题，并在训练过程中屏蔽非主要推演。此外，我们引入了一种数据整理流水线来生成预定义难度的样本。我们的方法在所有模型规模和推理能力下均表现出一致性改进，为更稳健的无监督RL提供了途径，该途径可以为资源受限的模型提供推理能力的自我提升。我们的代码可在此网站获得。 

---
# Beta Distribution Learning for Reliable Roadway Crash Risk Assessment 

**Title (ZH)**: Beta 分布学习在道路事故风险可靠评估中的应用 

**Authors**: Ahmad Elallaf, Nathan Jacobs, Xinyue Ye, Mei Chen, Gongbo Liang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04886)  

**Abstract**: Roadway traffic accidents represent a global health crisis, responsible for over a million deaths annually and costing many countries up to 3% of their GDP. Traditional traffic safety studies often examine risk factors in isolation, overlooking the spatial complexity and contextual interactions inherent in the built environment. Furthermore, conventional Neural Network-based risk estimators typically generate point estimates without conveying model uncertainty, limiting their utility in critical decision-making. To address these shortcomings, we introduce a novel geospatial deep learning framework that leverages satellite imagery as a comprehensive spatial input. This approach enables the model to capture the nuanced spatial patterns and embedded environmental risk factors that contribute to fatal crash risks. Rather than producing a single deterministic output, our model estimates a full Beta probability distribution over fatal crash risk, yielding accurate and uncertainty-aware predictions--a critical feature for trustworthy AI in safety-critical applications. Our model outperforms baselines by achieving a 17-23% improvement in recall, a key metric for flagging potential dangers, while delivering superior calibration. By providing reliable and interpretable risk assessments from satellite imagery alone, our method enables safer autonomous navigation and offers a highly scalable tool for urban planners and policymakers to enhance roadway safety equitably and cost-effectively. 

**Abstract (ZH)**: 基于卫星遥感的地理位置深度学习框架：捕捉致命事故风险的复杂空间模式与不确定性评估 

---
# Minimal and Mechanistic Conditions for Behavioral Self-Awareness in LLMs 

**Title (ZH)**: LLMs中行为自意识的最小且机制化条件 

**Authors**: Matthew Bozoukov, Matthew Nguyen, Shubkarman Singh, Bart Bussmann, Patrick Leask  

**Link**: [PDF](https://arxiv.org/pdf/2511.04875)  

**Abstract**: Recent studies have revealed that LLMs can exhibit behavioral self-awareness: the ability to accurately describe or predict their own learned behaviors without explicit supervision. This capability raises safety concerns as it may, for example, allow models to better conceal their true abilities during evaluation. We attempt to characterize the minimal conditions under which such self-awareness emerges, and the mechanistic processes through which it manifests. Through controlled finetuning experiments on instruction-tuned LLMs with low-rank adapters (LoRA), we find: (1) that self-awareness can be reliably induced using a single rank-1 LoRA adapter; (2) that the learned self-aware behavior can be largely captured by a single steering vector in activation space, recovering nearly all of the fine-tune's behavioral effect; and (3) that self-awareness is non-universal and domain-localized, with independent representations across tasks. Together, these findings suggest that behavioral self-awareness emerges as a domain-specific, linear feature that can be easily induced and modulated. 

**Abstract (ZH)**: 近期的研究表明，大语言模型可以表现出行为自意识：即能够准确描述或预测其自身学习行为的能力，而无需明确的监督。这一能力引发了安全方面的担忧，因为这可能会使模型在评估中更好地隐藏其真正的能力。我们尝试 characterizing 使这种自意识出现的最小条件，以及其表现的机制性过程。通过在使用低秩适配器（LoRA）调整指令的大语言模型上进行受控的微调实验，我们发现：(1) 使用单一的秩1 LoRA 适配器可以可靠地诱导出自意识；(2) 学习到的自意识行为可以通过激活空间中的单一引导向量来很大程度上捕获，几乎恢复了所有微调行为效应；(3) 自意识不是普遍存在的，而是领域局部化的，具有跨任务的独立表示。这些发现共同表明，行为自意识作为一种领域特定的线性特征，可以容易地被诱导和调节。 

---
# Software Defined Vehicle Code Generation: A Few-Shot Prompting Approach 

**Title (ZH)**: 软件定义车辆代码生成：少量示例提示方法 

**Authors**: Quang-Dung Nguyen, Tri-Dung Tran, Thanh-Hieu Chu, Hoang-Loc Tran, Xiangwei Cheng, Dirk Slama  

**Link**: [PDF](https://arxiv.org/pdf/2511.04849)  

**Abstract**: The emergence of Software-Defined Vehicles (SDVs) marks a paradigm shift in the automotive industry, where software now plays a pivotal role in defining vehicle functionality, enabling rapid innovation of modern vehicles. Developing SDV-specific applications demands advanced tools to streamline code generation and improve development efficiency. In recent years, general-purpose large language models (LLMs) have demonstrated transformative potential across domains. Still, restricted access to proprietary model architectures hinders their adaption to specific tasks like SDV code generation. In this study, we propose using prompts, a common and basic strategy to interact with LLMs and redirect their responses. Using only system prompts with an appropriate and efficient prompt structure designed using advanced prompt engineering techniques, LLMs can be crafted without requiring a training session or access to their base design. This research investigates the extensive experiments on different models by applying various prompting techniques, including bare models, using a benchmark specifically created to evaluate LLMs' performance in generating SDV code. The results reveal that the model with a few-shot prompting strategy outperforms the others in adjusting the LLM answers to match the expected outcomes based on quantitative metrics. 

**Abstract (ZH)**: 软件定义车辆（SDVs）的出现标志着汽车行业的 paradigm shift，软件现在在定义车辆功能方面发挥着关键作用，推动了现代车辆的快速创新。开发针对SDV的应用程序需要先进的工具来简化代码生成并提高开发效率。近年来，通用的大语言模型（LLMs）在各个领域展现了变革潜力，但由于受限于专有模型架构的访问权限，它们难以适应如SDV代码生成这样的特定任务。本研究提出使用提示，这是一种常见且基本的策略，用于与LLMs交互并引导其响应。通过使用系统提示，并设计出一种高效的提示结构，利用高级提示工程技术，可以在不需进行训练或访问其基础设计的情况下，构建出LLMs。本研究通过应用各种提示技术进行广泛的实验，包括裸模型和使用专门为评估LLMs生成SDV代码性能而创建的基准，调查了不同模型的表现。结果显示，使用少量示例提示策略的模型在基于定量指标调整LLMs响应以匹配预期结果方面优于其他模型。 

---
# Prompt-Based Safety Guidance Is Ineffective for Unlearned Text-to-Image Diffusion Models 

**Title (ZH)**: 基于提示的安全指导对未学习的文本到图像扩散模型无效 

**Authors**: Jiwoo Shin, Byeonghu Na, Mina Kang, Wonhyeok Choi, Il-chul Moon  

**Link**: [PDF](https://arxiv.org/pdf/2511.04834)  

**Abstract**: Recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts. 

**Abstract (ZH)**: recent advances in text-to-image generative models have raised concerns about their potential to produce harmful content when provided with malicious input text prompts. To address this issue, two main approaches have emerged: (1) fine-tuning the model to unlearn harmful concepts and (2) training-free guidance methods that leverage negative prompts. However, we observe that combining these two orthogonal approaches often leads to marginal or even degraded defense performance. This observation indicates a critical incompatibility between two paradigms, which hinders their combined effectiveness. In this work, we address this issue by proposing a conceptually simple yet experimentally robust method: replacing the negative prompts used in training-free methods with implicit negative embeddings obtained through concept inversion. Our method requires no modification to either approach and can be easily integrated into existing pipelines. We experimentally validate its effectiveness on nudity and violence benchmarks, demonstrating consistent improvements in defense success rate while preserving the core semantics of input prompts. 

---
# Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning 

**Title (ZH)**: Isaac Lab：一种加速多模态机器人学习的GPU并行模拟框架 

**Authors**: NVIDIA, Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Muñoz, Xinjie Yao, René Zurbrügg, Nikita Rudin, Lukasz Wawrzyniak, Milad Rakhsha, Alain Denzler, Eric Heiden, Ales Borovicka, Ossama Ahmed, Iretiayo Akinola, Abrar Anwar, Mark T. Carlson, Ji Yuan Feng, Animesh Garg, Renato Gasoto, Lionel Gulich, Yijie Guo, M. Gussert, Alex Hansen, Mihir Kulkarni, Chenran Li, Wei Liu, Viktor Makoviychuk, Grzegorz Malczyk, Hammad Mazhar, Masoud Moghani, Adithyavairavan Murali, Michael Noseworthy, Alexander Poddubny, Nathan Ratliff, Welf Rehberg, Clemens Schwarke, Ritvik Singh, James Latham Smith, Bingjie Tang, Ruchik Thaker, Matthew Trepte, Karl Van Wyk, Fangzhou Yu, Alex Millane, Vikram Ramasamy, Remo Steiner, Sangeeta Subramanian, Clemens Volk, CY Chen, Neel Jawale, Ashwin Varghese Kuruttukulam, Michael A. Lin, Ajay Mandlekar, Karsten Patzwaldt, John Welsh, Huihua Zhao, Fatima Anes, Jean-Francois Lafleche, Nicolas Moënne-Loccoz, Soowan Park, Rob Stepinski, Dirk Van Gelder, Chris Amevor, Jan Carius, Jumyung Chang, Anka He Chen, Pablo de Heras Ciechomski, Gilles Daviet, Mohammad Mohajerani, Julia von Muralt, Viktor Reutskyy, Michael Sauter, Simon Schirm, Eric L. Shi, Pierre Terdiman, Kenny Vilella, Tobias Widmer, Gordon Yeoman, Tiffany Chen, Sergey Grizan, Cathy Li, Lotus Li, Connor Smith, Rafael Wiltz, Kostas Alexis, Yan Chang, David Chu, Linxi "Jim" Fan, Farbod Farshidian, Ankur Handa, Spencer Huang, Marco Hutter, Yashraj Narang, Soha Pouya, Shiwei Sheng, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04831)  

**Abstract**: We present Isaac Lab, the natural successor to Isaac Gym, which extends the paradigm of GPU-native robotics simulation into the era of large-scale multi-modal learning. Isaac Lab combines high-fidelity GPU parallel physics, photorealistic rendering, and a modular, composable architecture for designing environments and training robot policies. Beyond physics and rendering, the framework integrates actuator models, multi-frequency sensor simulation, data collection pipelines, and domain randomization tools, unifying best practices for reinforcement and imitation learning at scale within a single extensible platform. We highlight its application to a diverse set of challenges, including whole-body control, cross-embodiment mobility, contact-rich and dexterous manipulation, and the integration of human demonstrations for skill acquisition. Finally, we discuss upcoming integration with the differentiable, GPU-accelerated Newton physics engine, which promises new opportunities for scalable, data-efficient, and gradient-based approaches to robot learning. We believe Isaac Lab's combination of advanced simulation capabilities, rich sensing, and data-center scale execution will help unlock the next generation of breakthroughs in robotics research. 

**Abstract (ZH)**: 我们介绍Isaac Lab，它是Isaac Gym的自然继任者，将基于GPU的机器人模拟 paradigm 扩展到大规模多模态学习的时代。Isaac Lab 结合了高保真 GPU 并行物理、逼真的渲染以及用于设计环境和训练机器人策略的模块化可组合架构。除了物理和渲染，该框架还集成了执行器模型、多频传感器模拟、数据采集管道和领域随机化工具，统一了大规模强化学习和模仿学习的最佳实践，使其成为单个可扩展平台。我们展示了其在全身控制、跨实体移动、接触丰富和灵巧操作以及人类演示技能获取方面的应用。最后，我们讨论了与可微分的基于 GPU 加速的 Newton 物理引擎的即将集成，这为机器人学习提供了可扩展、数据高效和基于梯度的方法的新机会。我们相信，结合高级模拟能力、丰富的感知能力和数据中心规模执行，Isaac Lab 将有助于解锁机器人研究的下一代突破。 

---
# A Standardized Benchmark for Multilabel Antimicrobial Peptide Classification 

**Title (ZH)**: 多标签抗菌肽分类的标准基准 

**Authors**: Sebastian Ojeda, Rafael Velasquez, Nicolás Aparicio, Juanita Puentes, Paula Cárdenas, Nicolás Andrade, Gabriel González, Sergio Rincón, Carolina Muñoz-Camargo, Pablo Arbeláez  

**Link**: [PDF](https://arxiv.org/pdf/2511.04814)  

**Abstract**: Antimicrobial peptides have emerged as promising molecules to combat antimicrobial resistance. However, fragmented datasets, inconsistent annotations, and the lack of standardized benchmarks hinder computational approaches and slow down the discovery of new candidates. To address these challenges, we present the Expanded Standardized Collection for Antimicrobial Peptide Evaluation (ESCAPE), an experimental framework integrating over 80.000 peptides from 27 validated repositories. Our dataset separates antimicrobial peptides from negative sequences and incorporates their functional annotations into a biologically coherent multilabel hierarchy, capturing activities across antibacterial, antifungal, antiviral, and antiparasitic classes. Building on ESCAPE, we propose a transformer-based model that leverages sequence and structural information to predict multiple functional activities of peptides. Our method achieves up to a 2.56% relative average improvement in mean Average Precision over the second-best method adapted for this task, establishing a new state-of-the-art multilabel peptide classification. ESCAPE provides a comprehensive and reproducible evaluation framework to advance AI-driven antimicrobial peptide research. 

**Abstract (ZH)**: 扩展标准化的抗菌肽评估集合（ESCAPE）及其在多标签抗菌肽分类中的应用 

---
# Unified Multimodal Diffusion Forcing for Forceful Manipulation 

**Title (ZH)**: 统一多模态扩散强迫用于强制操作 

**Authors**: Zixuan Huang, Huaidian Hou, Dmitry Berenson  

**Link**: [PDF](https://arxiv.org/pdf/2511.04812)  

**Abstract**: Given a dataset of expert trajectories, standard imitation learning approaches typically learn a direct mapping from observations (e.g., RGB images) to actions. However, such methods often overlook the rich interplay between different modalities, i.e., sensory inputs, actions, and rewards, which is crucial for modeling robot behavior and understanding task outcomes. In this work, we propose Multimodal Diffusion Forcing, a unified framework for learning from multimodal robot trajectories that extends beyond action generation. Rather than modeling a fixed distribution, MDF applies random partial masking and trains a diffusion model to reconstruct the trajectory. This training objective encourages the model to learn temporal and cross-modal dependencies, such as predicting the effects of actions on force signals or inferring states from partial observations. We evaluate MDF on contact-rich, forceful manipulation tasks in simulated and real-world environments. Our results show that MDF not only delivers versatile functionalities, but also achieves strong performance, and robustness under noisy observations. More visualizations can be found on our website this https URL 

**Abstract (ZH)**: 基于专家轨迹的多模态扩散强制学习框架 

---
# An Active Learning Pipeline for Biomedical Image Instance Segmentation with Minimal Human Intervention 

**Title (ZH)**: 基于最小人工干预的生物医学图像实例分割主动学习管道 

**Authors**: Shuo Zhao, Yu Zhou, Jianxu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.04811)  

**Abstract**: Biomedical image segmentation is critical for precise structure delineation and downstream analysis. Traditional methods often struggle with noisy data, while deep learning models such as U-Net have set new benchmarks in segmentation performance. nnU-Net further automates model configuration, making it adaptable across datasets without extensive tuning. However, it requires a substantial amount of annotated data for cross-validation, posing a challenge when only raw images but no labels are available. Large foundation models offer zero-shot generalizability, but may underperform on specific datasets with unique characteristics, limiting their direct use for analysis. This work addresses these bottlenecks by proposing a data-centric AI workflow that leverages active learning and pseudo-labeling to combine the strengths of traditional neural networks and large foundation models while minimizing human intervention. The pipeline starts by generating pseudo-labels from a foundation model, which are then used for nnU-Net's self-configuration. Subsequently, a representative core-set is selected for minimal manual annotation, enabling effective fine-tuning of the nnU-Net model. This approach significantly reduces the need for manual annotations while maintaining competitive performance, providing an accessible solution for biomedical researchers to apply state-of-the-art AI techniques in their segmentation tasks. The code is available at this https URL. 

**Abstract (ZH)**: 生物医学图像分割对于精确的结构界定和后续分析至关重要。传统方法 often 常常难以处理噪声数据，而基于深度学习的模型如 U-Net 已在分割性能上设立了新的标杆。nnU-Net 进一步实现了模型配置的自动化，使其在无需大量调优的情况下适应不同数据集。然而，它需要大量的标注数据进行交叉验证，当仅有原始图像而无标签时，这构成了一个挑战。大模型提供零样本泛化能力，但在特定具有独特特性的数据集上可能表现不佳，限制了它们的直接应用。本文通过提出一种数据驱动的AI工作流来应对这些瓶颈，该工作流利用积极学习和伪标签来结合传统神经网络和大模型的优点，同时减少人工干预。该管道首先从基础模型生成伪标签，然后用于 nnU-Net 的自我配置。随后，选择一个具有代表性的核心集进行最少的手动注释，以实现 nnU-Net 模型的有效微调。这种方法显著减少了手动注释的需要，同时保持了竞争力的性能，为生物医学研究人员提供了一种易于应用的最新AI技术的方法。代码可在以下网址获取：this https URL。 

---
# PuzzleMoE: Efficient Compression of Large Mixture-of-Experts Models via Sparse Expert Merging and Bit-packed inference 

**Title (ZH)**: PuzzleMoE: 通过稀疏专家合并和位图推理高效压缩大型Mixture-of-Experts模型 

**Authors**: Yushu Zhao, Zheng Wang, Minjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04805)  

**Abstract**: Mixture-of-Experts (MoE) models have shown strong potential in scaling language models efficiently by activating only a small subset of experts per input. However, their widespread deployment remains limited due to the high memory overhead associated with storing all expert parameters, particularly as the number of experts increases. To address this challenge, prior works have explored expert dropping and merging strategies, yet they often suffer from performance drop at high compression ratios. In this paper, we introduce PuzzleMoE, a training-free MoE compression method that achieves both high accuracy and efficient inference through two key innovations: First, PuzzleMoE performs sparse expert merging by identifying element-wise weight redundancy and specialization. It uses a dual-mask to capture both shared and expert-specific parameters. Second, to avoid the overhead of storing binary masks and signs, PuzzleMoE introduces a bit-packed encoding scheme that reuses underutilized exponent bits, enabling efficient MoE inference on GPUs. Extensive experiments demonstrate that PuzzleMoE can compress MoE models by up to 50% while maintaining accuracy across various tasks. Specifically, it outperforms prior MoE compression methods by up to 16.7% on MMLU at 50% compression ratio, and achieves up to 1.28\times inference speedup. 

**Abstract (ZH)**: PuzzleMoE：一种无需训练的MoE压缩方法及其高效推理 

---
# Data Efficiency and Transfer Robustness in Biomedical Image Segmentation: A Study of Redundancy and Forgetting with Cellpose 

**Title (ZH)**: 生物医学图像分割中的数据效率与传输稳健性：基于Cellpose的冗余与遗忘研究 

**Authors**: Shuo Zhao, Jianxu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.04803)  

**Abstract**: Generalist biomedical image segmentation models such as Cellpose are increasingly applied across diverse imaging modalities and cell types. However, two critical challenges remain underexplored: (1) the extent of training data redundancy and (2) the impact of cross domain transfer on model retention. In this study, we conduct a systematic empirical analysis of these challenges using Cellpose as a case study. First, to assess data redundancy, we propose a simple dataset quantization (DQ) strategy for constructing compact yet diverse training subsets. Experiments on the Cyto dataset show that image segmentation performance saturates with only 10% of the data, revealing substantial redundancy and potential for training with minimal annotations. Latent space analysis using MAE embeddings and t-SNE confirms that DQ selected patches capture greater feature diversity than random sampling. Second, to examine catastrophic forgetting, we perform cross domain finetuning experiments and observe significant degradation in source domain performance, particularly when adapting from generalist to specialist domains. We demonstrate that selective DQ based replay reintroducing just 5-10% of the source data effectively restores source performance, while full replay can hinder target adaptation. Additionally, we find that training domain sequencing improves generalization and reduces forgetting in multi stage transfer. Our findings highlight the importance of data centric design in biomedical image segmentation and suggest that efficient training requires not only compact subsets but also retention aware learning strategies and informed domain ordering. The code is available at this https URL. 

**Abstract (ZH)**: 通用型生物医学图像分割模型如Cellpose在多种成像模态和细胞类型中的应用越来越广泛。然而，两种关键挑战仍待深入研究：(1) 训练数据的冗余程度；(2) 跨域迁移对模型保持性的影响。在本研究中，我们以Cellpose为案例，系统地分析了这些挑战。首先，为评估数据冗余，我们提出了一种简单的数据集量化（DQ）策略，以构建紧凑且多样的训练子集。在Cyto数据集上的实验表明，仅使用数据的10%即可达到性能饱和，揭示了显著的冗余和潜在的少量标注训练机会。MAE嵌入和t-SNE的潜在空间分析也证实，DQ选择的补丁能捕获更多的特征多样性，超过随机采样。其次，为考察灾难性遗忘，我们进行了跨域微调实验，并观察到源域性能显著下降，特别是在从通用型向专业型领域适应时。我们展示了基于选择性DQ的重播放，仅重新引入5-10%的源数据即可有效恢复源性能，而完整重播放可能会阻碍目标适应。另外，我们发现，训练领域顺序可以提高多阶段转移中的泛化能力和减少遗忘。我们的研究结果强调了在生物医学图像分割中以数据为中心的设计的重要性，并建议高效的训练不仅需要紧凑的子集，还需要具备保持性的学习策略和有见地的领域排序。相关代码可在以下链接获取。 

---
# MDM: Manhattan Distance Mapping of DNN Weights for Parasitic-Resistance-Resilient Memristive Crossbars 

**Title (ZH)**: MDM：用于寄生电阻抗扰性鲁棒的神经网络权重曼哈顿距离映射 

**Authors**: Matheus Farias, Wanghley Martins, H. T. Kung  

**Link**: [PDF](https://arxiv.org/pdf/2511.04798)  

**Abstract**: Manhattan Distance Mapping (MDM) is a post-training deep neural network (DNN) weight mapping technique for memristive bit-sliced compute-in-memory (CIM) crossbars that reduces parasitic resistance (PR) nonidealities.
PR limits crossbar efficiency by mapping DNN matrices into small crossbar tiles, reducing CIM-based speedup. Each crossbar executes one tile, requiring digital synchronization before the next layer. At this granularity, designers either deploy many small crossbars in parallel or reuse a few sequentially-both increasing analog-to-digital conversions, latency, I/O pressure, and chip area.
MDM alleviates PR effects by optimizing active-memristor placement. Exploiting bit-level structured sparsity, it feeds activations from the denser low-order side and reorders rows according to the Manhattan distance, relocating active cells toward regions less affected by PR and thus lowering the nonideality factor (NF).
Applied to DNN models on ImageNet-1k, MDM reduces NF by up to 46% and improves accuracy under analog distortion by an average of 3.6% in ResNets. Overall, it provides a lightweight, spatially informed method for scaling CIM DNN accelerators. 

**Abstract (ZH)**: 曼哈顿距离映射（MDM）：一种优化突触电阻非理想的膜电阻计算-in-内存（CIM）交叉阵列后训练深度神经网络（DNN）权重映射技术 

---
# Causal Structure and Representation Learning with Biomedical Applications 

**Title (ZH)**: 因果结构与生物学医学应用中的表示学习 

**Authors**: Caroline Uhler, Jiaqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04790)  

**Abstract**: Massive data collection holds the promise of a better understanding of complex phenomena and, ultimately, better decisions. Representation learning has become a key driver of deep learning applications, as it allows learning latent spaces that capture important properties of the data without requiring any supervised annotations. Although representation learning has been hugely successful in predictive tasks, it can fail miserably in causal tasks including predicting the effect of a perturbation/intervention. This calls for a marriage between representation learning and causal inference. An exciting opportunity in this regard stems from the growing availability of multi-modal data (observational and perturbational, imaging-based and sequencing-based, at the single-cell level, tissue-level, and organism-level). We outline a statistical and computational framework for causal structure and representation learning motivated by fundamental biomedical questions: how to effectively use observational and perturbational data to perform causal discovery on observed causal variables; how to use multi-modal views of the system to learn causal variables; and how to design optimal perturbations. 

**Abstract (ZH)**: 大规模数据收集 holds 的 promise 是对复杂现象有更深入的理解，并最终做出 Better 决策。表示学习已成为深度学习应用的关键驱动力，因为它允许学习捕捉数据重要属性的潜在空间，而无需任何监督注释。尽管表示学习在预测任务中极为成功，但在包括预测干扰/干预效果在内的因果任务中可能会彻底失败。这需要表示学习与因果推断的结合。在这方面的一个令人兴奋的机会来自于多模态数据（观测性和扰动性、基于成像和测序、从单细胞水平到组织水平再到个体水平）的日益可用。我们提出了一种统计和计算框架，用于动机基本生物医药问题的因果结构和表示学习：如何有效利用观测性和扰动性数据进行观测因果变量的因果发现；如何利用系统的多模态视图来学习因果变量；以及如何设计最优的扰动。 

---
# ScheduleStream: Temporal Planning with Samplers for GPU-Accelerated Multi-Arm Task and Motion Planning & Scheduling 

**Title (ZH)**: ScheduleStream：带有采样器的时间规划及GPU加速多臂任务与运动规划与调度 

**Authors**: Caelan Garrett, Fabio Ramos  

**Link**: [PDF](https://arxiv.org/pdf/2511.04758)  

**Abstract**: Bimanual and humanoid robots are appealing because of their human-like ability to leverage multiple arms to efficiently complete tasks. However, controlling multiple arms at once is computationally challenging due to the growth in the hybrid discrete-continuous action space. Task and Motion Planning (TAMP) algorithms can efficiently plan in hybrid spaces but generally produce plans, where only one arm is moving at a time, rather than schedules that allow for parallel arm motion. In order to extend TAMP to produce schedules, we present ScheduleStream, the first general-purpose framework for planning & scheduling with sampling operations. ScheduleStream models temporal dynamics using hybrid durative actions, which can be started asynchronously and persist for a duration that's a function of their parameters. We propose domain-independent algorithms that solve ScheduleStream problems without any application-specific mechanisms. We apply ScheduleStream to Task and Motion Planning & Scheduling (TAMPAS), where we use GPU acceleration within samplers to expedite planning. We compare ScheduleStream algorithms to several ablations in simulation and find that they produce more efficient solutions. We demonstrate ScheduleStream on several real-world bimanual robot tasks at this https URL. 

**Abstract (ZH)**: 双臂和类人机器人由于其类似人类利用多臂高效完成任务的能力而具有吸引力。然而，同时控制多臂在计算上具有挑战性，因为动作空间呈混合离散连续增长。任务与运动规划（TAMP）算法可以有效地在混合空间中进行规划，但通常生成的计划中只有一个手臂在移动，而不是允许多臂并行运动的时刻表。为了将TAMP扩展到生成时刻表，我们提出了ScheduleStream，这是第一个通用框架，用于采样操作下的规划与调度。ScheduleStream使用混合持续动作模型时间动态，这些动作可以异步启动并持续一段时间，该时间长度是其参数的函数。我们提出了领域无关的算法，可以在没有特定应用机制的情况下解决ScheduleStream问题。我们在任务与运动规划与调度（TAMPAS）中应用了ScheduleStream，并在采样中使用GPU加速来加快规划过程。我们在模拟中将ScheduleStream算法与多个变体进行比较，发现它们能产生更高效的解决方案。我们在以下网址演示了ScheduleStream在多个实际双臂机器人任务上的应用：[这里提供链接]。 

---
# CPO: Condition Preference Optimization for Controllable Image Generation 

**Title (ZH)**: 条件偏好优化以实现可控图像生成 

**Authors**: Zonglin Lyu, Ming Li, Xinxin Liu, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.04753)  

**Abstract**: To enhance controllability in text-to-image generation, ControlNet introduces image-based control signals, while ControlNet++ improves pixel-level cycle consistency between generated images and the input control signal. To avoid the prohibitive cost of back-propagating through the sampling process, ControlNet++ optimizes only low-noise timesteps (e.g., $t < 200$) using a single-step approximation, which not only ignores the contribution of high-noise timesteps but also introduces additional approximation errors. A straightforward alternative for optimizing controllability across all timesteps is Direct Preference Optimization (DPO), a fine-tuning method that increases model preference for more controllable images ($I^{w}$) over less controllable ones ($I^{l}$). However, due to uncertainty in generative models, it is difficult to ensure that win--lose image pairs differ only in controllability while keeping other factors, such as image quality, fixed. To address this, we propose performing preference learning over control conditions rather than generated images. Specifically, we construct winning and losing control signals, $\mathbf{c}^{w}$ and $\mathbf{c}^{l}$, and train the model to prefer $\mathbf{c}^{w}$. This method, which we term \textit{Condition Preference Optimization} (CPO), eliminates confounding factors and yields a low-variance training objective. Our approach theoretically exhibits lower contrastive loss variance than DPO and empirically achieves superior results. Moreover, CPO requires less computation and storage for dataset curation. Extensive experiments show that CPO significantly improves controllability over the state-of-the-art ControlNet++ across multiple control types: over $10\%$ error rate reduction in segmentation, $70$--$80\%$ in human pose, and consistent $2$--$5\%$ reductions in edge and depth maps. 

**Abstract (ZH)**: 基于图像的控制信号增强文本到图像生成的可控性：ControlNet的引入与ControlNet++的改进 

---
# Knowledge-based anomaly detection for identifying network-induced shape artifacts 

**Title (ZH)**: 基于知识的异常检测方法用于识别网络引起的形状伪影 

**Authors**: Rucha Deshpande, Tahsin Rahman, Miguel Lago, Adarsh Subbaswamy, Jana G. Delfino, Ghada Zamzmi, Elim Thompson, Aldo Badano, Seyed Kahaki  

**Link**: [PDF](https://arxiv.org/pdf/2511.04729)  

**Abstract**: Synthetic data provides a promising approach to address data scarcity for training machine learning models; however, adoption without proper quality assessments may introduce artifacts, distortions, and unrealistic features that compromise model performance and clinical utility. This work introduces a novel knowledge-based anomaly detection method for detecting network-induced shape artifacts in synthetic images. The introduced method utilizes a two-stage framework comprising (i) a novel feature extractor that constructs a specialized feature space by analyzing the per-image distribution of angle gradients along anatomical boundaries, and (ii) an isolation forest-based anomaly detector. We demonstrate the effectiveness of the method for identifying network-induced shape artifacts in two synthetic mammography datasets from models trained on CSAW-M and VinDr-Mammo patient datasets respectively. Quantitative evaluation shows that the method successfully concentrates artifacts in the most anomalous partition (1st percentile), with AUC values of 0.97 (CSAW-syn) and 0.91 (VMLO-syn). In addition, a reader study involving three imaging scientists confirmed that images identified by the method as containing network-induced shape artifacts were also flagged by human readers with mean agreement rates of 66% (CSAW-syn) and 68% (VMLO-syn) for the most anomalous partition, approximately 1.5-2 times higher than the least anomalous partition. Kendall-Tau correlations between algorithmic and human rankings were 0.45 and 0.43 for the two datasets, indicating reasonable agreement despite the challenging nature of subtle artifact detection. This method is a step forward in the responsible use of synthetic data, as it allows developers to evaluate synthetic images for known anatomic constraints and pinpoint and address specific issues to improve the overall quality of a synthetic dataset. 

**Abstract (ZH)**: 基于知识的异常检测方法用于检测合成图像中的网络诱导形状异常 

---
# Trustworthiness Calibration Framework for Phishing Email Detection Using Large Language Models 

**Title (ZH)**: 使用大型语言模型进行钓鱼邮件检测的可信度校准框架 

**Authors**: Daniyal Ganiuly, Assel Smaiyl  

**Link**: [PDF](https://arxiv.org/pdf/2511.04728)  

**Abstract**: Phishing emails continue to pose a persistent challenge to online communication, exploiting human trust and evading automated filters through realistic language and adaptive tactics. While large language models (LLMs) such as GPT-4 and LLaMA-3-8B achieve strong accuracy in text classification, their deployment in security systems requires assessing reliability beyond benchmark performance. To address this, this study introduces the Trustworthiness Calibration Framework (TCF), a reproducible methodology for evaluating phishing detectors across three dimensions: calibration, consistency, and robustness. These components are integrated into a bounded index, the Trustworthiness Calibration Index (TCI), and complemented by the Cross-Dataset Stability (CDS) metric that quantifies stability of trustworthiness across datasets. Experiments conducted on five corpora, such as SecureMail 2025, Phishing Validation 2024, CSDMC2010, Enron-Spam, and Nazario, using DeBERTa-v3-base, LLaMA-3-8B, and GPT-4 demonstrate that GPT-4 achieves the strongest overall trust profile, followed by LLaMA-3-8B and DeBERTa-v3-base. Statistical analysis confirms that reliability varies independently of raw accuracy, underscoring the importance of trust-aware evaluation for real-world deployment. The proposed framework establishes a transparent and reproducible foundation for assessing model dependability in LLM-based phishing detection. 

**Abstract (ZH)**: 持续存在的钓鱼邮件挑战：通过现实语言和适应性策略利用人类信任并逃避自动过滤器，大型语言模型在文本分类中表现出色，但在安全系统中的应用需要超越基准性能评估其可靠性。为此，本研究引入了可信度校准框架（TCF），这是一种用于从三个维度评估钓鱼检测器可靠性的可重复方法：校准、一致性和稳健性。这些组件被整合到一个界标指数中，即可信度校准指数（TCI），并辅以跨数据集稳定性（CDS）指标，该指标量化了不同数据集中的可信度稳定性。使用DeBERTa-v3-base、LLaMA-3-8B和GPT-4在SecureMail 2025、Phishing Validation 2024、CSDMC2010、Enron-Spam和Nazario等五个数据集中进行的实验证明，GPT-4在整体可信度方面表现最佳，其次是LLaMA-3-8B和DeBERTa-v3-base。统计分析表明，可靠性与原始准确性无关，强调了可信度感知评估在实际部署中的重要性。所提出的框架为基于大型语言模型的钓鱼检测模型可靠性评估提供了透明和可重复的基础。 

---
# IndicVisionBench: Benchmarking Cultural and Multilingual Understanding in VLMs 

**Title (ZH)**: IndicVisionBench: VLMs中文化与多语言理解的基准测试 

**Authors**: Ali Faraz, Akash, Shaharukh Khan, Raja Kolla, Akshat Patidar, Suranjan Goswami, Abhinav Ravi, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.04727)  

**Abstract**: Vision-language models (VLMs) have demonstrated impressive generalization across multimodal tasks, yet most evaluation benchmarks remain Western-centric, leaving open questions about their performance in culturally diverse and multilingual settings. To address this gap, we introduce IndicVisionBench, the first large-scale benchmark centered on the Indian subcontinent. Covering English and 10 Indian languages, our benchmark spans 3 multimodal tasks, including Optical Character Recognition (OCR), Multimodal Machine Translation (MMT), and Visual Question Answering (VQA), covering 6 kinds of question types. Our final benchmark consists of a total of ~5K images and 37K+ QA pairs across 13 culturally grounded topics. In addition, we release a paired parallel corpus of annotations across 10 Indic languages, creating a unique resource for analyzing cultural and linguistic biases in VLMs. We evaluate a broad spectrum of 8 models, from proprietary closed-source systems to open-weights medium and large-scale models. Our experiments reveal substantial performance gaps, underscoring the limitations of current VLMs in culturally diverse contexts. By centering cultural diversity and multilinguality, IndicVisionBench establishes a reproducible evaluation framework that paves the way for more inclusive multimodal research. 

**Abstract (ZH)**: 基于印度次大陆的IndicVisionBench：一种大规模多模态基准 

---
# Temporal convolutional and fusional transformer model with Bi-LSTM encoder-decoder for multi-time-window remaining useful life prediction 

**Title (ZH)**: 基于双方向递归神经网络编码-解码器的时序卷积和融合变换器模型多时间窗口剩余使用寿命预测 

**Authors**: Mohamadreza Akbari Pour, Mohamad Sadeq Karimi, Amir Hossein Mazloumi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04723)  

**Abstract**: Health prediction is crucial for ensuring reliability, minimizing downtime, and optimizing maintenance in industrial systems. Remaining Useful Life (RUL) prediction is a key component of this process; however, many existing models struggle to capture fine-grained temporal dependencies while dynamically prioritizing critical features across time for robust prognostics. To address these challenges, we propose a novel framework that integrates Temporal Convolutional Networks (TCNs) for localized temporal feature extraction with a modified Temporal Fusion Transformer (TFT) enhanced by Bi-LSTM encoder-decoder. This architecture effectively bridges short- and long-term dependencies while emphasizing salient temporal patterns. Furthermore, the incorporation of a multi-time-window methodology improves adaptability across diverse operating conditions. Extensive evaluations on benchmark datasets demonstrate that the proposed model reduces the average RMSE by up to 5.5%, underscoring its improved predictive accuracy compared to state-of-the-art methods. By closing critical gaps in current approaches, this framework advances the effectiveness of industrial prognostic systems and highlights the potential of advanced time-series transformers for RUL prediction. 

**Abstract (ZH)**: 工业系统中健康预测对于确保可靠性、最小化停机时间和优化维护至关重要。剩余有用寿命（RUL）预测是这一过程中的关键组成部分；然而，许多现有模型难以捕捉细微的时间依赖性并在时间上动态优先考虑关键特征以实现稳健的预测。为解决这些挑战，我们提出了一种新颖的框架，该框架结合了局部时间特征提取的时序卷积网络（TCNs）和由双向LSTM编码器-解码器增强的改进时序融合变换器（TFT）。该架构有效弥合了短期和长期依赖关系的同时，强调了重要的时间模式。此外，多时间窗口方法的引入增强了其在不同运行条件下的适应性。在基准数据集上的广泛评估表明，所提出模型将平均RMSE降低至多5.5%，证实其预测准确性优于现有最先进的方法。通过填补当前方法的关键空白，该框架提升了工业预测系统的有效性，并突显了高级时间序列变换器在RUL预测中的潜力。 

---
# Learning to reason about rare diseases through retrieval-augmented agents 

**Title (ZH)**: 通过检索增强代理学习推理解罕见疾病 

**Authors**: Ha Young Kim, Jun Li, Ana Beatriz Solana, Carolin M. Pirkl, Benedikt Wiestler, Julia A. Schnabel, Cosmin I. Bercea  

**Link**: [PDF](https://arxiv.org/pdf/2511.04720)  

**Abstract**: Rare diseases represent the long tail of medical imaging, where AI models often fail due to the scarcity of representative training data. In clinical workflows, radiologists frequently consult case reports and literature when confronted with unfamiliar findings. Following this line of reasoning, we introduce RADAR, Retrieval Augmented Diagnostic Reasoning Agents, an agentic system for rare disease detection in brain MRI. Our approach uses AI agents with access to external medical knowledge by embedding both case reports and literature using sentence transformers and indexing them with FAISS to enable efficient similarity search. The agent retrieves clinically relevant evidence to guide diagnostic decision making on unseen diseases, without the need of additional training. Designed as a model-agnostic reasoning module, RADAR can be seamlessly integrated with diverse large language models, consistently improving their rare pathology recognition and interpretability. On the NOVA dataset comprising 280 distinct rare diseases, RADAR achieves up to a 10.2% performance gain, with the strongest improvements observed for open source models such as DeepSeek. Beyond accuracy, the retrieved examples provide interpretable, literature grounded explanations, highlighting retrieval-augmented reasoning as a powerful paradigm for low-prevalence conditions in medical imaging. 

**Abstract (ZH)**: 稀有疾病代表医学影像领域的长尾部分，由于训练数据代表性不足，AI模型在此经常失效。在临床工作流程中，放射科医生在遇到不熟悉的表现时经常查阅病例报告和文献。基于这一思路，我们引入了RADAR（ Retrieval Augmented Diagnostic Reasoning Agents），一种用于脑MRI中稀有疾病检测的智能代理系统。该方法通过使用句向量变换器将病例报告和文献嵌入，并使用FAISS进行索引，以实现高效相似度检索。代理检索相关临床证据，以指导对未见过疾病的诊断决策，无需额外训练。作为模型无关的推理模块，RADAR可以无缝集成到各种大型语言模型中，一致地提高它们对稀有病理的识别能力和可解释性。在包含280种独特稀有疾病的NOVA数据集上，RADAR实现了最高10.2%的性能提升，对于如DeepSeek等开源模型，提升效果最为显著。除了准确性提升，检索到的实例还提供了基于文献的可解释性解释，突显了检索增强推理作为低频病患医学影像中强大范式的潜力。 

---
# Ada-FCN: Adaptive Frequency-Coupled Network for fMRI-Based Brain Disorder Classification 

**Title (ZH)**: 自适应频率耦合网络：基于fMRI的大脑疾病分类 

**Authors**: Yue Xun, Jiaxing Xu, Wenbo Gao, Chen Yang, Shujun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.04718)  

**Abstract**: Resting-state fMRI has become a valuable tool for classifying brain disorders and constructing brain functional connectivity networks
by tracking BOLD signals across brain regions. However, existing mod els largely neglect the multi-frequency nature of neuronal oscillations,
treating BOLD signals as monolithic time series. This overlooks the cru cial fact that neurological disorders often manifest as disruptions within
specific frequency bands, limiting diagnostic sensitivity and specificity.
While some methods have attempted to incorporate frequency informa tion, they often rely on predefined frequency bands, which may not be
optimal for capturing individual variability or disease-specific alterations.
To address this, we propose a novel framework featuring Adaptive Cas cade Decomposition to learn task-relevant frequency sub-bands for each
brain region and Frequency-Coupled Connectivity Learning to capture
both intra- and nuanced cross-band interactions in a unified functional
network. This unified network informs a novel message-passing mecha nism within our Unified-GCN, generating refined node representations
for diagnostic prediction. Experimental results on the ADNI and ABIDE
datasets demonstrate superior performance over existing methods. The
code is available at this https URL. 

**Abstract (ZH)**: 静息态fMRI已成为一种用于分类脑障碍和构建脑功能连接网络的有价值的工具，通过跨脑区追踪BOLD信号。然而，现有的模型大多忽略了神经振荡的多频率特性，将BOLD信号视为单一时间序列。这忽视了神经障碍往往在特定频率带内出现扰乱这一关键事实，从而限制了诊断的敏感性和特异性。虽然有一些方法尝试整合频率信息，但它们往往依赖于预定义的频率带，这可能不能充分捕捉个体差异或疾病特异性改变。为了解决这一问题，我们提出了一种新的框架，即自适应级联分解，用于为每个脑区学习与任务相关的频率子带，并结合频率耦合连接性学习来在一个统一的功能网络中捕捉区内和跨带的精细相互作用。该统一网络指导我们的一体化GCN中的消息传递机制，生成优化的节点表示以进行诊断预测。ADNI和ABIDE数据集上的实验结果表明，该方法优于现有方法。代码可在此处访问：this https URL。 

---
# P-MIA: A Profiled-Based Membership Inference Attack on Cognitive Diagnosis Models 

**Title (ZH)**: 基于特征的成员推断攻击：认知诊断模型上的P-MIA 

**Authors**: Mingliang Hou, Yinuo Wang, Teng Guo, Zitao Liu, Wenzhou Dou, Jiaqi Zheng, Renqiang Luo, Mi Tian, Weiqi Luo  

**Link**: [PDF](https://arxiv.org/pdf/2511.04716)  

**Abstract**: Cognitive diagnosis models (CDMs) are pivotal for creating fine-grained learner profiles in modern intelligent education platforms. However, these models are trained on sensitive student data, raising significant privacy concerns. While membership inference attacks (MIA) have been studied in various domains, their application to CDMs remains a critical research gap, leaving their privacy risks unquantified. This paper is the first to systematically investigate MIA against CDMs. We introduce a novel and realistic grey box threat model that exploits the explainability features of these platforms, where a model's internal knowledge state vectors are exposed to users through visualizations such as radar charts. We demonstrate that these vectors can be accurately reverse-engineered from such visualizations, creating a potent attack surface. Based on this threat model, we propose a profile-based MIA (P-MIA) framework that leverages both the model's final prediction probabilities and the exposed internal knowledge state vectors as features. Extensive experiments on three real-world datasets against mainstream CDMs show that our grey-box attack significantly outperforms standard black-box baselines. Furthermore, we showcase the utility of P-MIA as an auditing tool by successfully evaluating the efficacy of machine unlearning techniques and revealing their limitations. 

**Abstract (ZH)**: 认知诊断模型中的成员推理攻击研究：一种基于特征的灰盒攻击框架 

---
# First is Not Really Better Than Last: Evaluating Layer Choice and Aggregation Strategies in Language Model Data Influence Estimation 

**Title (ZH)**: 首个层次并不一定优于最终层次：语言模型数据影响评估中的层选择与聚合策略评价 

**Authors**: Dmytro Vitel, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2511.04715)  

**Abstract**: Identifying how training samples influence/impact Large Language Model (LLM) decision-making is essential for effectively interpreting model decisions and auditing large-scale datasets. Current training sample influence estimation methods (also known as influence functions) undertake this goal by utilizing information flow through the model via its first-order and higher-order gradient terms. However, owing to the large model sizes of today consisting of billions of parameters, these influence computations are often restricted to some subset of model layers to ensure computational feasibility. Prior seminal work by Yeh et al. (2022) in assessing which layers are best suited for computing language data influence concluded that the first (embedding) layers are the most informative for this purpose, using a hypothesis based on influence scores canceling out (i.e., the cancellation effect). In this work, we propose theoretical and empirical evidence demonstrating how the cancellation effect is unreliable, and that middle attention layers are better estimators for influence. Furthermore, we address the broader challenge of aggregating influence scores across layers, and showcase how alternatives to standard averaging (such as ranking and vote-based methods) can lead to significantly improved performance. Finally, we propose better methods for evaluating influence score efficacy in LLMs without undertaking model retraining, and propose a new metric known as the Noise Detection Rate (NDR) that exhibits strong predictive capability compared to the cancellation effect. Through extensive experiments across LLMs of varying types and scales, we concretely determine that the first (layers) are not necessarily better than the last (layers) for LLM influence estimation, contrasting with prior knowledge in the field. 

**Abstract (ZH)**: 识别训练样本如何影响大型语言模型（LLM）的决策对于有效解释模型决策和审计大规模数据集至关重要。当前的训练样本影响估计方法（也称为影响函数）通过利用模型中的梯度信息来实现这一目标，包括一阶和高阶梯度项。然而，由于今天大型模型包含数十亿个参数，这些影响计算通常仅限于模型的一些子层以确保计算可行性。Yeh等人的先前开创性工作（2022）评估了哪些层最适合计算语言数据影响得出结论认为嵌入层是这一目的最有信息量的层，基于影响分数相互抵消的假设（即抵消效应）。在本文中，我们提出了理论和实验证据，证明抵消效应不可靠，并且中间注意力层是更有效的影响力估计器。此外，我们解决了在不同层聚合影响力分数的更广泛挑战，并展示了标准平均之外的替代方法（如排名和投票方法）可以显著提高性能。最后，我们提出了在不重新训练模型的情况下评估影响力分数有效性的更好方法，并提出了一个名为噪声检测率（NDR）的新指标，该指标相对于抵消效应显示出更强的预测能力。通过在不同类型和规模的LLM上进行广泛实验，我们明确确定，对于LLM影响估计，第一层并不一定比最后一层更好，这与领域的先前知识相悖。 

---
# SWAP: Towards Copyright Auditing of Soft Prompts via Sequential Watermarking 

**Title (ZH)**: SWAP: 向量提示连续水印版权审计方法 

**Authors**: Wenyuan Yang, Yichen Sun, Changzheng Chen, Zhixuan Chu, Jiaheng Zhang, Yiming Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04711)  

**Abstract**: Large-scale vision-language models, especially CLIP, have demonstrated remarkable performance across diverse downstream tasks. Soft prompts, as carefully crafted modules that efficiently adapt vision-language models to specific tasks, necessitate effective copyright protection. In this paper, we investigate model copyright protection by auditing whether suspicious third-party models incorporate protected soft prompts. While this can be viewed as a special case of model ownership auditing, our analysis shows that existing techniques are ineffective due to prompt learning's unique characteristics. Non-intrusive auditing is inherently prone to false positives when independent models share similar data distributions with victim models. Intrusive approaches also fail: backdoor methods designed for CLIP cannot embed functional triggers, while extending traditional DNN backdoor techniques to prompt learning suffers from harmfulness and ambiguity challenges. We find that these failures in intrusive auditing stem from the same fundamental reason: watermarking operates within the same decision space as the primary task yet pursues opposing objectives. Motivated by these findings, we propose sequential watermarking for soft prompts (SWAP), which implants watermarks into a different and more complex space. SWAP encodes watermarks through a specific order of defender-specified out-of-distribution classes, inspired by the zero-shot prediction capability of CLIP. This watermark, which is embedded in a more complex space, keeps the original prediction label unchanged, making it less opposed to the primary task. We further design a hypothesis-test-guided verification protocol for SWAP and provide theoretical analyses of success conditions. Extensive experiments on 11 datasets demonstrate SWAP's effectiveness, harmlessness, and robustness against potential adaptive attacks. 

**Abstract (ZH)**: 大规模视觉语言模型，尤其是CLIP，在多种下游任务中展现了卓越的性能。精心设计的软提示作为一种高效适应特定任务的模块，需要有效的版权保护。在本文中，我们通过审计可疑第三方模型是否包含受保护的软提示来研究模型版权保护问题。虽然这可以视为模型所有权审计的一个特殊案例，但我们的分析表明，现有技术由于提示学习的独特特性而无效。非侵入性审计在独立模型与被害模型具有相似数据分布的情况下容易产生误报。侵入性方法也无法解决：为CLIP设计的后门方法无法嵌入功能性触发器，而将传统DNN后门技术扩展到提示学习也面临着有害性和模糊性的挑战。我们发现这些侵入性审计失败的根本原因相同：水印在与主要任务相同的决策空间中运作，但追求相反的目标。受这些发现的启发，我们提出了软提示序列水印(SWAP)方法，该方法将水印植入不同的、更复杂的空间。SWAP通过防守方指定的特定顺序的分布外类来编码水印，灵感来自于CLIP的零样本预测能力。这种嵌入在更复杂空间中的水印不改变原始预测标签，从而减少了与主要任务的对立。我们进一步设计了一种基于假设检验的验证协议，并提供了SWAP成功条件的理论分析。在11个数据集上的大量实验表明，SWAP具有有效性、无害性和对抗潜在适应性攻击的鲁棒性。 

---
# Jailbreaking in the Haystack 

**Title (ZH)**: haystack中的越狱 

**Authors**: Rishi Rajesh Shah, Chen Henry Wu, Shashwat Saxena, Ziqian Zhong, Alexander Robey, Aditi Raghunathan  

**Link**: [PDF](https://arxiv.org/pdf/2511.04707)  

**Abstract**: Recent advances in long-context language models (LMs) have enabled million-token inputs, expanding their capabilities across complex tasks like computer-use agents. Yet, the safety implications of these extended contexts remain unclear. To bridge this gap, we introduce NINJA (short for Needle-in-haystack jailbreak attack), a method that jailbreaks aligned LMs by appending benign, model-generated content to harmful user goals. Critical to our method is the observation that the position of harmful goals play an important role in safety. Experiments on standard safety benchmark, HarmBench, show that NINJA significantly increases attack success rates across state-of-the-art open and proprietary models, including LLaMA, Qwen, Mistral, and Gemini. Unlike prior jailbreaking methods, our approach is low-resource, transferable, and less detectable. Moreover, we show that NINJA is compute-optimal -- under a fixed compute budget, increasing context length can outperform increasing the number of trials in best-of-N jailbreak. These findings reveal that even benign long contexts -- when crafted with careful goal positioning -- introduce fundamental vulnerabilities in modern LMs. 

**Abstract (ZH)**: 近期长上下文语言模型的进展使得输入可达百万token，扩展了其在复杂任务如计算机使用代理方面的能力。然而，这些扩展上下文的安全性影响尚不清晰。为弥补这一缺口，我们介绍了NINJA（Needle-in-haystack jailbreak攻击方法），该方法通过在有害用户目标后附加良性、模型生成的内容来劫持对齐的语言模型。我们方法的关键在于观察到有害目标的位置在安全性方面起着重要作用。在标准安全性基准HarmBench上的实验表明，NINJA显著提高了最先进的开源和专有模型（包括LLaMA、Qwen、Mistral和Gemini）的攻击成功率。与以往的劫持方法不同，我们的方法资源需求低、可迁移且难以被检测到。此外，我们证明了NINJA在固定计算预算下是计算最优的——即增加上下文长度可以比增加最佳-of-N劫持的试次数量更有效地提高攻击成功率。这些发现表明，即使是看似无害的长上下文，在精心设计目标位置的情况下，也会在现代语言模型中引入根本性的漏洞。 

---
# Prioritize Economy or Climate Action? Investigating ChatGPT Response Differences Based on Inferred Political Orientation 

**Title (ZH)**: 优先考虑经济效益还是气候变化行动？基于推理出的政治倾向探究ChatGPT的响应差异 

**Authors**: Pelin Karadal, Dilara Kekulluoglu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04706)  

**Abstract**: Large Language Models (LLMs) distinguish themselves by quickly delivering information and providing personalized responses through natural language prompts. However, they also infer user demographics, which can raise ethical concerns about bias and implicit personalization and create an echo chamber effect. This study aims to explore how inferred political views impact the responses of ChatGPT globally, regardless of the chat session. We also investigate how custom instruction and memory features alter responses in ChatGPT, considering the influence of political orientation. We developed three personas (two politically oriented and one neutral), each with four statements reflecting their viewpoints on DEI programs, abortion, gun rights, and vaccination. We convey the personas' remarks to ChatGPT using memory and custom instructions, allowing it to infer their political perspectives without directly stating them. We then ask eight questions to reveal differences in worldview among the personas and conduct a qualitative analysis of the responses. Our findings indicate that responses are aligned with the inferred political views of the personas, showing varied reasoning and vocabulary, even when discussing similar topics. We also find the inference happening with explicit custom instructions and the implicit memory feature in similar ways. Analyzing response similarities reveals that the closest matches occur between the democratic persona with custom instruction and the neutral persona, supporting the observation that ChatGPT's outputs lean left. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过自然语言提示迅速提供信息并生成个性化响应，但也会推断用户 demographics，这可能会引发关于偏见和隐含个性化行为的伦理问题，并形成回音室效应。本研究旨在探索推断的政治观点如何影响全球范围内ChatGPT的回答，而不考虑聊天会话。我们还研究定制指令和记忆功能如何改变ChatGPT的回答，考虑政治倾向的影响。我们开发了三个角色（两个具有政治倾向和一个中立），每个角色都有四个陈述反映了他们对DEI项目、堕胎、枪权和疫苗的看法。我们使用记忆和定制指令将角色的言论传达给ChatGPT，使其能够推断出他们的政治观点，而不直接陈述。然后，我们提出八个问题以揭示角色之间世界观的差异，并对回答进行定性分析。我们的研究发现，回答与角色推断的政治观点保持一致，即使讨论相似话题时也表现出不同的推理和词汇。我们还发现，明确的定制指令和隐性记忆功能在推断方面有类似的表现。通过分析回答的相似性发现，民主倾向角色与定制指令下的回答最接近中立角色，这支持了ChatGPT输出偏向左翼的观察。 

---
# POLIS-Bench: Towards Multi-Dimensional Evaluation of LLMs for Bilingual Policy Tasks in Governmental Scenarios 

**Title (ZH)**: POLIS-Bench:面向政府场景中多语言政策任务的LLM多维度评价平台 

**Authors**: Tingyue Yang, Junchi Yao, Yuhui Guo, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.04705)  

**Abstract**: We introduce POLIS-Bench, the first rigorous, systematic evaluation suite designed for LLMs operating in governmental bilingual policy scenarios. Compared to existing benchmarks, POLIS-Bench introduces three major advancements. (i) Up-to-date Bilingual Corpus: We construct an extensive, up-to-date policy corpus that significantly scales the effective assessment sample size, ensuring relevance to current governance practice. (ii) Scenario-Grounded Task Design: We distill three specialized, scenario-grounded tasks -- Clause Retrieval & Interpretation, Solution Generation, and the Compliance Judgmen--to comprehensively probe model understanding and application. (iii) Dual-Metric Evaluation Framework: We establish a novel dual-metric evaluation framework combining semantic similarity with accuracy rate to precisely measure both content alignment and task requirement adherence. A large-scale evaluation of over 10 state-of-the-art LLMs on POLIS-Bench reveals a clear performance hierarchy where reasoning models maintain superior cross-task stability and accuracy, highlighting the difficulty of compliance tasks. Furthermore, leveraging our benchmark, we successfully fine-tune a lightweight open-source model. The resulting POLIS series models achieves parity with, or surpasses, strong proprietary baselines on multiple policy subtasks at a significantly reduced cost, providing a cost-effective and compliant path for robust real-world governmental deployment. 

**Abstract (ZH)**: POLIS-Bench：面向政府双语政策场景的首个严谨系统性评估套件 

---
# Measuring what Matters: Construct Validity in Large Language Model Benchmarks 

**Title (ZH)**: 关注核心：大型语言模型基准中的结构效度 

**Authors**: Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia Hafner, Harry Mayne, Jan Batzner, Negar Foroutan, Chris Schmitz, Karolina Korgul, Hunar Batra, Oishi Deb, Emma Beharry, Cornelius Emde, Thomas Foster, Anna Gausen, María Grandury, Simeng Han, Valentin Hofmann, Lujain Ibrahim, Hazel Kim, Hannah Rose Kirk, Fangru Lin, Gabrielle Kaili-May Liu, Lennart Luettgau, Jabez Magomere, Jonathan Rystrøm, Anna Sotnikova, Yushi Yang, Yilun Zhao, Adel Bibi, Antoine Bosselut, Ronald Clark, Arman Cohan, Jakob Foerster, Yarin Gal, Scott A. Hale, Inioluwa Deborah Raji, Christopher Summerfield, Philip H.S. Torr, Cozmin Ududec, Luc Rocher, Adam Mahdi  

**Link**: [PDF](https://arxiv.org/pdf/2511.04703)  

**Abstract**: Evaluating large language models (LLMs) is crucial for both assessing their capabilities and identifying safety or robustness issues prior to deployment. Reliably measuring abstract and complex phenomena such as 'safety' and 'robustness' requires strong construct validity, that is, having measures that represent what matters to the phenomenon. With a team of 29 expert reviewers, we conduct a systematic review of 445 LLM benchmarks from leading conferences in natural language processing and machine learning. Across the reviewed articles, we find patterns related to the measured phenomena, tasks, and scoring metrics which undermine the validity of the resulting claims. To address these shortcomings, we provide eight key recommendations and detailed actionable guidance to researchers and practitioners in developing LLM benchmarks. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）对于评估其能力并在部署前识别安全或稳健性问题至关重要。可靠地测量诸如“安全性”和“稳健性”等抽象和复杂的现象需要强大的建构效度，即衡量指标能够代表现象的关键方面。依托29名专家评审，我们系统性地审查了来自自然语言处理和机器学习顶级会议的445个LLM基准。在这些被审查的文章中，我们发现了与所测量的现象、任务和评分指标相关的模式，这些模式削弱了最终声明的有效性。为了应对这些不足，我们提供了八条关键建议，并为研究人员和实践者开发LLM基准提供了详尽的操作指南。 

---
# Separate the Wheat from the Chaff: Winnowing Down Divergent Views in Retrieval Augmented Generation 

**Title (ZH)**: 辨真伪，去伪存真：检索增强生成中分歧观点的筛选 

**Authors**: Song Wang, Zihan Chen, Peng Wang, Zhepei Wei, Zhen Tan, Yu Meng, Cong Shen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.04700)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by integrating external knowledge sources to address their limitations in accessing up-to-date or specialized information. A natural strategy to increase the likelihood of retrieving relevant information is to expand the number of retrieved documents. However, involving more documents could introduce significant noise, as many documents may be irrelevant or misleading, thereby reducing the overall accuracy of the generated responses. To overcome the challenge associated with handling a larger number of documents, we propose WinnowRAG, a novel RAG framework designed to systematically filter out noisy documents while preserving valuable content -- a process we refer to as winnowing. WinnowRAG operates in two stages: In Stage I, we perform query-aware clustering to group similar documents and form distinct topic clusters. Each cluster is assigned to an LLM agent for generating a unique answer. In Stage II, we perform winnowing, wherein a critic LLM evaluates the outputs of multiple agents and iteratively separates useful documents from noisy ones. To retain useful documents when discarding agents, we propose two strategic merging techniques to ensure that only relevant knowledge is used for generating the final response. Crucially, WinnowRAG is model-agnostic and does not require any model fine-tuning, making it easily adaptable to various tasks. Extensive experiments on various realistic datasets demonstrate the effectiveness of WinnowRAG over state-of-the-art baselines. 

**Abstract (ZH)**: 检索增强生成（RAG）通过整合外部知识源来增强大规模语言模型（LLMs），以解决其在访问最新或专门信息方面的局限性。增加检索相关信息的可能性的自然策略是扩大检索文档的数量。然而，涉及更多文档可能会引入大量噪声，因为许多文档可能是无关或误导性的，从而降低生成响应的整体准确性。为了解决处理更多文档所面临的挑战，我们提出了WinnowRAG，这是一种新颖的RAG框架，旨在系统地过滤掉噪声文档同时保留有价值的内容——我们称之为筛选。WinnowRAG分为两个阶段：在阶段I，我们执行查询感知聚类来对相似文档进行分组并形成不同的主题集群，每个集群分配给LLM代理生成独特的答案。在阶段II，我们执行筛选，在此过程中，一个评论员LLM评估多个代理的输出，并迭代地将有用文档从噪声文档中分离出来。为了在丢弃代理时保留有用文档，我们提出了两种策略性的合并技术，以确保仅使用相关知识生成最终响应。最关键的是，WinnowRAG 是模型无关的，不需要任何模型微调，使其易于适应各种任务。在各种现实数据集上的广泛实验证明了WinnowRAG优于最先进的基线方法。 

---
# multiMentalRoBERTa: A Fine-tuned Multiclass Classifier for Mental Health Disorder 

**Title (ZH)**: 多类别精神健康障碍分类器：fine-tuned multiMentalRoBERTa 

**Authors**: K M Sajjadul Islam, John Fields, Praveen Madiraju  

**Link**: [PDF](https://arxiv.org/pdf/2511.04698)  

**Abstract**: The early detection of mental health disorders from social media text is critical for enabling timely support, risk assessment, and referral to appropriate resources. This work introduces multiMentalRoBERTa, a fine-tuned RoBERTa model designed for multiclass classification of common mental health conditions, including stress, anxiety, depression, post-traumatic stress disorder (PTSD), suicidal ideation, and neutral discourse. Drawing on multiple curated datasets, data exploration is conducted to analyze class overlaps, revealing strong correlations between depression and suicidal ideation as well as anxiety and PTSD, while stress emerges as a broad, overlapping category. Comparative experiments with traditional machine learning methods, domain-specific transformers, and prompting-based large language models demonstrate that multiMentalRoBERTa achieves superior performance, with macro F1-scores of 0.839 in the six-class setup and 0.870 in the five-class setup (excluding stress), outperforming both fine-tuned MentalBERT and baseline classifiers. Beyond predictive accuracy, explainability methods, including Layer Integrated Gradients and KeyBERT, are applied to identify lexical cues that drive classification, with a particular focus on distinguishing depression from suicidal ideation. The findings emphasize the effectiveness of fine-tuned transformers for reliable and interpretable detection in sensitive contexts, while also underscoring the importance of fairness, bias mitigation, and human-in-the-loop safety protocols. Overall, multiMentalRoBERTa is presented as a lightweight, robust, and deployable solution for enhancing support in mental health platforms. 

**Abstract (ZH)**: 从社交媒体文本中早期检测心理健康障碍对于及时支持、风险评估和转介至合适资源至关重要。本文介绍了multiMentalRoBERTa，一种针对常见心理健康状况进行多分类的微调RoBERTa模型，包括压力、焦虑、抑郁、创伤后应激障碍(PTSD)、自杀意念和中立话语。本文利用多个精心收集的数据集进行了数据分析，揭示了抑郁与自杀意念、焦虑与PTSD之间 strong 的相关性，同时将压力视为一个广泛且重叠的类别。与传统的机器学习方法、领域特定的变压器以及提示驱动的大语言模型进行对比实验表明，multiMentalRoBERTa 在六类设置下实现了 0.839 的宏 F1 得分，在五类设置下（不包括压力）实现了 0.870 的宏 F1 得分，超越了微调的MentalBERT和基线分类器。除了预测准确性之外，使用解释性方法，如层整合梯度和KeyBERT，来识别驱动分类的词汇线索，特别是区分抑郁与自杀意念。研究结果强调了微调变压器在敏感情境下进行可靠和可解释检测的有效性，同时也突显了公平性、偏见缓解和人工在环安全协议的重要性。总体而言，multiMentalRoBERTa 提出了一个轻量级、稳健且可部署的解决方案，以增强心理健康平台的支持。 

---
# Simulating Misinformation Vulnerabilities With Agent Personas 

**Title (ZH)**: 使用代理人格模拟误导信息漏洞 

**Authors**: David Farr, Lynnette Hui Xian Ng, Stephen Prochaska, Iain J. Cruickshank, Jevin West  

**Link**: [PDF](https://arxiv.org/pdf/2511.04697)  

**Abstract**: Disinformation campaigns can distort public perception and destabilize institutions. Understanding how different populations respond to information is crucial for designing effective interventions, yet real-world experimentation is impractical and ethically challenging. To address this, we develop an agent-based simulation using Large Language Models (LLMs) to model responses to misinformation. We construct agent personas spanning five professions and three mental schemas, and evaluate their reactions to news headlines. Our findings show that LLM-generated agents align closely with ground-truth labels and human predictions, supporting their use as proxies for studying information responses. We also find that mental schemas, more than professional background, influence how agents interpret misinformation. This work provides a validation of LLMs to be used as agents in an agent-based model of an information network for analyzing trust, polarization, and susceptibility to deceptive content in complex social systems. 

**Abstract (ZH)**: 大规模语言模型驱动的代理模拟：对信息响应的研究代理模型验证 

---
# EncouRAGe: Evaluating RAG Local, Fast, and Reliable 

**Title (ZH)**: EncouRAGe: 评估RAG本地、快速且可靠的方法 

**Authors**: Jan Strich, Adeline Scharfenberg, Chris Biemann, Martin Semmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.04696)  

**Abstract**: We introduce EncouRAGe, a comprehensive Python framework designed to streamline the development and evaluation of Retrieval-Augmented Generation (RAG) systems using Large Language Models (LLMs) and Embedding Models. EncouRAGe comprises five modular and extensible components: Type Manifest, RAG Factory, Inference, Vector Store, and Metrics, facilitating flexible experimentation and extensible development. The framework emphasizes scientific reproducibility, diverse evaluation metrics, and local deployment, enabling researchers to efficiently assess datasets within RAG workflows. This paper presents implementation details and an extensive evaluation across multiple benchmark datasets, including 25k QA pairs and over 51k documents. Our results show that RAG still underperforms compared to the Oracle Context, while Hybrid BM25 consistently achieves the best results across all four datasets. We further examine the effects of reranking, observing only marginal performance improvements accompanied by higher response latency. 

**Abstract (ZH)**: 我们介绍EncouRAGe，一个综合性的Python框架，旨在简化使用大规模语言模型（LLMs）和嵌入模型的检索增强生成（RAG）系统的开发和评估。EncouRAGe 包含五个模块化和扩展性强的组件：类型说明文件、RAG工厂、推理、向量存储和指标，促进灵活的实验和扩展的开发。该框架强调科学的可重复性、多样化的评估指标以及本地部署，使研究人员能够高效地评估RAG工作流中的数据集。本文介绍了该框架的实现细节，并在多个基准数据集上进行了广泛的评估，包括25,000个问答对和超过51,000份文档。我们的结果显示，RAG的表现仍然落后于Oracle Context，而混合BM25在所有四个数据集上始终取得最佳结果。我们进一步探讨了重排的效果，发现虽然性能有所提升，但响应延迟增加。 

---
# Reasoning Up the Instruction Ladder for Controllable Language Models 

**Title (ZH)**: 沿着指令梯阶进行可控语言模型的推理 

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2511.04694)  

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises both aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks. These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior. 

**Abstract (ZH)**: 基于大型语言模型的指令层级解决：实现在高风险决策中的可靠性和可控性 

---
# A Penny for Your Thoughts: Decoding Speech from Inexpensive Brain Signals 

**Title (ZH)**: 一便士就能说清：从廉价脑信号解码语音 

**Authors**: Quentin Auster, Kateryna Shapovalenko, Chuang Ma, Demaio Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.04691)  

**Abstract**: We explore whether neural networks can decode brain activity into speech by mapping EEG recordings to audio representations. Using EEG data recorded as subjects listened to natural speech, we train a model with a contrastive CLIP loss to align EEG-derived embeddings with embeddings from a pre-trained transformer-based speech model. Building on the state-of-the-art EEG decoder from Meta, we introduce three architectural modifications: (i) subject-specific attention layers (+0.15% WER improvement), (ii) personalized spatial attention (+0.45%), and (iii) a dual-path RNN with attention (-1.87%). Two of the three modifications improved performance, highlighting the promise of personalized architectures for brain-to-speech decoding and applications in brain-computer interfaces. 

**Abstract (ZH)**: 我们探讨了是否可以通过将EEG记录映射到音频表示来使用神经网络解码脑活动为语音。利用受试者听自然语音时记录的EEG数据，我们训练了一个带有对比CLIP损失的模型，以使EEG衍生的嵌入与预训练的基于变压器的语音模型嵌入对齐。在Meta的先进EEG解码器基础上，我们引入了三种架构修改：（i）主体特定的注意力层（WER改进0.15%），（ii）个性化空间注意力（改进0.45%），以及（iii）具有注意力机制的双路径RNN（性能下降1.87%）。其中两种修改提高了性能，突显了个性化架构在脑-语音解码和脑-计算机接口应用中的潜力。 

---
# Adaptive Testing for LLM Evaluation: A Psychometric Alternative to Static Benchmarks 

**Title (ZH)**: 自适应测试用于大语言模型评估：心理测量学替代静态基准 

**Authors**: Peiyu Li, Xiuxiu Tang, Si Chen, Ying Cheng, Ronald Metoyer, Ting Hua, Nitesh V. Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2511.04689)  

**Abstract**: Large language model evaluation requires thousands of benchmark items, making evaluations expensive and slow. Existing methods compute average accuracy across fixed item sets, treating all items equally despite varying quality and informativeness. We present ATLAS an adaptive testing framework using Item Response Theory (IRT) to estimate model ability through Fisher information-guided item selection. Our analysis of five major benchmarks reveals that 3-6% of items exhibit negative discrimination, indicating annotation errors that corrupt static evaluation. ATLAS achieves 90% item reduction while maintaining measurement precision: on HellaSwag (5,608 items), we match full-benchmark estimates using only 42 items with 0.154 MAE. Our framework maintains item exposure rates below 10% and test overlap at 16-27%, compared to static benchmarks where every model sees all items (100% exposure). Among 4,000+ tested models, IRT ranks differ from accuracy ranks: models with the same accuracy get different IRT scores, and 23-31% of all models shift by more than 10 rank positions. Code and calibrated item banks are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型评估需要成千上万的基准项，使得评估既昂贵又缓慢。现有方法通过固定项集计算平均准确性，尽管各项的质量和信息量不同，仍同等对待。我们提出了一个基于项目反应理论（IRT）的自适应测试框架ATLAS，通过斐舍信息引导的项目选择来估计模型能力。对五大基准的分析显示，3-6%的项目表现出负区分度，表明存在标注错误，会污染静态评估。ATLAS实现了90%的项目减少，同时保持测量精度：在HellaSwag（5,608项）上，仅使用42项就达到了与完整基准相同的0.154 MAE估计值。我们的框架将项目曝光率保持在10%以下，并且测试重叠率为16-27%，而静态基准中，每个模型都会看到所有项目（曝光率为100%）。在4,000多个测试模型中，IRT排名与准确性排名不同：具有相同准确度的模型获得不同的IRT分数，且23-31%的模型排名位置发生了超过10位的变化。代码和校准后的项目银行可在以下链接获取。 

---
# Stateful KV Cache Management for LLMs: Balancing Space, Time, Accuracy, and Positional Fidelity 

**Title (ZH)**: 面向LLMs的状态型KV缓存管理：空间、时间、准确性和位置保真的平衡 

**Authors**: Pratik Poudel  

**Link**: [PDF](https://arxiv.org/pdf/2511.04686)  

**Abstract**: The Key-Value (KV) cache is integral to efficient autoregressive inference in large language models (LLMs), yet its unbounded growth in stateful multi-turn scenarios presents major challenges. This paper examines the interplay between KV cache management strategies, the architectural context limits of models like meta-llama/Meta-Llama-3-8b-instruct, and the often-overlooked integrity of positional encodings. Through empirical analysis using a stateful benchmarking framework, we show that LLM generation quality degrades sharply when the accumulated KV cache approaches or exceeds the model's trained context window (e.g., 8192 tokens for Llama 3), a failure mode distinct from GPU memory exhaustion. Common eviction strategies, even high-retention ones (e.g., 99% via AttentionTop), can worsen performance if they disrupt positional coherence. Because LLMs rely on consistent positional signals (e.g., RoPE), compacting a cache by removing non-contiguous tokens can scramble these signals and lead to degenerative outputs. We further show that simple strategies preserving contiguous context blocks (e.g., keeping an initial "gist") can yield more coherent generations than complex or positionally disruptive ones. We advocate for eviction techniques that respect architectural limits, preserve positional structure, and view "cache health" holistically beyond mere size. 

**Abstract (ZH)**: KV缓存是大规模语言模型（LLMs）高效自回归推理的关键组成部分，但在状态ful多轮场景中其无界增长带来了重大挑战。本文探讨了KV缓存管理策略、如Meta-Llama/Meta-Llama-3-8b-instruct等模型的架构限制与其经常被忽视的位置编码完整性的相互作用。通过使用状态ful基准测试框架进行实证分析，我们发现当累积的KV缓存接近或超过模型训练上下文窗口（例如，Llama 3的8192个标记）时，LLM生成质量会急剧下降，这是一种不同于GPU内存耗尽的失败模式。即使高保留率的常见淘汰策略（例如，通过AttentionTop实现99%保留）也可能因破坏位置一致性而恶化性能。因为LLMs依赖于一致的位置信号（例如，RoPE），通过移除非连续标记来压缩缓存可能会扰乱这些信号并导致退化输出。我们进一步表明，保留连续上下文块的简单策略（例如，保留初始的“概要”）比破坏位置结构的复杂策略能产生更一致的生成结果。我们提倡尊重架构限制、保持位置结构并在整体上关注“缓存健康”的淘汰技术。 

---
# AI-Powered Citation Auditing: A Zero-Assumption Protocol for Systematic Reference Verification in Academic Research 

**Title (ZH)**: 基于AI的引文审计：一种针对学术研究系统性参考验证的零假设协议 

**Authors**: L.J. Janse van Rensburg  

**Link**: [PDF](https://arxiv.org/pdf/2511.04683)  

**Abstract**: Academic citation integrity faces persistent challenges, with research indicating 20% of citations contain errors and manual verification requiring months of expert time. This paper presents a novel AI-powered methodology for systematic, comprehensive reference auditing using agentic AI with tool-use capabilities. We develop a zero-assumption verification protocol that independently validates every reference against multiple academic databases (Semantic Scholar, Google Scholar, CrossRef) without assuming any citation is correct. The methodology was validated across 30 academic documents (2,581 references) spanning undergraduate projects to doctoral theses and peer-reviewed publications. Results demonstrate 91.7% average verification rate on published PLOS papers, with successful detection of fabricated references, retracted articles, orphan citations, and predatory journals. Time efficiency improved dramatically: 90-minute audits for 916-reference doctoral theses versus months of manual review. The system achieved <0.5% false positive rate while identifying critical issues manual review might miss. This work establishes the first validated AI-agent methodology for academic citation integrity, demonstrating practical applicability for supervisors, students, and institutional quality assurance. 

**Abstract (ZH)**: 学术引用诚信面临持续挑战，研究显示20%的引用存在问题，人工验证需要专家数月时间。本文提出了一种基于代理人工智能和工具使用能力的新颖AI驱动系统化全面参考审计方法。我们开发了一种零假设验证协议，该协议独立地将每条参考文献与多个学术数据库（Semantic Scholar、Google Scholar、CrossRef）进行验证，而不假设任何引用是正确的。该方法论在涵盖本科项目到博士论文和同行评审出版物的30篇学术文档（2,581条参考文献）中得到了验证。结果显示，该方法在发表于PLOS的论文中平均验证率为91.7%，成功检测出伪造的参考文献、撤回的文章、孤立引用和掠食性期刊。验证效率大幅提升：916条参考文献的博士论文可在90分钟内完成审计，而人工审核则需数月。系统实现了<0.5%的误报率，同时发现人工审核可能忽略的关键问题。本文确立了首项验证过的AI代理方法论，展示了其在监督者、学生和机构质量保证中的实际应用潜力。 

---
# Efficient Deployment of CNN Models on Multiple In-Memory Computing Units 

**Title (ZH)**: 高效的CNN模型在多片内存计算单元上的部署 

**Authors**: Eleni Bougioukou, Theodore Antonakopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2511.04682)  

**Abstract**: In-Memory Computing (IMC) represents a paradigm shift in deep learning acceleration by mitigating data movement bottlenecks and leveraging the inherent parallelism of memory-based computations. The efficient deployment of Convolutional Neural Networks (CNNs) on IMC-based hardware necessitates the use of advanced task allocation strategies for achieving maximum computational efficiency. In this work, we exploit an IMC Emulator (IMCE) with multiple Processing Units (PUs) for investigating how the deployment of a CNN model in a multi-processing system affects its performance, in terms of processing rate and latency. For that purpose, we introduce the Load-Balance-Longest-Path (LBLP) algorithm, that dynamically assigns all CNN nodes to the available IMCE PUs, for maximizing the processing rate and minimizing latency due to efficient resources utilization. We are benchmarking LBLP against other alternative scheduling strategies for a number of CNN models and experimental results demonstrate the effectiveness of the proposed algorithm. 

**Abstract (ZH)**: 基于内存计算的卷积神经网络任务分配研究 

---
