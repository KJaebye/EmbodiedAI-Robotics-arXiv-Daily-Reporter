# The Decrypto Benchmark for Multi-Agent Reasoning and Theory of Mind 

**Title (ZH)**: 多智能体推理与心智理论的Decrypto基准 

**Authors**: Andrei Lupu, Timon Willi, Jakob Foerster  

**Link**: [PDF](https://arxiv.org/pdf/2506.20664)  

**Abstract**: As Large Language Models (LLMs) gain agentic abilities, they will have to navigate complex multi-agent scenarios, interacting with human users and other agents in cooperative and competitive settings. This will require new reasoning skills, chief amongst them being theory of mind (ToM), or the ability to reason about the "mental" states of other agents. However, ToM and other multi-agent abilities in LLMs are poorly understood, since existing benchmarks suffer from narrow scope, data leakage, saturation, and lack of interactivity. We thus propose Decrypto, a game-based benchmark for multi-agent reasoning and ToM drawing inspiration from cognitive science, computational pragmatics and multi-agent reinforcement learning. It is designed to be as easy as possible in all other dimensions, eliminating confounding factors commonly found in other benchmarks. To our knowledge, it is also the first platform for designing interactive ToM experiments.
We validate the benchmark design through comprehensive empirical evaluations of frontier LLMs, robustness studies, and human-AI cross-play experiments. We find that LLM game-playing abilities lag behind humans and simple word-embedding baselines. We then create variants of two classic cognitive science experiments within Decrypto to evaluate three key ToM abilities. Surprisingly, we find that state-of-the-art reasoning models are significantly worse at those tasks than their older counterparts. This demonstrates that Decrypto addresses a crucial gap in current reasoning and ToM evaluations, and paves the path towards better artificial agents. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）获得自主能力，它们将不得不在复杂多智能体场景中导航，与人类用户和其他智能体在合作和竞争环境中互动。这将需要新的推理能力，其中最主要的是心理理论（ToM），即推理其他智能体“心理”状态的能力。然而，LLMs中的ToM和其他多智能体能力尚未充分理解，因为现有基准测试存在范围狭窄、数据泄露、饱和和缺乏互动性等问题。因此，我们提出了Decrypto，这是一种借鉴认知科学、计算语用学和多智能体强化学习的基于游戏的基准测试，设计上在所有其他维度尽可能简单，消除其他基准测试中常见的混淆因素。据我们所知，这也是首个用于设计互动心理理论实验的平台。 

---
# Towards Community-Driven Agents for Machine Learning Engineering 

**Title (ZH)**: 面向社区驱动的机器学习工程代理 

**Authors**: Sijie Li, Weiwei Sun, Shanda Li, Ameet Talwalkar, Yiming Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20640)  

**Abstract**: Large language model-based machine learning (ML) agents have shown great promise in automating ML research. However, existing agents typically operate in isolation on a given research problem, without engaging with the broader research community, where human researchers often gain insights and contribute by sharing knowledge. To bridge this gap, we introduce MLE-Live, a live evaluation framework designed to assess an agent's ability to communicate with and leverage collective knowledge from a simulated Kaggle research community. Building on this framework, we propose CoMind, a novel agent that excels at exchanging insights and developing novel solutions within a community context. CoMind achieves state-of-the-art performance on MLE-Live and outperforms 79.2% human competitors on average across four ongoing Kaggle competitions. Our code is released at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的机器学习代理在自动化机器学习研究方面展现了巨大的潜力。然而，现有的代理通常孤立地在特定研究问题上操作，而不与更广泛的科研社区互动，人类研究人员常常通过分享知识来获取见解和贡献。为弥合这一差距，我们介绍了一种实时评估框架MLE-Live，旨在评估代理与模拟的Kaggle研究社区进行沟通并利用集体知识的能力。在此基础上，我们提出了一种名为CoMind的新型代理，它在社区背景下擅长交换见解并发展新颖的解决方案。CoMind在MLE-Live上达到了最先进的性能，并且在四项正在进行的Kaggle竞赛中平均优于79.2%的人类对手。我们的代码发布在该网址：https://github.com/alibaba/CoMind。 

---
# AI Assistants to Enhance and Exploit the PETSc Knowledge Base 

**Title (ZH)**: AI助手以增强和利用PETSc知识库 

**Authors**: Barry Smith, Junchao Zhang, Hong Zhang, Lois Curfman McInnes, Murat Keceli, Archit Vasan, Satish Balay, Toby Isaac, Le Chen, Venkatram Vishwanath  

**Link**: [PDF](https://arxiv.org/pdf/2506.20608)  

**Abstract**: Generative AI, especially through large language models (LLMs), is transforming how technical knowledge can be accessed, reused, and extended. PETSc, a widely used numerical library for high-performance scientific computing, has accumulated a rich but fragmented knowledge base over its three decades of development, spanning source code, documentation, mailing lists, GitLab issues, Discord conversations, technical papers, and more. Much of this knowledge remains informal and inaccessible to users and new developers. To activate and utilize this knowledge base more effectively, the PETSc team has begun building an LLM-powered system that combines PETSc content with custom LLM tools -- including retrieval-augmented generation (RAG), reranking algorithms, and chatbots -- to assist users, support developers, and propose updates to formal documentation. This paper presents initial experiences designing and evaluating these tools, focusing on system architecture, using RAG and reranking for PETSc-specific information, evaluation methodologies for various LLMs and embedding models, and user interface design. Leveraging the Argonne Leadership Computing Facility resources, we analyze how LLM responses can enhance the development and use of numerical software, with an initial focus on scalable Krylov solvers. Our goal is to establish an extensible framework for knowledge-centered AI in scientific software, enabling scalable support, enriched documentation, and enhanced workflows for research and development. We conclude by outlining directions for expanding this system into a robust, evolving platform that advances software ecosystems to accelerate scientific discovery. 

**Abstract (ZH)**: Generative AI，尤其是在大型语言模型（LLMs）的帮助下，正在 transforming 如何获取、重用和技术扩展技术知识。PETSc，一个广泛用于高性能科学计算的数值库，在其三十余年的开发过程中积累了丰富的但又分散的知识基础，涵盖源代码、文档、邮件列表、GitLab问题、Discord对话、技术论文等。其中大量知识仍处于非正式状态且难以为用户和新开发者访问。为更有效地激活和利用这一知识基础，PETSc 团队已经开始构建一个基于 LLM 的系统，将 PETSc 内容与自定义 LLM 工具相结合——包括检索增强生成（RAG）、重排序算法和聊天机器人——以协助用户、支持开发人员，并提议更新正式文档。本文介绍了这些工具的初步设计与评估经验，重点在于系统架构，利用 RAG 和重排序处理 PETSc 特定信息，各种 LLM 和嵌入式模型的评估方法，以及用户界面设计。利用阿贡领导力计算设施的资源，我们分析了 LLM 响应如何增强数值软件的开发和使用，初期重点放在可扩展的共轭梯度求解器上。我们的目标是建立一个可扩展的知识中心型 AI 框架，以促进科学软件的支持、丰富文档和增强的研究与开发工作流。最后，我们概括了扩展该系统的方向，以便构建一个稳健且不断演进的平台，从而推动软件生态系统的发展，加快科学研究的发现过程。 

---
# CogGen: A Learner-Centered Generative AI Architecture for Intelligent Tutoring with Programming Video 

**Title (ZH)**: CogGen: 以学习者为中心的编程视频智能辅导生成AI架构 

**Authors**: Wengxi Li, Roy Pea, Nick Haber, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20600)  

**Abstract**: We introduce CogGen, a learner-centered AI architecture that transforms programming videos into interactive, adaptive learning experiences by integrating student modeling with generative AI tutoring based on the Cognitive Apprenticeship framework. The architecture consists of three components: (1) video segmentation by learning goals, (2) a conversational tutoring engine applying Cognitive Apprenticeship strategies, and (3) a student model using Bayesian Knowledge Tracing to adapt instruction. Our technical evaluation demonstrates effective video segmentation accuracy and strong pedagogical alignment across knowledge, method, action, and interaction layers. Ablation studies confirm the necessity of each component in generating effective guidance. This work advances AI-powered tutoring by bridging structured student modeling with interactive AI conversations, offering a scalable approach to enhancing video-based programming education. 

**Abstract (ZH)**: 我们介绍了一种以学习者为中心的AI架构CogGen，该架构通过将认知学徒制框架下的生成AI辅导与学生建模相结合，将编程视频转换为互动式、自适应的学习体验。该架构包括三个组成部分：(1) 依据学习目标进行视频分割，(2) 采用认知学徒制策略的对话式辅导引擎，以及(3) 利用贝叶斯知识追踪的学生模型，以适应性地调整教学。我们的技术评估展示了有效的视频分割准确性和在知识、方法、行动和互动层面上强大的教学一致性。消融研究证实了每个组成部分在生成有效指导方面的重要性。本研究通过将结构化学生建模与互动AI对话相结合，推进了基于AI的辅导技术，并提供了一种增强基于视频的编程教育的可扩展方法。 

---
# Fine-Tuning and Prompt Engineering of LLMs, for the Creation of Multi-Agent AI for Addressing Sustainable Protein Production Challenges 

**Title (ZH)**: 细调和指令工程化大语言模型，以创建多智能体AI应对可持续蛋白质生产挑战 

**Authors**: Alexander D. Kalian, Jaewook Lee, Stefan P. Johannesson, Lennart Otte, Christer Hogstrand, Miao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.20598)  

**Abstract**: The global demand for sustainable protein sources has accelerated the need for intelligent tools that can rapidly process and synthesise domain-specific scientific knowledge. In this study, we present a proof-of-concept multi-agent Artificial Intelligence (AI) framework designed to support sustainable protein production research, with an initial focus on microbial protein sources. Our Retrieval-Augmented Generation (RAG)-oriented system consists of two GPT-based LLM agents: (1) a literature search agent that retrieves relevant scientific literature on microbial protein production for a specified microbial strain, and (2) an information extraction agent that processes the retrieved content to extract relevant biological and chemical information. Two parallel methodologies, fine-tuning and prompt engineering, were explored for agent optimisation. Both methods demonstrated effectiveness at improving the performance of the information extraction agent in terms of transformer-based cosine similarity scores between obtained and ideal outputs. Mean cosine similarity scores were increased by up to 25%, while universally reaching mean scores of $\geq 0.89$ against ideal output text. Fine-tuning overall improved the mean scores to a greater extent (consistently of $\geq 0.94$) compared to prompt engineering, although lower statistical uncertainties were observed with the latter approach. A user interface was developed and published for enabling the use of the multi-agent AI system, alongside preliminary exploration of additional chemical safety-based search capabilities 

**Abstract (ZH)**: 全球可持续蛋白质来源的需求加速了智能工具的发展，这些工具可以快速处理和合成特定领域的科学知识。本研究介绍了一种概念验证的多代理人工智能（AI）框架，旨在支持可持续蛋白质生产研究，初期重点关注微生物蛋白质来源。基于检索增强生成（RAG）的系统包括两个基于GPT的大语言模型代理：（1）文献搜索代理，用于检索指定微生物菌株的微生物蛋白质生产相关科学文献；（2）信息提取代理，处理检索内容以提取相关生物和化学信息。探索了两种并行方法——微调和提示工程——以优化代理性能。两种方法在基于变压器的余弦相似度评分上均显示出有效性，其中检索到的内容与理想输出之间的相似度得分最高提高了25%，普遍达到≥0.89。总体而言，微调方法在平均得分上的改进程度更大（≥0.94），尽管与之相比，提示工程方法的统计不确定性更低。开发并发布了用户界面，以启用多代理AI系统的使用，并初步探索了额外的基于化学安全的搜索功能。

标题：
基于RAG的多代理AI框架支持微生物蛋白质生产的研究 

---
# Case-based Reasoning Augmented Large Language Model Framework for Decision Making in Realistic Safety-Critical Driving Scenarios 

**Title (ZH)**: 基于案例推理增强的大语言模型决策框架在实际安全关键驾驶场景中的应用 

**Authors**: Wenbin Gan, Minh-Son Dao, Koji Zettsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20531)  

**Abstract**: Driving in safety-critical scenarios requires quick, context-aware decision-making grounded in both situational understanding and experiential reasoning. Large Language Models (LLMs), with their powerful general-purpose reasoning capabilities, offer a promising foundation for such decision-making. However, their direct application to autonomous driving remains limited due to challenges in domain adaptation, contextual grounding, and the lack of experiential knowledge needed to make reliable and interpretable decisions in dynamic, high-risk environments. To address this gap, this paper presents a Case-Based Reasoning Augmented Large Language Model (CBR-LLM) framework for evasive maneuver decision-making in complex risk scenarios. Our approach integrates semantic scene understanding from dashcam video inputs with the retrieval of relevant past driving cases, enabling LLMs to generate maneuver recommendations that are both context-sensitive and human-aligned. Experiments across multiple open-source LLMs show that our framework improves decision accuracy, justification quality, and alignment with human expert behavior. Risk-aware prompting strategies further enhance performance across diverse risk types, while similarity-based case retrieval consistently outperforms random sampling in guiding in-context learning. Case studies further demonstrate the framework's robustness in challenging real-world conditions, underscoring its potential as an adaptive and trustworthy decision-support tool for intelligent driving systems. 

**Abstract (ZH)**: 基于案例推理增强的大语言模型在复杂风险场景中的回避 maneuvers决策框架 

---
# Engineering Sentience 

**Title (ZH)**: 工程智能 

**Authors**: Konstantin Demin, Taylor Webb, Eric Elmoznino, Hakwan Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.20504)  

**Abstract**: We spell out a definition of sentience that may be useful for designing and building it in machines. We propose that for sentience to be meaningful for AI, it must be fleshed out in functional, computational terms, in enough detail to allow for implementation. Yet, this notion of sentience must also reflect something essentially 'subjective', beyond just having the general capacity to encode perceptual content. For this specific functional notion of sentience to occur, we propose that certain sensory signals need to be both assertoric (persistent) and qualitative. To illustrate the definition in more concrete terms, we sketch out some ways for potential implementation, given current technology. Understanding what it takes for artificial agents to be functionally sentient can also help us avoid creating them inadvertently, or at least, realize that we have created them in a timely manner. 

**Abstract (ZH)**: 我们定义了一种可能适用于机器设计和构建的意识概念。我们提出，对于人工智能来说，意识必须在功能和计算的层面上得到具体阐述，详细到可以实施的程度。然而，这一意识的概念也必须反映一些本质上主观的东西，而不仅仅是具备编码感知内容的一般能力。为了实现这种特定的功能性意识，我们提出某些感觉信号需要既是断言性的（持久的）又是质性的。为了更具体地说明这一定义，我们概述了一些当前技术条件下潜在实现方式。了解使人工代理具备功能性意识所需条件也可以帮助我们避免无意中创造它们，或者至少在创造它们时及时意识到这一点。 

---
# Mixtures of Neural Cellular Automata: A Stochastic Framework for Growth Modelling and Self-Organization 

**Title (ZH)**: 神经细胞自动机的混合模型：一种生长建模和自我组织的随机框架 

**Authors**: Salvatore Milite, Giulio Caravagna, Andrea Sottoriva  

**Link**: [PDF](https://arxiv.org/pdf/2506.20486)  

**Abstract**: Neural Cellular Automata (NCAs) are a promising new approach to model self-organizing processes, with potential applications in life science. However, their deterministic nature limits their ability to capture the stochasticity of real-world biological and physical systems.
We propose the Mixture of Neural Cellular Automata (MNCA), a novel framework incorporating the idea of mixture models into the NCA paradigm. By combining probabilistic rule assignments with intrinsic noise, MNCAs can model diverse local behaviors and reproduce the stochastic dynamics observed in biological processes.
We evaluate the effectiveness of MNCAs in three key domains: (1) synthetic simulations of tissue growth and differentiation, (2) image morphogenesis robustness, and (3) microscopy image segmentation. Results show that MNCAs achieve superior robustness to perturbations, better recapitulate real biological growth patterns, and provide interpretable rule segmentation. These findings position MNCAs as a promising tool for modeling stochastic dynamical systems and studying self-growth processes. 

**Abstract (ZH)**: 混合神经细胞自动机（Mixture of Neural Cellular Automata, MNCA）：一种新的自组织过程建模框架 

---
# GymPN: A Library for Decision-Making in Process Management Systems 

**Title (ZH)**: GymPN: 一种过程管理系统的决策库 

**Authors**: Riccardo Lo Bianco, Willem van Jaarsveld, Remco Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20404)  

**Abstract**: Process management systems support key decisions about the way work is allocated in organizations. This includes decisions on which task to perform next, when to execute the task, and who to assign the task to. Suitable software tools are required to support these decisions in a way that is optimal for the organization. This paper presents a software library, called GymPN, that supports optimal decision-making in business processes using Deep Reinforcement Learning. GymPN builds on previous work that supports task assignment in business processes, introducing two key novelties: support for partial process observability and the ability to model multiple decisions in a business process. These novel elements address fundamental limitations of previous work and thus enable the representation of more realistic process decisions. We evaluate the library on eight typical business process decision-making problem patterns, showing that GymPN allows for easy modeling of the desired problems, as well as learning optimal decision policies. 

**Abstract (ZH)**: 基于深度强化学习的业务流程最优决策软件库GymPN 

---
# Smart Ride and Delivery Services with Electric Vehicles: Leveraging Bidirectional Charging for Profit Optimisation 

**Title (ZH)**: 基于双向充电的智能出行与配送服务：盈利优化研究 

**Authors**: Jinchun Du, Bojie Shen, Muhammad Aamir Cheema, Adel N. Toosi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20401)  

**Abstract**: With the rising popularity of electric vehicles (EVs), modern service systems, such as ride-hailing delivery services, are increasingly integrating EVs into their operations. Unlike conventional vehicles, EVs often have a shorter driving range, necessitating careful consideration of charging when fulfilling requests. With recent advances in Vehicle-to-Grid (V2G) technology - allowing EVs to also discharge energy back to the grid - new opportunities and complexities emerge. We introduce the Electric Vehicle Orienteering Problem with V2G (EVOP-V2G): a profit-maximization problem where EV drivers must select customer requests or orders while managing when and where to charge or discharge. This involves navigating dynamic electricity prices, charging station selection, and route constraints. We formulate the problem as a Mixed Integer Programming (MIP) model and propose two near-optimal metaheuristic algorithms: one evolutionary (EA) and the other based on large neighborhood search (LNS). Experiments on real-world data show our methods can double driver profits compared to baselines, while maintaining near-optimal performance on small instances and excellent scalability on larger ones. Our work highlights a promising path toward smarter, more profitable EV-based mobility systems that actively support the energy grid. 

**Abstract (ZH)**: 基于V2G的电动车辆 orienteering 问题（EVOP-V2G） 

---
# Paladin-mini: A Compact and Efficient Grounding Model Excelling in Real-World Scenarios 

**Title (ZH)**: Paladin-mini：一个紧凑高效且适用于实际场景的语义 grounding 模型 

**Authors**: Dror Ivry, Oran Nahum  

**Link**: [PDF](https://arxiv.org/pdf/2506.20384)  

**Abstract**: This paper introduces two significant contributions to address the issue of grounding claims in a given context. Grounding means that given a context (document) and a claim, there's at least one supportive evidence for the claim in the document. We will introduce Paladin-mini, a compact (3.8B parameters) open-source classifier model (used for labeling data as grounded or ungrounded) engineered for robust performance in real-world scenarios, and the grounding-benchmark, a new evaluation dataset designed to assess performance on critical reasoning tasks. We'll also demonstrate the results of Paladin-mini with benchmarks against the current State-of-the-art and share clear and reproducible results. 

**Abstract (ZH)**: 本文介绍了两项重要贡献，以解决在给定上下文中接地断言的问题。接地意味着给定一个上下文（文档）和一个断言，在文档中至少有一条支持证据。我们将介绍Paladin-mini，这是一个紧凑型（3.8B参数）开源分类模型（用于将数据标记为接地或未接地），它旨在在实际场景中实现稳健性能，并介绍用于评估关键推理任务性能的新评价数据集grounding-benchmark。我们还将展示Paladin-mini与当前最佳性能的基准测试结果，并分享清晰可重复的结果。 

---
# Tabular Feature Discovery With Reasoning Type Exploration 

**Title (ZH)**: 表格特征发现与推理类型探索 

**Authors**: Sungwon Han, Sungkyu Park, Seungeon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20357)  

**Abstract**: Feature engineering for tabular data remains a critical yet challenging step in machine learning. Recently, large language models (LLMs) have been used to automatically generate new features by leveraging their vast knowledge. However, existing LLM-based approaches often produce overly simple or repetitive features, partly due to inherent biases in the transformations the LLM chooses and the lack of structured reasoning guidance during generation. In this paper, we propose a novel method REFeat, which guides an LLM to discover diverse and informative features by leveraging multiple types of reasoning to steer the feature generation process. Experiments on 59 benchmark datasets demonstrate that our approach not only achieves higher predictive accuracy on average, but also discovers more diverse and meaningful features. These results highlight the promise of incorporating rich reasoning paradigms and adaptive strategy selection into LLM-driven feature discovery for tabular data. 

**Abstract (ZH)**: 基于大规模语言模型的特征工程方法通过多种推理引导发现多样化和有信息量的特征 

---
# Mobile-R1: Towards Interactive Reinforcement Learning for VLM-Based Mobile Agent via Task-Level Rewards 

**Title (ZH)**: Mobile-R1：面向基于VLM的移动代理的交互式强化学习方法及其任务级奖励机制 

**Authors**: Jihao Gu, Qihang Ai, Yingyao Wang, Pi Bu, Jingxuan Xing, Zekun Zhu, Wei Jiang, Ziming Wang, Yingxiu Zhao, Ming-Liang Zhang, Jun Song, Yuning Jiang, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20332)  

**Abstract**: Vision-language model-based mobile agents have gained the ability to not only understand complex instructions and mobile screenshots, but also optimize their action outputs via thinking and reasoning, benefiting from reinforcement learning, such as Group Relative Policy Optimization (GRPO). However, existing research centers on offline reinforcement learning training or online optimization using action-level rewards, which limits the agent's dynamic interaction with the environment. This often results in agents settling into local optima, thereby weakening their ability for exploration and error action correction. To address these challenges, we introduce an approach called Mobile-R1, which employs interactive multi-turn reinforcement learning with task-level rewards for mobile agents. Our training framework consists of three stages: initial format finetuning, single-step online training via action-level reward, followed by online training via task-level reward based on multi-turn trajectories. This strategy is designed to enhance the exploration and error correction capabilities of Mobile-R1, leading to significant performance improvements. Moreover, we have collected a dataset covering 28 Chinese applications with 24,521 high-quality manual annotations and established a new benchmark with 500 trajectories. We will open source all resources, including the dataset, benchmark, model weight, and codes: this https URL. 

**Abstract (ZH)**: 基于视觉-语言模型的移动代理通过交互式多轮强化学习和任务级奖励优化探索与错误纠正能力 

---
# Enterprise Large Language Model Evaluation Benchmark 

**Title (ZH)**: 企业级大型语言模型评估基准 

**Authors**: Liya Wang, David Yi, Damien Jose, John Passarelli, James Gao, Jordan Leventis, Kang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.20274)  

**Abstract**: Large Language Models (LLMs) ) have demonstrated promise in boosting productivity across AI-powered tools, yet existing benchmarks like Massive Multitask Language Understanding (MMLU) inadequately assess enterprise-specific task complexities. We propose a 14-task framework grounded in Bloom's Taxonomy to holistically evaluate LLM capabilities in enterprise contexts. To address challenges of noisy data and costly annotation, we develop a scalable pipeline combining LLM-as-a-Labeler, LLM-as-a-Judge, and corrective retrieval-augmented generation (CRAG), curating a robust 9,700-sample benchmark. Evaluation of six leading models shows open-source contenders like DeepSeek R1 rival proprietary models in reasoning tasks but lag in judgment-based scenarios, likely due to overthinking. Our benchmark reveals critical enterprise performance gaps and offers actionable insights for model optimization. This work provides enterprises a blueprint for tailored evaluations and advances practical LLM deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）在提升AI工具生产力方面展示了潜力，但现有的基准测试如大规模多任务语言理解（MMLU）未能充分评估企业特定的任务复杂性。我们提出一个基于布卢姆分类法的14任务框架，以全面评估LLM在企业环境中的能力。为应对嘈杂数据和昂贵标注的挑战，我们开发了一个可扩展的流水线，结合了LLM-as-a-标签器、LLM-as-a-裁判和纠正检索增强生成（CRAG）技术，构建了一个包含9,700个样本的坚实基准。对六种领先模型的评估显示，开源竞争者如DeepSeek R1在推理任务中媲美专有模型，但在基于判断的场景中表现较弱，可能是因为过度思考。我们的基准揭示了关键的企业性能缺口，并提供了可操作的模型优化建议。本研究为企业提供了一个定制评估的蓝图，并推动了实用的LLM部署。 

---
# Language Modeling by Language Models 

**Title (ZH)**: 语言模型_BY_语言模型 

**Authors**: Junyan Cheng, Peter Clark, Kyle Richardson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20249)  

**Abstract**: Can we leverage LLMs to model the process of discovering novel language model (LM) architectures? Inspired by real research, we propose a multi-agent LLM approach that simulates the conventional stages of research, from ideation and literature search (proposal stage) to design implementation (code generation), generative pre-training, and downstream evaluation (verification). Using ideas from scaling laws, our system, Genesys, employs a Ladder of Scales approach; new designs are proposed, adversarially reviewed, implemented, and selectively verified at increasingly larger model scales (14M$\sim$350M parameters) with a narrowing budget (the number of models we can train at each scale). To help make discovery efficient and factorizable, Genesys uses a novel genetic programming backbone, which we show has empirical advantages over commonly used direct prompt generation workflows (e.g., $\sim$86\% percentage point improvement in successful design generation, a key bottleneck). We report experiments involving 1,162 newly discovered designs (1,062 fully verified through pre-training) and find the best designs to be highly competitive with known architectures (e.g., outperform GPT2, Mamba2, etc., on 6/9 common benchmarks). We couple these results with comprehensive system-level ablations and formal results, which give broader insights into the design of effective autonomous discovery systems. 

**Abstract (ZH)**: 能否利用大语言模型来建模发现新型语言模型架构的过程？受真实研究的启发，我们提出了一种多智能体大语言模型方法，模拟了传统研究阶段，从理念构思和文献搜索（提案阶段）到设计实现（代码生成）、生成预训练和下游评估（验证）。借鉴扩展律的思想，我们的系统Genesys采用阶梯尺度方法；新设计被提出、对抗性审核、实施，并在越来越大的模型规模（14M至350M参数）下选择性验证，预算逐渐减少（每个规模下我们可以训练的模型数量）。为了提高发现的效率和可分解性，Genesys使用了一种新型的遗传编程架构，我们证明其在实验上比常用的直接提示生成工作流（例如，成功设计生成的百分点提升约86%）具有优势。我们报道了涉及1,162个新发现的设计（1,062个通过预训练完全验证）的实验，发现最优设计与已知架构具有很强的竞争力（例如，在6/9个常见基准上优于GPT2、Mamba2等）。我们结合了全面的系统级消融实验和正式结果，这为设计有效的自主发现系统提供了更广泛的见解。 

---
# AI Copilots for Reproducibility in Science: A Case Study 

**Title (ZH)**: AI副驾助力科学的可重复性：一个案例研究 

**Authors**: Adrien Bibal, Steven N. Minton, Deborah Khider, Yolanda Gil  

**Link**: [PDF](https://arxiv.org/pdf/2506.20130)  

**Abstract**: Open science initiatives seek to make research outputs more transparent, accessible, and reusable, but ensuring that published findings can be independently reproduced remains a persistent challenge. This paper introduces OpenPub, an AI-powered platform that supports researchers, reviewers, and readers through a suite of modular copilots focused on key open science tasks. In this work, we present the Reproducibility Copilot, which analyzes manuscripts, code, and supplementary materials to generate structured Jupyter Notebooks and recommendations aimed at facilitating computational, or "rote", reproducibility. We conducted feasibility tests using previously studied research papers with known reproducibility benchmarks. Results indicate that OpenPub can substantially reduce reproduction time - from over 30 hours to about 1 hour - while achieving high coverage of figures, tables, and results suitable for computational reproduction. The system systematically detects barriers to reproducibility, including missing hyperparameters, undocumented preprocessing steps, and incomplete or inaccessible datasets. These findings suggest that AI-driven tools can meaningfully reduce the burden of reproducibility efforts and contribute to more transparent and verifiable scientific communication. The modular copilot architecture also provides a foundation for extending AI assistance to additional open science objectives beyond reproducibility. 

**Abstract (ZH)**: 开源科学倡议旨在使研究成果更加透明、可访问和可重用，但确保已发表的研究结果可以独立重现仍然是一个持续的挑战。本文介绍了OpenPub，这是一个基于AI的平台，通过专注于关键开源任务的一系列模块化副驾驶，支持研究人员、审稿人和读者。在这项工作中，我们提出了重现性副驾驶，它可以分析手稿、代码和补充材料，生成结构化Jupyter Notebook和建议，以促进计算性或“机械性”的重现性。我们在具有已知重现性基准的先前研究论文上进行了可行性测试。结果表明，OpenPub可以显著减少重现时间——从超过30小时减少到约1小时——同时实现对适合计算性重现的图形、表格和结果的高水平覆盖。该系统系统地检测重现性障碍，包括缺失的超参数、未记录的预处理步骤以及不完整或不可访问的数据集。这些发现表明，以AI驱动的工具可以在减少重现性努力的负担方面发挥实质性作用，并有助于更透明和可验证的科学交流。模块化副驾驶架构也为将AI辅助扩展到重现性之外的其他开源目标奠定了基础。 

---
# DiaLLMs: EHR Enhanced Clinical Conversational System for Clinical Test Recommendation and Diagnosis Prediction 

**Title (ZH)**: DiaLLMs: 基于EHR的临床对话系统，用于临床检测推荐和诊断预测 

**Authors**: Weijieying Ren, Tianxiang Zhao, Lei Wang, Tianchun Wang, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20059)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to remarkable progresses in medical consultation. However, existing medical LLMs overlook the essential role of Electronic Health Records (EHR) and focus primarily on diagnosis recommendation, limiting their clinical applicability. We propose DiaLLM, the first medical LLM that integrates heterogeneous EHR data into clinically grounded dialogues, enabling clinical test recommendation, result interpretation, and diagnosis prediction to better align with real-world medical practice. To construct clinically grounded dialogues from EHR, we design a Clinical Test Reference (CTR) strategy that maps each clinical code to its corresponding description and classifies test results as "normal" or "abnormal". Additionally, DiaLLM employs a reinforcement learning framework for evidence acquisition and automated diagnosis. To handle the large action space, we introduce a reject sampling strategy to reduce redundancy and improve exploration efficiency. Furthermore, a confirmation reward and a class-sensitive diagnosis reward are designed to guide accurate diagnosis prediction. Extensive experimental results demonstrate that DiaLLM outperforms baselines in clinical test recommendation and diagnosis prediction. 

**Abstract (ZH)**: Recent Advances in Large Language Models (LLMs) Have Led to Remarkable Progresses in Medical Consultation: DiaLLM, the First Medical LLM That Integrates Heterogeneous EHR Data into Clinically Grounded Dialogues 

---
# Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning 

**Title (ZH)**: 个性化赋权的大语言模型显示出类似人类的动力型推理。 

**Authors**: Saloni Dash, Amélie Reymond, Emma S. Spiro, Aylin Caliskan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20020)  

**Abstract**: Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This motivated reasoning at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically, are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans. 

**Abstract (ZH)**: 大型语言模型受人格化身份动机影响的归纳推理：实证研究 

---
# Achieving Trustworthy Real-Time Decision Support Systems with Low-Latency Interpretable AI Models 

**Title (ZH)**: 实现低延迟可解释人工智能模型以建立可信的实时决策支持系统 

**Authors**: Zechun Deng, Ziwei Liu, Ziqian Bi, Junhao Song, Chia Xin Liang, Joe Yeong, Junfeng Hao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20018)  

**Abstract**: This paper investigates real-time decision support systems that leverage low-latency AI models, bringing together recent progress in holistic AI-driven decision tools, integration with Edge-IoT technologies, and approaches for effective human-AI teamwork. It looks into how large language models can assist decision-making, especially when resources are limited. The research also examines the effects of technical developments such as DeLLMa, methods for compressing models, and improvements for analytics on edge devices, while also addressing issues like limited resources and the need for adaptable frameworks. Through a detailed review, the paper offers practical perspectives on development strategies and areas of application, adding to the field by pointing out opportunities for more efficient and flexible AI-supported systems. The conclusions set the stage for future breakthroughs in this fast-changing area, highlighting how AI can reshape real-time decision support. 

**Abstract (ZH)**: 本文探讨了利用低延迟AI模型的实时决策支持系统，综述了整体AI驱动决策工具的进步、Edge-IoT技术的集成以及有效的人机团队合作方法。研究了大型语言模型在资源受限条件下如何辅助决策，并探讨了如DeLLMa等技术发展、模型压缩方法以及边缘设备上数据分析改进的效果，同时关注资源限制和可适应框架的需求。通过详细综述，本文提供了开发策略和应用领域的实用观点，为更高效和灵活的AI支持系统指出了机会，并为这一快速变化领域未来的突破奠定了基础，强调了AI如何重塑实时决策支持。 

---
# Accurate and Energy Efficient: Local Retrieval-Augmented Generation Models Outperform Commercial Large Language Models in Medical Tasks 

**Title (ZH)**: 准确且节能：局部检索增强生成模型在医疗任务中优于商用大型语言模型 

**Authors**: Konstantinos Vrettos, Michail E. Klontzas  

**Link**: [PDF](https://arxiv.org/pdf/2506.20009)  

**Abstract**: Background The increasing adoption of Artificial Intelligence (AI) in healthcare has sparked growing concerns about its environmental and ethical implications. Commercial Large Language Models (LLMs), such as ChatGPT and DeepSeek, require substantial resources, while the utilization of these systems for medical purposes raises critical issues regarding patient privacy and safety. Methods We developed a customizable Retrieval-Augmented Generation (RAG) framework for medical tasks, which monitors its energy usage and CO2 emissions. This system was then used to create RAGs based on various open-source LLMs. The tested models included both general purpose models like llama3.1:8b and medgemma-4b-it, which is medical-domain specific. The best RAGs performance and energy consumption was compared to DeepSeekV3-R1 and OpenAIs o4-mini model. A dataset of medical questions was used for the evaluation. Results Custom RAG models outperformed commercial models in accuracy and energy consumption. The RAG model built on llama3.1:8B achieved the highest accuracy (58.5%) and was significantly better than other models, including o4-mini and DeepSeekV3-R1. The llama3.1-RAG also exhibited the lowest energy consumption and CO2 footprint among all models, with a Performance per kWh of 0.52 and a total CO2 emission of 473g. Compared to o4-mini, the llama3.1-RAG achieved 2.7x times more accuracy points per kWh and 172% less electricity usage while maintaining higher accuracy. Conclusion Our study demonstrates that local LLMs can be leveraged to develop RAGs that outperform commercial, online LLMs in medical tasks, while having a smaller environmental impact. Our modular framework promotes sustainable AI development, reducing electricity usage and aligning with the UNs Sustainable Development Goals. 

**Abstract (ZH)**: 背景：人工智能（AI）在医疗领域的广泛应用引发了对其环境和伦理影响的关注。商用大语言模型（LLMs），如ChatGPT和DeepSeek，需要大量资源，而这些系统在医疗领域的应用则引发了关于患者隐私和安全的 critical 问题。

方法：我们开发了一种可定制的检索增强生成（RAG）框架，用于医疗任务，该框架监控其能源使用和二氧化碳排放。然后，利用该系统基于各种开源LLMs创建了RAGs。测试的模型包括通用模型如llama3.1:8b和针对医疗领域的medgemma-4b-it。最佳RAG性能和能源消耗与DeepSeekV3-R1和OpenAI的o4-mini模型进行了比较。使用医疗问题数据集进行了评估。

结果：定制的RAG模型在准确性和能源消耗方面优于商用模型。基于llama3.1:8B构建的RAG模型达到了最高准确率（58.5%），并显著优于其他模型，包括o4-mini和DeepSeekV3-R1。llama3.1-RAG也是所有模型中能源消耗和二氧化碳足迹最低的，每千瓦时性能为0.52，总二氧化碳排放量为473克。与o4-mini相比，llama3.1-RAG在每千瓦时获得了2.7倍的准确率点，并减少了172%的电能使用，同时保持了更高的准确率。

结论：我们的研究表明，利用本地LLMs可以开发出在医疗任务中优于商用在线LLMs的RAGs，同时对环境影响较小。我们的模块化框架促进了可持续的人工智能开发，减少了电能使用，并与联合国可持续发展目标一致。 

---
# QHackBench: Benchmarking Large Language Models for Quantum Code Generation Using PennyLane Hackathon Challenges 

**Title (ZH)**: QHackBench：基于PennyLane黑客马拉松挑战任务的大型语言模型量子代码生成性能评测 

**Authors**: Abdul Basit, Minghao Shao, Haider Asif, Nouhaila Innan, Muhammad Kashif, Alberto Marchisio, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2506.20008)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated strong potential in code generation, yet their effectiveness in quantum computing remains underexplored. This paper benchmarks LLMs for PennyLane-based quantum code generation using real-world challenges from the Quantum Hackathon (QHack). We introduce QHackBench, a novel benchmark dataset derived from QHack competitions, and evaluate model performance under vanilla prompting and Retrieval-Augmented Generation (RAG). Our structured evaluation framework assesses functional correctness, syntactic validity, and execution success across varying challenge difficulties. Results indicate that RAG-enhanced models, supplemented with an augmented PennyLane dataset, approximately generate similar results as the standard prompting, particularly in complex quantum algorithms. Additionally, we introduce a multi-agent evaluation pipeline that iteratively refines incorrect solutions, further enhancing execution success rates. To foster further research, we commit to publicly releasing QHackBench, along with our evaluation framework and experimental results, enabling continued advancements in AI-assisted quantum programming. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在代码生成领域展示了强大的潜力，但在量子计算领域的有效性仍鲜有探索。本文使用Quantum Hackathon（QHack）中的真实挑战，评估PennyLane为基础的量子代码生成的大型语言模型。我们引入了QHackBench，一个基于QHack竞赛的新基准数据集，并对其在vanilla提示和检索增强生成（RAG）下的模型性能进行了评估。我们的结构化评估框架评估了功能正确性、语法有效性以及在不同挑战难度下的执行成功率。结果表明，结合增强PennyLane数据集的RAG增强模型，生成的结果与标准提示方法接近，尤其是在复杂量子算法中。此外，我们提出了一个迭代校正不正确解的多智能体评估流水线，进一步提高了执行成功率。为促进进一步研究，我们承诺公开发布QHackBench，以及我们的评估框架和实验结果，推动AI辅助量子编程的持续发展。 

---
# Context Attribution with Multi-Armed Bandit Optimization 

**Title (ZH)**: 多臂老虎机优化的上下文归因 

**Authors**: Deng Pan, Keerthiram Murugesan, Nuno Moniz, Nitesh Chawla  

**Link**: [PDF](https://arxiv.org/pdf/2506.19977)  

**Abstract**: Understanding which parts of the retrieved context contribute to a large language model's generated answer is essential for building interpretable and trustworthy generative QA systems. We propose a novel framework that formulates context attribution as a combinatorial multi-armed bandit (CMAB) problem. Each context segment is treated as a bandit arm, and we employ Combinatorial Thompson Sampling (CTS) to efficiently explore the exponentially large space of context subsets under a limited query budget. Our method defines a reward function based on normalized token likelihoods, capturing how well a subset of segments supports the original model response. Unlike traditional perturbation-based attribution methods such as SHAP, which sample subsets uniformly and incur high computational costs, our approach adaptively balances exploration and exploitation by leveraging posterior estimates of segment relevance. This leads to substantially improved query efficiency while maintaining high attribution fidelity. Extensive experiments on diverse datasets and LLMs demonstrate that our method achieves competitive attribution quality with fewer model queries. 

**Abstract (ZH)**: 理解检索到的上下文哪些部分对大型语言模型生成的答案有贡献对于构建可解释和可信赖的生成式问答系统至关重要。我们提出了一种新的框架，将上下文归因问题形式化为组合多臂bandit (CMAB) 问题。每个上下文片段被视为一个bandit臂，并采用组合 Thompson 抽样 (CTS) 有效地在有限的查询预算下探索上下文子集的指数级空间。我们的方法基于归一化令牌可能性定义奖励函数，捕捉片段子集如何支持原始模型响应。与传统的基于扰动的归因方法（如SHAP），后者均匀抽样子集并产生高计算成本不同，我们的方法通过利用片段相关性的后验估计，适应性地平衡探索与利用。这在提高查询效率的同时保持了高归因保真度。在多样化的数据集和LLMs上的广泛实验表明，我们的方法在较少的模型查询下实现与传统方法相当的归因质量。 

---
# Prover Agent: An Agent-based Framework for Formal Mathematical Proofs 

**Title (ZH)**: 证明代理：基于代理的正式数学证明框架 

**Authors**: Kaito Baba, Chaoran Liu, Shuhei Kurita, Akiyoshi Sannai  

**Link**: [PDF](https://arxiv.org/pdf/2506.19923)  

**Abstract**: We present Prover Agent, a novel AI agent for automated theorem proving that integrates large language models (LLMs) with a formal proof assistant, Lean. Prover Agent coordinates an informal reasoning LLM, a formal prover model, and feedback from Lean while also generating auxiliary lemmas to assist in discovering the overall proof strategy. It achieves an 86.1% success rate on the MiniF2F benchmark, establishing a new state-of-the-art among methods using small language models (SLMs) with a much lower sample budget than previous approaches. We also present case studies illustrating how these generated lemmas contribute to solving challenging problems. 

**Abstract (ZH)**: 我们介绍了Prover Agent，这是一种将大型语言模型与形式证明助手Lean集成的新型AI代理，用于自动化定理证明。Prover Agent协调非形式推理的大语言模型、形式证明模型以及来自Lean的反馈，同时生成辅助引理以协助发现整体证明策略。它在MiniF2F基准测试中的成功率达到86.1%，是使用小型语言模型的方法中新的前沿，同时比之前的方法使用了更低的数据样本预算。我们还介绍了案例研究，说明这些生成的引理如何有助于解决复杂问题。 

---
# Inside you are many wolves: Using cognitive models to interpret value trade-offs in LLMs 

**Title (ZH)**: 内在你中有许多狼：使用认知模型解释LLMs的价值权衡 

**Authors**: Sonia K. Murthy, Rosie Zhao, Jennifer Hu, Sham Kakade, Markus Wulfmeier, Peng Qian, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20666)  

**Abstract**: Navigating everyday social situations often requires juggling conflicting goals, such as conveying a harsh truth, maintaining trust, all while still being mindful of another person's feelings. These value trade-offs are an integral part of human decision-making and language use, however, current tools for interpreting such dynamic and multi-faceted notions of values in LLMs are limited. In cognitive science, so-called "cognitive models" provide formal accounts of these trade-offs in humans, by modeling the weighting of a speaker's competing utility functions in choosing an action or utterance. In this work, we use a leading cognitive model of polite speech to interpret the extent to which LLMs represent human-like trade-offs. We apply this lens to systematically evaluate value trade-offs in two encompassing model settings: degrees of reasoning "effort" in frontier black-box models, and RL post-training dynamics of open-source models. Our results highlight patterns of higher informational utility than social utility in reasoning models, and in open-source models shown to be stronger in mathematical reasoning. Our findings from LLMs' training dynamics suggest large shifts in utility values early on in training with persistent effects of the choice of base model and pretraining data, compared to feedback dataset or alignment method. We show that our method is responsive to diverse aspects of the rapidly evolving LLM landscape, with insights for forming hypotheses about other high-level behaviors, shaping training regimes for reasoning models, and better controlling trade-offs between values during model training. 

**Abstract (ZH)**: 导航日常社交情境往往需要权衡相互冲突的目标，如传达严峻的事实、维持信任等方面，同时还要考虑到对方的感受。然而，当下的工具在解释此类动态且多维度的价值观念时仍是有限的。在认知科学中，所谓的“认知模型”通过建模说话人在选择行动或言语时竞争的利益函数的权重，提供人类这些权衡的正式解释。在本工作中，我们利用礼貌言语的认知模型来解释LLMs在人类样式的权衡中的表现程度。我们将这一视角应用于系统地评估两种广泛的模型设置中的价值权衡：推理“努力”程度的等级在前沿的黑盒模型中，以及开放源代码模型的强化学习后训练动态。我们的结果强调了推理模型中的信息效用高于社会效用的模式，并且在显示出更强数学推理能力的开放源代码模型中也是如此。从LLMs的训练动态中，我们的发现表明，大型模型在训练早期发生显著的效用值转变，这些效果持续影响基础模型和预训练数据的选择，相对于反馈数据集或对齐方法而言。我们展示了我们的方法能够响应快速变化的LLM景观中的各种方面，并为形成其他高层次行为的假设、塑造推理模型的训练制度以及更好地控制模型训练中的价值权衡提供了见解。 

---
# Disentangled representations of microscopy images 

**Title (ZH)**: 显微图像的解耦表示 

**Authors**: Jacopo Dapueto, Vito Paolo Pastore, Nicoletta Noceti, Francesca Odone  

**Link**: [PDF](https://arxiv.org/pdf/2506.20649)  

**Abstract**: Microscopy image analysis is fundamental for different applications, from diagnosis to synthetic engineering and environmental monitoring. Modern acquisition systems have granted the possibility to acquire an escalating amount of images, requiring a consequent development of a large collection of deep learning-based automatic image analysis methods. Although deep neural networks have demonstrated great performance in this field, interpretability, an essential requirement for microscopy image analysis, remains an open challenge.
This work proposes a Disentangled Representation Learning (DRL) methodology to enhance model interpretability for microscopy image classification. Exploiting benchmark datasets from three different microscopic image domains (plankton, yeast vacuoles, and human cells), we show how a DRL framework, based on transferring a representation learnt from synthetic data, can provide a good trade-off between accuracy and interpretability in this domain. 

**Abstract (ZH)**: 显微镜图像分析对于从诊断到合成工程和环境监测等不同应用至关重要。现代采集系统使得获取越来越多的图像成为可能，这要求开发大量的基于深度学习的自动图像分析方法。尽管深度神经网络在这个领域表现出了出色的效果，但对于显微镜图像分析至关重要的可解释性仍是一个开放的挑战。

本文提出了一种解耦表示学习（DRL）方法，以增强显微镜图像分类模型的可解释性。利用来自三个不同显微镜图像领域的基准数据集（浮游生物、酵母Vacuoles和人类细胞），我们展示了基于从合成数据迁移学习得到的表示的DRL框架，能够在该领域提供准确性和可解释性之间的良好权衡。 

---
# Define-ML: An Approach to Ideate Machine Learning-Enabled Systems 

**Title (ZH)**: Define-ML：一种机器学习赋能系统构思的方法 

**Authors**: Silvio Alonso, Antonio Pedro Santos Alves, Lucas Romao, Hélio Lopes, Marcos Kalinowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.20621)  

**Abstract**: [Context] The increasing adoption of machine learning (ML) in software systems demands specialized ideation approaches that address ML-specific challenges, including data dependencies, technical feasibility, and alignment between business objectives and probabilistic system behavior. Traditional ideation methods like Lean Inception lack structured support for these ML considerations, which can result in misaligned product visions and unrealistic expectations. [Goal] This paper presents Define-ML, a framework that extends Lean Inception with tailored activities - Data Source Mapping, Feature-to-Data Source Mapping, and ML Mapping - to systematically integrate data and technical constraints into early-stage ML product ideation. [Method] We developed and validated Define-ML following the Technology Transfer Model, conducting both static validation (with a toy problem) and dynamic validation (in a real-world industrial case study). The analysis combined quantitative surveys with qualitative feedback, assessing utility, ease of use, and intent of adoption. [Results] Participants found Define-ML effective for clarifying data concerns, aligning ML capabilities with business goals, and fostering cross-functional collaboration. The approach's structured activities reduced ideation ambiguity, though some noted a learning curve for ML-specific components, which can be mitigated by expert facilitation. All participants expressed the intention to adopt Define-ML. [Conclusion] Define-ML provides an openly available, validated approach for ML product ideation, building on Lean Inception's agility while aligning features with available data and increasing awareness of technical feasibility. 

**Abstract (ZH)**: 定义-机器学习：Lean Inception的扩展框架以系统地将数据和技术约束纳入早期机器学习产品构想 

---
# Weighted Mean Frequencies: a handcraft Fourier feature for 4D Flow MRI segmentation 

**Title (ZH)**: 加权平均频率：一种手工制作的傅里叶特征用于4D流MRI分割 

**Authors**: Simon Perrin, Sébastien Levilly, Huajun Sun, Harold Mouchère, Jean-Michel Serfaty  

**Link**: [PDF](https://arxiv.org/pdf/2506.20614)  

**Abstract**: In recent decades, the use of 4D Flow MRI images has enabled the quantification of velocity fields within a volume of interest and along the cardiac cycle. However, the lack of resolution and the presence of noise in these biomarkers are significant issues. As indicated by recent studies, it appears that biomarkers such as wall shear stress are particularly impacted by the poor resolution of vessel segmentation. The Phase Contrast Magnetic Resonance Angiography (PC-MRA) is the state-of-the-art method to facilitate segmentation. The objective of this work is to introduce a new handcraft feature that provides a novel visualisation of 4D Flow MRI images, which is useful in the segmentation task. This feature, termed Weighted Mean Frequencies (WMF), is capable of revealing the region in three dimensions where a voxel has been passed by pulsatile flow. Indeed, this feature is representative of the hull of all pulsatile velocity voxels. The value of the feature under discussion is illustrated by two experiments. The experiments involved segmenting 4D Flow MRI images using optimal thresholding and deep learning methods. The results obtained demonstrate a substantial enhancement in terms of IoU and Dice, with a respective increase of 0.12 and 0.13 in comparison with the PC-MRA feature, as evidenced by the deep learning task. This feature has the potential to yield valuable insights that could inform future segmentation processes in other vascular regions, such as the heart or the brain. 

**Abstract (ZH)**: 近几十年来，4D Flow MRI图像的使用使得在感兴趣体积和整个心脏周期内定量分析速度场成为可能。然而，这些生物标志物分辨率低且存在噪声是重要问题。最近的研究表明，诸如壁剪应力等生物标志物特别受到血管分割低分辨率的影响。相位对比磁共振血管造影（PC-MRA）是目前最先进的分割方法。本文的目的在于介绍一种新的手工特征，该特征为4D Flow MRI图像提供了一种新颖的可视化方法，有助于分割任务。该特征称为加权均值频率（WMF），能够揭示三维空间中脉动流体通过的体元区域。实际上，该特征代表了所有脉动速度体元的包络。通过两组实验说明了该特征的价值。实验使用了最优阈值分割和深度学习方法来分割4D Flow MRI图像。结果显示，相较于PC-MRA特征，在深度学习任务中IoU和Dice值分别提高了0.12和0.13，证明了该特征的有效性。该特征在其他血管区域，如心脏或大脑的分割过程中有可能提供有价值的见解。 

---
# Deciphering GunType Hierarchy through Acoustic Analysis of Gunshot Recordings 

**Title (ZH)**: 通过枪声录音的声学分析 deciphering 枪型层次结构 

**Authors**: Ankit Shah, Rita Singh, Bhiksha Raj, Alexander Hauptmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.20609)  

**Abstract**: The escalating rates of gun-related violence and mass shootings represent a significant threat to public safety. Timely and accurate information for law enforcement agencies is crucial in mitigating these incidents. Current commercial gunshot detection systems, while effective, often come with prohibitive costs. This research explores a cost-effective alternative by leveraging acoustic analysis of gunshot recordings, potentially obtainable from ubiquitous devices like cell phones, to not only detect gunshots but also classify the type of firearm used. This paper details a study on deciphering gun type hierarchies using a curated dataset of 3459 recordings. We investigate the fundamental acoustic characteristics of gunshots, including muzzle blasts and shockwaves, which vary based on firearm type, ammunition, and shooting direction. We propose and evaluate machine learning frameworks, including Support Vector Machines (SVMs) as a baseline and a more advanced Convolutional Neural Network (CNN) architecture for joint gunshot detection and gun type classification. Results indicate that our deep learning approach achieves a mean average precision (mAP) of 0.58 on clean labeled data, outperforming the SVM baseline (mAP 0.39). Challenges related to data quality, environmental noise, and the generalization capabilities when using noisy web-sourced data (mAP 0.35) are also discussed. The long-term vision is to develop a highly accurate, real-time system deployable on common recording devices, significantly reducing detection costs and providing critical intelligence to first responders. 

**Abstract (ZH)**: escalating 枪击事件和群体性枪击事件频率的上升对公共安全构成了重大威胁。及时准确的信息对于执法机构减轻这些事件的影响至关重要。当前的商业枪声检测系统虽然有效，但往往成本高昂。本研究通过利用来自常见设备（如手机）的枪声录音声学分析探索了一种成本效益更高的替代方案，不仅用于检测枪声，还用于识别所使用的枪械类型。本文详细介绍了使用3459条录音制作的精心挑选数据集研究枪械类型层次结构的研究。我们探讨了枪声的基本声学特征，包括枪口爆裂和冲击波，这些特征根据枪械类型、弹药和射击方向而变化。我们提出了机器学习框架，包括支持向量机（SVM）作为基线和更先进的卷积神经网络（CNN）结构，用于联合枪声检测和枪械类型分类。结果表明，在干净标签数据上，我们的深度学习方法平均准确率（mAP）达到0.58，优于SVM基线（mAP 0.39）。还讨论了数据质量、环境噪音以及在使用嘈杂的网络来源数据时的泛化能力（mAP 0.35）带来的挑战。长期愿景是开发一种高度准确的实时系统，可部署在常见录音设备上，显著降低检测成本并为一线紧急响应人员提供关键情报。 

---
# AI in the Writing Process: How Purposeful AI Support Fosters Student Writing 

**Title (ZH)**: AI在写作过程中的应用：目标导向的AI支持对提升学生写作能力的影响 

**Authors**: Momin N. Siddiqui, Roy Pea, Hari Subramonyam  

**Link**: [PDF](https://arxiv.org/pdf/2506.20595)  

**Abstract**: The ubiquity of technologies like ChatGPT has raised concerns about their impact on student writing, particularly regarding reduced learner agency and superficial engagement with content. While standalone chat-based LLMs often produce suboptimal writing outcomes, evidence suggests that purposefully designed AI writing support tools can enhance the writing process. This paper investigates how different AI support approaches affect writers' sense of agency and depth of knowledge transformation. Through a randomized control trial with 90 undergraduate students, we compare three conditions: (1) a chat-based LLM writing assistant, (2) an integrated AI writing tool to support diverse subprocesses, and (3) a standard writing interface (control). Our findings demonstrate that, among AI-supported conditions, students using the integrated AI writing tool exhibited greater agency over their writing process and engaged in deeper knowledge transformation overall. These results suggest that thoughtfully designed AI writing support targeting specific aspects of the writing process can help students maintain ownership of their work while facilitating improved engagement with content. 

**Abstract (ZH)**: 像ChatGPT这样的技术的普遍存在引发了对其对学生写作影响的担忧，特别是关于学习者自主性的降低和内容的浅层参与。虽然独立的基于聊天的大型语言模型往往产生次优的写作效果，但证据表明，目的设计的AI写作支持工具可以增强写作过程。本文探讨不同AI支持方法如何影响作者的自主感和知识深度转换。通过一项包含90名本科生的随机对照试验，我们比较了三种条件：（1）基于聊天的LLM写作助手，（2）集成AI写作工具以支持多样化的次过程，以及（3）标准写作界面（对照）。研究发现，在AI支持的条件下，使用集成AI写作工具的学生在写作过程中表现出更大的自主性，并且总体上参与了更深层次的知识转换。这些结果表明，针对写作过程特定方面的精心设计的AI写作支持可以帮助学生保持对自己作品的所有权，同时促进他们与内容的更好参与。 

---
# Dense Video Captioning using Graph-based Sentence Summarization 

**Title (ZH)**: 基于图结构句子总结的密集视频字幕生成 

**Authors**: Zhiwang Zhang, Dong Xu, Wanli Ouyang, Luping Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20583)  

**Abstract**: Recently, dense video captioning has made attractive progress in detecting and captioning all events in a long untrimmed video. Despite promising results were achieved, most existing methods do not sufficiently explore the scene evolution within an event temporal proposal for captioning, and therefore perform less satisfactorily when the scenes and objects change over a relatively long proposal. To address this problem, we propose a graph-based partition-and-summarization (GPaS) framework for dense video captioning within two stages. For the ``partition" stage, a whole event proposal is split into short video segments for captioning at a finer level. For the ``summarization" stage, the generated sentences carrying rich description information for each segment are summarized into one sentence to describe the whole event. We particularly focus on the ``summarization" stage, and propose a framework that effectively exploits the relationship between semantic words for summarization. We achieve this goal by treating semantic words as nodes in a graph and learning their interactions by coupling Graph Convolutional Network (GCN) and Long Short Term Memory (LSTM), with the aid of visual cues. Two schemes of GCN-LSTM Interaction (GLI) modules are proposed for seamless integration of GCN and LSTM. The effectiveness of our approach is demonstrated via an extensive comparison with the state-of-the-arts methods on the two benchmarks ActivityNet Captions dataset and YouCook II dataset. 

**Abstract (ZH)**: 基于图的分割与总结框架（GPaS）用于视频事件描述的密集视频字幕 

---
# Causal Representation Learning with Observational Grouping for CXR Classification 

**Title (ZH)**: 基于观测分组的因果表示学习在胸部X光分类中的应用 

**Authors**: Rajat Rasal, Avinash Kori, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20582)  

**Abstract**: Identifiable causal representation learning seeks to uncover the true causal relationships underlying a data generation process. In medical imaging, this presents opportunities to improve the generalisability and robustness of task-specific latent features. This work introduces the concept of grouping observations to learn identifiable representations for disease classification in chest X-rays via an end-to-end framework. Our experiments demonstrate that these causal representations improve generalisability and robustness across multiple classification tasks when grouping is used to enforce invariance w.r.t race, sex, and imaging views. 

**Abstract (ZH)**: 可识别的因果表示学习旨在揭示数据生成过程中真实的因果关系。在医学成像领域，这为提高特定任务的潜变量特征的一般化能力和稳健性提供了机会。本工作通过端到端框架引入了将观测分组以学习可识别表示的思想，用于胸片疾病分类。我们的实验表明，在分组时强制不变性（针对种族、性别和成像视角），这些因果表示能够提高多种分类任务的一般化能力和稳健性。 

---
# Vulnerability Disclosure through Adaptive Black-Box Adversarial Attacks on NIDS 

**Title (ZH)**: 基于自适应黑盒 adversarial 攻击的 NIDS 漏洞披露 

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi V. Mancini  

**Link**: [PDF](https://arxiv.org/pdf/2506.20576)  

**Abstract**: Adversarial attacks, wherein slight inputs are carefully crafted to mislead intelligent models, have attracted increasing attention. However, a critical gap persists between theoretical advancements and practical application, particularly in structured data like network traffic, where interdependent features complicate effective adversarial manipulations. Moreover, ambiguity in current approaches restricts reproducibility and limits progress in this field. Hence, existing defenses often fail to handle evolving adversarial attacks. This paper proposes a novel approach for black-box adversarial attacks, that addresses these limitations. Unlike prior work, which often assumes system access or relies on repeated probing, our method strictly respect black-box constraints, reducing interaction to avoid detection and better reflect real-world scenarios. We present an adaptive feature selection strategy using change-point detection and causality analysis to identify and target sensitive features to perturbations. This lightweight design ensures low computational cost and high deployability. Our comprehensive experiments show the attack's effectiveness in evading detection with minimal interaction, enhancing its adaptability and applicability in real-world scenarios. By advancing the understanding of adversarial attacks in network traffic, this work lays a foundation for developing robust defenses. 

**Abstract (ZH)**: adversarial 攻击中的黑盒攻击方法：针对结构化数据的新型策略 

---
# Show, Tell and Summarize: Dense Video Captioning Using Visual Cue Aided Sentence Summarization 

**Title (ZH)**: 展示、描述和总结：基于视觉线索辅助句子总结的密集视频描述 

**Authors**: Zhiwang Zhang, Dong Xu, Wanli Ouyang, Chuanqi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20567)  

**Abstract**: In this work, we propose a division-and-summarization (DaS) framework for dense video captioning. After partitioning each untrimmed long video as multiple event proposals, where each event proposal consists of a set of short video segments, we extract visual feature (e.g., C3D feature) from each segment and use the existing image/video captioning approach to generate one sentence description for this segment. Considering that the generated sentences contain rich semantic descriptions about the whole event proposal, we formulate the dense video captioning task as a visual cue aided sentence summarization problem and propose a new two stage Long Short Term Memory (LSTM) approach equipped with a new hierarchical attention mechanism to summarize all generated sentences as one descriptive sentence with the aid of visual features. Specifically, the first-stage LSTM network takes all semantic words from the generated sentences and the visual features from all segments within one event proposal as the input, and acts as the encoder to effectively summarize both semantic and visual information related to this event proposal. The second-stage LSTM network takes the output from the first-stage LSTM network and the visual features from all video segments within one event proposal as the input, and acts as the decoder to generate one descriptive sentence for this event proposal. Our comprehensive experiments on the ActivityNet Captions dataset demonstrate the effectiveness of our newly proposed DaS framework for dense video captioning. 

**Abstract (ZH)**: 基于分段和总结的密集视频描述框架 

---
# DeepQuark: deep-neural-network approach to multiquark bound states 

**Title (ZH)**: DeepQuark：深度神经网络方法研究多夸克束缚态 

**Authors**: Wei-Lin Wu, Lu Meng, Shi-Lin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20555)  

**Abstract**: For the first time, we implement the deep-neural-network-based variational Monte Carlo approach for the multiquark bound states, whose complexity surpasses that of electron or nucleon systems due to strong SU(3) color interactions. We design a novel and high-efficiency architecture, DeepQuark, to address the unique challenges in multiquark systems such as stronger correlations, extra discrete quantum numbers, and intractable confinement interaction. Our method demonstrates competitive performance with state-of-the-art approaches, including diffusion Monte Carlo and Gaussian expansion method, in the nucleon, doubly heavy tetraquark, and fully heavy tetraquark systems. Notably, it outperforms existing calculations for pentaquarks, exemplified by the triply heavy pentaquark. For the nucleon, we successfully incorporate three-body flux-tube confinement interactions without additional computational costs. In tetraquark systems, we consistently describe hadronic molecule $T_{cc}$ and compact tetraquark $T_{bb}$ with an unbiased form of wave function ansatz. In the pentaquark sector, we obtain weakly bound $\bar D^*\Xi_{cc}^*$ molecule $P_{cc\bar c}(5715)$ with $S=\frac{5}{2}$ and its bottom partner $P_{bb\bar b}(15569)$. They can be viewed as the analogs of the molecular $T_{cc}$. We recommend experimental search of $P_{cc\bar c}(5715)$ in the D-wave $J/\psi \Lambda_c$ channel. DeepQuark holds great promise for extension to larger multiquark systems, overcoming the computational barriers in conventional methods. It also serves as a powerful framework for exploring confining mechanism beyond two-body interactions in multiquark states, which may offer valuable insights into nonperturbative QCD and general many-body physics. 

**Abstract (ZH)**: 基于深度神经网络的变分蒙特卡罗方法首次应用于多夸克束缚态，因其复杂性超出电子或核子系统，主要原因在于强烈的SU(3)色相互作用。我们设计了一种新颖且高效的架构DeepQuark，以解决多夸克系统中更强的相关性、额外的离散量子数以及难以处理的束缚相互作用等独特挑战。我们的方法在核子、双重重夸克四夸克系统和完全重夸克四夸克系统中展示了与最先进的方法（包括扩散蒙特卡罗和高斯展开方法）竞争的性能。特别是在五夸克系统中，DeepQuark在三重重夸克五夸克$P_{cc\bar{c}}(5715)$的计算中超越了现有方法，$S=\frac{5}{2}$，以及其底夸克伙伴$P_{bb\bar{b}}(15569)$。它们可以视为四夸克分子$T_{cc}$的类比。我们建议在$D$波$J/\psi \Lambda_c$通道中寻找$P_{cc\bar{c}}(5715)$的实验。DeepQuark在扩展到更大规模的多夸克系统以及探索多夸克态中超越二体相互作用的束缚机制方面具有巨大潜力，这可能为非微扰量子色动力学和广义多体物理学提供宝贵见解。 

---
# Large Language Model-Driven Code Compliance Checking in Building Information Modeling 

**Title (ZH)**: 大型语言模型驱动的建筑信息建模代码合规性检查 

**Authors**: Soumya Madireddy, Lu Gao, Zia Din, Kinam Kim, Ahmed Senouci, Zhe Han, Yunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20551)  

**Abstract**: This research addresses the time-consuming and error-prone nature of manual code compliance checking in Building Information Modeling (BIM) by introducing a Large Language Model (LLM)-driven approach to semi-automate this critical process. The developed system integrates LLMs such as GPT, Claude, Gemini, and Llama, with Revit software to interpret building codes, generate Python scripts, and perform semi-automated compliance checks within the BIM environment. Case studies on a single-family residential project and an office building project demonstrated the system's ability to reduce the time and effort required for compliance checks while improving accuracy. It streamlined the identification of violations, such as non-compliant room dimensions, material usage, and object placements, by automatically assessing relationships and generating actionable reports. Compared to manual methods, the system eliminated repetitive tasks, simplified complex regulations, and ensured reliable adherence to standards. By offering a comprehensive, adaptable, and cost-effective solution, this proposed approach offers a promising advancement in BIM-based compliance checking, with potential applications across diverse regulatory documents in construction projects. 

**Abstract (ZH)**: 通过大型语言模型驱动的半自动化方法解决建筑信息建模（BIM）中手动代码合规检查的时间 consuming 和易出错性质 

---
# Pay Less Attention to Deceptive Artifacts: Robust Detection of Compressed Deepfakes on Online Social Networks 

**Title (ZH)**: 较少关注欺骗性伪影：在线社交网络中压缩深度伪造的鲁棒检测 

**Authors**: Manyi Li, Renshuai Tao, Yufan Liu, Chuangchuang Tan, Haotong Qin, Bing Li, Yunchao Wei, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.20548)  

**Abstract**: With the rapid advancement of deep learning, particularly through generative adversarial networks (GANs) and diffusion models (DMs), AI-generated images, or ``deepfakes", have become nearly indistinguishable from real ones. These images are widely shared across Online Social Networks (OSNs), raising concerns about their misuse. Existing deepfake detection methods overlook the ``block effects" introduced by compression in OSNs, which obscure deepfake artifacts, and primarily focus on raw images, rarely encountered in real-world scenarios. To address these challenges, we propose PLADA (Pay Less Attention to Deceptive Artifacts), a novel framework designed to tackle the lack of paired data and the ineffective use of compressed images. PLADA consists of two core modules: Block Effect Eraser (B2E), which uses a dual-stage attention mechanism to handle block effects, and Open Data Aggregation (ODA), which processes both paired and unpaired data to improve detection. Extensive experiments across 26 datasets demonstrate that PLADA achieves a remarkable balance in deepfake detection, outperforming SoTA methods in detecting deepfakes on OSNs, even with limited paired data and compression. More importantly, this work introduces the ``block effect" as a critical factor in deepfake detection, providing a robust solution for open-world scenarios. Our code is available at this https URL. 

**Abstract (ZH)**: 随着深度学习的迅速发展，尤其是生成对抗网络（GANs）和扩散模型（DMs）的应用，AI生成的图像或“换脸”图像越来越难以与真实图像区分开来。这些图像在在线社交网络（OSNs）上广泛传播，引发了对其误用的担忧。现有换脸检测方法忽略了OSNs压缩引入的“块效应”，这些效应会掩盖换脸痕迹，并主要关注原始图像，而在真实场景中很少遇到。为应对这些挑战，我们提出了PLADA（忽略欺骗性伪影的关注），一个新框架，旨在解决配对数据缺乏和压缩图像无效使用的问题。PLADA由两个核心模块组成：块效应消除器（B2E），它采用双阶段注意力机制处理块效应，以及开放数据聚合器（ODA），它处理配对和未配对数据以提高检测效果。在26个数据集中进行的广泛实验表明，PLADA在OSNs中的换脸检测上表现出色，即使配对数据有限且存在压缩，其检测性能也优于当前最佳方法。更重要的是，本工作将“块效应”引入换脸检测的关键因素，为开放世界场景提供了稳健的解决方案。我们的代码可在以下链接获取。 

---
# When Life Gives You Samples: The Benefits of Scaling up Inference Compute for Multilingual LLMs 

**Title (ZH)**: Life给我们带来了样本：扩大推理计算规模对多语言LLM的益处 

**Authors**: Ammar Khairi, Daniel D'souza, Ye Shen, Julia Kreutzer, Sara Hooker  

**Link**: [PDF](https://arxiv.org/pdf/2506.20544)  

**Abstract**: Recent advancements in large language models (LLMs) have shifted focus toward scaling inference-time compute, improving performance without retraining the model. A common approach is to sample multiple outputs in parallel, and select one of these as the final output. However, work to date has focused on English and a handful of domains such as math and code. In contrast, we are most interested in techniques that generalize across open-ended tasks, formally verifiable tasks, and across languages. In this work, we study how to robustly scale inference-time compute for open-ended generative tasks in a multilingual, multi-task setting.
Our findings show that both sampling strategy based on temperature variation and selection strategy must be adapted to account for diverse domains and varied language settings. We evaluate existing selection methods, revealing that strategies effective in English often fail to generalize across languages. We propose novel sampling and selection strategies specifically adapted for multilingual and multi-task inference scenarios, and show they yield notable gains across languages and tasks. In particular, our combined sampling and selection methods lead to an average +6.8 jump in win-rates for our 8B models on m-ArenaHard-v2.0 prompts, against proprietary models such as Gemini. At larger scale, Command-A (111B model) equipped with our methods, shows +9.0 improvement in win-rates on the same benchmark with just five samples against single-sample decoding, a substantial increase at minimal cost. Our results underscore the need for language- and task-aware approaches to inference-time compute, aiming to democratize performance improvements in underrepresented languages. 

**Abstract (ZH)**: recent 进展大语言模型 (LLMs)的最新进展已将重点转向扩展推断时的计算能力，通过温度变化调整采样策略和选择策略以适应多语言和多任务场景，而不必重新训练模型。当前工作研究如何在多语言、多任务设置中稳健地扩展开放生成任务的推断计算能力。 

---
# WattsOnAI: Measuring, Analyzing, and Visualizing Energy and Carbon Footprint of AI Workloads 

**Title (ZH)**: WattsOnAI：测量、分析和可视化人工智能工作负载的能耗与碳足迹 

**Authors**: Hongzhen Huang, Kunming Zhang, Hanlong Liao, Kui Wu, Guoming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20535)  

**Abstract**: The rapid advancement of AI, particularly large language models (LLMs), has raised significant concerns about the energy use and carbon emissions associated with model training and inference. However, existing tools for measuring and reporting such impacts are often fragmented, lacking systematic metric integration and offering limited support for correlation analysis among them. This paper presents WattsOnAI, a comprehensive software toolkit for the measurement, analysis, and visualization of energy use, power draw, hardware performance, and carbon emissions across AI workloads. By seamlessly integrating with existing AI frameworks, WattsOnAI offers standardized reports and exports fine-grained time-series data to support benchmarking and reproducibility in a lightweight manner. It further enables in-depth correlation analysis between hardware metrics and model performance and thus facilitates bottleneck identification and performance enhancement. By addressing critical limitations in existing tools, WattsOnAI encourages the research community to weigh environmental impact alongside raw performance of AI workloads and advances the shift toward more sustainable "Green AI" practices. The code is available at this https URL. 

**Abstract (ZH)**: AI的迅猛发展，尤其是大型语言模型（LLMs），引起了人们对模型训练和推理过程中能源使用和碳排放的关注。然而，现有的测量和报告此类影响的工具往往支离破碎，缺乏系统性的度量整合，并且对它们之间的相关性分析支持有限。本文介绍了WattsOnAI，一个全面的软件工具包，用于测量、分析和可视化跨AI工作负载的能源使用、功耗、硬件性能和碳排放。通过与现有AI框架无缝集成，WattsOnAI提供了标准化的报告并导出细粒度的时间序列数据，以轻量级的方式支持基准测试和可重复性。该工具还能够深入分析硬件指标与模型性能之间的相关性，从而促进瓶颈识别和性能提升。通过解决现有工具的关键局限性，WattsOnAI鼓励研究社区在评估AI工作负载的原始性能的同时考虑其环境影响，并推动向更可持续的“绿色AI”实践转变。代码可在以下链接获取：this https URL。 

---
# Industrial Energy Disaggregation with Digital Twin-generated Dataset and Efficient Data Augmentation 

**Title (ZH)**: 基于数字孪生生成数据集和高效数据增强的工业能源拆分 

**Authors**: Christian Internò, Andrea Castellani, Sebastian Schmitt, Fabio Stella, Barbara Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2506.20525)  

**Abstract**: Industrial Non-Intrusive Load Monitoring (NILM) is limited by the scarcity of high-quality datasets and the complex variability of industrial energy consumption patterns. To address data scarcity and privacy issues, we introduce the Synthetic Industrial Dataset for Energy Disaggregation (SIDED), an open-source dataset generated using Digital Twin simulations. SIDED includes three types of industrial facilities across three different geographic locations, capturing diverse appliance behaviors, weather conditions, and load profiles. We also propose the Appliance-Modulated Data Augmentation (AMDA) method, a computationally efficient technique that enhances NILM model generalization by intelligently scaling appliance power contributions based on their relative impact. We show in experiments that NILM models trained with AMDA-augmented data significantly improve the disaggregation of energy consumption of complex industrial appliances like combined heat and power systems. Specifically, in our out-of-sample scenarios, models trained with AMDA achieved a Normalized Disaggregation Error of 0.093, outperforming models trained without data augmentation (0.451) and those trained with random data augmentation (0.290). Data distribution analyses confirm that AMDA effectively aligns training and test data distributions, enhancing model generalization. 

**Abstract (ZH)**: 工业非侵入式负荷监测（NILM）受限于高质量数据集的稀缺性和工业能源消耗模式的复杂多变性。为了解决数据稀缺性和隐私问题，我们引入了基于数字孪生模拟生成的开源合成工业分解数据集（SIDED）。SIDED 包含三种不同类型工业设施，覆盖三个不同的地理区域，涵盖了多样化的电器行为、天气条件和负载特性。我们还提出了电器调制数据增强（AMDA）方法，这是一种计算高效的增强技术，通过智能调整电器功率贡献的比例，提高 NILM 模型的泛化能力。实验结果表明，使用 AMDA 增强的数据训练的 NILM 模型显著提高了复杂工业电器（如热电联供系统）的分解性能。具体来说，在我们的泛化场景中，使用 AMDA 增强的数据训练的模型取得了归一化分解误差为 0.093 的成绩，优于未使用数据增强（0.451）和随机数据增强（0.290）的模型。数据分析证实，AMDA 有效地对齐了训练和测试数据分布，提升了模型的泛化能力。 

---
# OctoThinker: Mid-training Incentivizes Reinforcement Learning Scaling 

**Title (ZH)**: OctoThinker: 中断训练激励强化学习扩展 

**Authors**: Zengzhi Wang, Fan Zhou, Xuefeng Li, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20512)  

**Abstract**: Different base language model families, such as Llama and Qwen, exhibit divergent behaviors during post-training with reinforcement learning (RL), especially on reasoning-intensive tasks. What makes a base language model suitable for reinforcement learning? Gaining deeper insight into this question is essential for developing RL-scalable foundation models of the next generation. In this work, we investigate how mid-training strategies shape RL dynamics, focusing on two representative model families: Qwen and Llama. Our study reveals that (1) high-quality mathematical corpora, such as MegaMath-Web-Pro, significantly improve both base model and RL performance, while existing alternatives (e.g., FineMath-4plus) fail to do so; (2) further adding QA-style data, particularly long chain-of-thought (CoT) reasoning examples, enhances RL outcomes, and instruction data further unlocks this effect; (3) while long-CoT improves reasoning depth, it can also induce verbosity of model responses and unstability of RL training, underscoring the importance of data formatting; (4) scaling mid-training consistently leads to stronger downstream RL performance. Building on these insights, we introduce a two-stage mid-training strategy, Stable-then-Decay, in which base models are first trained on 200B tokens with a constant learning rate, followed by 20B tokens across three CoT-focused branches with learning rate decay. This yields OctoThinker, a family of models demonstrating strong RL compatibility and closing the performance gap with more RL-friendly model families, i.e., Qwen. We hope our work will help shape pre-training strategies for foundation models in the RL era. To support further research, we release our open-source models along with a curated math reasoning-intensive corpus of over 70 billion tokens (i.e., MegaMath-Web-Pro-Max). 

**Abstract (ZH)**: 不同的基础语言模型家族（如Llama和Qwen）在强化学习（RL）微调阶段表现出不同的行为，尤其是在推理密集型任务上。什么是适合强化学习的基础语言模型？深入了解这一问题是开发下一代RL可扩展基础模型的关键。在本工作中，我们探讨了中期训练策略如何影响RL动态，重点关注两种代表性的模型家族：Qwen和Llama。我们的研究揭示了以下几点：（1）高质量的数学语料库（如MegaMath-Web-Pro）能显著提升基础模型和RL性能，而现有的替代方案（如FineMath-4plus）则未能达到这一效果；（2）进一步加入问答风格的数据，尤其是长链推理例子，能增强RL效果，而指令数据进一步解锁了这一效果；（3）虽然长链推理能提升推理深度，但也可能导致模型响应的冗长和RL训练的不稳定性，强调了数据格式化的重要性；（4）中期训练的扩量一致地提升了下游RL性能。基于这些见解，我们提出了一个两阶段中期训练策略，Stable-then-Decay，首先在200B tokens上以恒定的学习率训练基础模型，然后在三个链推理（CoT）重点分支上用学习率衰减训练20B tokens。这产生了OctoThinker家族模型，该家族模型展现出强大的RL兼容性，并与更友好的RL模型家族（如Qwen）缩小了性能差距。我们希望这项工作能帮助塑造RL时代的预训练策略。为支持进一步研究，我们发布了开源模型以及一个超过700亿个令牌的精选数学推理语料库（即MegaMath-Web-Pro-Max）。 

---
# ReCode: Updating Code API Knowledge with Reinforcement Learning 

**Title (ZH)**: ReCode: 使用强化学习更新代码API知识 

**Authors**: Haoze Wu, Yunzhi Yao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20495)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成方面表现出色，但在应对外部库API频繁更新时却表现不佳。这一关键限制源自对训练数据中过时API知识的依赖，即使有当前文档可供参考，也阻碍了在动态环境中的可靠代码生成。为了解决这一问题，我们提出了ReCode（基于规则的强化学习代码更新框架），该框架模拟了人类程序员适应API变更的过程。具体来说，我们构建了一个包含约2,000条数据条目的数据集，用于训练LLMs根据更新信息执行版本迁移。然后，我们引入了一种修改后的字符串相似度度量作为强化学习的奖励标准。我们的实验表明，ReCode在动态API场景中显著提升了LLMs的代码生成性能，特别是在未见过的CodeUpdateArena任务上。关键的是，与监督微调相比，ReCode对LLMs的一般代码生成能力影响较小。我们在多种LLMs和强化学习算法（GRPO和DAPO）上应用了ReCode，所有模型都取得了一致的改进。值得注意的是，经过训练后，Qwen2.5-Coder-7B的表现超过了参数量为32B的代码指令微调模型和具有相同架构的推理模型。代码可在以下链接获取。 

---
# Counterfactual Influence as a Distributional Quantity 

**Title (ZH)**: 反事实影响作为分布量纲 

**Authors**: Matthieu Meeus, Igor Shilov, Georgios Kaissis, Yves-Alexandre de Montjoye  

**Link**: [PDF](https://arxiv.org/pdf/2506.20481)  

**Abstract**: Machine learning models are known to memorize samples from their training data, raising concerns around privacy and generalization. Counterfactual self-influence is a popular metric to study memorization, quantifying how the model's prediction for a sample changes depending on the sample's inclusion in the training dataset. However, recent work has shown memorization to be affected by factors beyond self-influence, with other training samples, in particular (near-)duplicates, having a large impact. We here study memorization treating counterfactual influence as a distributional quantity, taking into account how all training samples influence how a sample is memorized. For a small language model, we compute the full influence distribution of training samples on each other and analyze its properties. We find that solely looking at self-influence can severely underestimate tangible risks associated with memorization: the presence of (near-)duplicates seriously reduces self-influence, while we find these samples to be (near-)extractable. We observe similar patterns for image classification, where simply looking at the influence distributions reveals the presence of near-duplicates in CIFAR-10. Our findings highlight that memorization stems from complex interactions across training data and is better captured by the full influence distribution than by self-influence alone. 

**Abstract (ZH)**: 机器学习模型由于记忆训练数据中的样本而存在隐私和泛化方面的顾虑。事实上的影响反事实自影响是研究记忆的一种流行度量，量化模型对样本的预测如何依赖于该样本是否包含在训练数据集中。然而，近期研究表明，记忆不仅受自影响因素的影响，其他训练样本，尤其是（准）副本，也对其有重大影响。我们通过将反事实影响视为分布量来研究记忆，考虑到所有训练样本如何影响一个样本的记忆过程。对于一个小语言模型，我们计算了每个训练样本之间的影响分布，并分析了其性质。我们发现，仅关注自影响严重低估了记忆所带来的实际风险：存在（准）副本会显著降低自影响，而我们发现这些样本是（准）可提取的。我们在图像分类中也观察到类似模式，在 CIFAR-10 中通过观察影响分布揭示了存在近似副本的现象。我们的研究结果强调，记忆源于训练数据间的复杂交互，而完整的影響分布能更好地捕捉记忆现象，而不仅仅是依赖于自影响。 

---
# Automatic Demonstration Selection for LLM-based Tabular Data Classification 

**Title (ZH)**: 基于LLM的表格数据分类的自动演示文稿选择 

**Authors**: Shuchu Han, Wolfgang Bruckner  

**Link**: [PDF](https://arxiv.org/pdf/2506.20451)  

**Abstract**: A fundamental question in applying In-Context Learning (ICL) for tabular data classification is how to determine the ideal number of demonstrations in the prompt. This work addresses this challenge by presenting an algorithm to automatically select a reasonable number of required demonstrations. Our method distinguishes itself by integrating not only the tabular data's distribution but also the user's selected prompt template and the specific Large Language Model (LLM) into its estimation. Rooted in Spectral Graph Theory, our proposed algorithm defines a novel metric to quantify the similarities between different demonstrations. We then construct a similarity graph and analyze the eigenvalues of its Laplacian to derive the minimum number of demonstrations capable of representing the data within the LLM's intrinsic representation space. We validate the efficacy of our approach through experiments comparing its performance against conventional random selection algorithms on diverse datasets and LLMs. 

**Abstract (ZH)**: 应用In-Context Learning（ICL）进行表格数据分类时的一个基本问题是如何确定提示中所需演示的数量。本项工作通过提出一种算法来自动选择合理数量的演示来应对这一挑战。该方法不仅考虑表格数据的分布，还结合用户的选定提示模板和特定的大语言模型（LLM）来进行估算。基于谱图理论，我们提出的算法定义了一个新的度量来量化不同演示之间的相似性。然后构建相似性图，并分析其拉普拉斯矩阵的特征值，以推导出能够在LLM固有表示空间中代表数据的最小演示数量。通过在不同数据集和LLM上将我们的方法与传统随机选择算法的性能进行比较实验，验证了我们方法的有效性。 

---
# An Agentic System for Rare Disease Diagnosis with Traceable Reasoning 

**Title (ZH)**: 罕见疾病诊断的可追溯推理代理系统 

**Authors**: Weike Zhao, Chaoyi Wu, Yanjie Fan, Xiaoman Zhang, Pengcheng Qiu, Yuze Sun, Xiao Zhou, Yanfeng Wang, Ya Zhang, Yongguo Yu, Kun Sun, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20430)  

**Abstract**: Rare diseases collectively affect over 300 million individuals worldwide, yet timely and accurate diagnosis remains a pervasive challenge. This is largely due to their clinical heterogeneity, low individual prevalence, and the limited familiarity most clinicians have with rare conditions. Here, we introduce DeepRare, the first rare disease diagnosis agentic system powered by a large language model (LLM), capable of processing heterogeneous clinical inputs. The system generates ranked diagnostic hypotheses for rare diseases, each accompanied by a transparent chain of reasoning that links intermediate analytic steps to verifiable medical evidence.
DeepRare comprises three key components: a central host with a long-term memory module; specialized agent servers responsible for domain-specific analytical tasks integrating over 40 specialized tools and web-scale, up-to-date medical knowledge sources, ensuring access to the most current clinical information. This modular and scalable design enables complex diagnostic reasoning while maintaining traceability and adaptability. We evaluate DeepRare on eight datasets. The system demonstrates exceptional diagnostic performance among 2,919 diseases, achieving 100% accuracy for 1013 diseases. In HPO-based evaluations, DeepRare significantly outperforms other 15 methods, like traditional bioinformatics diagnostic tools, LLMs, and other agentic systems, achieving an average Recall@1 score of 57.18% and surpassing the second-best method (Reasoning LLM) by a substantial margin of 23.79 percentage points. For multi-modal input scenarios, DeepRare achieves 70.60% at Recall@1 compared to Exomiser's 53.20% in 109 cases. Manual verification of reasoning chains by clinical experts achieves 95.40% agreements. Furthermore, the DeepRare system has been implemented as a user-friendly web application this http URL. 

**Abstract (ZH)**: 罕见疾病集体影响 worldwide 超过 30 亿人，但及时和准确的诊断仍然是一个普遍的挑战。这主要归因于其临床异质性、低个体发病率以及大多数临床医生对罕见疾病的不熟悉。这里，我们介绍了 DeepRare，这是一种由大型语言模型（LLM）驱动的第一个罕见疾病诊断代理系统，能够处理异质性的临床输入。该系统生成针对罕见疾病的分级诊断假设，每个假设都配有透明的推理链，将中间分析步骤与可验证的医学证据联系起来。 

---
# Off-Policy Evaluation and Learning for the Future under Non-Stationarity 

**Title (ZH)**: 未来非稳态下的离策评估与学习 

**Authors**: Tatsuhiro Shimizu, Kazuki Kawamura, Takanori Muroi, Yusuke Narita, Kei Tateno, Takuma Udagawa, Yuta Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.20417)  

**Abstract**: We study the novel problem of future off-policy evaluation (F-OPE) and learning (F-OPL) for estimating and optimizing the future value of policies in non-stationary environments, where distributions vary over time. In e-commerce recommendations, for instance, our goal is often to estimate and optimize the policy value for the upcoming month using data collected by an old policy in the previous month. A critical challenge is that data related to the future environment is not observed in the historical data. Existing methods assume stationarity or depend on restrictive reward-modeling assumptions, leading to significant bias. To address these limitations, we propose a novel estimator named \textit{\textbf{O}ff-\textbf{P}olicy Estimator for the \textbf{F}uture \textbf{V}alue (\textbf{\textit{OPFV}})}, designed for accurately estimating policy values at any future time point. The key feature of OPFV is its ability to leverage the useful structure within time-series data. While future data might not be present in the historical log, we can leverage, for example, seasonal, weekly, or holiday effects that are consistent in both the historical and future data. Our estimator is the first to exploit these time-related structures via a new type of importance weighting, enabling effective F-OPE. Theoretical analysis identifies the conditions under which OPFV becomes low-bias. In addition, we extend our estimator to develop a new policy-gradient method to proactively learn a good future policy using only historical data. Empirical results show that our methods substantially outperform existing methods in estimating and optimizing the future policy value under non-stationarity for various experimental setups. 

**Abstract (ZH)**: 未来离策评估与学习（F-OPE和F-OPL）及其在非平稳环境下的未来策略价值估计与优化 

---
# SV-LLM: An Agentic Approach for SoC Security Verification using Large Language Models 

**Title (ZH)**: SV-LLM：基于大型语言模型的SoC安全性验证方法 

**Authors**: Dipayan Saha, Shams Tarek, Hasan Al Shaikh, Khan Thamid Hasan, Pavan Sai Nalluri, Md. Ajoad Hasan, Nashmin Alam, Jingbo Zhou, Sujan Kumar Saha, Mark Tehranipoor, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20415)  

**Abstract**: Ensuring the security of complex system-on-chips (SoCs) designs is a critical imperative, yet traditional verification techniques struggle to keep pace due to significant challenges in automation, scalability, comprehensiveness, and adaptability. The advent of large language models (LLMs), with their remarkable capabilities in natural language understanding, code generation, and advanced reasoning, presents a new paradigm for tackling these issues. Moving beyond monolithic models, an agentic approach allows for the creation of multi-agent systems where specialized LLMs collaborate to solve complex problems more effectively. Recognizing this opportunity, we introduce SV-LLM, a novel multi-agent assistant system designed to automate and enhance SoC security verification. By integrating specialized agents for tasks like verification question answering, security asset identification, threat modeling, test plan and property generation, vulnerability detection, and simulation-based bug validation, SV-LLM streamlines the workflow. To optimize their performance in these diverse tasks, agents leverage different learning paradigms, such as in-context learning, fine-tuning, and retrieval-augmented generation (RAG). The system aims to reduce manual intervention, improve accuracy, and accelerate security analysis, supporting proactive identification and mitigation of risks early in the design cycle. We demonstrate its potential to transform hardware security practices through illustrative case studies and experiments that showcase its applicability and efficacy. 

**Abstract (ZH)**: 确保复杂系统级芯片（SoCs）设计的安全性是一项关键需求，但由于传统验证技术在自动化、扩展性、全面性和适应性方面的重大挑战，传统的验证技术难以跟上发展的步伐。大型语言模型（LLMs）凭借其在自然语言理解、代码生成和高级推理方面的出色能力，为解决这些问题提供了新的范式。超越单一模型，采取一种代理式的方法，可以创建由专业LLM组成的多智能体系统，以更有效地解决复杂问题。认识到这一机遇，我们引入了SV-LLM，这是一种新型的多智能体助手系统，旨在自动化并增强SoC安全验证。通过集成专门负责验证问题回答、安全资产识别、威胁建模、测试计划和属性生成、漏洞检测以及基于仿真的缺陷验证等任务的智能体，SV-LLM简化了工作流程。为了在这些多样化的任务中优化智能体的性能，它们利用不同的学习范式，如上下文学习、微调和检索增强生成（RAG）。该系统旨在减少人工干预、提高准确性并加速安全分析，从而支持在设计周期早期积极识别和缓解风险。我们通过示范案例研究和展示其应用效果和有效性来证明其潜力，从而转型硬件安全性实践。 

---
# Client Clustering Meets Knowledge Sharing: Enhancing Privacy and Robustness in Personalized Peer-to-Peer Learning 

**Title (ZH)**: 客户端聚类与知识共享：增强个性化peer-to-peer学习中的隐私性和稳健性 

**Authors**: Mohammad Mahdi Maheri, Denys Herasymuk, Hamed Haddadi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20413)  

**Abstract**: The growing adoption of Artificial Intelligence (AI) in Internet of Things (IoT) ecosystems has intensified the need for personalized learning methods that can operate efficiently and privately across heterogeneous, resource-constrained devices. However, enabling effective personalized learning in decentralized settings introduces several challenges, including efficient knowledge transfer between clients, protection of data privacy, and resilience against poisoning attacks. In this paper, we address these challenges by developing P4 (Personalized, Private, Peer-to-Peer) -- a method designed to deliver personalized models for resource-constrained IoT devices while ensuring differential privacy and robustness against poisoning attacks. Our solution employs a lightweight, fully decentralized algorithm to privately detect client similarity and form collaborative groups. Within each group, clients leverage differentially private knowledge distillation to co-train their models, maintaining high accuracy while ensuring robustness to the presence of malicious clients. We evaluate P4 on popular benchmark datasets using both linear and CNN-based architectures across various heterogeneity settings and attack scenarios. Experimental results show that P4 achieves 5% to 30% higher accuracy than leading differentially private peer-to-peer approaches and maintains robustness with up to 30% malicious clients. Additionally, we demonstrate its practicality by deploying it on resource-constrained devices, where collaborative training between two clients adds only ~7 seconds of overhead. 

**Abstract (ZH)**: 人工智能在物联网生态系统中的广泛应用加剧了对高效私密个性化学习方法的需求。然而，在去中心化的环境中实现有效的个性化学习带来了若干挑战，包括在客户端之间高效的知识转移、保护数据隐私以及抵御投毒攻击的鲁棒性。本文通过开发P4（个性化、私密、点对点）方法来应对这些挑战，该方法旨在为资源受限的物联网设备提供个性化模型，同时确保差分隐私并抵御投毒攻击。我们的解决方案采用一种轻量级的完全去中心化算法，以私密方式检测客户端相似性并形成协作组。在每个组内，客户端利用差分隐私的知识蒸馏共同训练模型，在保证高准确率的同时确保在恶意客户端存在时具备鲁棒性。我们在多种异构设置和攻击场景下使用线性架构和CNN架构的流行基准数据集评估了P4。实验结果表明，P4的准确率比领先的差分隐私点对点方法高出5%至30%，并能够容忍高达30%的恶意客户端。此外，我们通过部署在资源受限的设备上展示了其实用性，其中两个客户端之间的协作训练仅增加了约7秒的额外开销。 

---
# CARMA: Context-Aware Situational Grounding of Human-Robot Group Interactions by Combining Vision-Language Models with Object and Action Recognition 

**Title (ZH)**: CARMA：结合视觉语言模型、物体识别与动作识别的基于上下文的情境化人类-机器人组交互接地技术 

**Authors**: Joerg Deigmoeller, Stephan Hasler, Nakul Agarwal, Daniel Tanneberg, Anna Belardinelli, Reza Ghoddoosian, Chao Wang, Felix Ocker, Fan Zhang, Behzad Dariush, Michael Gienger  

**Link**: [PDF](https://arxiv.org/pdf/2506.20373)  

**Abstract**: We introduce CARMA, a system for situational grounding in human-robot group interactions. Effective collaboration in such group settings requires situational awareness based on a consistent representation of present persons and objects coupled with an episodic abstraction of events regarding actors and manipulated objects. This calls for a clear and consistent assignment of instances, ensuring that robots correctly recognize and track actors, objects, and their interactions over time. To achieve this, CARMA uniquely identifies physical instances of such entities in the real world and organizes them into grounded triplets of actors, objects, and actions.
To validate our approach, we conducted three experiments, where multiple humans and a robot interact: collaborative pouring, handovers, and sorting. These scenarios allow the assessment of the system's capabilities as to role distinction, multi-actor awareness, and consistent instance identification. Our experiments demonstrate that the system can reliably generate accurate actor-action-object triplets, providing a structured and robust foundation for applications requiring spatiotemporal reasoning and situated decision-making in collaborative settings. 

**Abstract (ZH)**: CARMA：人类-机器人团队互动中的情境接地系统 

---
# Self-Supervised Graph Learning via Spectral Bootstrapping and Laplacian-Based Augmentations 

**Title (ZH)**: 基于谱自助强化和拉普拉斯基扩增的自监督图学习 

**Authors**: Lorenzo Bini, Stephane Marchand-Maillet  

**Link**: [PDF](https://arxiv.org/pdf/2506.20362)  

**Abstract**: We present LaplaceGNN, a novel self-supervised graph learning framework that bypasses the need for negative sampling by leveraging spectral bootstrapping techniques. Our method integrates Laplacian-based signals into the learning process, allowing the model to effectively capture rich structural representations without relying on contrastive objectives or handcrafted augmentations. By focusing on positive alignment, LaplaceGNN achieves linear scaling while offering a simpler, more efficient, self-supervised alternative for graph neural networks, applicable across diverse domains. Our contributions are twofold: we precompute spectral augmentations through max-min centrality-guided optimization, enabling rich structural supervision without relying on handcrafted augmentations, then we integrate an adversarial bootstrapped training scheme that further strengthens feature learning and robustness. Our extensive experiments on different benchmark datasets show that LaplaceGNN achieves superior performance compared to state-of-the-art self-supervised graph methods, offering a promising direction for efficiently learning expressive graph representations. 

**Abstract (ZH)**: LaplaceGNN：一种基于谱增强的自监督图学习框架 

---
# A foundation model with multi-variate parallel attention to generate neuronal activity 

**Title (ZH)**: 具有多变量并行注意力的基础模型生成神经活动 

**Authors**: Francesco Carzaniga, Michael Hersche, Abu Sebastian, Kaspar Schindler, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20354)  

**Abstract**: Learning from multi-variate time-series with heterogeneous channel configurations remains a fundamental challenge for deep neural networks (DNNs), particularly in clinical domains such as intracranial electroencephalography (iEEG), where channel setups vary widely across subjects. In this work, we introduce multi-variate parallel attention (MVPA), a novel self-attention mechanism that disentangles content, temporal, and spatial attention, enabling flexible, generalizable, and efficient modeling of time-series data with varying channel counts and configurations. We use MVPA to build MVPFormer, a generative foundation model for human electrophysiology, trained to predict the evolution of iEEG signals across diverse subjects. To support this and future effort by the community, we release the SWEC iEEG dataset, the largest publicly available iEEG dataset to date, comprising nearly 10,000 hours of recordings from heterogeneous clinical sources. MVPFormer leverages MVPA to achieve strong generalization across subjects, demonstrating expert-level performance in seizure detection and outperforming state-of-the-art Transformer baselines on our SWEC, the MAYO, and the FNUSA dataset. We further validate MVPA on standard time-series forecasting and classification tasks, where it matches or exceeds existing attention-based models. Together, our contributions establish MVPA as a general-purpose attention mechanism for heterogeneous time-series and MVPFormer as the first open-source, open-weights, and open-data iEEG foundation model with state-of-the-art clinical performance. The code is available at this https URL. The SWEC iEEG dataset is available at this https URL. 

**Abstract (ZH)**: 多变量时间序列在异构通道配置下的学习仍然是深度神经网络（DNNs）的基本挑战，尤其是在如颅内电encephalography (iEEG)等临床领域，其中各个被试的通道设置差异很大。在这项工作中，我们介绍了多变量并行注意（MVPA）机制，这是一种新型的自注意力机制，能够解耦内容、时间和空间注意力，从而实现对具有不同通道数量和配置的时间序列数据的灵活、泛化和高效建模。我们使用MVPA构建了MVPFormer，一个用于人类电生理学的生成基础模型，该模型经过训练，可以预测跨不同类型被试的iEEG信号演变。为了支持这一工作中以及未来的社区工作，我们发布了SWEC iEEG数据集，这是迄今为止最大的公开可用的iEEG数据集，包含来自异构临床来源的近10000小时的记录。MVPFormer利用MVPA实现了跨被试的强泛化能力，在癫痫检测方面达到专家级性能，并在我们的SWEC、MAYO和FNUSA数据集上优于最先进的Transformer基线。我们进一步在标准的时间序列预测和分类任务上验证了MVPA，结果显示它与现有的注意机制模型相当或更优。我们的贡献确立了MVPA作为异构时间序列的通用注意机制的地位，同时将MVPFormer确立为首个具有先进临床性能的开源、开放权重、开放数据的iEEG基础模型。代码可在以下链接获取：这个 https URL 数据集可在以下链接获取：这个 https URL 

---
# DipSVD: Dual-importance Protected SVD for Efficient LLM Compression 

**Title (ZH)**: DipSVD: 双重要性保护的SVD方法及其在高效大型语言模型压缩中的应用 

**Authors**: Xuan Ding, Rui Sun, Yunjian Zhang, Xiu Yan, Yueqi Zhou, Kaihao Huang, Suzhong Fu, Chuanlong Xie, Yao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20353)  

**Abstract**: The ever-increasing computational demands and deployment costs of large language models (LLMs) have spurred numerous compressing methods. Compared to quantization and unstructured pruning, SVD compression offers superior hardware compatibility and theoretical guarantees. However, existing SVD-based methods focus on the overall discrepancy between the original and compressed matrices while overlooking the protection of critical components within the matrix, which leads to inferior performance in the compressed models. This paper proposes a dual-level importance protection mechanism to enhance SVD-based compression methods: (1) local importance protection: preserving the most critical singular vectors within each weight matrix through channel-weighted data whitening; and (2) global importance protection: enabling less important layers to bear a greater portion of the compression burden through either a heuristic or optimization-based approach, thereby minimizing the impact of compression on critical layers. Extensive experiments demonstrate that DipSVD outperforms existing SVD-based compression approaches across multiple benchmarks, achieving superior model performance especially at high model compression ratios. 

**Abstract (ZH)**: 大型语言模型（LLMs）不断增加的计算需求和部署成本促使了众多压缩方法的发展。与量化的和未结构化剪枝方法相比，SVD压缩在硬件兼容性和理论保证方面具有优势。然而，现有的基于SVD的方法主要关注原始矩阵和压缩矩阵的整体差异，而忽视了保护矩阵中的关键组件，导致压缩模型的性能较差。本文提出了一种双层重要性保护机制以增强基于SVD的压缩方法：（1）局部重要性保护：通过通道加权数据去相关保留每个权重矩阵中最关键的奇异向量；（2）全局重要性保护：通过启发式或优化方法使不太重要的层承担更多的压缩负担，从而最小化压缩对关键层的影响。广泛实验证明，DipSVD在多个基准上优于现有基于SVD的压缩方法，特别是在高模型压缩比下实现了更优异的模型性能。 

---
# Feature Hallucination for Self-supervised Action Recognition 

**Title (ZH)**: 自监督动作识别的特征 hallucination 

**Authors**: Lei Wang, Piotr Koniusz  

**Link**: [PDF](https://arxiv.org/pdf/2506.20342)  

**Abstract**: Understanding human actions in videos requires more than raw pixel analysis; it relies on high-level semantic reasoning and effective integration of multimodal features. We propose a deep translational action recognition framework that enhances recognition accuracy by jointly predicting action concepts and auxiliary features from RGB video frames. At test time, hallucination streams infer missing cues, enriching feature representations without increasing computational overhead. To focus on action-relevant regions beyond raw pixels, we introduce two novel domain-specific descriptors. Object Detection Features (ODF) aggregate outputs from multiple object detectors to capture contextual cues, while Saliency Detection Features (SDF) highlight spatial and intensity patterns crucial for action recognition. Our framework seamlessly integrates these descriptors with auxiliary modalities such as optical flow, Improved Dense Trajectories, skeleton data, and audio cues. It remains compatible with state-of-the-art architectures, including I3D, AssembleNet, Video Transformer Network, FASTER, and recent models like VideoMAE V2 and InternVideo2. To handle uncertainty in auxiliary features, we incorporate aleatoric uncertainty modeling in the hallucination step and introduce a robust loss function to mitigate feature noise. Our multimodal self-supervised action recognition framework achieves state-of-the-art performance on multiple benchmarks, including Kinetics-400, Kinetics-600, and Something-Something V2, demonstrating its effectiveness in capturing fine-grained action dynamics. 

**Abstract (ZH)**: 基于多模态自监督的方法在视频中的精细粒度动作动态理解中取得最佳性能 

---
# Comparative Analysis of Deep Learning Models for Crop Disease Detection: A Transfer Learning Approach 

**Title (ZH)**: 基于迁移学习的作物病害检测深度学习模型比较分析 

**Authors**: Saundarya Subramaniam, Shalini Majumdar, Shantanu Nadar, Kaustubh Kulkarni  

**Link**: [PDF](https://arxiv.org/pdf/2506.20323)  

**Abstract**: This research presents the development of an Artificial Intelligence (AI) - driven crop disease detection system designed to assist farmers in rural areas with limited resources. We aim to compare different deep learning models for a comparative analysis, focusing on their efficacy in transfer learning. By leveraging deep learning models, including EfficientNet, ResNet101, MobileNetV2, and our custom CNN, which achieved a validation accuracy of 95.76%, the system effectively classifies plant diseases. This research demonstrates the potential of transfer learning in reshaping agricultural practices, improving crop health management, and supporting sustainable farming in rural environments. 

**Abstract (ZH)**: 本研究提出了一种基于人工智能的作物疾病检测系统，旨在帮助资源有限的农村地区农民。我们旨在比较不同深度学习模型，侧重于它们在迁移学习中的有效性。通过利用包括EfficientNet、ResNet101、MobileNetV2以及我们自定义的CNN在内的深度学习模型，系统实现了95.76%的验证准确率，有效地分类植物疾病。本研究展示了迁移学习在重塑农业实践、改善作物健康管理以及支持农村可持续农业方面的潜力。 

---
# Beyond-Expert Performance with Limited Demonstrations: Efficient Imitation Learning with Double Exploration 

**Title (ZH)**: 在有限示范下的超越专家性能：双探索增强的高效 imitation 学习 

**Authors**: Heyang Zhao, Xingrui Yu, David M. Bossens, Ivor W. Tsang, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20307)  

**Abstract**: Imitation learning is a central problem in reinforcement learning where the goal is to learn a policy that mimics the expert's behavior. In practice, it is often challenging to learn the expert policy from a limited number of demonstrations accurately due to the complexity of the state space. Moreover, it is essential to explore the environment and collect data to achieve beyond-expert performance. To overcome these challenges, we propose a novel imitation learning algorithm called Imitation Learning with Double Exploration (ILDE), which implements exploration in two aspects: (1) optimistic policy optimization via an exploration bonus that rewards state-action pairs with high uncertainty to potentially improve the convergence to the expert policy, and (2) curiosity-driven exploration of the states that deviate from the demonstration trajectories to potentially yield beyond-expert performance. Empirically, we demonstrate that ILDE outperforms the state-of-the-art imitation learning algorithms in terms of sample efficiency and achieves beyond-expert performance on Atari and MuJoCo tasks with fewer demonstrations than in previous work. We also provide a theoretical justification of ILDE as an uncertainty-regularized policy optimization method with optimistic exploration, leading to a regret growing sublinearly in the number of episodes. 

**Abstract (ZH)**: 模仿学习是强化学习中的一个核心问题，目标是学习一个模仿专家行为的策略。在实践中，由于状态空间的复杂性，从有限的演示中准确学习专家策略往往颇具挑战。此外，探索环境和收集数据以实现超越专家的表现至关重要。为克服这些挑战，我们提出了一种新颖的模仿学习算法——双重探索模仿学习（ILDE），该算法从两个方面实现探索：（1）通过探索奖励来乐观的政策优化，奖励具有高不确定性的状态-动作对，以潜在地提高向专家策略收敛的速度；（2）针对与演示轨迹偏差的状态进行好奇心驱动的探索，以潜在地实现超越专家的表现。实验结果表明，ILDE 在样本效率方面优于现有的模仿学习算法，并在使用较少演示的情况下实现了超越专家的表现，特别是在Atari和MuJoCo任务上。我们还从理论上证明了ILDE 是一种正则化不确定性优化方法，具有乐观探索性，其后悔的增长率在episode数量增加时呈次线性增长。 

---
# Argumentative Ensembling for Robust Recourse under Model Multiplicity 

**Title (ZH)**: 基于模型多样性的情况下稳健反事实生成的论证集成方法 

**Authors**: Junqi Jiang, Antonio Rago, Francesco Leofante, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2506.20260)  

**Abstract**: In machine learning, it is common to obtain multiple equally performing models for the same prediction task, e.g., when training neural networks with different random seeds. Model multiplicity (MM) is the situation which arises when these competing models differ in their predictions for the same input, for which ensembling is often employed to determine an aggregation of the outputs. Providing recourse recommendations via counterfactual explanations (CEs) under MM thus becomes complex, since the CE may not be valid across all models, i.e., the CEs are not robust under MM. In this work, we formalise the problem of providing recourse under MM, which we name recourse-aware ensembling (RAE). We propose the idea that under MM, CEs for each individual model should be considered alongside their predictions so that the aggregated prediction and recourse are decided in tandem. Centred around this intuition, we introduce six desirable properties for solutions to this problem. For solving RAE, we propose a novel argumentative ensembling method which guarantees the robustness of CEs under MM. Specifically, our method leverages computational argumentation to explicitly represent the conflicts between models and counterfactuals regarding prediction results and CE validity. It then uses argumentation semantics to resolve the conflicts and obtain the final solution, in a manner which is parametric to the chosen semantics. Our method also allows for the specification of preferences over the models under MM, allowing further customisation of the ensemble. In a comprehensive theoretical analysis, we characterise the behaviour of argumentative ensembling with four different argumentation semantics. We then empirically demonstrate the effectiveness of our approach in satisfying desirable properties with eight instantiations of our method. (Abstract is shortened for arXiv.) 

**Abstract (ZH)**: 在机器学习中，当使用不同的随机种子训练神经网络时，常常会得到多个等效模型。模型多样性（MD）是指这些竞争模型对同一输入的预测结果不一致的情况，常常通过集成方法来确定预测结果的聚合。在模型多样性的情况下，通过反事实解释（CEs）提供纠正建议变得复杂，因为CE可能不适用于所有模型，即CE在MD下缺乏鲁棒性。在本文中，我们形式化了在MD情况下提供纠正建议的问题，并将其命名为意识多样性的集成（RAE）。我们提出，在模型多样性的情况下，每个模型的CE应与其预测一起考虑，以便同时决定聚合预测和纠正建议。围绕这一直觉，我们提出了六种解决方案应具备的 desirable 属性。为了实现RAE，我们提出了一种新的论证型集成方法，确保在模型多样性的情况下CE的鲁棒性。具体而言，我们的方法利用计算论证来明确表示模型和反事实之间的冲突，包括预测结果和CE有效性。然后通过使用论证语义来解决冲突并获得最终解决方案，该过程对所选语义是参数化的。我们的方法还允许指定模型多样性下的偏好，从而进一步定制集成。在全面的理论分析中，我们使用四种不同的论证语义来刻画论证型集成的行为。然后通过八个实例的实验证明，我们的方法能够满足 desirable 属性。 

---
# Generating and Customizing Robotic Arm Trajectories using Neural Networks 

**Title (ZH)**: 使用神经网络生成和定制机器人臂轨迹 

**Authors**: Andrej Lúčny, Matilde Antonj, Carlo Mazzola, Hana Hornáčková, Igor Farkaš  

**Link**: [PDF](https://arxiv.org/pdf/2506.20259)  

**Abstract**: We introduce a neural network approach for generating and customizing the trajectory of a robotic arm, that guarantees precision and repeatability. To highlight the potential of this novel method, we describe the design and implementation of the technique and show its application in an experimental setting of cognitive robotics. In this scenario, the NICO robot was characterized by the ability to point to specific points in space with precise linear movements, increasing the predictability of the robotic action during its interaction with humans. To achieve this goal, the neural network computes the forward kinematics of the robot arm. By integrating it with a generator of joint angles, another neural network was developed and trained on an artificial dataset created from suitable start and end poses of the robotic arm. Through the computation of angular velocities, the robot was characterized by its ability to perform the movement, and the quality of its action was evaluated in terms of shape and accuracy. Thanks to its broad applicability, our approach successfully generates precise trajectories that could be customized in their shape and adapted to different settings. 

**Abstract (ZH)**: 一种确保精确性和可重复性的机器人臂轨迹生成与定制的神经网络方法：应用与评估 

---
# Time-series surrogates from energy consumers generated by machine learning approaches for long-term forecasting scenarios 

**Title (ZH)**: 由机器学习方法生成的能源消费者时序替代数据用于长期预测场景 

**Authors**: Ben Gerhards, Nikita Popkov, Annekatrin König, Marcel Arpogaus, Bastian Schäfermeier, Leonie Riedl, Stephan Vogt, Philip Hehlert  

**Link**: [PDF](https://arxiv.org/pdf/2506.20253)  

**Abstract**: Forecasting attracts a lot of research attention in the electricity value chain. However, most studies concentrate on short-term forecasting of generation or consumption with a focus on systems and less on individual consumers. Even more neglected is the topic of long-term forecasting of individual power consumption.
Here, we provide an in-depth comparative evaluation of data-driven methods for generating synthetic time series data tailored to energy consumption long-term forecasting. High-fidelity synthetic data is crucial for a wide range of applications, including state estimations in energy systems or power grid planning. In this study, we assess and compare the performance of multiple state-of-the-art but less common techniques: a hybrid Wasserstein Generative Adversarial Network (WGAN), Denoising Diffusion Probabilistic Model (DDPM), Hidden Markov Model (HMM), and Masked Autoregressive Bernstein polynomial normalizing Flows (MABF). We analyze the ability of each method to replicate the temporal dynamics, long-range dependencies, and probabilistic transitions characteristic of individual energy consumption profiles. Our comparative evaluation highlights the strengths and limitations of: WGAN, DDPM, HMM and MABF aiding in selecting the most suitable approach for state estimations and other energy-related tasks. Our generation and analysis framework aims to enhance the accuracy and reliability of synthetic power consumption data while generating data that fulfills criteria like anonymisation - preserving privacy concerns mitigating risks of specific profiling of single customers. This study utilizes an open-source dataset from households in Germany with 15min time resolution. The generated synthetic power profiles can readily be used in applications like state estimations or consumption forecasting. 

**Abstract (ZH)**: 基于数据驱动方法的电力消费长周期预测合成时间序列数据生成对比研究 

---
# Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models 

**Title (ZH)**: Q-resafe: 评估安全风险和量化感知的安全修补对于量化的大语言模型 

**Authors**: Kejia Chen, Jiawen Zhang, Jiacong Hu, Yu Wang, Jian Lou, Zunlei Feng, Mingli Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.20251)  

**Abstract**: Quantized large language models (LLMs) have gained increasing attention and significance for enabling deployment in resource-constrained environments. However, emerging studies on a few calibration dataset-free quantization methods suggest that quantization may compromise the safety capabilities of LLMs, underscoring the urgent need for systematic safety evaluations and effective mitigation strategies. In this paper, we present comprehensive safety evaluations across various mainstream quantization techniques and diverse calibration datasets, utilizing widely accepted safety benchmarks. To address the identified safety vulnerabilities, we propose a quantization-aware safety patching framework, Q-resafe, to efficiently restore the safety capabilities of quantized LLMs while minimizing any adverse impact on utility. Extensive experimental results demonstrate that Q-resafe successfully re-aligns the safety of quantized LLMs with their pre-quantization counterparts, even under challenging evaluation scenarios. Project page is available at: this https URL. 

**Abstract (ZH)**: 量化大型语言模型（LLMs）在资源受限环境中部署愈发受到关注，但新兴的研究表明，量化可能会牺牲LLMs的安全能力，突显了系统安全评估和有效缓解策略的迫切需求。在本文中，我们全面评估了各种主流量化技术与多样化的校准数据集的安全性，采用广泛接受的安全基准。为应对识别出的安全漏洞，我们提出了一种量化感知安全性修补框架Q-resafe，以高效恢复量化LLMs的安全能力并尽量减小对实用性的负面影响。广泛实验结果显示，Q-resafe能够即使在严峻的评估场景下，成功使量化LLMs的安全性重新与量化前的版本保持一致。项目页面详见：this https URL。 

---
# FedBKD: Distilled Federated Learning to Embrace Gerneralization and Personalization on Non-IID Data 

**Title (ZH)**: FedBKD：提炼联邦学习以拥抱非IID数据上的泛化能力和个性化能力 

**Authors**: Yushan Zhao, Jinyuan He, Donglai Chen, Weijie Luo, Chong Xie, Ri Zhang, Yonghong Chen, Yan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20245)  

**Abstract**: Federated learning (FL) is a decentralized collaborative machine learning (ML) technique. It provides a solution to the issues of isolated data islands and data privacy leakage in industrial ML practices. One major challenge in FL is handling the non-identical and independent distributed (non-IID) data. Current solutions either focus on constructing an all-powerful global model, or customizing personalized local models. Few of them can provide both a well-generalized global model and well-performed local models at the same time. Additionally, many FL solutions to the non-IID problem are benefited from introducing public datasets. However, this will also increase the risk of data leakage. To tackle the problems, we propose a novel data-free distillation framework, Federated Bidirectional Knowledge Distillation (FedBKD). Specifically, we train Generative Adversarial Networks (GAN) for synthetic data. During the GAN training, local models serve as discriminators and their parameters are frozen. The synthetic data is then used for bidirectional distillation between global and local models to achieve knowledge interactions so that performances for both sides are improved. We conduct extensive experiments on 4 benchmarks under different non-IID settings. The results show that FedBKD achieves SOTA performances in every case. 

**Abstract (ZH)**: 联邦学习（FL）是一种去中心化的协作机器学习（ML）技术。它提供了一种解决工业ML实践中孤立的数据孤岛和数据隐私泄露问题的方案。FL面临的一个主要挑战是如何处理非同质独立分布（non-IID）数据。目前的解决方案要么侧重于构建全能的全局模型，要么定制个性化的局部模型。很少有方法能在同时提供泛化良好的全局模型和性能良好的局部模型方面同时取得成功。此外，许多解决非-IID问题的FL方案得益于引入公共数据集，但这也增加了数据泄露的风险。为了解决这些问题，我们提出了一种新颖的数据免费蒸馏框架，联邦双向知识蒸馏（FedBKD）。具体而言，我们训练生成对抗网络（GAN）生成合成数据。在GAN训练过程中，局部模型充当鉴别器且参数被冻结。合成数据随后用于全局模型和局部模型之间的双向蒸馏，以实现知识交互，从而提高双方的性能。我们在不同非-IID设置下的4个基准上进行了广泛的实验。结果表明，FedBKD在所有情况下都取得了最佳性能。 

---
# Enhancing Large Language Models through Structured Reasoning 

**Title (ZH)**: 通过结构化推理增强大型语言模型 

**Authors**: Yubo Dong, Hehe Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20241)  

**Abstract**: Recent Large Language Models (LLMs) have significantly advanced natural language processing and automated decision-making. However, these models still encounter difficulties when performing complex reasoning tasks involving logical deduction and systematic planning, primarily due to their reliance on implicit statistical relationships without structured knowledge this http URL by cognitive science and neurosymbolic AI, we introduce a novel approach to enhance LLMs through explicit structured reasoning. First, we convert unstructured data into structured formats by explicitly annotating reasoning steps. We then employ this structured dataset to train LLMs through Supervised Fine-Tuning (SFT). Additionally, we enhance the structured reasoning capabilities of LLMs using Group Relative Policy Optimization (GRPO), incorporating two innovative algorithms--MAX-Flow and Longest Common Subsequence (LCS)--which notably improve reasoning effectiveness and reduce computational complexity. Experimental results from fine-tuning a DeepSeek-R1-Distill-Qwen-1.5B model demonstrate concise reasoning, robust performance across various scenarios, and improved compatibility with optimization techniques, validating the efficacy of structured reasoning integration in LLMs. 

**Abstract (ZH)**: Recent Large Language Models (LLMs)在自然语言处理和自动化决策方面取得了显著进展，但仍然难以执行涉及逻辑推理和系统规划的复杂任务，主要原因是它们依赖于无结构的知识和隐含的统计关系。通过认知科学和神经符号AI，我们提出了一种新的方法来增强LLMs的结构化推理能力。首先，通过明确标注推理步骤，我们将非结构化数据转换为结构化格式。然后，我们使用监督微调(SFT)来训练LLMs。此外，我们通过Group Relative Policy Optimization (GRPO)增强LLMs的结构化推理能力，并引入了两种创新算法——MAX-Flow和Longest Common Subsequence (LCS)——显著提高了推理效果并降低了计算复杂度。对DeepSeek-R1-Distill-Qwen-1.5B模型进行微调的实验结果表明，这种结构化推理方法具有简洁的推理能力、在各种场景中表现稳健，并且能更好地与优化技术兼容，验证了结构化推理在LLMs中的有效性。 

---
# Directed Link Prediction using GNN with Local and Global Feature Fusion 

**Title (ZH)**: 基于局部和全局特征融合的图神经网络定向链接预测 

**Authors**: Yuyang Zhang, Xu Shen, Yu Xie, Ka-Chun Wong, Weidun Xie, Chengbin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.20235)  

**Abstract**: Link prediction is a classical problem in graph analysis with many practical applications. For directed graphs, recently developed deep learning approaches typically analyze node similarities through contrastive learning and aggregate neighborhood information through graph convolutions. In this work, we propose a novel graph neural network (GNN) framework to fuse feature embedding with community information. We theoretically demonstrate that such hybrid features can improve the performance of directed link prediction. To utilize such features efficiently, we also propose an approach to transform input graphs into directed line graphs so that nodes in the transformed graph can aggregate more information during graph convolutions. Experiments on benchmark datasets show that our approach outperforms the state-of-the-art in most cases when 30%, 40%, 50%, and 60% of the connected links are used as training data, respectively. 

**Abstract (ZH)**: 基于图神经网络的特征嵌入与社区信息融合的有向链接预测方法 

---
# Perspectives in Play: A Multi-Perspective Approach for More Inclusive NLP Systems 

**Title (ZH)**: 玩的视角：一种更加包容的NLP系统多视角方法 

**Authors**: Benedetta Muscato, Lucia Passaro, Gizem Gezici, Fosca Giannotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.20209)  

**Abstract**: In the realm of Natural Language Processing (NLP), common approaches for handling human disagreement consist of aggregating annotators' viewpoints to establish a single ground truth. However, prior studies show that disregarding individual opinions can lead can lead to the side effect of underrepresenting minority perspectives, especially in subjective tasks, where annotators may systematically disagree because of their preferences. Recognizing that labels reflect the diverse backgrounds, life experiences, and values of individuals, this study proposes a new multi-perspective approach using soft labels to encourage the development of the next generation of perspective aware models, more inclusive and pluralistic. We conduct an extensive analysis across diverse subjective text classification tasks, including hate speech, irony, abusive language, and stance detection, to highlight the importance of capturing human disagreements, often overlooked by traditional aggregation methods. Results show that the multi-perspective approach not only better approximates human label distributions, as measured by Jensen-Shannon Divergence (JSD), but also achieves superior classification performance (higher F1 scores), outperforming traditional approaches. However, our approach exhibits lower confidence in tasks like irony and stance detection, likely due to the inherent subjectivity present in the texts. Lastly, leveraging Explainable AI (XAI), we explore model uncertainty and uncover meaningful insights into model predictions. 

**Abstract (ZH)**: 在自然语言处理（NLP）领域，处理人类分歧的常见方法是汇总标注者的观点以建立单一的Ground Truth。然而，先前的研究表明，忽略个人意见可能会导致少数派视角的代表性不足，尤其是在主观任务中，标注者可能会由于其偏好系统性地产生分歧。鉴于标签反映了个体的多样背景、生活经历和价值观，本研究提出了一种新的多视角方法，利用软标签来促进下一代具有视角意识模型的发展，更具包容性和多元性。我们在包括仇恨言论、讽刺、攻击性语言和观点检测在内的多种主观文本分类任务中进行了广泛分析，以强调捕捉人类分歧的重要性，这往往是传统汇总方法忽略的。结果表明，多视角方法不仅在 Jensen-Shannon 散度（JSD）衡量的人类标签分布近似方面更优，同时在分类性能（更高的 F1 分数）方面也优于传统方法。然而，在讽刺和观点检测等任务中，我们的方法表现出较低的置信度，这可能归因于文本中存在的固有主观性。最后，利用可解释人工智能（XAI），我们探索了模型的不确定性并揭示了有关模型预测的有意义见解。 

---
# Affective Priming Score: A Data-Driven Method to Detect Priming in Sequential Datasets 

**Title (ZH)**: 情感启动分数：一种基于数据的方法用于检测序列数据中的启动效应 

**Authors**: Eduardo Gutierrez Maestro, Hadi Banaee, Amy Loutfi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20204)  

**Abstract**: Affective priming exemplifies the challenge of ambiguity in affective computing. While the community has largely addressed this issue from a label-based perspective, identifying data points in the sequence affected by the priming effect, the impact of priming on data itself, particularly in physiological signals, remains underexplored. Data affected by priming can lead to misclassifications when used in learning models. This study proposes the Affective Priming Score (APS), a data-driven method to detect data points influenced by the priming effect. The APS assigns a score to each data point, quantifying the extent to which it is affected by priming. To validate this method, we apply it to the SEED and SEED-VII datasets, which contain sufficient transitions between emotional events to exhibit priming effects. We train models with the same configuration using both the original data and priming-free sequences. The misclassification rate is significantly reduced when using priming-free sequences compared to the original data. This work contributes to the broader challenge of ambiguity by identifying and mitigating priming effects at the data level, enhancing model robustness, and offering valuable insights for the design and collection of affective computing datasets. 

**Abstract (ZH)**: 情感启动体现了情感计算中不确定性的挑战。尽管社区主要从标签的角度来解决这一问题，标识序列中受启动效应影响的数据点，启动效应对数据本身的影响，尤其是在生理信号方面，仍需进一步探索。受启动效应影响的数据可能导致在学习模型中出现误分类。本研究提出情感启动评分（APS），这是一种数据驱动的方法，用于检测受启动效应影响的数据点。APS为每个数据点分配一个得分，量化其受启动效应影响的程度。为验证该方法，我们在SEED和SEED-VII数据集上应用此方法，这两个数据集包含足够的情绪事件转换，以展示启动效应。使用去启动序列与原始数据训练相同配置的模型。使用去启动序列的错误分类率显著低于使用原始数据的模型。本研究通过在数据层面识别和缓解启动效应，为更广泛的不确定性挑战做出了贡献，增强了模型的稳健性，并提供了有关情感计算数据集设计和收集的宝贵见解。 

---
# How to Retrieve Examples in In-context Learning to Improve Conversational Emotion Recognition using Large Language Models? 

**Title (ZH)**: 在上下文学习中利用大型语言模型提高 conversational emotion recognition 的例子检索方法探究 

**Authors**: Mengqi Wang, Tiantian Feng, Shrikanth Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20199)  

**Abstract**: Large language models (LLMs) have enabled a wide variety of real-world applications in various domains. However, creating a high-performing application with high accuracy remains challenging, particularly for subjective tasks like emotion recognition. Inspired by the SLT 2024 GenSER Challenge, this study investigates approaches to improving conversational emotion recognition (CER) by LLMs. Specifically, we explore how to retrieve high-quality examples in in-context learning (ICL) to enhance CER. We propose various strategies based on random and augmented example retrieval and also analyze the impact of conversational context on CER accuracy. Experiments were conducted on the three datasets including IEMOCAP, MELD and EmoryNLP. The results show that augmented example retrieval consistently outperforms other techniques under investigation across all datasets, highlighting the importance of retrieving coherent targeted examples and enhancing them through paraphrasing. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域-enable了广泛的实际应用。然而，创建具有高准确性的高performance应用仍然颇具挑战，特别是在情感识别等主观任务上。受SLT 2024 GenSER挑战的启发，本研究探讨了通过在上下文学习（ICL）中检索高质量示例来提高对话情感识别（CER）的方法。具体而言，我们研究了如何在ICL中检索高质量示例以增强CER。我们提出了基于随机和增强示例检索的各种策略，并分析了对话上下文对CER准确性的影响。实验在IEMOCAP、MELD和EmoryNLP三个数据集上进行。结果表明，增强示例检索在所有数据集上均一致优于其他调查的技术，突显了检索一致的相关示例并通过改述增强它们的重要性。 

---
# Zero-Shot Attribution for Large Language Models: A Distribution Testing Approach 

**Title (ZH)**: 大型语言模型的零样本归属分析：一种分布测试方法 

**Authors**: Clément L. Canonne, Yash Pote, Uddalok Sarkar  

**Link**: [PDF](https://arxiv.org/pdf/2506.20197)  

**Abstract**: A growing fraction of all code is sampled from Large Language Models (LLMs). We investigate the problem of attributing code generated by language models using hypothesis testing to leverage established techniques and guarantees. Given a set of samples $S$ and a suspect model $\mathcal{L}^*$, our goal is to assess the likelihood of $S$ originating from $\mathcal{L}^*$. Due to the curse of dimensionality, this is intractable when only samples from the LLM are given: to circumvent this, we use both samples and density estimates from the LLM, a form of access commonly available.
We introduce $\mathsf{Anubis}$, a zero-shot attribution tool that frames attribution as a distribution testing problem. Our experiments on a benchmark of code samples show that $\mathsf{Anubis}$ achieves high AUROC scores ( $\ge0.9$) when distinguishing between LLMs like DeepSeek-Coder, CodeGemma, and Stable-Code using only $\approx 2000$ samples. 

**Abstract (ZH)**: 越来越多的代码片段来自于大型语言模型（LLMs）。我们研究使用假设检验对语言模型生成的代码进行归属的问题，以利用已有的技术和保证。给定一个样本集$S$和一个嫌疑模型$\mathcal{L}^*$，我们的目标是评估$S$来源于$\mathcal{L}^*$的概率。由于维数灾难，仅给定LLM的样本时，这一任务无法解决：为了克服这一问题，我们使用了来自LLM的样本和密度估计，这是通常可用的一种访问形式。

我们引入了$\mathsf{Anubis}$，一种零样本归属工具，将其归属问题框架化为分布检验问题。我们在一个代码样本基准测试中进行的实验表明，$\mathsf{Anubis}$仅使用$\approx 2000$个样本就能够成功地区分DeepSeek-Coder、CodeGemma和Stable-Code等LLM，其AUC-ROC分数达到$\ge0.9$。 

---
# Progressive Alignment Degradation Learning for Pansharpening 

**Title (ZH)**: 渐进对齐退化学习融合 

**Authors**: Enzhe Zhao, Zhichang Guo, Yao Li, Fanghui Song, Boying Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20179)  

**Abstract**: Deep learning-based pansharpening has been shown to effectively generate high-resolution multispectral (HRMS) images. To create supervised ground-truth HRMS images, synthetic data generated using the Wald protocol is commonly employed. This protocol assumes that networks trained on artificial low-resolution data will perform equally well on high-resolution data. However, well-trained models typically exhibit a trade-off in performance between reduced-resolution and full-resolution datasets. In this paper, we delve into the Wald protocol and find that its inaccurate approximation of real-world degradation patterns limits the generalization of deep pansharpening models. To address this issue, we propose the Progressive Alignment Degradation Module (PADM), which uses mutual iteration between two sub-networks, PAlignNet and PDegradeNet, to adaptively learn accurate degradation processes without relying on predefined operators. Building on this, we introduce HFreqdiff, which embeds high-frequency details into a diffusion framework and incorporates CFB and BACM modules for frequency-selective detail extraction and precise reverse process learning. These innovations enable effective integration of high-resolution panchromatic and multispectral images, significantly enhancing spatial sharpness and quality. Experiments and ablation studies demonstrate the proposed method's superior performance compared to state-of-the-art techniques. 

**Abstract (ZH)**: 基于深度学习的多光谱高分辨率化已证明能有效生成高分辨率多光谱（HRMS）图像。为了创建监督地面真实HRMS图像，通常使用Wald协议生成的合成数据。该协议假设在低分辨率数据上训练的网络在高分辨率数据上表现相同。然而， 잘 훈련된 모델들은 줄怊分辨率和全分辨率数据集之间的性能之间通常存在权衡。本文探讨了Wald协议，并发现其对现实降解模式的不准确近似限制了深度多光谱高分辨率化模型的泛化能力。为了解决这一问题，我们提出了渐进对齐降解模块（PADM），该模块通过两个子网络PAlignNet和PDegradeNet之间的相互迭代，自适应地学习准确的降解过程，而不依赖于预定义的操作符。在此基础上，我们引入了HFreqdiff，该方法嵌入了高频细节到扩散框架中，并结合了CFB和BACM模块进行频率选择性细节提取和精确反向过程学习。这些创新使高分辨率全色和多光谱图像的有效集成成为可能，显著提升了空间锐度和质量。实验和消融研究表明，所提出的方法在与最新技术相比时表现出优越性。 

---
# COIN: Uncertainty-Guarding Selective Question Answering for Foundation Models with Provable Risk Guarantees 

**Title (ZH)**: COIN：具有可证明风险保证的不确定性保护选择性问答基础模型 

**Authors**: Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20178)  

**Abstract**: Uncertainty quantification (UQ) for foundation models is essential to identify and mitigate potential hallucinations in automatically generated text. However, heuristic UQ approaches lack formal guarantees for key metrics such as the false discovery rate (FDR) in selective prediction. Previous work adopts the split conformal prediction (SCP) framework to ensure desired coverage of admissible answers by constructing prediction sets, but these sets often contain incorrect candidates, limiting their practical utility. To address this, we propose COIN, an uncertainty-guarding selection framework that calibrates statistically valid thresholds to filter a single generated answer per question under user-specified FDR constraints. COIN estimates the empirical error rate on a calibration set and applies confidence interval methods such as Clopper-Pearson to establish a high-probability upper bound on the true error rate (i.e., FDR). This enables the selection of the largest uncertainty threshold that ensures FDR control on test data while significantly increasing sample retention. We demonstrate COIN's robustness in risk control, strong test-time power in retaining admissible answers, and predictive efficiency under limited calibration data across both general and multimodal text generation tasks. Furthermore, we show that employing alternative upper bound constructions and UQ strategies can further boost COIN's power performance, which underscores its extensibility and adaptability to diverse application scenarios. 

**Abstract (ZH)**: 基础模型的不确定性量化（UQ）对于识别和缓解自动生成文本中的幻觉至关重要。然而，启发式的UQ方法在确保选择性预测中的假发现率（FDR）等方面缺乏正式保证。先前工作采用分割置信预测（SCP）框架通过构建预测集来确保可接受答案的覆盖范围，但这些集往往包含错误的候选项，限制了其实用价值。为解决这一问题，我们提出COIN，这是一种不确定性保护的选择框架，通过在用户指定的FDR约束下校准统计有效的阈值来筛选每个问题的单个生成答案。COIN在校准集上估计经验错误率，并应用Clopper-Pearson等置信区间方法来建立真实错误率（即FDR）的高概率上限。这使得在显著增加样本保留的情况下，通过选择确保测试数据中的FDR控制的不确定性阈值成为可能。我们展示了COIN在风险控制中的稳健性、在保留可接受答案方面的强大测试时效能以及在有限校准数据下对通用和多模态文本生成任务的预测效率。此外，我们表明采用替代的上限构建方法和不确定性量化策略可以进一步提升COIN的效能，这突显了其在多种应用场景中的扩展性和适应性。 

---
# Valid Selection among Conformal Sets 

**Title (ZH)**: 在同构集中的有效选择 

**Authors**: Mahmoud Hegazy, Liviu Aolaritei, Michael I. Jordan, Aymeric Dieuleveut  

**Link**: [PDF](https://arxiv.org/pdf/2506.20173)  

**Abstract**: Conformal prediction offers a distribution-free framework for constructing prediction sets with coverage guarantees. In practice, multiple valid conformal prediction sets may be available, arising from different models or methodologies. However, selecting the most desirable set, such as the smallest, can invalidate the coverage guarantees. To address this challenge, we propose a stability-based approach that ensures coverage for the selected prediction set. We extend our results to the online conformal setting, propose several refinements in settings where additional structure is available, and demonstrate its effectiveness through experiments. 

**Abstract (ZH)**: 基于稳定性的可覆盖性预测集选择方法及其在线 conformal 设置的拓展 

---
# SEED: A Structural Encoder for Embedding-Driven Decoding in Time Series Prediction with LLMs 

**Title (ZH)**: SEED：一种结构编码器，用于时间序列预测的嵌入驱动解码 

**Authors**: Fengze Li, Yue Wang, Yangle Liu, Ming Huang, Dou Hong, Jieming Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.20167)  

**Abstract**: Multivariate time series forecasting requires models to simultaneously capture variable-wise structural dependencies and generalize across diverse tasks. While structural encoders are effective in modeling feature interactions, they lack the capacity to support semantic-level reasoning or task adaptation. Conversely, large language models (LLMs) possess strong generalization capabilities but remain incompatible with raw time series inputs. This gap limits the development of unified, transferable prediction systems. Therefore, we introduce SEED, a structural encoder for embedding-driven decoding, which integrates four stages: a token-aware encoder for patch extraction, a projection module that aligns patches with language model embeddings, a semantic reprogramming mechanism that maps patches to task-aware prototypes, and a frozen language model for prediction. This modular architecture decouples representation learning from inference, enabling efficient alignment between numerical patterns and semantic reasoning. Empirical results demonstrate that the proposed method achieves consistent improvements over strong baselines, and comparative studies on various datasets confirm SEED's role in addressing the structural-semantic modeling gap. 

**Abstract (ZH)**: 多变量时间序列预测要求模型同时捕捉变量级别的结构依赖并跨多种任务进行泛化。虽然结构编码器能够有效建模特征交互，但缺乏支持语义级推理或任务适配的能力。相反，大规模语言模型（LLMs）具备强大的泛化能力，但与原始时间序列输入不兼容。这一差距限制了统一可迁移预测系统的开发。因此，我们引入了SEED，一种用于嵌入驱动解码的结构编码器，集成四个阶段：一个具有标记意识的编码器用于片段提取，一个投影模块将片段与语言模型嵌入对齐，一个语义重编程机制将片段映射到任务感知原型，以及一个冻结的语言模型用于预测。这种模块化架构将表示学习与推理解耦，允许高效地对齐数值模式与语义推理。实验结果表明，所提出的方法在强基线方法上实现了一致的改进，并且在各种数据集上的对比研究证实了SEED在解决结构-语义建模差距中的作用。 

---
# Do psychic cells generate consciousness? 

**Title (ZH)**: 灵性细胞是否产生意识？ 

**Authors**: Mototaka Suzuki, Jaan Aru  

**Link**: [PDF](https://arxiv.org/pdf/2506.20164)  

**Abstract**: Technological advances in the past decades have begun to enable neuroscientists to address fundamental questions about consciousness in an unprecedented way. Here we review remarkable recent progress in our understanding of cellular-level mechanisms of conscious processing in the brain. Of particular interest are the cortical pyramidal neurons -- or "psychic cells" called by Ramón y Cajal more than 100 years ago -- which have an intriguing cellular mechanism that accounts for selective disruption of feedback signaling in the brain upon anesthetic-induced loss of consciousness. Importantly, a particular class of metabotropic receptors distributed over the dendrites of pyramidal cells are highlighted as the key cellular mechanism. After all, Cajal's instinct over a century ago may turn out to be correct -- we may have just begun to understand whether and how psychic cells indeed generate and control our consciousness. 

**Abstract (ZH)**: 过去的几十年中，技术的进步已经开始使神经科学家以前所未有的方式探究意识的基本问题。本文回顾了对大脑中意识处理细胞机制的最新理解进展。特别是皮层尖*spiny*神经元——拉蒙·耶·卡哈尔一百多年前称其为“心理细胞”——具有一个引人注目的细胞机制，可以解释在麻醉导致意识丧失时反馈信号的选择性中断。重要的是，分布在尖神经元树突上的特定类型代谢型受体被突出显示为关键的细胞机制。毕竟，一个多世纪前卡哈尔的直觉可能是正确的——我们可能才刚刚开始理解心理细胞是否以及如何生成和控制我们的意识。 

---
# AI and Agile Software Development: From Frustration to Success -- XP2025 Workshop Summary 

**Title (ZH)**: AI和敏捷软件开发：从挫折到成功——XP2025研讨会摘要 

**Authors**: Tomas Herda, Victoria Pichler, Zheying Zhang, Pekka Abrahamsson, Geir K. Hanssen  

**Link**: [PDF](https://arxiv.org/pdf/2506.20159)  

**Abstract**: The full-day workshop on AI and Agile at XP 2025 convened a diverse group of researchers and industry practitioners to address the practical challenges and opportunities of integrating Artificial Intelligence into Agile software development. Through interactive sessions, participants identified shared frustrations related to integrating AI into Agile Software Development practices, including challenges with tooling, governance, data quality, and critical skill gaps. These challenges were systematically prioritized and analyzed to uncover root causes. The workshop culminated in the collaborative development of a research roadmap that pinpoints actionable directions for future work, including both immediate solutions and ambitious long-term goals. The key outcome is a structured agenda designed to foster joint industry-academic efforts to move from identified frustrations to successful implementation. 

**Abstract (ZH)**: 全日工作坊：AI与极限编程2025中的敏捷开发 

---
# Irec: A Metacognitive Scaffolding for Self-Regulated Learning through Just-in-Time Insight Recall: A Conceptual Framework and System Prototype 

**Title (ZH)**: Irec：一种即时洞察回忆的元认知支架以促进自我调节学习：一个概念框架及系统原型 

**Authors**: Xuefei Hou, Xizhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20156)  

**Abstract**: The core challenge in learning has shifted from knowledge acquisition to effective Self-Regulated Learning (SRL): planning, monitoring, and reflecting on one's learning. Existing digital tools, however, inadequately support metacognitive reflection. Spaced Repetition Systems (SRS) use de-contextualized review, overlooking the role of context, while Personal Knowledge Management (PKM) tools require high manual maintenance.
To address these challenges, this paper introduces "Insight Recall," a novel paradigm that conceptualizes the context-triggered retrieval of personal past insights as a metacognitive scaffold to promote SRL. We formalize this paradigm using the Just-in-Time Adaptive Intervention (JITAI) framework and implement a prototype system, Irec, to demonstrate its feasibility. At its core, Irec uses a dynamic knowledge graph of the user's learning history. When a user faces a new problem, a hybrid retrieval engine recalls relevant personal "insights." Subsequently, a large language model (LLM) performs a deep similarity assessment to filter and present the most relevant scaffold in a just-in-time manner. To reduce cognitive load, Irec features a human-in-the-loop pipeline for LLM-based knowledge graph construction. We also propose an optional "Guided Inquiry" module, where users can engage in a Socratic dialogue with an expert LLM, using the current problem and recalled insights as context. The contribution of this paper is a solid theoretical framework and a usable system platform for designing next-generation intelligent learning systems that enhance metacognition and self-regulation. 

**Abstract (ZH)**: 学习的核心挑战已从知识获取转变为有效的自我调节学习（SRL）：规划、监控和反思自己的学习。然而，现有数字工具在支持元认知反思方面做得不够。间隔重复系统（SRS）采用脱嵌式的复习，忽视了情境的作用，而个人知识管理（PKM）工具需要大量的手动维护。

为应对这些挑战，本文提出了“洞察回忆”这一新的范式，将情境触发的个人过往洞察回忆概念化为元认知支架，以促进自我调节学习（SRL）。本文使用即时适配干预（JITAI）框架形式化了这一范式，并构建了一个原型系统Irec，以证明其实用性。Irec的核心在于使用用户学习历史的动态知识图谱。当用户遇到新问题时，混合检索引擎会召回相关个人“洞察”。随后，大型语言模型（LLM）执行深度相似性评估，以实时方式过滤并呈现最相关的支架。为减轻认知负担，Irec配备了一个包含人类在环的流水线，用于基于LLM的知识图谱构建。此外，本文还提出了一个可选的“引导性探究”模块，用户可以在其中与专家LLM进行苏格拉底式的对话，以当前问题和召回的洞察作为背景。本文的贡献在于提供了一个坚实理论框架和一个实用系统平台，用于设计增强元认知和自我调节能力的下一代智能学习系统。 

---
# Loss-Aware Automatic Selection of Structured Pruning Criteria for Deep Neural Network Acceleration 

**Title (ZH)**: 基于损失感知的结构化剪枝标准自动选择方法以加速深度神经网络 

**Authors**: Deepak Ghimire, Kilho Lee, Seong-heum Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.20152)  

**Abstract**: Structured pruning is a well-established technique for compressing neural networks, making it suitable for deployment in resource-limited edge devices. This paper presents an efficient Loss-Aware Automatic Selection of Structured Pruning Criteria (LAASP) for slimming and accelerating deep neural networks. The majority of pruning methodologies employ a sequential process consisting of three stages: 1) training, 2) pruning, and 3) fine-tuning, whereas the proposed pruning technique adopts a pruning-while-training approach that eliminates the first stage and integrates the second and third stages into a single cycle. The automatic selection of magnitude or similarity-based filter pruning criteria from a specified pool of criteria and the specific pruning layer at each pruning iteration is guided by the network's overall loss on a small subset of the training data. To mitigate the abrupt accuracy drop due to pruning, the network is retrained briefly after each reduction of a predefined number of floating-point operations (FLOPs). The optimal pruning rates for each layer in the network are automatically determined, eliminating the need for manual allocation of fixed or variable pruning rates for each layer. Experiments on the VGGNet and ResNet models on the CIFAR-10 and ImageNet benchmark datasets demonstrate the effectiveness of the proposed method. In particular, the ResNet56 and ResNet110 models on the CIFAR-10 dataset significantly improve the top-1 accuracy compared to state-of-the-art methods while reducing the network FLOPs by 52\%. Furthermore, the ResNet50 model on the ImageNet dataset reduces FLOPs by more than 42\% with a negligible 0.33\% drop in top-5 accuracy. The source code of this paper is publicly available online - this https URL. 

**Abstract (ZH)**: 基于损失感知的结构剪枝自动选择标准（LAASP）: 一种高效 slimming 和加速深度神经网络的方法 

---
# EAR: Erasing Concepts from Unified Autoregressive Models 

**Title (ZH)**: EAR: 消除统一自回归模型中的概念 

**Authors**: Haipeng Fan, Shiyuan Zhang, Baohunesitu, Zihang Guo, Huaiwen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20151)  

**Abstract**: Autoregressive (AR) models have achieved unified and strong performance across both visual understanding and image generation tasks. However, removing undesired concepts from AR models while maintaining overall generation quality remains an open challenge. In this paper, we propose Erasure Autoregressive Model (EAR), a fine-tuning method for effective and utility-preserving concept erasure in AR models. Specifically, we introduce Windowed Gradient Accumulation (WGA) strategy to align patch-level decoding with erasure objectives, and Thresholded Loss Masking (TLM) strategy to protect content unrelated to the target concept during fine-tuning. Furthermore, we propose a novel benchmark, Erase Concept Generator and Visual Filter (ECGVF), aim at provide a more rigorous and comprehensive foundation for evaluating concept erasure in AR models. Specifically, we first employ structured templates across diverse large language models (LLMs) to pre-generate a large-scale corpus of target-replacement concept prompt pairs. Subsequently, we generate images from these prompts and subject them to rigorous filtering via a visual classifier to ensure concept fidelity and alignment. Extensive experimental results conducted on the ECGVF benchmark with the AR model Janus-Pro demonstrate that EAR achieves marked improvements in both erasure effectiveness and model utility preservation. Code is available at: this https URL 

**Abstract (ZH)**: 自回归（AR）模型在视觉理解和图像生成任务中均取得了统一且强大的表现。然而，在移除AR模型中的不需要概念的同时保持整体生成质量仍然是一个开放的挑战。本文提出了一种名为Erasure Autoregressive Model（EAR）的方法，用于在不损害模型实用性的前提下有效移除概念。具体而言，我们引入了Windowed Gradient Accumulation（WGA）策略以与移除目标对局部解码进行对齐，并提出了Thresholded Loss Masking（TLM）策略以在微调过程中保护与目标概念无关的内容。此外，我们提出了一个新的基准Erase Concept Generator and Visual Filter（ECGVF），旨在为评估AR模型中的概念移除提供更严谨和全面的基础。具体而言，我们利用跨多种大型语言模型（LLMs）的结构化模板预先生成了大量的目标替换概念提示对。随后，我们从这些提示生成图像，并通过视觉分类器进行严格的过滤以确保概念的准确性和一致性。在使用AR模型Janus-Pro进行的ECGVF基准上进行的广泛实验结果表明，EAR在移除效果和模型实用性保留方面均取得了显著改进。代码可访问：this https URL。 

---
# CCRS: A Zero-Shot LLM-as-a-Judge Framework for Comprehensive RAG Evaluation 

**Title (ZH)**: CCRS：一种全面RAG评估的零样本LLM-as-a-Judge框架 

**Authors**: Aashiq Muhamed  

**Link**: [PDF](https://arxiv.org/pdf/2506.20128)  

**Abstract**: RAG systems enhance LLMs by incorporating external knowledge, which is crucial for domains that demand factual accuracy and up-to-date information. However, evaluating the multifaceted quality of RAG outputs, spanning aspects such as contextual coherence, query relevance, factual correctness, and informational completeness, poses significant challenges. Existing evaluation methods often rely on simple lexical overlap metrics, which are inadequate for capturing these nuances, or involve complex multi-stage pipelines with intermediate steps like claim extraction or require finetuning specialized judge models, hindering practical efficiency. To address these limitations, we propose CCRS (Contextual Coherence and Relevance Score), a novel suite of five metrics that utilizes a single, powerful, pretrained LLM as a zero-shot, end-to-end judge. CCRS evaluates: Contextual Coherence (CC), Question Relevance (QR), Information Density (ID), Answer Correctness (AC), and Information Recall (IR). We apply CCRS to evaluate six diverse RAG system configurations on the challenging BioASQ dataset. Our analysis demonstrates that CCRS effectively discriminates between system performances, confirming, for instance, that the Mistral-7B reader outperforms Llama variants. We provide a detailed analysis of CCRS metric properties, including score distributions, convergent/discriminant validity, tie rates, population statistics, and discriminative power. Compared to the complex RAGChecker framework, CCRS offers comparable or superior discriminative power for key aspects like recall and faithfulness, while being significantly more computationally efficient. CCRS thus provides a practical, comprehensive, and efficient framework for evaluating and iteratively improving RAG systems. 

**Abstract (ZH)**: RAG系统通过整合外部知识增强语言模型，这对于需要事实准确性和最新信息的领域至关重要。然而，评估RAG输出的多方面质量，涵盖上下文连贯性、查询相关性、事实正确性和信息完整性等方面，面临着重大挑战。现有的评估方法往往依赖简单的词频重叠度量，这些方法不足以捕捉这些微妙之处，或者涉及复杂的多阶段管道，包含中间步骤如主张提取，或者需要微调专门的法官模型，这阻碍了其实用效率。为了应对这些局限性，我们提出了一种新的评分方案CCRS（上下文连贯性和相关性评分），这是一种利用单一预训练语言模型作为零样本、端到端法官的五个新指标集合。CCRS评估：上下文连贯性（CC）、查询相关性（QR）、信息密度（ID）、答案正确性（AC）和信息召回率（IR）。我们将CCRS应用于六个不同的RAG系统配置在具有挑战性的BioASQ数据集上的评估。我们的分析表明，CCRS有效地区分了系统性能，例如证实Mistral-7B阅读器优于Llama变体。我们详细分析了CCRS指标属性，包括分数分布、收敛/区分效度、平局率、人口统计学和区分能力。与复杂的RAGChecker框架相比，CCRS在关键方面如召回率和忠实度方面的区分能力具有可比性或更优，并且计算效率显著更高。CCRS因此提供了一种实际、全面且高效的框架，用于评估和逐步改进RAG系统。 

---
# BrokenVideos: A Benchmark Dataset for Fine-Grained Artifact Localization in AI-Generated Videos 

**Title (ZH)**: 破碎视频：一种用于AI生成视频中细微瑕疵定位的数据集 

**Authors**: Jiahao Lin, Weixuan Peng, Bojia Zi, Yifeng Gao, Xianbiao Qi, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20103)  

**Abstract**: Recent advances in deep generative models have led to significant progress in video generation, yet the fidelity of AI-generated videos remains limited. Synthesized content often exhibits visual artifacts such as temporally inconsistent motion, physically implausible trajectories, unnatural object deformations, and local blurring that undermine realism and user trust. Accurate detection and spatial localization of these artifacts are crucial for both automated quality control and for guiding the development of improved generative models. However, the research community currently lacks a comprehensive benchmark specifically designed for artifact localization in AI generated videos. Existing datasets either restrict themselves to video or frame level detection or lack the fine-grained spatial annotations necessary for evaluating localization methods. To address this gap, we introduce BrokenVideos, a benchmark dataset of 3,254 AI-generated videos with meticulously annotated, pixel-level masks highlighting regions of visual corruption. Each annotation is validated through detailed human inspection to ensure high quality ground truth. Our experiments show that training state of the art artifact detection models and multi modal large language models (MLLMs) on BrokenVideos significantly improves their ability to localize corrupted regions. Through extensive evaluation, we demonstrate that BrokenVideos establishes a critical foundation for benchmarking and advancing research on artifact localization in generative video models. The dataset is available at: this https URL. 

**Abstract (ZH)**: 近期深度生成模型的进展在视频生成领域取得了显著成果，但AI生成视频的保真度仍然有限。合成内容中常常出现视觉伪影，如时间上不一致的运动、物理上不合理轨迹、不自然的对象变形和局部模糊，这些都会削弱视频的真实感和用户的信任度。准确检测和空间定位这些伪影对于自动化质量控制和改进生成模型的发展至关重要。然而，当前研究社区缺乏专门用于AI生成视频伪影定位的综合性基准。现有数据集要么仅关注视频或帧级检测，要么缺乏用于评估定位方法所需的精细空间注释。为解决这一问题，我们介绍了BrokenVideos，这是一个包含3,254个AI生成视频的基准数据集，这些视频具有详细像素级别的标注，标记出视觉损坏区域。每个注释都通过详细的人工检查来确保高质量的地面真相。我们的实验表明，将最先进的伪影检测模型和多模态大规模语言模型（MLLMs）训练在BrokenVideos上，可以显著提高它们定位损坏区域的能力。通过广泛的评估，我们证明BrokenVideos为生成视频模型中伪影定位的研究提供了关键的基础。数据集可在以下链接获取：this https URL。 

---
# MIRAGE: A Benchmark for Multimodal Information-Seeking and Reasoning in Agricultural Expert-Guided Conversations 

**Title (ZH)**: MIRAGE：农业专家指导对话中多模态信息查询与推理基准测试 

**Authors**: Vardhan Dongre, Chi Gui, Shubham Garg, Hooshang Nayyeri, Gokhan Tur, Dilek Hakkani-Tür, Vikram S. Adve  

**Link**: [PDF](https://arxiv.org/pdf/2506.20100)  

**Abstract**: We introduce MIRAGE, a new benchmark for multimodal expert-level reasoning and decision-making in consultative interaction settings. Designed for the agriculture domain, MIRAGE captures the full complexity of expert consultations by combining natural user queries, expert-authored responses, and image-based context, offering a high-fidelity benchmark for evaluating models on grounded reasoning, clarification strategies, and long-form generation in a real-world, knowledge-intensive domain. Grounded in over 35,000 real user-expert interactions and curated through a carefully designed multi-step pipeline, MIRAGE spans diverse crop health, pest diagnosis, and crop management scenarios. The benchmark includes more than 7,000 unique biological entities, covering plant species, pests, and diseases, making it one of the most taxonomically diverse benchmarks available for vision-language models, grounded in the real world. Unlike existing benchmarks that rely on well-specified user inputs and closed-set taxonomies, MIRAGE features underspecified, context-rich scenarios with open-world settings, requiring models to infer latent knowledge gaps, handle rare entities, and either proactively guide the interaction or respond. Project Page: this https URL 

**Abstract (ZH)**: 我们介绍了MIRAGE，一个新的多模态专家级推理与决策基准，适用于咨询交互场景。MIRAGE针对农业领域，通过结合自然用户查询、专家撰写的回应以及基于图像的上下文，捕捉专家咨询的全部复杂性，为评估模型在现实世界、知识密集型领域中的基于事实推理、澄清策略和长文生成能力提供了一个高保真基准。MIRAGE基于超过35,000个真实的用户-专家交互，并通过精心设计的多步骤管道进行筛选，涵盖了作物健康、病虫害诊断和作物管理等多种场景。该基准包括超过7,000种独特的生物学实体，涵盖了植物种类、害虫和疾病，是目前可用于视觉-语言模型的、基于真实世界最多样化的基准之一。与依赖于明确用户输入和封闭分类体系的现有基准不同，MIRAGE包括未充分指定、上下文丰富的场景，要求模型推断潜在的知识空白、处理稀有实体，并主动引导交互或做出回应。项目页面：this [链接]。 

---
# SACL: Understanding and Combating Textual Bias in Code Retrieval with Semantic-Augmented Reranking and Localization 

**Title (ZH)**: SACL：通过语义增强重排和定位理解并对抗代码检索中的文本偏见 

**Authors**: Dhruv Gupta, Gayathri Ganesh Lakshmy, Yiqing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20081)  

**Abstract**: Retrieval-Augmented Code Generation (RACG) is a critical technique for enhancing code generation by retrieving relevant information. In this work, we conduct an in-depth analysis of code retrieval by systematically masking specific features while preserving code functionality. Our discoveries include: (1) although trained on code, current retrievers heavily rely on surface-level textual features (e.g., docstrings, identifier names), and (2) they exhibit a strong bias towards well-documented code, even if the documentation is this http URL on our discoveries, we propose SACL, a framework that enriches textual information and reduces bias by augmenting code or structural knowledge with semantic information. Extensive experiments show that SACL substantially improves code retrieval (e.g., by 12.8% / 9.4% / 7.0% Recall@1 on HumanEval / MBPP / SWE-Bench-Lite), which also leads to better code generation performance (e.g., by 4.88% Pass@1 on HumanEval). 

**Abstract (ZH)**: 检索增强代码生成（RACG）是通过检索相关信息来提升代码生成的关键技术。在本工作中，我们系统地屏蔽特定特征以保留代码功能，进行了深入的代码检索分析。我们的发现包括：(1) 尽管基于代码训练，当前的检索器严重依赖表面级文本特征（例如，文档字符串、标识符名称），(2) 并且它们强烈偏向于有良好文档的代码，即使这些文档存在问题。鉴于上述发现，我们提出了SACL框架，该框架通过用语义信息丰富文本信息并增加代码或结构知识来减轻偏见。广泛的实验表明，SACL显著改进了代码检索（例如，在HumanEval / MBPP / SWE-Bench-Lite上的Recall@1分别提高了12.8% / 9.4% / 7.0%），这也导致了更好的代码生成性能（例如，在HumanEval上的Pass@1提高了4.88%）。 

---
# A Modular Multitask Reasoning Framework Integrating Spatio-temporal Models and LLMs 

**Title (ZH)**: 集成空间-时间模型和大语言模型的模块化多任务推理框架 

**Authors**: Kethmi Hirushini Hettige, Jiahao Ji, Cheng Long, Shili Xiang, Gao Cong, Jingyuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20073)  

**Abstract**: Spatio-temporal data mining plays a pivotal role in informed decision making across diverse domains. However, existing models are often restricted to narrow tasks, lacking the capacity for multi-task inference and complex long-form reasoning that require generation of in-depth, explanatory outputs. These limitations restrict their applicability to real-world, multi-faceted decision scenarios. In this work, we introduce STReason, a novel framework that integrates the reasoning strengths of large language models (LLMs) with the analytical capabilities of spatio-temporal models for multi-task inference and execution. Without requiring task-specific finetuning, STReason leverages in-context learning to decompose complex natural language queries into modular, interpretable programs, which are then systematically executed to generate both solutions and detailed rationales. To facilitate rigorous evaluation, we construct a new benchmark dataset and propose a unified evaluation framework with metrics specifically designed for long-form spatio-temporal reasoning. Experimental results show that STReason significantly outperforms advanced LLM baselines across all metrics, particularly excelling in complex, reasoning-intensive spatio-temporal scenarios. Human evaluations further validate STReason's credibility and practical utility, demonstrating its potential to reduce expert workload and broaden the applicability to real-world spatio-temporal tasks. We believe STReason provides a promising direction for developing more capable and generalizable spatio-temporal reasoning systems. 

**Abstract (ZH)**: 时空数据挖掘在不同领域的知情决策中发挥着关键作用。然而，现有的模型往往局限于狭窄的任务，缺乏进行多任务推理和复杂长篇推理的能力，这些能力需要生成深入解释性的输出。这些限制限制了它们在现实世界多方面决策场景中的应用。本文介绍了一种名为STReason的新型框架，该框架将大型语言模型（LLMs）的推理优势与时空模型的分析能力相结合，用于多任务推理和执行。STReason无需特定任务的微调，利用上下文学习将复杂的自然语言查询分解为模块化、可解释的程序，然后系统地执行这些程序以生成解决方案和详细的推理过程。为了进行严格的评估，我们构建了一个新的基准数据集，并提出了一个统一的评估框架，包含专门设计用于长篇时空推理的评估指标。实验结果表明，STReason在所有指标上均显著优于先进的LLM基线模型，尤其在复杂的、推理密集型的时空场景中表现优异。进一步的人类评估证实了STReason的可信度和实际应用价值，展示了其在减轻专家工作负担和扩大现实世界时空任务应用范围方面的潜力。我们认为STReason为开发更强大和通用的时空推理系统指明了有前景的方向。 

---
# Beyond Autocomplete: Designing CopilotLens Towards Transparent and Explainable AI Coding Agents 

**Title (ZH)**: 超越自动补全：设计透明可解释的CopilotLens代码伴侣agents 

**Authors**: Runlong Ye, Zeling Zhang, Boushra Almazroua, Michael Liut  

**Link**: [PDF](https://arxiv.org/pdf/2506.20062)  

**Abstract**: AI-powered code assistants are widely used to generate code completions, significantly boosting developer productivity. However, these tools typically present suggestions without explaining their rationale, leaving their decision-making process inscrutable. This opacity hinders developers' ability to critically evaluate the output, form accurate mental models, and build calibrated trust in the system. To address this, we introduce CopilotLens, a novel interactive framework that reframes code completion from a simple suggestion into a transparent, explainable event. CopilotLens operates as an explanation layer that reveals the AI agent's "thought process" through a dynamic two-level interface, surfacing everything from its reconstructed high-level plans to the specific codebase context influencing the code. This paper presents the design and rationale of CopilotLens, offering a concrete framework for building future agentic code assistants that prioritize clarity of reasoning over speed of suggestion, thereby fostering deeper comprehension and more robust human-AI collaboration. 

**Abstract (ZH)**: 基于AI的代码助手广泛用于生成代码补全，显著提升开发者生产力。然而，这些工具通常不解释其建议的理由，使开发者难以评估输出，形成准确的心理模型，并建立对系统的校准信任。为解决这一问题，我们介绍了CopilotLens，一个新颖的交互框架，将代码补全从简单的建议重新定义为透明可解释的事件。CopilotLens 作为一个解释层，通过动态的双层用户界面揭示AI代理的“思维过程”，从其重建的高阶计划到具体代码上下文的影响机制。本文阐述了CopilotLens的设计与 rationale，提供了一个具体的框架，用于构建优先考虑推理清晰度而非建议速度的未来代理型代码助手，从而促进更深入的理解和更稳健的人机协作。 

---
# Robust Robotic Exploration and Mapping Using Generative Occupancy Map Synthesis 

**Title (ZH)**: 基于生成占用地图合成的鲁棒机器人探索与建图 

**Authors**: Lorin Achey, Alec Reed, Brendan Crowe, Bradley Hayes, Christoffer Heckman  

**Link**: [PDF](https://arxiv.org/pdf/2506.20049)  

**Abstract**: We present a novel approach for enhancing robotic exploration by using generative occupancy mapping. We introduce SceneSense, a diffusion model designed and trained for predicting 3D occupancy maps given partial observations. Our proposed approach probabilistically fuses these predictions into a running occupancy map in real-time, resulting in significant improvements in map quality and traversability. We implement SceneSense onboard a quadruped robot and validate its performance with real-world experiments to demonstrate the effectiveness of the model. In these experiments, we show that occupancy maps enhanced with SceneSense predictions better represent our fully observed ground truth data (24.44% FID improvement around the robot and 75.59% improvement at range). We additionally show that integrating SceneSense-enhanced maps into our robotic exploration stack as a "drop-in" map improvement, utilizing an existing off-the-shelf planner, results in improvements in robustness and traversability time. Finally we show results of full exploration evaluations with our proposed system in two dissimilar environments and find that locally enhanced maps provide more consistent exploration results than maps constructed only from direct sensor measurements. 

**Abstract (ZH)**: 我们提出了一种利用生成式占用映射增强机器人探索的新方法。我们引入了SceneSense，这是一种用于预测3D占用映射的扩散模型，该模型经过设计和训练，可以处理部分观测数据。我们的方法将这些预测以概率方式实时融合到运行中的占用映射中，从而显著提高了地图的质量和可通行性。我们在腿式机器人上实现了SceneSense，并通过实地实验验证了该模型的性能，展示了其有效性。在这些实验中，我们展示了使用SceneSense预测增强的占用映射更好地代表了我们完全观测到的真实数据（机器人周围区域FID改进24.44%，远距离区域改进75.59%）。此外，我们将SceneSense增强的地图作为“即插即用”的地图改进集成到我们的机器人探索框架中，利用现有的现成规划器，结果表明这提高了鲁棒性和可通行性时间。最后，我们在两种不同的环境中对所提出的系统进行了全面探索评估，并发现局部增强的地图提供了比仅从直接传感器测量构建的地图更一致的探索结果。 

---
# GNN's Uncertainty Quantification using Self-Distillation 

**Title (ZH)**: GNN的不确定性量化研究：基于自我蒸馏的方法 

**Authors**: Hirad Daneshvar, Reza Samavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.20046)  

**Abstract**: Graph Neural Networks (GNNs) have shown remarkable performance in the healthcare domain. However, what remained challenging is quantifying the predictive uncertainty of GNNs, which is an important aspect of trustworthiness in clinical settings. While Bayesian and ensemble methods can be used to quantify uncertainty, they are computationally expensive. Additionally, the disagreement metric used by ensemble methods to compute uncertainty cannot capture the diversity of models in an ensemble network. In this paper, we propose a novel method, based on knowledge distillation, to quantify GNNs' uncertainty more efficiently and with higher precision. We apply self-distillation, where the same network serves as both the teacher and student models, thereby avoiding the need to train several networks independently. To ensure the impact of self-distillation, we develop an uncertainty metric that captures the diverse nature of the network by assigning different weights to each GNN classifier. We experimentally evaluate the precision, performance, and ability of our approach in distinguishing out-of-distribution data on two graph datasets: MIMIC-IV and Enzymes. The evaluation results demonstrate that the proposed method can effectively capture the predictive uncertainty of the model while having performance similar to that of the MC Dropout and ensemble methods. The code is publicly available at this https URL. 

**Abstract (ZH)**: 基于知识蒸馏的图神经网络不确定性量化方法 

---
# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification 

**Title (ZH)**: LSH-DynED：一种基于LSH下采样的动态集成框架用于 evolving 多类别不平衡分类 

**Authors**: Soheil Abadifard, Fazli Can  

**Link**: [PDF](https://arxiv.org/pdf/2506.20041)  

**Abstract**: The classification of imbalanced data streams, which have unequal class distributions, is a key difficulty in machine learning, especially when dealing with multiple classes. While binary imbalanced data stream classification tasks have received considerable attention, only a few studies have focused on multi-class imbalanced data streams. Effectively managing the dynamic imbalance ratio is a key challenge in this domain. This study introduces a novel, robust, and resilient approach to address these challenges by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework. To the best of our knowledge, we present the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams. The proposed method undersamples the majority classes by utilizing LSH-RHP, provides a balanced training set, and improves the ensemble's prediction performance. We conduct comprehensive experiments on 23 real-world and ten semi-synthetic datasets and compare LSH-DynED with 15 state-of-the-art methods. The results reveal that LSH-DynED outperforms other approaches in terms of both Kappa and mG-Mean effectiveness measures, demonstrating its capability in dealing with multi-class imbalanced non-stationary data streams. Notably, LSH-DynED performs well in large-scale, high-dimensional datasets with considerable class imbalances and demonstrates adaptation and robustness in real-world circumstances. To motivate our design, we review existing methods for imbalanced data streams, outline key challenges, and offer guidance for future work. For the reproducibility of our results, we have made our implementation available on GitHub. 

**Abstract (ZH)**: 不平衡数据流的分类：一种基于局部敏感哈希与随机超平面投影的动态ensemble多样化方法 

---
# Cross-Layer Discrete Concept Discovery for Interpreting Language Models 

**Title (ZH)**: 跨层离散概念发现用于解释语言模型 

**Authors**: Ankur Garg, Xuemin Yu, Hassan Sajjad, Samira Ebrahimi Kahou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20040)  

**Abstract**: Uncovering emergent concepts across transformer layers remains a significant challenge because the residual stream linearly mixes and duplicates information, obscuring how features evolve within large language models. Current research efforts primarily inspect neural representations at single layers, thereby overlooking this cross-layer superposition and the redundancy it introduces. These representations are typically either analyzed directly for activation patterns or passed to probing classifiers that map them to a limited set of predefined concepts. To address these limitations, we propose \gls{clvqvae}, a framework that uses vector quantization to map representations across layers and in the process collapse duplicated residual-stream features into compact, interpretable concept vectors. Our approach uniquely combines top-$k$ temperature-based sampling during quantization with EMA codebook updates, providing controlled exploration of the discrete latent space while maintaining code-book diversity. We further enhance the framework with scaled-spherical k-means++ for codebook initialization, which clusters by directional similarity rather than magnitude, better aligning with semantic structure in word embedding space. 

**Abstract (ZH)**: 揭示变压器层间涌现的概念依然是一项重大挑战，因为残差流线性地混合和复制信息，掩盖了大型语言模型中特征的发展过程。当前的研究主要检查单层的神经表示，从而忽视了层间的叠加以及由此引入的冗余性。这些表示通常要么直接分析其激活模式，要么传递给探针分类器映射到一组预定义的概念中。为解决这些局限性，我们提出了一种名为\gls{clvqvae}的框架，该框架使用向量量化在层间映射表示，并在此过程中将重复的残差流特征压缩为紧凑且可解释的概念向量。我们的方法独特地结合了量化过程中的基于top-$k$温度的采样与指数移动平均码书更新，提供了对离散潜在空间可控的探索，同时保持码书多样性。我们进一步通过缩放球形k-means++初始化码书，该方法按方向相似性而不是幅度进行聚类，更好地与词嵌入空间中的语义结构对齐。 

---
# Learning Bilateral Team Formation in Cooperative Multi-Agent Reinforcement Learning 

**Title (ZH)**: 学习合作多智能体强化学习中的双边团队形成 

**Authors**: Koorosh Moslemi, Chi-Guhn Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.20039)  

**Abstract**: Team formation and the dynamics of team-based learning have drawn significant interest in the context of Multi-Agent Reinforcement Learning (MARL). However, existing studies primarily focus on unilateral groupings, predefined teams, or fixed-population settings, leaving the effects of algorithmic bilateral grouping choices in dynamic populations underexplored. To address this gap, we introduce a framework for learning two-sided team formation in dynamic multi-agent systems. Through this study, we gain insight into what algorithmic properties in bilateral team formation influence policy performance and generalization. We validate our approach using widely adopted multi-agent scenarios, demonstrating competitive performance and improved generalization in most scenarios. 

**Abstract (ZH)**: 多Agent强化学习（MARL）背景下动态团队形成及动态团队学习的机制和动态性引起了广泛关注。然而，现有研究主要集中在单边分组、预定义团队或固定群体设置上，忽视了动态群体中算法双边分组选择的影响。为填补这一空白，我们提出了一种学习动态多Agent系统中双边团队形成的方法。通过本研究，我们探究了双边团队形成中算法属性如何影响策略性能和泛化能力。我们利用广泛采用的多Agent场景验证了该方法，展示了在大多数场景中的竞争性能和增强的泛化能力。 

---
# Hierarchical Reinforcement Learning and Value Optimization for Challenging Quadruped Locomotion 

**Title (ZH)**: 层次强化学习与值优化在挑战性 quadruped 运动控制中的应用 

**Authors**: Jeremiah Coholich, Muhammad Ali Murtaza, Seth Hutchinson, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2506.20036)  

**Abstract**: We propose a novel hierarchical reinforcement learning framework for quadruped locomotion over challenging terrain. Our approach incorporates a two-layer hierarchy in which a high-level policy (HLP) selects optimal goals for a low-level policy (LLP). The LLP is trained using an on-policy actor-critic RL algorithm and is given footstep placements as goals. We propose an HLP that does not require any additional training or environment samples and instead operates via an online optimization process over the learned value function of the LLP. We demonstrate the benefits of this framework by comparing it with an end-to-end reinforcement learning (RL) approach. We observe improvements in its ability to achieve higher rewards with fewer collisions across an array of different terrains, including terrains more difficult than any encountered during training. 

**Abstract (ZH)**: 我们提出了一种新型分层强化学习框架，用于在挑战性地形上实现四足运动。该方法采用两层结构，高层策略（HLP）选择适合低层策略（LLP）执行的任务目标。LLP使用基于策略的演员-评论家RL算法进行训练，并由脚印放置位置作为目标。我们提出了一种HLP，它不需要额外的训练或环境样本，而是通过在线优化过程在LLP学习的价值函数上运行。我们通过将其与端到端的强化学习（RL）方法进行比较，展示了该框架的优势。我们观察到，该框架在不同地形（包括训练中遇到的更难的地形）上实现了更高的奖励并减少了碰撞。 

---
# Automated Generation of Diverse Courses of Actions for Multi-Agent Operations using Binary Optimization and Graph Learning 

**Title (ZH)**: 使用二元优化和图学习生成多智能体操作的多样化行动方案的自动化生成 

**Authors**: Prithvi Poddar, Ehsan Tarkesh Esfahani, Karthik Dantu, Souma Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2506.20031)  

**Abstract**: Operations in disaster response, search \& rescue, and military missions that involve multiple agents demand automated processes to support the planning of the courses of action (COA). Moreover, traverse-affecting changes in the environment (rain, snow, blockades, etc.) may impact the expected performance of a COA, making it desirable to have a pool of COAs that are diverse in task distributions across agents. Further, variations in agent capabilities, which could be human crews and/or autonomous systems, present practical opportunities and computational challenges to the planning process. This paper presents a new theoretical formulation and computational framework to generate such diverse pools of COAs for operations with soft variations in agent-task compatibility. Key to the problem formulation is a graph abstraction of the task space and the pool of COAs itself to quantify its diversity. Formulating the COAs as a centralized multi-robot task allocation problem, a genetic algorithm is used for (order-ignoring) allocations of tasks to each agent that jointly maximize diversity within the COA pool and overall compatibility of the agent-task mappings. A graph neural network is trained using a policy gradient approach to then perform single agent task sequencing in each COA, which maximizes completion rates adaptive to task features. Our tests of the COA generation process in a simulated environment demonstrate significant performance gain over a random walk baseline, small optimality gap in task sequencing, and execution time of about 50 minutes to plan up to 20 COAs for 5 agent/100 task operations. 

**Abstract (ZH)**: 灾害响应、搜索与救援及军事任务中涉及多agent的操作需要自动化流程支持行动方案（COA）的规划。环境变化（如降雨、降雪、封锁等）可能影响COA的预期性能，因此需要一个在agent间任务分配上多样化的COA池。此外，agent能力的差异，包括人类班组和/or自主系统，为规划过程带来了实际机遇和计算挑战。本文提出了一种新的理论框架和计算方法，以生成在agent-task兼容性软变化情况下的多样化COA池。问题的核心在于任务空间及COA池的图抽象表示，以量化多样性。将COA表示为集中式多机器人任务分配问题，使用遗传算法进行忽略顺序的任务分配，以在COA池内最大化多样性，并优化agent-task映射的整体兼容性。通过策略梯度方法训练图神经网络，以在每个COA中进行单个agent的任务序列，从而最大化适应任务特征的完成率。在模拟环境中测试COA生成过程显示，相比于随机漫步基准，显著提高了性能，任务序列的优化差距较小，规划20个COA的执行时间为约50分钟。 

---
# Elucidated Rolling Diffusion Models for Probabilistic Weather Forecasting 

**Title (ZH)**: 阐述性滚动扩散模型在概率天气预报中的应用 

**Authors**: Salva Rühling Cachay, Miika Aittala, Karsten Kreis, Noah Brenowitz, Arash Vahdat, Morteza Mardani, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20024)  

**Abstract**: Diffusion models are a powerful tool for probabilistic forecasting, yet most applications in high-dimensional chaotic systems predict future snapshots one-by-one. This common approach struggles to model complex temporal dependencies and fails to explicitly account for the progressive growth of uncertainty inherent to such systems. While rolling diffusion frameworks, which apply increasing noise to forecasts at longer lead times, have been proposed to address this, their integration with state-of-the-art, high-fidelity diffusion techniques remains a significant challenge. We tackle this problem by introducing Elucidated Rolling Diffusion Models (ERDM), the first framework to successfully unify a rolling forecast structure with the principled, performant design of Elucidated Diffusion Models (EDM). To do this, we adapt the core EDM components-its noise schedule, network preconditioning, and Heun sampler-to the rolling forecast setting. The success of this integration is driven by three key contributions: (i) a novel loss weighting scheme that focuses model capacity on the mid-range forecast horizons where determinism gives way to stochasticity; (ii) an efficient initialization strategy using a pre-trained EDM for the initial window; and (iii) a bespoke hybrid sequence architecture for robust spatiotemporal feature extraction under progressive denoising. On 2D Navier-Stokes simulations and ERA5 global weather forecasting at 1.5^\circ resolution, ERDM consistently outperforms key diffusion-based baselines, including conditional autoregressive EDM. ERDM offers a flexible and powerful general framework for tackling diffusion-based sequence generation problems where modeling escalating uncertainty is paramount. Code is available at: this https URL 

**Abstract (ZH)**: 阐明滚动扩散模型（ERDM）: 首个结合滚动预测结构与原则性高性能设计的扩散模型框架 

---
# New Insights on Unfolding and Fine-tuning Quantum Federated Learning 

**Title (ZH)**: 新见解：展开与细调量子联邦学习 

**Authors**: Shanika Iroshi Nanayakkara, Shiva Raj Pokhrel  

**Link**: [PDF](https://arxiv.org/pdf/2506.20016)  

**Abstract**: Client heterogeneity poses significant challenges to the performance of Quantum Federated Learning (QFL). To overcome these limitations, we propose a new approach leveraging deep unfolding, which enables clients to autonomously optimize hyperparameters, such as learning rates and regularization factors, based on their specific training behavior. This dynamic adaptation mitigates overfitting and ensures robust optimization in highly heterogeneous environments where standard aggregation methods often fail. Our framework achieves approximately 90% accuracy, significantly outperforming traditional methods, which typically yield around 55% accuracy, as demonstrated through real-time training on IBM quantum hardware and Qiskit Aer simulators. By developing self adaptive fine tuning, the proposed method proves particularly effective in critical applications such as gene expression analysis and cancer detection, enhancing diagnostic precision and predictive modeling within quantum systems. Our results are attributed to convergence-aware, learnable optimization steps intrinsic to the deep unfolded framework, which maintains the generalization. Hence, this study addresses the core limitations of conventional QFL, streamlining its applicability to any complex challenges such as healthcare and genomic research. 

**Abstract (ZH)**: 客户端异质性对量子联邦学习（QFL）的性能构成了重大挑战。为克服这些限制，我们提出了一种基于深度展开的新方法，该方法使客户端能够根据其特定的训练行为自主优化超参数，如学习率和正则化因子。这种动态适应性减轻了过拟合，确保在标准聚合方法常失败的高度异质环境中实现稳健优化。我们的框架实现了约90%的准确率，显著优于传统的约55%的准确率，这已在IBM量子硬件和Qiskit Aer仿真器上的实时训练中得到验证。通过开发自我适应的精细调整，所提出的方法尤其有效于基因表达分析和癌症检测等关键应用中，增强了量子系统中的诊断精确度和预测建模。本研究将归因于深度展开框架中固有的、关注收敛性的可学习优化步骤，从而保持泛化能力。因此，这项研究解决了传统QFL的核心局限，使其更易于应用于任何复杂的挑战，如医疗保健和基因组研究。 

---
# TRACED: Transition-aware Regret Approximation with Co-learnability for Environment Design 

**Title (ZH)**: TRACED: 考虑转换的遗憾近似与协同学习的环境设计 

**Authors**: Geonwoo Cho, Jaegyun Im, Jihwan Lee, Hojun Yi, Sejin Kim, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19997)  

**Abstract**: Generalizing deep reinforcement learning agents to unseen environments remains a significant challenge. One promising solution is Unsupervised Environment Design (UED), a co-evolutionary framework in which a teacher adaptively generates tasks with high learning potential, while a student learns a robust policy from this evolving curriculum. Existing UED methods typically measure learning potential via regret, the gap between optimal and current performance, approximated solely by value-function loss. Building on these approaches, we introduce the transition prediction error as an additional term in our regret approximation. To capture how training on one task affects performance on others, we further propose a lightweight metric called co-learnability. By combining these two measures, we present Transition-aware Regret Approximation with Co-learnability for Environment Design (TRACED). Empirical evaluations show that TRACED yields curricula that improve zero-shot generalization across multiple benchmarks while requiring up to 2x fewer environment interactions than strong baselines. Ablation studies confirm that the transition prediction error drives rapid complexity ramp-up and that co-learnability delivers additional gains when paired with the transition prediction error. These results demonstrate how refined regret approximation and explicit modeling of task relationships can be leveraged for sample-efficient curriculum design in UED. 

**Abstract (ZH)**: 基于转换预测误差与共学习能力的环境设计后悔近似方法（TRACED） 

---
# HERCULES: Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization 

**Title (ZH)**: HERCULES：基于嵌入的分层递归聚类及大规模语言模型高效摘要方法 

**Authors**: Gabor Petnehazi, Bernadett Aradi  

**Link**: [PDF](https://arxiv.org/pdf/2506.19992)  

**Abstract**: The explosive growth of complex datasets across various modalities necessitates advanced analytical tools that not only group data effectively but also provide human-understandable insights into the discovered structures. We introduce HERCULES (Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization), a novel algorithm and Python package designed for hierarchical k-means clustering of diverse data types, including text, images, and numeric data (processed one modality per run). HERCULES constructs a cluster hierarchy by recursively applying k-means clustering, starting from individual data points at level 0. A key innovation is its deep integration of Large Language Models (LLMs) to generate semantically rich titles and descriptions for clusters at each level of the hierarchy, significantly enhancing interpretability. The algorithm supports two main representation modes: `direct' mode, which clusters based on original data embeddings or scaled numeric features, and `description' mode, which clusters based on embeddings derived from LLM-generated summaries. Users can provide a `topic\_seed' to guide LLM-generated summaries towards specific themes. An interactive visualization tool facilitates thorough analysis and understanding of the clustering results. We demonstrate HERCULES's capabilities and discuss its potential for extracting meaningful, hierarchical knowledge from complex datasets. 

**Abstract (ZH)**: 复杂模态下的爆炸性增长数据集需先进的分析工具，不仅要有效地分组数据，还能提供可理解的发现结构洞察。我们引入HERCULES（基于层次嵌入的递归聚类算法，利用LLM进行高效总结），这是一种新型算法及Python包，用于对包括文本、图像和数值数据在内的多种类型数据进行层次k-means聚类。HERCULES通过递归应用k-means聚类，从第0级的单个数据点开始构建聚类层次结构。关键创新在于其深度集成了大型语言模型（LLM）以生成具有丰富语义的聚类标题和描述，显著提高了可解释性。算法支持两种主要的表示模式：“直接”模式，基于原始数据嵌入或缩放的数值特征进行聚类；“描述”模式，基于LLM生成的摘要的嵌入进行聚类。用户可以提供“主题种子”以引导LLM生成的摘要向特定主题靠拢。交互式可视化工具有助于深入了解聚类结果。我们展示了HERCULES的能力，并讨论其从复杂数据集中提取有意义的层次知识的潜力。 

---
# VoxelOpt: Voxel-Adaptive Message Passing for Discrete Optimization in Deformable Abdominal CT Registration 

**Title (ZH)**: VoxelOpt：用于可变形腹部CT配准的体素自适应消息传递离散优化 

**Authors**: Hang Zhang, Yuxi Zhang, Jiazheng Wang, Xiang Chen, Renjiu Hu, Xin Tian, Gaolei Li, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19975)  

**Abstract**: Recent developments in neural networks have improved deformable image registration (DIR) by amortizing iterative optimization, enabling fast and accurate DIR results. However, learning-based methods often face challenges with limited training data, large deformations, and tend to underperform compared to iterative approaches when label supervision is unavailable. While iterative methods can achieve higher accuracy in such scenarios, they are considerably slower than learning-based methods. To address these limitations, we propose VoxelOpt, a discrete optimization-based DIR framework that combines the strengths of learning-based and iterative methods to achieve a better balance between registration accuracy and runtime. VoxelOpt uses displacement entropy from local cost volumes to measure displacement signal strength at each voxel, which differs from earlier approaches in three key aspects. First, it introduces voxel-wise adaptive message passing, where voxels with lower entropy receives less influence from their neighbors. Second, it employs a multi-level image pyramid with 27-neighbor cost volumes at each level, avoiding exponential complexity growth. Third, it replaces hand-crafted features or contrastive learning with a pretrained foundational segmentation model for feature extraction. In abdominal CT registration, these changes allow VoxelOpt to outperform leading iterative in both efficiency and accuracy, while matching state-of-the-art learning-based methods trained with label supervision. The source code will be available at this https URL 

**Abstract (ZH)**: 最近神经网络的发展通过减轻迭代优化提高了可变形图像配准（DIR）的速度和准确性，但基于学习的方法在有限的训练数据、大形变和无标签监督的情况下往往表现不佳，不如迭代方法准确。为了解决这些局限性，我们提出了一种名为VoxelOpt的离散优化基于DIR框架，结合了基于学习和迭代方法的优点，以实现更高的注册准确率和运行时间平衡。VoxelOpt使用局部代价体素中超移熵来衡量每个体素的超移信号强度，并从三个方面区别于早期方法：首先，引入了体素级自适应消息传递，低熵的体素受到邻居的影响较小；其次，使用多级图像金字塔，每一级采用27邻居代价体素，避免了复杂度的指数增长；最后，用预训练的基础分割模型替代手工特征或对比学习来进行特征提取。在腹部CT配准中，这些变化使得VoxelOpt在效率和准确性上优于领先的迭代方法，同时匹配使用标签监督训练的最先进基于学习的方法。源代码将在此处提供。 

---
# Quantum Neural Networks for Propensity Score Estimation and Survival Analysis in Observational Biomedical Studies 

**Title (ZH)**: 量子神经网络在观察 biomedical 研究中倾向评分估计和生存分析中的应用 

**Authors**: Vojtěch Novák, Ivan Zelinka, Lenka Přibylová, Lubomír Martínek  

**Link**: [PDF](https://arxiv.org/pdf/2506.19973)  

**Abstract**: This study investigates the application of quantum neural networks (QNNs) for propensity score estimation to address selection bias in comparing survival outcomes between laparoscopic and open surgical techniques in a cohort of 1177 colorectal carcinoma patients treated at University Hospital Ostrava (2001-2009). Using a dataset with 77 variables, including patient demographics and tumor characteristics, we developed QNN-based propensity score models focusing on four key covariates (Age, Sex, Stage, BMI). The QNN architecture employed a linear ZFeatureMap for data encoding, a SummedPaulis operator for predictions, and the Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for robust, gradient-free optimization in noisy quantum environments. Variance regularization was integrated to mitigate quantum measurement noise, with simulations conducted under exact, sampling (1024 shots), and noisy hardware (FakeManhattanV2) conditions. QNNs, particularly with simulated hardware noise, outperformed classical logistic regression and gradient boosted machines in small samples (AUC up to 0.750 for n=100), with noise modeling enhancing predictive stability. Propensity score matching and weighting, optimized via genetic matching and matching weights, achieved covariate balance with standardized mean differences of 0.0849 and 0.0869, respectively. Survival analyses using Kaplan-Meier estimation, Cox proportional hazards, and Aalen additive regression revealed no significant survival differences post-adjustment (p-values 0.287-0.851), indicating confounding bias in unadjusted outcomes. These results highlight QNNs' potential, enhanced by CMA-ES and noise-aware strategies, to improve causal inference in biomedical research, particularly for small-sample, high-dimensional datasets. 

**Abstract (ZH)**: 量子神经网络在队列研究中用于腹腔镜与开放手术技术生存结果比较中的倾向评分估计应用：大学医院奥strava（2001-2009年）1177例结直肠癌患者的倾向评分建模研究 

---
# Inference Scaled GraphRAG: Improving Multi Hop Question Answering on Knowledge Graphs 

**Title (ZH)**: Inference Scaled GraphRAG：改进知识图上的多跳问答 

**Authors**: Travis Thompson, Seung-Hwan Lim, Paul Liu, Ruoying He, Dongkuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19967)  

**Abstract**: Large Language Models (LLMs) have achieved impressive capabilities in language understanding and generation, yet they continue to underperform on knowledge-intensive reasoning tasks due to limited access to structured context and multi-hop information. Retrieval-Augmented Generation (RAG) partially mitigates this by grounding generation in retrieved context, but conventional RAG and GraphRAG methods often fail to capture relational structure across nodes in knowledge graphs. We introduce Inference-Scaled GraphRAG, a novel framework that enhances LLM-based graph reasoning by applying inference-time compute scaling. Our method combines sequential scaling with deep chain-of-thought graph traversal, and parallel scaling with majority voting over sampled trajectories within an interleaved reasoning-execution loop. Experiments on the GRBench benchmark demonstrate that our approach significantly improves multi-hop question answering performance, achieving substantial gains over both traditional GraphRAG and prior graph traversal baselines. These findings suggest that inference-time scaling is a practical and architecture-agnostic solution for structured knowledge reasoning with LLMs 

**Abstract (ZH)**: 基于推理缩放的GraphRAG：一种增强的大语言模型图推理框架 

---
# An ab initio foundation model of wavefunctions that accurately describes chemical bond breaking 

**Title (ZH)**: 从头算原理模型的波函数基础：准确描述化学键断裂 

**Authors**: Adam Foster, Zeno Schätzle, P. Bernát Szabó, Lixue Cheng, Jonas Köhler, Gino Cassella, Nicholas Gao, Jiawei Li, Frank Noé, Jan Hermann  

**Link**: [PDF](https://arxiv.org/pdf/2506.19960)  

**Abstract**: Reliable description of bond breaking remains a major challenge for quantum chemistry due to the multireferential character of the electronic structure in dissociating species. Multireferential methods in particular suffer from large computational cost, which under the normal paradigm has to be paid anew for each system at a full price, ignoring commonalities in electronic structure across molecules. Quantum Monte Carlo with deep neural networks (deep QMC) uniquely offers to exploit such commonalities by pretraining transferable wavefunction models, but all such attempts were so far limited in scope. Here, we bring this new paradigm to fruition with Orbformer, a novel transferable wavefunction model pretrained on 22,000 equilibrium and dissociating structures that can be fine-tuned on unseen molecules reaching an accuracy-cost ratio rivalling classical multireferential methods. On established benchmarks as well as more challenging bond dissociations and Diels-Alder reactions, Orbformer is the only method that consistently converges to chemical accuracy (1 kcal/mol). This work turns the idea of amortizing the cost of solving the Schrödinger equation over many molecules into a practical approach in quantum chemistry. 

**Abstract (ZH)**: 可靠的键断裂描述仍然是量子化学中的一个重大挑战，尤其是在解离物种的多重参量电子结构特性下。特别是多重参量方法受到大量计算成本的困扰，通常在处理每个系统时都需要重新计算，无视分子间电子结构的共同性。通过深度神经网络进行量子蒙特卡罗（深度QMC）唯一地提供了通过先验训练可转移波函数模型利用这些共同性的机会，但迄今为止此类尝试的范围都有限。在此，我们通过Orbformer——一种在22,000个平衡和解离结构上预训练的新型可转移波函数模型，实现了这一新范式，该模型可以对未见过的分子进行微调，达到与经典多重参量方法媲美的准确度-成本比。在标准基准以及更具挑战性的键解离和狄耳 erklärt-阿尔德反应中，Orbformer是唯一一种能够一致地达到化学精度（1 kcal/mol）的方法。这项工作将解决薛定谔方程的计算成本在多个分子上的分摊理念转变为量子化学中的实用方法。 

---
# CycleDistill: Bootstrapping Machine Translation using LLMs with Cyclical Distillation 

**Title (ZH)**: CycleDistill：使用循环蒸馏的大型语言模型bootstrapping机器翻译 

**Authors**: Deepon Halder, Thanmay Jayakumar, Raj Dabre  

**Link**: [PDF](https://arxiv.org/pdf/2506.19952)  

**Abstract**: Large language models (LLMs), despite their ability to perform few-shot machine translation (MT), often lag behind dedicated MT systems trained on parallel corpora, which are crucial for high quality machine translation (MT). However, parallel corpora are often scarce or non-existent for low-resource languages. In this paper, we propose CycleDistill, a bootstrapping approach leveraging LLMs and few-shot translation to obtain high-quality MT systems. CycleDistill involves iteratively generating synthetic parallel corpora from monolingual corpora via zero- or few-shot MT, which is then used to fine-tune the model that was used for generating said data for MT. CycleDistill does not need parallel corpora beyond 1 to 4 few-shot examples, and in our experiments focusing on three Indian languages, by relying solely on monolingual corpora, it can achieve high-quality machine translation, improving upon a few-shot baseline model by over 20-30 chrF points on average in the first iteration. We also study the effect of leveraging softmax activations during the distillation process and observe mild improvements in translation quality. 

**Abstract (ZH)**: 基于循环蒸馏的低资源语言机器翻译方法 

---
# Can LLMs Replace Humans During Code Chunking? 

**Title (ZH)**: 大规模语言模型能否替代人类进行代码分割？ 

**Authors**: Christopher Glasz, Emily Escamilla, Eric O. Scott, Anand Patel, Jacob Zimmer, Colin Diggs, Michael Doyle, Scott Rosen, Nitin Naik, Justin F. Brunelle, Samruddhi Thaker, Parthav Poudel, Arun Sridharan, Amit Madan, Doug Wendt, William Macke, Thomas Schill  

**Link**: [PDF](https://arxiv.org/pdf/2506.19897)  

**Abstract**: Large language models (LLMs) have become essential tools in computer science, especially for tasks involving code understanding and generation. However, existing work does not address many of the unique challenges presented by code written for government applications. In particular, government enterprise software is often written in legacy languages like MUMPS or assembly language code (ALC) and the overall token lengths of these systems exceed the context window size for current commercially available LLMs. Additionally, LLMs are primarily trained on modern software languages and have undergone limited testing with legacy languages, making their ability to understand legacy languages unknown and, hence, an area for empirical study. This paper examines the application of LLMs in the modernization of legacy government code written in ALC and MUMPS, addressing the challenges of input limitations. We investigate various code-chunking methods to optimize the generation of summary module comments for legacy code files, evaluating the impact of code-chunking methods on the quality of documentation produced by different LLMs, including GPT-4o, Claude 3 Sonnet, Mixtral, and Llama 3. Our results indicate that LLMs can select partition points closely aligned with human expert partitioning. We also find that chunking approaches have significant impact on downstream tasks such as documentation generation. LLM-created partitions produce comments that are up to 20% more factual and up to 10% more useful than when humans create partitions. Therefore, we conclude that LLMs can be used as suitable replacements for human partitioning of large codebases during LLM-aided modernization. 

**Abstract (ZH)**: 大型语言模型在政府应用legacy代码现代化中的应用：解决输入限制挑战 

---
# A Framework for Uncertainty Quantification Based on Nearest Neighbors Across Layers 

**Title (ZH)**: 基于跨层最近邻的不确定性量化框架 

**Authors**: Miguel N. Font, José L. Jorro-Aragoneses, Carlos M. Alaíz  

**Link**: [PDF](https://arxiv.org/pdf/2506.19895)  

**Abstract**: Neural Networks have high accuracy in solving problems where it is difficult to detect patterns or create a logical model. However, these algorithms sometimes return wrong solutions, which become problematic in high-risk domains like medical diagnosis or autonomous driving. One strategy to detect and mitigate these errors is the measurement of the uncertainty over neural network decisions. In this paper, we present a novel post-hoc framework for measuring the uncertainty of a decision based on retrieved training cases that have a similar activation vector to the query for each layer. Based on these retrieved cases, we propose two new metrics: Decision Change and Layer Uncertainty, which capture changes in nearest-neighbor class distributions across layers. We evaluated our approach in a classification model for two datasets: CIFAR-10 and MNIST. The results show that these metrics enhance uncertainty estimation, especially in challenging classification tasks, outperforming softmax-based confidence. 

**Abstract (ZH)**: 神经网络在难以检测模式或创建逻辑模型的问题上具有高精度，但在医疗诊断或自动驾驶等高风险领域中有时会返回错误的解，这成为问题。一种检测和减轻这些错误的策略是对神经网络决策的不确定性进行测量。本文提出了一种新的后处理框架，基于查询在同一层中具有相似激活向量的训练案例来衡量决策的不确定性。基于这些检索到的案例，我们提出了两个新的度量标准：决策变化和层不确定性，它们捕捉各层之间最近邻类别分布的变化。我们在CIFAR-10和MNIST两个数据集上的分类模型中评估了该方法。结果表明，这些指标在困难的分类任务中增强了不确定性估计，优于基于softmax的置信度。 

---
# Explaining deep neural network models for electricity price forecasting with XAI 

**Title (ZH)**: 使用XAI解释深度神经网络模型的电力价格预测 

**Authors**: Antoine Pesenti, Aidan OSullivan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19894)  

**Abstract**: Electricity markets are highly complex, involving lots of interactions and complex dependencies that make it hard to understand the inner workings of the market and what is driving prices. Econometric methods have been developed for this, white-box models, however, they are not as powerful as deep neural network models (DNN). In this paper, we use a DNN to forecast the price and then use XAI methods to understand the factors driving the price dynamics in the market. The objective is to increase our understanding of how different electricity markets work. To do that, we apply explainable methods such as SHAP and Gradient, combined with visual techniques like heatmaps (saliency maps) to analyse the behaviour and contributions of various features across five electricity markets. We introduce the novel concepts of SSHAP values and SSHAP lines to enhance the complex representation of high-dimensional tabular models. 

**Abstract (ZH)**: 电市场极为复杂，涉及众多交互和复杂依赖，使得理解市场的内在运作机制和价格驱动因素颇具挑战。虽然已经发展了计量经济学方法，但白盒模型的效力不如深度神经网络模型（DNN）。本文利用DNN进行价格预测，并结合XAI方法理解市场中价格动态的因素。旨在增加我们对不同电力市场运作机制的理解。为此，我们应用可解释方法，如SHAP和梯度，结合热图（可注意力图）等可视化技术，分析五个电力市场中各种特征的行为和贡献。我们引入了SSHAP值和SSHAP线的概念，以增强高维表型模型的复杂表示。 

---
# Distillation-Enabled Knowledge Alignment for Generative Semantic Communications in AIGC Provisioning Tasks 

**Title (ZH)**: 基于蒸馏的知识对齐在AIGC生成语义通信任务中的应用 

**Authors**: Jingzhi Hu, Geoffrey Ye Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19893)  

**Abstract**: Due to the surging amount of AI-generated content (AIGC), its provisioning to edges and mobile users from the cloud incurs substantial traffic on networks. Generative semantic communication (GSC) offers a promising solution by transmitting highly compact information, i.e., prompt text and latent representations, instead of high-dimensional AIGC data. However, GSC relies on the alignment between the knowledge in the cloud generative AI (GAI) and that possessed by the edges and users, and between the knowledge for wireless transmission and that of actual channels, which remains challenging. In this paper, we propose DeKA-g, a distillation-enabled knowledge alignment algorithm for GSC systems. The core idea is to distill the generation knowledge from the cloud-GAI into low-rank matrices, which can be incorporated by the edge and used to adapt the transmission knowledge to diverse wireless channel conditions. DeKA-g comprises two novel methods: metaword-aided knowledge distillation (MAKD) and variable-rate grouped SNR adaptation (VGSA). For MAKD, an optimized metaword is employed to enhance the efficiency of knowledge distillation, while VGSA enables efficient adaptation to diverse compression rates and SNR ranges. From simulation results, DeKA-g improves the alignment between the edge-generated images and the cloud-generated ones by 44%. Moreover, it adapts to compression rates with 116% higher efficiency than the baseline and enhances the performance in low-SNR conditions by 28%. 

**Abstract (ZH)**: 由于生成式人工智能内容（AIGC）的数量激增，将其从云端传输到边缘和移动用户导致网络流量显著增加。生成式语义通信（GSC）通过传输高度压缩的信息，即提示文本和潜在表示，而非高维AIGC数据，提供了一种有前景的解决方案。然而，GSC依赖于云端生成式人工智能（GAI）的知识与边缘设备和用户所具备的知识之间的对齐，以及无线传输知识与实际信道知识之间的对齐，这仍然是一个挑战。在本文中，我们提出了一种名为DeKA-g的蒸馏增强知识对齐算法，用于GSC系统。核心思想是从云端GAI中蒸馏生成知识并将其转换为低秩矩阵，边缘设备可以采用这些矩阵来适应各种无线信道条件。DeKA-g包括两种新颖的方法：元词辅助知识蒸馏（MAKD）和可变速率分组信噪比适应（VGSA）。MAKD通过优化元词增强知识蒸馏的效率，VGSA允许高效地适应不同的压缩率和信噪比范围。从仿真实验结果来看，DeKA-g将边缘生成的图像与云端生成的图像之间的对齐提高了44%。此外，它在压缩率适应性上比基线提高了116%的效率，并在低信噪比条件下提升了28%的性能。 

---
# RepuNet: A Reputation System for Mitigating Malicious Clients in DFL 

**Title (ZH)**: RepuNet：一种缓解DFL中恶意客户端的声誉系统 

**Authors**: Isaac Marroqui Penalva, Enrique Tomás Martínez Beltrán, Manuel Gil Pérez, Alberto Huertas Celdrán  

**Link**: [PDF](https://arxiv.org/pdf/2506.19892)  

**Abstract**: Decentralized Federated Learning (DFL) enables nodes to collaboratively train models without a central server, introducing new vulnerabilities since each node independently selects peers for model aggregation. Malicious nodes may exploit this autonomy by sending corrupted models (model poisoning), delaying model submissions (delay attack), or flooding the network with excessive messages, negatively affecting system performance. Existing solutions often depend on rigid configurations or additional infrastructures such as blockchain, leading to computational overhead, scalability issues, or limited adaptability. To overcome these limitations, this paper proposes RepuNet, a decentralized reputation system that categorizes threats in DFL and dynamically evaluates node behavior using metrics like model similarity, parameter changes, message latency, and communication volume. Nodes' influence in model aggregation is adjusted based on their reputation scores. RepuNet was integrated into the Nebula DFL platform and experimentally evaluated with MNIST and CIFAR-10 datasets under non-IID distributions, using federations of up to 25 nodes in both fully connected and random topologies. Different attack intensities, frequencies, and activation intervals were tested. Results demonstrated that RepuNet effectively detects and mitigates malicious behavior, achieving F1 scores above 95% for MNIST scenarios and approximately 76% for CIFAR-10 cases. These outcomes highlight RepuNet's adaptability, robustness, and practical potential for mitigating threats in decentralized federated learning environments. 

**Abstract (ZH)**: 去中心化联邦学习中的信誉网络（RepuNet）：一种动态评估节点行为的去中心化声誉系统 

---
# Orthogonal Soft Pruning for Efficient Class Unlearning 

**Title (ZH)**: 正交软剪枝以实现高效类遗忘 

**Authors**: Qinghui Gong, Xue Yang, Xiaohu Tang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19891)  

**Abstract**: Machine unlearning aims to selectively remove class-specific knowledge from pretrained neural networks to satisfy privacy regulations such as the GDPR. Existing methods typically face a trade-off between unlearning speed and preservation of predictive accuracy, often incurring either high computational overhead or significant performance degradation on retained classes. In this paper, we propose a novel class-aware soft pruning framework leveraging orthogonal convolutional kernel regularization to achieve rapid and precise forgetting with millisecond-level response times. By enforcing orthogonality constraints during training, our method decorrelates convolutional filters and disentangles feature representations, while efficiently identifying class-specific channels through activation difference analysis. Extensive evaluations across multiple architectures and datasets demonstrate stable pruning with near-instant execution, complete forgetting of targeted classes, and minimal accuracy loss on retained data. Experiments on CIFAR-10, CIFAR-100, and TinyImageNet confirm that our approach substantially reduces membership inference attack risks and accelerates unlearning by orders of magnitude compared to state-of-the-art baselines. This framework provides an efficient, practical solution for real-time machine unlearning in Machine Learning as a Service (MLaaS) scenarios. 

**Abstract (ZH)**: 机器遗忘旨在从预训练神经网络中选择性地移除特定类别的知识，以满足GDPR等隐私法规的要求。现有方法通常在遗忘速度和预测准确性的保留之间存在权衡，往往导致较高的计算开销或在保留类别的显著性能下降。本文提出了一种新颖的类自意识软剪枝框架，利用正交卷积核正则化来实现毫秒级响应时间的快速且精确的遗忘。通过在训练过程中施加正交约束，我们的方法解耦卷积滤波器并分离特征表示，同时通过激活差异分析高效地识别特定类别的通道。在多个架构和数据集上的广泛评估表明，该方法具有稳定的剪枝性能，近乎即时的执行速度，目标类别的完全遗忘，并且在保留数据上的准确率损失最小。在CIFAR-10、CIFAR-100和TinyImageNet上的实验确认，与最先进的基线相比，我们的方法大大降低了成员推理攻击的风险并极大地加速了遗忘过程。该框架为机器学习即服务（MLaaS）场景中的实时机器遗忘提供了一种高效且实用的解决方案。 

---
# Causal-Aware Intelligent QoE Optimization for VR Interaction with Adaptive Keyframe Extraction 

**Title (ZH)**: 基于因果意识的自适应关键帧提取以优化VR交互的QoE 

**Authors**: Ziru Zhang, Jiadong Yu, Danny H.K. Tsang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19890)  

**Abstract**: The optimization of quality of experience (QoE) in multi-user virtual reality (VR) interactions demands a delicate balance between ultra-low latency, high-fidelity motion synchronization, and equitable resource allocation. While adaptive keyframe extraction mitigates transmission overhead, existing approaches often overlook the causal relationships among allocated bandwidth, CPU frequency, and user perception, limiting QoE gains. This paper proposes an intelligent framework to maximize QoE by integrating adaptive keyframe extraction with causal-aware reinforcement learning (RL). First, a novel QoE metric is formulated using the Weber-Fechner Law, combining perceptual sensitivity, attention-driven priorities, and motion reconstruction accuracy. The QoE optimization problem is then modeled as a mixed integer programming (MIP) task, jointly optimizing keyframe ratios, bandwidth, and computational resources under horizon-fairness constraints. We propose Partial State Causal Deep Deterministic Policy Gradient (PS-CDDPG), which integrates the Deep Deterministic Policy Gradient (DDPG) method with causal influence detection. By leveraging causal information regarding how QoE is influenced and determined by various actions, we explore actions guided by weights calculated from causal inference (CI), which in turn improves training efficiency. Experiments conducted with the CMU Motion Capture Database demonstrate that our framework significantly reduces interactive latency, enhances QoE, and maintains fairness, achieving superior performance compared to benchmark methods. 

**Abstract (ZH)**: 多用户虚拟现实（VR）交互中体验质量（QoE）的优化需要在超低延迟、高保真运动同步和资源公平分配之间实现精细平衡。现有方法通过自适应关键帧提取减少传输开销，但往往忽视分配带宽、CPU频率与用户感知之间的因果关系，限制了QoE的提升。本文提出了一种智能框架，通过结合自适应关键帧提取与因果感知强化学习（RL）来最大化QoE。首先，利用韦伯-费希纳定律（Weber-Fechner Law）制定一个新型QoE度量标准，结合感知敏感度、注意力驱动的优先级与运动重构精度。随后，将QoE优化问题建模为混合整数规划（MIP）任务，在时间公平约束下联合优化关键帧比例、带宽与计算资源。我们提出了部分状态因果深度确定性策略梯度（PS-CDDPG），将深度确定性策略梯度（DDPG）方法与因果影响检测相结合。通过利用因果信息来指导QoE受不同动作影响的方式，我们探索由因果推断（CI）计算权重引导的动作，从而提高训练效率。实验结果表明，与基准方法相比，我们的框架显著降低了交互延迟、提升了QoE并保持了公平性，性能更优。 

---
# Retrieval-Confused Generation is a Good Defender for Privacy Violation Attack of Large Language Models 

**Title (ZH)**: 检索混淆生成是大型语言模型隐私侵犯攻击的良好防御方法 

**Authors**: Wanli Peng, Xin Chen, Hang Fu, XinYu He, Xue Yiming, Juan Wen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19889)  

**Abstract**: Recent advances in large language models (LLMs) have made a profound impact on our society and also raised new security concerns. Particularly, due to the remarkable inference ability of LLMs, the privacy violation attack (PVA), revealed by Staab et al., introduces serious personal privacy issues. Existing defense methods mainly leverage LLMs to anonymize the input query, which requires costly inference time and cannot gain satisfactory defense performance. Moreover, directly rejecting the PVA query seems like an effective defense method, while the defense method is exposed, promoting the evolution of PVA. In this paper, we propose a novel defense paradigm based on retrieval-confused generation (RCG) of LLMs, which can efficiently and covertly defend the PVA. We first design a paraphrasing prompt to induce the LLM to rewrite the "user comments" of the attack query to construct a disturbed database. Then, we propose the most irrelevant retrieval strategy to retrieve the desired user data from the disturbed database. Finally, the "data comments" are replaced with the retrieved user data to form a defended query, leading to responding to the adversary with some wrong personal attributes, i.e., the attack fails. Extensive experiments are conducted on two datasets and eight popular LLMs to comprehensively evaluate the feasibility and the superiority of the proposed defense method. 

**Abstract (ZH)**: Recent Advances in Large Language Models: A Retrieval-Confused Generation Paradigm for Defending Privacy Violation Attacks 

---
# MATER: Multi-level Acoustic and Textual Emotion Representation for Interpretable Speech Emotion Recognition 

**Title (ZH)**: 多层级声学和文本情绪表示用于可解释的语音情绪识别 

**Authors**: Hyo Jin Jon, Longbin Jin, Hyuntaek Jung, Hyunseo Kim, Donghun Min, Eun Yi Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.19887)  

**Abstract**: This paper presents our contributions to the Speech Emotion Recognition in Naturalistic Conditions (SERNC) Challenge, where we address categorical emotion recognition and emotional attribute prediction. To handle the complexities of natural speech, including intra- and inter-subject variability, we propose Multi-level Acoustic-Textual Emotion Representation (MATER), a novel hierarchical framework that integrates acoustic and textual features at the word, utterance, and embedding levels. By fusing low-level lexical and acoustic cues with high-level contextualized representations, MATER effectively captures both fine-grained prosodic variations and semantic nuances. Additionally, we introduce an uncertainty-aware ensemble strategy to mitigate annotator inconsistencies, improving robustness in ambiguous emotional expressions. MATER ranks fourth in both tasks with a Macro-F1 of 41.01% and an average CCC of 0.5928, securing second place in valence prediction with an impressive CCC of 0.6941. 

**Abstract (ZH)**: 本研究对于自然语境下的语音情感识别挑战（SERNC）作出了贡献，主要涉及情感类别识别和情感属性预测。为处理自然语音的复杂性，包括个体间和个体内的 variability，我们提出了一种多级声学-文本情感表示（MATER），这是一种新颖的分层框架，集成了一句话、一句话和嵌入表示层次上的声学和文本特征。通过结合低层次的词汇和声学线索与高层次的上下文表示，MATER 有效地捕捉了细微的音调变化和语义 nuance。此外，我们引入了一种意识不确定性的集成策略，以缓解注释者不一致的问题，提升了在模糊情感表达方面的能力。MATER 在两个任务中均排名第四，宏 F1 得分为 41.01%，平均一致性系数（CCC）为 0.5928，并在情感倾向预测中取得了令人印象深刻的 CCC 得分 0.6941，获得第二名。 

---
# FlightKooba: A Fast Interpretable FTP Model 

**Title (ZH)**: FlightKooba：一种快速可解释的FTP模型 

**Authors**: Jing Lu, Xuan Wu, Yizhun Tian, Songhan Fan, Yali Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19885)  

**Abstract**: The Koopman theory is a powerful and effective modeling tool for converting nonlinear systems into linear representations, and flight trajectory prediction (FTP) is a complex nonlinear system. However, current models applying the Koopman theory to FTP tasks are not very effective, model interpretability is indeed an issue, and the Koopman operators are computationally intensive, resulting in long training times. To address this issue, this paper proposes a new modeling and control framework based on the HIPPO method, the Koopman theory, and state space equations from cybernetics: FlightKooba. Inspired by the idea of structural state space equations, FlightKooba directly constructs the Koopman operators from data. This makes the framework highly interpretable and significantly reduces the number of trainable parameters in the module, thereby greatly reducing training time. Experiments have demonstrated the superiority of the FlightKooba modeling method in terms of time and memory consumption (training time comparable to the Mamba module without using CUDA-level acceleration; memory reduced by more than 50% on most datasets, with a tenfold reduction in the number of parameters), essentially completing the FTP task. It provides a new method for the fast computation of the Koopman operators, opening up new possibilities for the combination of time series forecasting and control. 

**Abstract (ZH)**: 基于HIPPO方法、库普曼理论和控制论状态空间方程的FlightKooba建模与控制框架 

---
# MNN-AECS: Energy Optimization for LLM Decoding on Mobile Devices via Adaptive Core Selection 

**Title (ZH)**: MNN-AECS: 移动设备上LLM解码的自适应核心选择能源优化 

**Authors**: Zhengxiang Huang, Chaoyue Niu, Zhaode Wang, Jiarui Xue, Hanming Zhang, Yugang Wang, Zewei Xin, Xiaotang Jiang, Chengfei Lv, Fan Wu, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.19884)  

**Abstract**: As the demand for on-device Large Language Model (LLM) inference grows, energy efficiency has become a major concern, especially for battery-limited mobile devices. Our analysis shows that the memory-bound LLM decode phase dominates energy use, and yet most existing works focus on accelerating the prefill phase, neglecting energy concerns. We introduce Adaptive Energy-Centric Core Selection (AECS) and integrate it into MNN to create the energy-efficient version, MNN-AECS, the first engine-level system solution without requiring root access or OS modifications for energy-efficient LLM decoding. MNN-AECS is designed to reduce LLM decoding energy while keeping decode speed within an acceptable slowdown threshold by dynamically selecting low-power CPU cores. MNN-AECS is evaluated across 5 Android and 2 iOS devices on 5 popular LLMs of various sizes. Compared to original MNN, MNN-AECS cuts down energy use by 23% without slowdown averaged over all 7 devices and 4 datasets. Against other engines, including this http URL, executorch, mllm, and MediaPipe, MNN-AECS delivers 39% to 78% energy saving and 12% to 363% speedup on average. 

**Abstract (ZH)**: 基于适应性的能量中心核心选择（AECS）驱动的能效大语言模型解码器：无需_root访问或OS修改的端侧能效解决方案 

---
# STIMULUS: Achieving Fast Convergence and Low Sample Complexity in Stochastic Multi-Objective Learning 

**Title (ZH)**: STIMULUS: 实现随机多目标学习的快速收敛和低样本复杂性 

**Authors**: Zhuqing Liu, Chaosheng Dong, Michinari Momma, Simone Shao, Shaoyuan Xu, Yan Gao, Haibo Yang, Jia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19883)  

**Abstract**: Recently, multi-objective optimization (MOO) has gained attention for its broad applications in ML, operations research, and engineering. However, MOO algorithm design remains in its infancy and many existing MOO methods suffer from unsatisfactory convergence rate and sample complexity performance. To address this challenge, in this paper, we propose an algorithm called STIMULUS( stochastic path-integrated multi-gradient recursive e\ulstimator), a new and robust approach for solving MOO problems. Different from the traditional methods, STIMULUS introduces a simple yet powerful recursive framework for updating stochastic gradient estimates to improve convergence performance with low sample complexity. In addition, we introduce an enhanced version of STIMULUS, termed STIMULUS-M, which incorporates a momentum term to further expedite convergence. We establish $O(1/T)$ convergence rates of the proposed methods for non-convex settings and $O (\exp{-\mu T})$ for strongly convex settings, where $T$ is the total number of iteration rounds. Additionally, we achieve the state-of-the-art $O \left(n+\sqrt{n}\epsilon^{-1}\right)$ sample complexities for non-convex settings and $O\left(n+ \sqrt{n} \ln ({\mu/\epsilon})\right)$ for strongly convex settings, where $\epsilon>0$ is a desired stationarity error. Moreover, to alleviate the periodic full gradient evaluation requirement in STIMULUS and STIMULUS-M, we further propose enhanced versions with adaptive batching called STIMULUS+/ STIMULUS-M+ and provide their theoretical analysis. 

**Abstract (ZH)**: 最近，多目标优化（MOO）因其在机器学习、运筹学和工程领域的广泛应用而受到关注。然而，MOO算法设计仍处于初级阶段，许多现有的MOO方法在收敛速度和样本复杂性方面表现不佳。为了解决这一挑战，本文 propose 一种名为 STIMULUS（随机路径集成多梯度递归估计器）的新颖且稳健的解决多目标优化问题的方法。与传统方法不同，STIMULUS 引入了一种简单的高效递归框架，用于更新随机梯度估计，从而在低样本复杂性的情况下提高收敛性能。此外，我们还提出了一种增强的 STIMULUS 版本，称为 STIMULUS-M，该版本引入动量项以进一步加快收敛速度。我们建立了非凸环境下提出方法的 $O(1/T)$ 收敛率和强凸环境下 $O(\exp{-\mu T})$ 的收敛率，其中 $T$ 是总的迭代轮数。此外，我们在非凸环境下实现了最先进的 $O \left(n+\sqrt{n}\epsilon^{-1}\right)$ 样本复杂性和强凸环境下 $O\left(n+ \sqrt{n} \ln ({\mu/\epsilon})\right)$ 的样本复杂性，其中 $\epsilon>0$ 是期望的稳定误差。此外，为进一步缓解 STIMULUS 和 STIMULUS-M 中周期性的完整梯度评估要求，我们还提出了具有自适应批次的增强版本，称为 STIMULUS+/STIMULUS-M+，并提供了它们的理论分析。 

---
# Position: Machine Learning Conferences Should Establish a "Refutations and Critiques" Track 

**Title (ZH)**: 机器学习会议应设立“反驳与批评”track 

**Authors**: Rylan Schaeffer, Joshua Kazdan, Yegor Denisov-Blanch, Brando Miranda, Matthias Gerstgrasser, Susan Zhang, Andreas Haupt, Isha Gupta, Elyas Obbad, Jesse Dodge, Jessica Zosa Forde, Koustuv Sinha, Francesco Orabona, Sanmi Koyejo, David Donoho  

**Link**: [PDF](https://arxiv.org/pdf/2506.19882)  

**Abstract**: Science progresses by iteratively advancing and correcting humanity's understanding of the world. In machine learning (ML) research, rapid advancements have led to an explosion of publications, but have also led to misleading, incorrect, flawed or perhaps even fraudulent studies being accepted and sometimes highlighted at ML conferences due to the fallibility of peer review. While such mistakes are understandable, ML conferences do not offer robust processes to help the field systematically correct when such errors are this http URL position paper argues that ML conferences should establish a dedicated "Refutations and Critiques" (R & C) Track. This R & C Track would provide a high-profile, reputable platform to support vital research that critically challenges prior research, thereby fostering a dynamic self-correcting research ecosystem. We discuss key considerations including track design, review principles, potential pitfalls, and provide an illustrative example submission concerning a recent ICLR 2025 Oral. We conclude that ML conferences should create official, reputable mechanisms to help ML research self-correct. 

**Abstract (ZH)**: 科学通过迭代地推进和纠正人类对世界的理解而发展。在机器学习（ML）研究中，快速的进步导致了大量出版物的涌现，但也导致了一些误导性、不正确、有缺陷甚至可能是虚假的研究被接受并在机器学习会议上受到关注，这反映了同行评审的局限性。虽然这些错误是可以理解的，但机器学习会议缺乏系统纠正这类错误的 robust 过程。本文建议机器学习会议应设立专门的“反驳与批判”（R & C）赛道。该R & C赛道将提供一个高知名度、信誉良好的平台，支持对先前研究进行批判性挑战的重要研究，从而促进动态的自我纠正研究生态系统。我们讨论了赛道设计、评审原则、潜在陷阱，并提供了关于ICLR 2025 Oral的一项实例提交。我们得出结论，机器学习会议应创建正式的、可信赖的机制，以帮助机器学习研究自我纠正。 

---
# Physics-Guided Radiotherapy Treatment Planning with Deep Learning 

**Title (ZH)**: 基于物理引导的放射治疗规划深度学习方法 

**Authors**: Stefanos Achlatis, Efstratios Gavves, Jan-Jakob Sonke  

**Link**: [PDF](https://arxiv.org/pdf/2506.19880)  

**Abstract**: Radiotherapy (RT) is a critical cancer treatment, with volumetric modulated arc therapy (VMAT) being a commonly used technique that enhances dose conformity by dynamically adjusting multileaf collimator (MLC) positions and monitor units (MU) throughout gantry rotation. Adaptive radiotherapy requires frequent modifications to treatment plans to account for anatomical variations, necessitating time-efficient solutions. Deep learning offers a promising solution to automate this process. To this end, we propose a two-stage, physics-guided deep learning pipeline for radiotherapy planning. In the first stage, our network is trained with direct supervision on treatment plan parameters, consisting of MLC and MU values. In the second stage, we incorporate an additional supervision signal derived from the predicted 3D dose distribution, integrating physics-based guidance into the training process. We train and evaluate our approach on 133 prostate cancer patients treated with a uniform 2-arc VMAT protocol delivering a dose of 62 Gy to the planning target volume (PTV). Our results demonstrate that the proposed approach, implemented using both 3D U-Net and UNETR architectures, consistently produces treatment plans that closely match clinical ground truths. Our method achieves a mean difference of D95% = 0.42 +/- 1.83 Gy and V95% = -0.22 +/- 1.87% at the PTV while generating dose distributions that reduce radiation exposure to organs at risk. These findings highlight the potential of physics-guided deep learning in RT planning. 

**Abstract (ZH)**: 基于物理引导的深度学习在放射治疗规划中的两阶段方法 

---
# Robust Anomaly Detection in Network Traffic: Evaluating Machine Learning Models on CICIDS2017 

**Title (ZH)**: 网络流量中鲁棒异常检测：评估CICIDS2017上的机器学习模型 

**Authors**: Zhaoyang Xu, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.19877)  

**Abstract**: Identifying suitable machine learning paradigms for intrusion detection remains critical for building effective and generalizable security solutions. In this study, we present a controlled comparison of four representative models - Multi-Layer Perceptron (MLP), 1D Convolutional Neural Network (CNN), One-Class Support Vector Machine (OCSVM) and Local Outlier Factor (LOF) - on the CICIDS2017 dataset under two scenarios: detecting known attack types and generalizing to previously unseen threats. Our results show that supervised MLP and CNN achieve near-perfect accuracy on familiar attacks but suffer drastic recall drops on novel attacks. Unsupervised LOF attains moderate overall accuracy and high recall on unknown threats at the cost of elevated false alarms, while boundary-based OCSVM balances precision and recall best, demonstrating robust detection across both scenarios. These findings offer practical guidance for selecting IDS models in dynamic network environments. 

**Abstract (ZH)**: 识别适合入侵检测的机器学习范式对于构建有效且具泛化性的安全解决方案仍至关重要。在本研究中，我们在两种情景下对比了四种代表性模型——多层感知机（MLP）、一维卷积神经网络（CNN）、一类支持向量机（OCSVM）和局部异常因子（LOF）——在CICIDS2017数据集上的性能：检测已知攻击类型和泛化到之前未见过的威胁。研究结果表明，监督MLP和CNN在熟悉攻击上的准确率接近完美，但在新型攻击上的召回率显著下降。无监督的LOF在未知威胁上总体准确率和召回率较高，但伴随着较高的误报率，而基于边界的一类支持向量机在精确率和召回率之间取得了最佳平衡，在两种情景中均展现出稳健的检测能力。这些发现为在动态网络环境中选择IDS模型提供了实用指导。 

---
# Speaker Embeddings to Improve Tracking of Intermittent and Moving Speakers 

**Title (ZH)**: 基于演讲者嵌入提高间歇性和移动演讲者跟踪性能 

**Authors**: Taous Iatariene, Can Cui, Alexandre Guérin, Romain Serizel  

**Link**: [PDF](https://arxiv.org/pdf/2506.19875)  

**Abstract**: Speaker tracking methods often rely on spatial observations to assign coherent track identities over time. This raises limits in scenarios with intermittent and moving speakers, i.e., speakers that may change position when they are inactive, thus leading to discontinuous spatial trajectories. This paper proposes to investigate the use of speaker embeddings, in a simple solution to this issue. We propose to perform identity reassignment post-tracking, using speaker embeddings. We leverage trajectory-related information provided by an initial tracking step and multichannel audio signal. Beamforming is used to enhance the signal towards the speakers' positions in order to compute speaker embeddings. These are then used to assign new track identities based on an enrollment pool. We evaluate the performance of the proposed speaker embedding-based identity reassignment method on a dataset where speakers change position during inactivity periods. Results show that it consistently improves the identity assignment performance of neural and standard tracking systems. In particular, we study the impact of beamforming and input duration for embedding extraction. 

**Abstract (ZH)**: 基于说话人嵌入的身份重新指派方法研究 

---
# Towards Provable (In)Secure Model Weight Release Schemes 

**Title (ZH)**: 关于可验证的安全性模型权重发布方案 

**Authors**: Xing Yang, Bingtao Wang, Yuhao Wang, Zimo Ji, Terry Jingchen Zhang, Wenyuan Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19874)  

**Abstract**: Recent secure weight release schemes claim to enable open-source model distribution while protecting model ownership and preventing misuse. However, these approaches lack rigorous security foundations and provide only informal security guarantees. Inspired by established works in cryptography, we formalize the security of weight release schemes by introducing several concrete security definitions. We then demonstrate our definition's utility through a case study of TaylorMLP, a prominent secure weight release scheme. Our analysis reveals vulnerabilities that allow parameter extraction thus showing that TaylorMLP fails to achieve its informal security goals. We hope this work will advocate for rigorous research at the intersection of machine learning and security communities and provide a blueprint for how future weight release schemes should be designed and evaluated. 

**Abstract (ZH)**: 近期的 secure weight release 方案声称能够在保护模型所有权和防止滥用的情况下实现开源模型分发，但这些方法缺乏严格的.security 基础，并仅提供非正式的安全保证。受密码学中现有工作的启发，我们通过引入几种具体的 security 定义来正式化 weight release 方案的安全性。我们通过一个典型的 secure weight release 方案 TaylorMLP 的案例研究展示了我们定义的实用性。我们的分析揭示了漏洞，允许参数提取，从而证明 TaylorMLP 未能实现其非正式的安全目标。我们希望这项工作能够促进机器学习与安全社区之间的严谨研究，并为未来 weight release 方案的设计与评估提供蓝图。 

---
# An Attack Method for Medical Insurance Claim Fraud Detection based on Generative Adversarial Network 

**Title (ZH)**: 基于生成对抗网络的医疗保险理赔欺诈检测攻击方法 

**Authors**: Yining Pang, Chenghan Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.19871)  

**Abstract**: Insurance fraud detection represents a pivotal advancement in modern insurance service, providing intelligent and digitalized monitoring to enhance management and prevent fraud. It is crucial for ensuring the security and efficiency of insurance systems. Although AI and machine learning algorithms have demonstrated strong performance in detecting fraudulent claims, the absence of standardized defense mechanisms renders current systems vulnerable to emerging adversarial threats. In this paper, we propose a GAN-based approach to conduct adversarial attacks on fraud detection systems. Our results indicate that an attacker, without knowledge of the training data or internal model details, can generate fraudulent cases that are classified as legitimate with a 99\% attack success rate (ASR). By subtly modifying real insurance records and claims, adversaries can significantly increase the fraud risk, potentially bypassing compromised detection systems. These findings underscore the urgent need to enhance the robustness of insurance fraud detection models against adversarial manipulation, thereby ensuring the stability and reliability of different insurance systems. 

**Abstract (ZH)**: 基于GAN的保险欺诈检测系统 adversarial 攻击研究 

---
# Secure Energy Transactions Using Blockchain Leveraging AI for Fraud Detection and Energy Market Stability 

**Title (ZH)**: 利用AI进行 fraud detection 和能源市场稳定性的区块链安全能源交易 

**Authors**: Md Asif Ul Hoq Khan, MD Zahedul Islam, Istiaq Ahmed, Md Masud Karim Rabbi, Farhana Rahman Anonna, MD Abdul Fahim Zeeshan, Mehedi Hasan Ridoy, Bivash Ranjan Chowdhury, Md Nazmul Shakir Rabbi, GM Alamin Sadnan  

**Link**: [PDF](https://arxiv.org/pdf/2506.19870)  

**Abstract**: Peer-to-peer trading and the move to decentralized grids have reshaped the energy markets in the United States. Notwithstanding, such developments lead to new challenges, mainly regarding the safety and authenticity of energy trade. This study aimed to develop and build a secure, intelligent, and efficient energy transaction system for the decentralized US energy market. This research interlinks the technological prowess of blockchain and artificial intelligence (AI) in a novel way to solve long-standing challenges in the distributed energy market, specifically those of security, fraudulent behavior detection, and market reliability. The dataset for this research is comprised of more than 1.2 million anonymized energy transaction records from a simulated peer-to-peer (P2P) energy exchange network emulating real-life blockchain-based American microgrids, including those tested by LO3 Energy and Grid+ Labs. Each record contains detailed fields of transaction identifier, timestamp, energy volume (kWh), transaction type (buy/sell), unit price, prosumer/consumer identifier (hashed for privacy), smart meter readings, geolocation regions, and settlement confirmation status. The dataset also includes system-calculated behavior metrics of transaction rate, variability of energy production, and historical pricing patterns. The system architecture proposed involves the integration of two layers, namely a blockchain layer and artificial intelligence (AI) layer, each playing a unique but complementary function in energy transaction securing and market intelligence improvement. The machine learning models used in this research were specifically chosen for their established high performance in classification tasks, specifically in the identification of energy transaction fraud in decentralized markets. 

**Abstract (ZH)**: P2P交易和去中心化电网的兴起重塑了美国能源市场：一种安全、智能、高效的去中心化能源交易系统的研究 

---
# Scalable and Cost-Efficient de Novo Template-Based Molecular Generation 

**Title (ZH)**: 基于模板的分子生成的可扩展和低成本从头设计方法 

**Authors**: Piotr Gaiński, Oussama Boussif, Andrei Rekesh, Dmytro Shevchuk, Ali Parviz, Mike Tyers, Robert A. Batey, Michał Koziarski  

**Link**: [PDF](https://arxiv.org/pdf/2506.19865)  

**Abstract**: Template-based molecular generation offers a promising avenue for drug design by ensuring generated compounds are synthetically accessible through predefined reaction templates and building blocks. In this work, we tackle three core challenges in template-based GFlowNets: (1) minimizing synthesis cost, (2) scaling to large building block libraries, and (3) effectively utilizing small fragment sets. We propose \textbf{Recursive Cost Guidance}, a backward policy framework that employs auxiliary machine learning models to approximate synthesis cost and viability. This guidance steers generation toward low-cost synthesis pathways, significantly enhancing cost-efficiency, molecular diversity, and quality, especially when paired with an \textbf{Exploitation Penalty} that balances the trade-off between exploration and exploitation. To enhance performance in smaller building block libraries, we develop a \textbf{Dynamic Library} mechanism that reuses intermediate high-reward states to construct full synthesis trees. Our approach establishes state-of-the-art results in template-based molecular generation. 

**Abstract (ZH)**: 基于模板的分子生成为通过预定义反应模板和构建块确保生成化合物的合成可行性提供了有前途的设计途径。本文攻克了基于模板的GFlowNets中的三大核心挑战：(1) 减少合成成本，(2) 扩展到大型构建块库，(3) 有效利用小片段集。我们提出了一种递归成本指导方法，这是一种.backward策略框架，利用辅助机器学习模型来近似合成成本和可行性。这种指导使生成偏向低成本合成路径，大幅提升了成本效率、分子多样性和质量，特别是在与探索与利用之间的平衡惩罚项（Exploitation Penalty）结合使用时更为显著。为了在较小的构建块库中增强性能，我们开发了一种动态库机制，通过重用中间高奖励状态来构建完整的合成树。我们的方法在基于模板的分子生成中达到了最先进的成果。 

---
# Exploring the Capabilities of the Frontier Large Language Models for Nuclear Energy Research 

**Title (ZH)**: 探索前沿大型语言模型在核能研究中的能力 

**Authors**: Ahmed Almeldein, Mohammed Alnaggar, Rick Archibald, Tom Beck, Arpan Biswas, Rike Bostelmann, Wes Brewer, Chris Bryan, Christopher Calle, Cihangir Celik, Rajni Chahal, Jong Youl Choi, Arindam Chowdhury, Mark Cianciosa, Franklin Curtis, Gregory Davidson, Sebastian De Pascuale, Lisa Fassino, Ana Gainaru, Yashika Ghai, Luke Gibson, Qian Gong, Christopher Greulich, Scott Greenwood, Cory Hauck, Ehab Hassan, Rinkle Juneja, Soyoung Kang, Scott Klasky, Atul Kumar, Vineet Kumar, Paul Laiu, Calvin Lear, Yan-Ru Lin, Jono McConnell, Furkan Oz, Anant Raj, Pradeep Ramuhalli, Marie Romedenne, Samantha Sabatino, José Salcedo-Pérez, Nathan D. See, Arpan Sircar, Punam Thankur, Tim Younkin, Xiao-Ying Yu, Prashant Jain, Tom Evans, Prasanna Balaprakash  

**Link**: [PDF](https://arxiv.org/pdf/2506.19863)  

**Abstract**: The AI for Nuclear Energy workshop at Oak Ridge National Laboratory evaluated the potential of Large Language Models (LLMs) to accelerate fusion and fission research. Fourteen interdisciplinary teams explored diverse nuclear science challenges using ChatGPT, Gemini, Claude, and other AI models over a single day. Applications ranged from developing foundation models for fusion reactor control to automating Monte Carlo simulations, predicting material degradation, and designing experimental programs for advanced reactors. Teams employed structured workflows combining prompt engineering, deep research capabilities, and iterative refinement to generate hypotheses, prototype code, and research strategies. Key findings demonstrate that LLMs excel at early-stage exploration, literature synthesis, and workflow design, successfully identifying research gaps and generating plausible experimental frameworks. However, significant limitations emerged, including difficulties with novel materials designs, advanced code generation for modeling and simulation, and domain-specific details requiring expert validation. The successful outcomes resulted from expert-driven prompt engineering and treating AI as a complementary tool rather than a replacement for physics-based methods. The workshop validated AI's potential to accelerate nuclear energy research through rapid iteration and cross-disciplinary synthesis while highlighting the need for curated nuclear-specific datasets, workflow automation, and specialized model development. These results provide a roadmap for integrating AI tools into nuclear science workflows, potentially reducing development cycles for safer, more efficient nuclear energy systems while maintaining rigorous scientific standards. 

**Abstract (ZH)**: Oak Ridge National Laboratory的AI在核能领域的工作shop评估了大型语言模型在加速聚变和裂变研究方面的潜力。十四个跨学科团队在一天内使用ChatGPT、Gemini、Claude及其他AI模型探索了多样的核科学挑战。应用范围包括为聚变反应堆控制开发基础模型、自动化蒙特卡洛模拟、预测材料退化以及设计先进反应堆的实验计划。团队采用了结构化的流程结合提示工程、深度研究能力和迭代优化生成假设、原型代码和研究策略。关键发现表明，大型语言模型在早期探索、文献综合和工作流程设计方面表现出色，成功识别了研究缺口并生成了可能的实验框架。然而，也出现了显著的局限性，包括新材料设计的难题、高级建模和模拟代码生成的挑战以及需要专家验证的领域特定细节。成功的成果来自于专家驱动的提示工程以及将AI视为物理方法的补充工具而非替代品。研讨会验证了AI通过快速迭代和跨学科综合加速核能研究的潜力，同时强调了需要制作化的核特定数据集、工作流程自动化和专门模型开发的需求。这些结果为将AI工具整合到核科学工作流程中提供了蓝图，可能在保持严格科学标准的同时，减少安全性和效率更高的核能系统的发展周期。 

---
# DualEquiNet: A Dual-Space Hierarchical Equivariant Network for Large Biomolecules 

**Title (ZH)**: DualEquiNet: 一种双空间分层等变网络，用于大生物分子 

**Authors**: Junjie Xu, Jiahao Zhang, Mangal Prakash, Xiang Zhang, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.19862)  

**Abstract**: Geometric graph neural networks (GNNs) that respect E(3) symmetries have achieved strong performance on small molecule modeling, but they face scalability and expressiveness challenges when applied to large biomolecules such as RNA and proteins. These systems require models that can simultaneously capture fine-grained atomic interactions, long-range dependencies across spatially distant components, and biologically relevant hierarchical structure, such as atoms forming residues, which in turn form higher-order domains. Existing geometric GNNs, which typically operate exclusively in either Euclidean or Spherical Harmonics space, are limited in their ability to capture both the fine-scale atomic details and the long-range, symmetry-aware dependencies required for modeling the multi-scale structure of large biomolecules. We introduce DualEquiNet, a Dual-Space Hierarchical Equivariant Network that constructs complementary representations in both Euclidean and Spherical Harmonics spaces to capture local geometry and global symmetry-aware features. DualEquiNet employs bidirectional cross-space message passing and a novel Cross-Space Interaction Pooling mechanism to hierarchically aggregate atomic features into biologically meaningful units, such as residues, enabling efficient and expressive multi-scale modeling for large biomolecular systems. DualEquiNet achieves state-of-the-art performance on multiple existing benchmarks for RNA property prediction and protein modeling, and outperforms prior methods on two newly introduced 3D structural benchmarks demonstrating its broad effectiveness across a range of large biomolecule modeling tasks. 

**Abstract (ZH)**: 几何图神经网络（GNNs）若符合E(3)对称性，在小分子建模中取得了显著成果，但在应用于大型生物分子如RNA和蛋白质时面临可扩展性和表达能力的挑战。现有的几何GNNs通常只能在同一空间（欧几里得空间或球谐空间）中操作，这限制了它们捕捉精细原子细节和长程、对称性 Awareness 的依赖性的能力，这两者对于建模大型生物分子的多层次结构至关重要。我们提出了DualEquiNet，这是一个在欧几里得和球谐空间构建互补表示的层次等变网络，用于捕捉局部几何结构和全局对称性敏感的特征。DualEquiNet 使用双向跨空间消息传递和一种新颖的跨空间交互聚池机制，分层聚合原子特征为生物意义单位，如残基，从而实现对大型生物分子系统的高效且表达性强的多层次建模。DualEquiNet 在多个现有基准上的RNA性质预测和蛋白质建模任务中取得了最先进的性能，并在两个新引入的三维结构基准上优于先前方法，展示了其在大型生物分子建模任务范围内的广泛有效性。 

---
