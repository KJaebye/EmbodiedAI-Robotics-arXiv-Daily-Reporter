# AD-VF: LLM-Automatic Differentiation Enables Fine-Tuning-Free Robot Planning from Formal Methods Feedback 

**Title (ZH)**: AD-VF: LLM-自动微分使通过形式方法反馈实现无需微调的机器人规划成为可能 

**Authors**: Yunhao Yang, Junyuan Hong, Gabriel Jacob Perin, Zhiwen Fan, Li Yin, Zhangyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18384)  

**Abstract**: Large language models (LLMs) can translate natural language instructions into executable action plans for robotics, autonomous driving, and other domains. Yet, deploying LLM-driven planning in the physical world demands strict adherence to safety and regulatory constraints, which current models often violate due to hallucination or weak alignment. Traditional data-driven alignment methods, such as Direct Preference Optimization (DPO), require costly human labeling, while recent formal-feedback approaches still depend on resource-intensive fine-tuning. In this paper, we propose LAD-VF, a fine-tuning-free framework that leverages formal verification feedback for automated prompt engineering. By introducing a formal-verification-informed text loss integrated with LLM-AutoDiff, LAD-VF iteratively refines prompts rather than model parameters. This yields three key benefits: (i) scalable adaptation without fine-tuning; (ii) compatibility with modular LLM architectures; and (iii) interpretable refinement via auditable prompts. Experiments in robot navigation and manipulation tasks demonstrate that LAD-VF substantially enhances specification compliance, improving success rates from 60% to over 90%. Our method thus presents a scalable and interpretable pathway toward trustworthy, formally-verified LLM-driven control systems. 

**Abstract (ZH)**: 大型语言模型可以通过将自然语言指令转换为可执行的动作计划来应用于机器人学、自主驾驶和其他领域。然而，在物理世界中部署由大型语言模型驱动的规划方案需要严格遵守安全和监管约束，当前的模型经常因幻觉或对齐不足而违反这些约束。传统的数据驱动对齐方法，如直接偏好优化（DPO），需要昂贵的人工标注，而近期的形式反馈方法仍然依赖于资源密集型微调。在本文中，我们提出了一种名为LAD-VF的无需微调框架，该框架利用形式验证反馈进行自动提示工程。通过结合LLM-AutoDiff并引入形式验证指导的文本损失，LAD-VF逐步优化提示而非模型参数。这带来了三个关键优势：(i) 不需要微调的可扩展适应；(ii) 兼容模块化的大规模语言模型架构；以及(iii) 通过可审计的提示进行可解释的优化。在机器人导航和操作任务中的实验表明，LAD-VF显著提高了规范符合性，成功率从60%提高到超过90%。因此，我们的方法为可信赖的形式验证大型语言模型驱动控制系统提供了一条可扩展且可解释的路径。 

---
# Cross-Cultural Transfer of Commonsense Reasoning in LLMs: Evidence from the Arab World 

**Title (ZH)**: 跨文化的常识推理转移在LLMs中的证据：阿拉伯世界的研究 

**Authors**: Saeed Almheiri, Rania Hossam, Mena Attia, Chenxi Wang, Preslav Nakov, Timothy Baldwin, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2509.19265)  

**Abstract**: Large language models (LLMs) often reflect Western-centric biases, limiting their effectiveness in diverse cultural contexts. Although some work has explored cultural alignment, the potential for cross-cultural transfer, using alignment in one culture to improve performance in others, remains underexplored. This paper investigates cross-cultural transfer of commonsense reasoning in the Arab world, where linguistic and historical similarities coexist with local cultural differences. Using a culturally grounded commonsense reasoning dataset covering 13 Arab countries, we evaluate lightweight alignment methods such as in-context learning and demonstration-based reinforcement (DITTO), alongside baselines like supervised fine-tuning and direct preference optimization. Our results show that merely 12 culture-specific examples from one country can improve performance in others by 10\% on average, within multilingual models. In addition, we demonstrate that out-of-culture demonstrations from Indonesia and US contexts can match or surpass in-culture alignment for MCQ reasoning, highlighting cultural commonsense transferability beyond the Arab world. These findings demonstrate that efficient cross-cultural alignment is possible and offer a promising approach to adapt LLMs to low-resource cultural settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往反映出西方中心主义偏差，限制了其在多元文化环境中的有效性。尽管已有部分研究探索了文化对齐，但利用一个文化中的对齐来改善其他文化的性能的跨文化转移潜力尚未得到充分探索。本文调查了阿拉伯世界中的常识推理跨文化转移，该地区存在语言和历史上的相似性同时伴随着当地文化差异。利用涵盖13个阿拉伯国家的文化基础常识推理数据集，我们评估了轻量级对齐方法，如上下文学习和示范强化（DITTO），并与监督微调和直接偏好优化等基准方法进行了比较。结果显示，仅从一个国家获取12个文化特定示例即可在多语言模型中平均提高10%的性能。此外，我们展示了来自印度尼西亚和美国背景的跨文化示范与阿拉伯背景对齐在选择题推理方面持平或超越，突显了常识推理在阿拉伯世界之外的文化转移潜力。这些发现表明，高效的跨文化对齐是可能的，并为适应LLM在低资源文化环境下的应用提供了有希望的方法。 

---
# AgentInit: Initializing LLM-based Multi-Agent Systems via Diversity and Expertise Orchestration for Effective and Efficient Collaboration 

**Title (ZH)**: AgentInit：通过多样性和专长 orchestration 初始化基于LLM的多agent系统，以实现有效高效的合作 

**Authors**: Chunhao Tian, Yutong Wang, Xuebo Liu, Zhexuan Wang, Liang Ding, Miao Zhang, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19236)  

**Abstract**: Proper initialization is crucial for any system, particularly in multi-agent systems (MAS), where it plays a pivotal role in determining both the system's efficiency and effectiveness. However, existing MAS initialization methods do not fully account for the collaborative needs of the generated agents in subsequent stages. Inspired by the principles of effective team composition, we propose AgentInit, which aims to optimize the structure of agent teams. Specifically, in addition to multi-round interactions and reflections between agents during agent generation, AgentInit incorporates a Natural Language to Format mechanism to ensure consistency and standardization. Balanced team selection strategies using Pareto principles are subsequently applied to jointly consider agent team diversity and task relevance to promote effective and efficient collaboration and enhance overall system performance. Experiments show that AgentInit consistently outperforms state-of-the-art initialization methods and pre-defined strategies across various frameworks and tasks, achieving an overall performance improvement of up to 1.2 and 1.6, respectively, while also significantly reducing token consumption. Further analysis confirms its strong transferability to similar tasks and verifies the effectiveness of its key components, demonstrating its capability and adaptability as a reliable MAS initialization method. Source code and models are available at this https URL. 

**Abstract (ZH)**: 合适的初始化对任何系统至关重要，特别是在多智能体系统中，它在决定系统效率和效果方面发挥着关键作用。然而，现有的多智能体系统初始化方法并未充分考虑到生成智能体在后续阶段的协作需求。受有效团队构成原则的启发，我们提出AgentInit，旨在优化智能体团队的结构。具体而言，在智能体生成过程中，除了多轮次的智能体互动和反思，AgentInit还引入自然语言格式化机制以确保一致性和标准化。随后应用帕累托原则导向的平衡团队选择策略，共同考虑智能体团队多样性和任务相关性，以促进有效和高效的协作，提升整体系统性能。实验结果显示，AgentInit在各种框架和任务中均优于最先进的初始化方法和预定义策略，分别在性能上提高了1.2和1.6，同时显著减少了令牌消耗。进一步分析证实其在类似任务中的强泛化能力和关键组件的有效性，展示了其作为可靠多智能体系统初始化方法的能力和适应性。代码和模型可在以下链接获取。 

---
# Code Driven Planning with Domain-Adaptive Critic 

**Title (ZH)**: 基于域自适应评论的代码驱动规划 

**Authors**: Zikang Tian, Shaohui Peng, Du Huang, Jiaming Guo, Ruizhi Chen, Rui Zhang, Xishan Zhang, Yuxuan Guo, Zidong Du, Qi Guo, Ling Li, Yewen Pu, Xing Hu, Yunji Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.19077)  

**Abstract**: Large Language Models (LLMs) have been widely adopted as task planners for AI agents in sequential decision-making problems, leveraging their extensive world knowledge. However, the gap between their general knowledge and environment-specific requirements often leads to inaccurate plans. To address this, existing approaches rely on frequent LLM queries to iteratively refine plans based on immediate environmental feedback, which incurs substantial query costs. However, this refinement is typically guided by short-term environmental feedback, limiting LLMs from developing plans aligned with long-term rewards. We propose Code Driven Planning with Domain-Adaptive Critic (CoPiC). Instead of relying on frequent queries, CoPiC employs LLMs to generate a diverse set of high-level planning programs, which iteratively produce and refine candidate plans. A trained domain-adaptive critic then evaluates these candidates and selects the one most aligned with long-term rewards for execution. Using high-level planning programs as planner and domain-adaptive critic as estimator, CoPiC improves planning while significantly reducing query costs. Results in ALFWorld, NetHack, and StarCraft II Unit Building show that CoPiC outperforms advanced LLM-based baselines, AdaPlanner and Reflexion, achieving an average (1) 23.33% improvement in success rate and (2) 91.27% reduction in query costs. 

**Abstract (ZH)**: 基于域自适应评价器的代码驱动规划（CoPiC） 

---
# From latent factors to language: a user study on LLM-generated explanations for an inherently interpretable matrix-based recommender system 

**Title (ZH)**: 从潜在因素到语言：一个关于LLM生成解释的用户研究，针对固有可解释的矩阵推荐系统 

**Authors**: Maxime Manderlier, Fabian Lecron, Olivier Vu Thanh, Nicolas Gillis  

**Link**: [PDF](https://arxiv.org/pdf/2509.18980)  

**Abstract**: We investigate whether large language models (LLMs) can generate effective, user-facing explanations from a mathematically interpretable recommendation model. The model is based on constrained matrix factorization, where user types are explicitly represented and predicted item scores share the same scale as observed ratings, making the model's internal representations and predicted scores directly interpretable. This structure is translated into natural language explanations using carefully designed LLM prompts. Many works in explainable AI rely on automatic evaluation metrics, which often fail to capture users' actual needs and perceptions. In contrast, we adopt a user-centered approach: we conduct a study with 326 participants who assessed the quality of the explanations across five key dimensions-transparency, effectiveness, persuasion, trust, and satisfaction-as well as the recommendations this http URL evaluate how different explanation strategies are perceived, we generate multiple explanation types from the same underlying model, varying the input information provided to the LLM. Our analysis reveals that all explanation types are generally well received, with moderate statistical differences between strategies. User comments further underscore how participants react to each type of explanation, offering complementary insights beyond the quantitative results. 

**Abstract (ZH)**: 我们研究大型语言模型（LLMs）是否能够生成有效的、面向用户的来自一个数学可解释推荐模型的解释。该模型基于受约束的矩阵分解，其中用户类型被明确表示，并且预测项评分与观测评分具有相同的量纲，从而使模型的内部表示和预测评分直接可解释。该结构通过精心设计的LLM提示转化为自然语言解释。许多可解释人工智能的研究依赖于自动评估指标，这些指标往往未能捕捉到用户的实际需求和感知。相反，我们采取以用户为中心的方法：我们对326名参与者进行了研究，他们按照透明度、有效性、说服力、信任和满意度五个关键维度评估解释的质量以及推荐。为了评估不同的解释策略，我们从相同的底层模型生成多种解释类型，改变提供给LLM的输入信息。我们的分析揭示了所有解释类型通常都受到欢迎，尽管不同策略之间存在适度的统计差异。用户评论进一步强调了参与者对每种解释类型的反应，为定量结果提供了补充见解。 

---
# LLM-based Agents Suffer from Hallucinations: A Survey of Taxonomy, Methods, and Directions 

**Title (ZH)**: 基于LLM的代理遭受幻觉：分类、方法和发展方向调查 

**Authors**: Xixun Lin, Yucheng Ning, Jingwen Zhang, Yan Dong, Yilong Liu, Yongxuan Wu, Xiaohua Qi, Nan Sun, Yanmin Shang, Pengfei Cao, Lixin Zou, Xu Chen, Chuan Zhou, Jia Wu, Shirui Pan, Bin Wang, Yanan Cao, Kai Chen, Songlin Hu, Li Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.18970)  

**Abstract**: Driven by the rapid advancements of Large Language Models (LLMs), LLM-based agents have emerged as powerful intelligent systems capable of human-like cognition, reasoning, and interaction. These agents are increasingly being deployed across diverse real-world applications, including student education, scientific research, and financial analysis. However, despite their remarkable potential, LLM-based agents remain vulnerable to hallucination issues, which can result in erroneous task execution and undermine the reliability of the overall system design. Addressing this critical challenge requires a deep understanding and a systematic consolidation of recent advances on LLM-based agents. To this end, we present the first comprehensive survey of hallucinations in LLM-based agents. By carefully analyzing the complete workflow of agents, we propose a new taxonomy that identifies different types of agent hallucinations occurring at different stages. Furthermore, we conduct an in-depth examination of eighteen triggering causes underlying the emergence of agent hallucinations. Through a detailed review of a large number of existing studies, we summarize approaches for hallucination mitigation and detection, and highlight promising directions for future research. We hope this survey will inspire further efforts toward addressing hallucinations in LLM-based agents, ultimately contributing to the development of more robust and reliable agent systems. 

**Abstract (ZH)**: 基于大规模语言模型的智能代理中的幻觉现象综述 

---
# Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning 

**Title (ZH)**: 大规模语言模型通过连续低秩微调实现的数据高效适应 

**Authors**: Xiao Han, Zimo Zhao, Wanyu Wang, Maolin Wang, Zitao Liu, Yi Chang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18942)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have emphasized the critical role of fine-tuning (FT) techniques in adapting LLMs to specific tasks, especially when retraining from scratch is computationally infeasible. Fine-tuning enables LLMs to leverage task- or domain-specific data, producing models that more effectively meet the requirements of targeted applications. However, con- ventional FT approaches often suffer from catastrophic forgetting and suboptimal data efficiency, limiting their real-world applicability. To address these challenges, this paper proposes DEAL, a novel framework that integrates Low-Rank Adapta- tion (LoRA) with a continuous fine-tuning strategy. By incorporating knowledge retention and adaptive parameter update modules, the framework mitigates the lim- itations of existing FT methods while maintaining efficiency in privacy-preserving settings. Experiments on 15 diverse datasets show that DEAL consistently outper- forms baseline methods, yielding substantial gains in task accuracy and resource efficiency. These findings demonstrate the potential of our approach to advance continual adaptation in LLMs by enhancing task performance while improving resource efficiency. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）强调了微调（FT）技术在将LLMs适应特定任务中的关键作用，特别是在重新训练从头开始计算上不可行的情况下。微调使LLMs能够利用任务或领域特定的数据，从而生产出更能满足目标应用需求的模型。然而，传统的FT方法往往遭受灾难性遗忘和次优数据效率的困扰，限制了它们在现实世界中的应用。为了应对这些挑战，本文提出了一种名为DEAL的新框架，该框架将低秩适应（LoRA）与连续微调策略相结合。通过结合知识保留和自适应参数更新模块，该框架克服了现有FT方法的限制，同时在隐私保护环境中保持了效率。在15个不同的数据集上的实验结果显示，DEAL在任务准确性和资源效率方面始终优于基线方法，取得了显著的进步。这些发现展示了我们方法在通过增强任务性能同时提高资源效率来推动LLMs持续适应方面具有潜在的优势。 

---
# LongCat-Flash-Thinking Technical Report 

**Title (ZH)**: 长猫-闪思技术报告 

**Authors**: Meituan LongCat Team, Anchun Gui, Bei Li, Bingyang Tao, Bole Zhou, Borun Chen, Chao Zhang, Chao Zhang, Chengcheng Han, Chenhui Yang, Chi Zhang, Chong Peng, Chuyu Zhang, Cong Chen, Fengcun Li, Gang Xu, Guoyuan Lin, Hao Jiang, Hao Liang, Haomin Fu, Haoxiang Ma, Hong Liu, Hongyan Hao, Hongyin Tang, Hongyu Zang, Hongzhi Ni, Hui Su, Jiahao Liu, Jiahuan Li, Jialin Liu, Jianfei Zhang, Jianhao Xu, Jianing Wang, Jiaqi Sun, Jiaqi Zhang, Jiarong Shi, Jiawei Yang, Jingang Wang, Jinrui Ding, Jun Kuang, Jun Xu, Ke He, Kefeng Zhang, Keheng Wang, Keqing He, Li Wei, Liang Shi, Lin Qiu, Lingbin Kong, Lingchuan Liu, Linsen Guo, Longfei An, Mai Xia, Meng Zhou, Mengshen Zhu, Peng Pei, Pengcheng Jia, Qi Gu, Qi Guo, Qiong Huang, Quan Chen, Quanchi Weng, Rongxiang Weng, Ruichen Shao, Rumei Li, Shanglin Lei, Shuai Du, Shuaikang Liu, Shuang Zhou, Shuhao Hu, Siyu Xu, Songshan Gong, Tao Liang, Tianhao Hu, Wei He, Wei Shi, Wei Wang, Wei Wu, Wei Zhuo, Weifeng Tang, Wenjie Shi, Wenlong Zhu, Xi Su, Xiangcheng Liu, Xiangyu Xi, Xiangzhou Huang, Xiao Liu, Xiaochen Jiang, Xiaowei Shi, Xiaowen Shi, Xiaoyu Li, Xin Chen, Xinyue Zhao, Xuan Huang, Xuemiao Zhang, Xuezhi Cao, Xunliang Cai, Yajie Zhang, Yang Chen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18883)  

**Abstract**: We present LongCat-Flash-Thinking, an efficient 560-billion-parameter open-source Mixture-of-Experts (MoE) reasoning model. Its advanced capabilities are cultivated through a meticulously crafted training process, beginning with long Chain-of-Thought (CoT) data cold-start and culminating in large-scale Reinforcement Learning (RL). We first employ a well-designed cold-start training strategy, which significantly enhances the reasoning potential and equips the model with specialized skills in both formal and agentic reasoning. Then, a core innovation is our domain-parallel training scheme, which decouples optimization across distinct domains (e.g., STEM, Code, Agentic) and subsequently fuses the resulting expert models into a single, nearly Pareto-optimal model. This entire process is powered by our Dynamic ORchestration for Asynchronous rollout (DORA) system, a large-scale RL framework that delivers a greater than threefold training speedup over synchronous methods on tens of thousands of accelerators. As a result, LongCat-Flash-Thinking achieves state-of-the-art performance among open-source models on a suite of complex reasoning tasks. The model exhibits exceptional efficiency in agentic reasoning, reducing average token consumption by 64.5% (from 19, 653 to 6, 965) on AIME-25, without degrading task accuracy. We release LongCat-Flash-Thinking to promote further advances in reasoning systems and agentic AI research. 

**Abstract (ZH)**: 长猫-闪电思考：一个高效的560亿参数开源混合专家推理模型 

---
# Memory in Large Language Models: Mechanisms, Evaluation and Evolution 

**Title (ZH)**: 大型语言模型中的记忆：机制、评估与进化 

**Authors**: Dianxing Zhang, Wendong Li, Kani Song, Jiaye Lu, Gang Li, Liuchun Yang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18868)  

**Abstract**: Under a unified operational definition, we define LLM memory as a persistent state written during pretraining, finetuning, or inference that can later be addressed and that stably influences outputs. We propose a four-part taxonomy (parametric, contextual, external, procedural/episodic) and a memory quadruple (location, persistence, write/access path, controllability). We link mechanism, evaluation, and governance via the chain write -> read -> inhibit/update. To avoid distorted comparisons across heterogeneous setups, we adopt a three-setting protocol (parametric only, offline retrieval, online retrieval) that decouples capability from information availability on the same data and timeline. On this basis we build a layered evaluation: parametric (closed-book recall, edit differential, memorization/privacy), contextual (position curves and the mid-sequence drop), external (answer correctness vs snippet attribution/faithfulness), and procedural/episodic (cross-session consistency and timeline replay, E MARS+). The framework integrates temporal governance and leakage auditing (freshness hits, outdated answers, refusal slices) and uncertainty reporting via inter-rater agreement plus paired tests with multiple-comparison correction. For updating and forgetting, we present DMM Gov: coordinating DAPT/TAPT, PEFT, model editing (ROME, MEND, MEMIT, SERAC), and RAG to form an auditable loop covering admission thresholds, rollout, monitoring, rollback, and change audits, with specs for timeliness, conflict handling, and long-horizon consistency. Finally, we give four testable propositions: minimum identifiability; a minimal evaluation card; causally constrained editing with verifiable forgetting; and when retrieval with small-window replay outperforms ultra-long-context reading. This yields a reproducible, comparable, and governable coordinate system for research and deployment. 

**Abstract (ZH)**: 在统一的操作定义下，我们将LLM记忆定义为在预训练、微调或推理过程中写入的持久状态，可以后期访问并稳定影响输出。我们提出了一种四部分分类法（参数化、上下文、外部、程序性/情景性）和一个记忆四元组（位置、持久性、写/访问路径、可控性）。我们通过写->读->抑制/更新的链条将机制、评估和治理联系起来。为了避免异构设置间的失真比较，我们采用了一种三设置协议（仅参数化、离线检索、在线检索），从而解耦能力与相同数据和时间线上的信息可用性。在此基础上，我们构建了一种分层评估体系：参数化（闭卷回忆、编辑差异、记忆/隐私）、上下文（位置曲线和中间序列下降）、外部（答案正确性 vs 摘要归因/忠实性）、程序性/情景性（跨会话一致性与时序重放、E MARS+）。框架整合了时序治理和泄漏审计（新鲜度命中、过时回答、拒绝片段）以及通过跨评判者一致性和配对检验进行的不确定性报告。对于记忆更新和遗忘，我们介绍了DMM Gov：协调DAPT/TAPT、PEFT、模型编辑（ROME、MEND、MEMIT、SERAC）和RAG，形成一个可审计循环，涵盖准入门槛、部署、监控、回滚和变更审计，并有时间敏感性、冲突处理和长周期一致性的规格说明。最后，我们提出了四项可测试的命题：最小可识别性；最小评估卡片；因果约束编辑与可验证遗忘；以及何时小窗口检索优于超长上下文阅读。这为研究和部署提供了可重现、可比较和可治理的坐标体系。 

---
# Conf-Profile: A Confidence-Driven Reasoning Paradigm for Label-Free User Profiling 

**Title (ZH)**: 基于信心驱动的推理范式：无标签用户画像方法 

**Authors**: Yingxin Li, Jianbo Zhao, Xueyu Ren, Jie Tang, Wangjie You, Xu Chen, Kan Zhou, Chao Feng, Jiao Ran, Yuan Meng, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18864)  

**Abstract**: User profiling, as a core technique for user understanding, aims to infer structural attributes from user information. Large Language Models (LLMs) provide a promising avenue for user profiling, yet the progress is hindered by the lack of comprehensive benchmarks. To bridge this gap, we propose ProfileBench, an industrial benchmark derived from a real-world video platform, encompassing heterogeneous user data and a well-structured profiling taxonomy. However, the profiling task remains challenging due to the difficulty of collecting large-scale ground-truth labels, and the heterogeneous and noisy user information can compromise the reliability of LLMs. To approach label-free and reliable user profiling, we propose a Confidence-driven Profile reasoning framework Conf-Profile, featuring a two-stage paradigm. We first synthesize high-quality labels by leveraging advanced LLMs with confidence hints, followed by confidence-weighted voting for accuracy improvement and confidence calibration for a balanced distribution. The multiple profile results, rationales, and confidence scores are aggregated and distilled into a lightweight LLM. We further enhance the reasoning ability via confidence-guided unsupervised reinforcement learning, which exploits confidence for difficulty filtering, quasi-ground truth voting, and reward weighting. Experimental results demonstrate that Conf-Profile delivers substantial performance through the two-stage training, improving F1 by 13.97 on Qwen3-8B. 

**Abstract (ZH)**: 用户画像构建：一种基于信心驱动的框架 

---
# Model selection meets clinical semantics: Optimizing ICD-10-CM prediction via LLM-as-Judge evaluation, redundancy-aware sampling, and section-aware fine-tuning 

**Title (ZH)**: 模型选择遇见临床语义：基于LLM-as-Judge评估、冗余感知采样和章节感知微调的ICD-10-CM预测优化 

**Authors**: Hong-Jie Dai, Zheng-Hao Li, An-Tai Lu, Bo-Tsz Shain, Ming-Ta Li, Tatheer Hussain Mir, Kuang-Te Wang, Min-I Su, Pei-Kang Liu, Ming-Ju Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2509.18846)  

**Abstract**: Accurate International Classification of Diseases (ICD) coding is critical for clinical documentation, billing, and healthcare analytics, yet it remains a labour-intensive and error-prone task. Although large language models (LLMs) show promise in automating ICD coding, their challenges in base model selection, input contextualization, and training data redundancy limit their effectiveness. We propose a modular framework for ICD-10 Clinical Modification (ICD-10-CM) code prediction that addresses these challenges through principled model selection, redundancy-aware data sampling, and structured input design. The framework integrates an LLM-as-judge evaluation protocol with Plackett-Luce aggregation to assess and rank open-source LLMs based on their intrinsic comprehension of ICD-10-CM code definitions. We introduced embedding-based similarity measures, a redundancy-aware sampling strategy to remove semantically duplicated discharge summaries. We leverage structured discharge summaries from Taiwanese hospitals to evaluate contextual effects and examine section-wise content inclusion under universal and section-specific modelling paradigms. Experiments across two institutional datasets demonstrate that the selected base model after fine-tuning consistently outperforms baseline LLMs in internal and external evaluations. Incorporating more clinical sections consistently improves prediction performance. This study uses open-source LLMs to establish a practical and principled approach to ICD-10-CM code prediction. The proposed framework provides a scalable, institution-ready solution for real-world deployment of automated medical coding systems by combining informed model selection, efficient data refinement, and context-aware prompting. 

**Abstract (ZH)**: 准确的国际疾病分类（ICD）编码对于临床记录、计费和医疗健康数据分析至关重要，然而这一过程仍然是劳动密集型且容易出错的任务。尽管大型语言模型（LLMs）在自动化ICD编码方面显示出潜力，但在基础模型选择、输入上下文中化以及训练数据冗余方面的挑战限制了其效果。我们提出了一种模块化框架，通过原理性的模型选择、冗余感知数据采样和结构化输入设计来解决这些挑战。该框架整合了LLM-as-judge评估协议和Plackett-Luce聚合，基于其对ICD-10-CM代码定义的内在理解来评估和排名开源LLMs。我们引入了基于嵌入的相似性度量，提出了一种冗余感知的采样策略以去除语义上重复的出院总结。我们利用台湾医院的结构化出院总结来评估上下文效应，并在通用和板块特定建模范式下检查板块级内容的纳入情况。在两个机构数据集中进行的实验显示，调整后的基础模型在内部和外部评估中均优于基础LLMs。增加更多临床板块的一致性改善了预测性能。本研究使用开源LLMs建立了ICD-10-CM编码预测的实用和原则性方法。提出的框架通过结合知情的模型选择、高效的数据精炼以及上下文感知的提示，提供了面向实际部署的自动化医疗编码系统的可扩展、机构级解决方案。 

---
# Bounded PCTL Model Checking of Large Language Model Outputs 

**Title (ZH)**: 大规模语言模型输出的有界PCTL模型检验 

**Authors**: Dennis Gross, Helge Spieker, Arnaud Gotlieb  

**Link**: [PDF](https://arxiv.org/pdf/2509.18836)  

**Abstract**: In this paper, we introduce LLMCHECKER, a model-checking-based verification method to verify the probabilistic computation tree logic (PCTL) properties of an LLM text generation process. We empirically show that only a limited number of tokens are typically chosen during text generation, which are not always the same. This insight drives the creation of $\alpha$-$k$-bounded text generation, narrowing the focus to the $\alpha$ maximal cumulative probability on the top-$k$ tokens at every step of the text generation process. Our verification method considers an initial string and the subsequent top-$k$ tokens while accommodating diverse text quantification methods, such as evaluating text quality and biases. The threshold $\alpha$ further reduces the selected tokens, only choosing those that exceed or meet it in cumulative probability. LLMCHECKER then allows us to formally verify the PCTL properties of $\alpha$-$k$-bounded LLMs. We demonstrate the applicability of our method in several LLMs, including Llama, Gemma, Mistral, Genstruct, and BERT. To our knowledge, this is the first time PCTL-based model checking has been used to check the consistency of the LLM text generation process. 

**Abstract (ZH)**: 基于模型检查的LLM文本生成过程的概率计算树逻辑（PCTL）属性验证方法：$\alpha$-$k$-受限文本生成的形式验证 

---
# Experience Scaling: Post-Deployment Evolution For Large Language Models 

**Title (ZH)**: 经验扩展：大型语言模型的部署后演化 

**Authors**: Xingkun Yin, Kaibin Huang, Dong In Kim, Hongyang Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.18771)  

**Abstract**: Scaling model size, training data, and compute power have driven advances in large language models (LLMs), but these approaches are reaching saturation as human-generated text is exhausted and further gains diminish. We propose experience scaling, a framework for continuous post-deployment evolution for LLMs through autonomous interaction with the environment and collaborative sharing of accumulated experience. The framework captures raw interactions, distills them into compact, reusable knowledge, and periodically refines stored content to preserve relevance and efficiency. We validate the framework in simulated real-world scenarios involving generalization to previously unseen but related tasks, repetitive queries, and over-saturated knowledge stores. Across all settings, experience scaling improves accuracy, sustains performance over time, and maintains gains when applied to novel situations. These results demonstrate that structured post-deployment learning can extend LLM capabilities beyond the limits of static human-generated data, offering a scalable path for continued intelligence progress. 

**Abstract (ZH)**: 经验扩展：通过与环境自主交互及累积经验的协作共享，持续演进大规模语言模型 

---
# Autonomous Data Agents: A New Opportunity for Smart Data 

**Title (ZH)**: 自主数据代理：智能数据的新机遇 

**Authors**: Yanjie Fu, Dongjie Wang, Wangyang Ying, Xiangliang Zhang, Huan Liu, Jian Pei  

**Link**: [PDF](https://arxiv.org/pdf/2509.18710)  

**Abstract**: As data continues to grow in scale and complexity, preparing, transforming, and analyzing it remains labor-intensive, repetitive, and difficult to scale. Since data contains knowledge and AI learns knowledge from it, the alignment between AI and data is essential. However, data is often not structured in ways that are optimal for AI utilization. Moreover, an important question arises: how much knowledge can we pack into data through intensive data operations? Autonomous data agents (DataAgents), which integrate LLM reasoning with task decomposition, action reasoning and grounding, and tool calling, can autonomously interpret data task descriptions, decompose tasks into subtasks, reason over actions, ground actions into python code or tool calling, and execute operations. Unlike traditional data management and engineering tools, DataAgents dynamically plan workflows, call powerful tools, and adapt to diverse data tasks at scale. This report argues that DataAgents represent a paradigm shift toward autonomous data-to-knowledge systems. DataAgents are capable of handling collection, integration, preprocessing, selection, transformation, reweighing, augmentation, reprogramming, repairs, and retrieval. Through these capabilities, DataAgents transform complex and unstructured data into coherent and actionable knowledge. We first examine why the convergence of agentic AI and data-to-knowledge systems has emerged as a critical trend. We then define the concept of DataAgents and discuss their architectural design, training strategies, as well as the new skills and capabilities they enable. Finally, we call for concerted efforts to advance action workflow optimization, establish open datasets and benchmark ecosystems, safeguard privacy, balance efficiency with scalability, and develop trustworthy DataAgent guardrails to prevent malicious actions. 

**Abstract (ZH)**: 随着数据在规模和复杂性上不断增长，数据的准备、转换和分析仍然是劳动密集型、重复性和难以扩展的任务。由于数据中蕴含知识，而AI通过学习数据中的知识来进行学习，因此AI与数据之间的对齐至关重要。然而，数据往往并未以最适合AI利用的方式结构化。此外，一个重要的问题是：通过密集的数据操作，我们能将多少知识整合进数据中？自主数据代理（DataAgents），结合了LLM推理、任务分解、动作推理与定位以及工具调用，能够自主解析数据任务描述，将任务分解为子任务，推理动作，将动作定位为Python代码或工具调用，并执行操作。与传统数据管理和工程工具不同，DataAgents能够动态规划工作流、调用强大工具，并在大规模环境下适应各种数据任务。本报告认为，DataAgents标志着向自主数据到知识系统的范式转变。DataAgents能够处理数据收集、集成、预处理、选择、转换、重新权重、增强、重新编程、修复和检索。通过这些能力，DataAgents将复杂且未结构化数据转变成连贯且可操作的知识。我们首先探讨为什么代理AI与数据到知识系统交汇成为一项关键趋势。然后定义DataAgents的概念，讨论其架构设计、训练策略，以及它们所赋予的新技能和能力。最后，我们呼吁协同努力，以优化行动工作流、建立开放数据集和基准生态系统、保障隐私、在效率与扩展性之间取得平衡，并开发值得信赖的DataAgent护栏，以阻止恶意行为。 

---
# Advances in Large Language Models for Medicine 

**Title (ZH)**: 大型语言模型在医学领域的进展 

**Authors**: Zhiyu Kan, Wensheng Gan, Zhenlian Qi, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18690)  

**Abstract**: Artificial intelligence (AI) technology has advanced rapidly in recent years, with large language models (LLMs) emerging as a significant breakthrough. LLMs are increasingly making an impact across various industries, with the medical field standing out as the most prominent application area. This paper systematically reviews the up-to-date research progress of LLMs in the medical field, providing an in-depth analysis of training techniques for large medical models, their adaptation in healthcare settings, related applications, as well as their strengths and limitations. Furthermore, it innovatively categorizes medical LLMs into three distinct types based on their training methodologies and classifies their evaluation approaches into two categories. Finally, the study proposes solutions to existing challenges and outlines future research directions based on identified issues in the field of medical LLMs. By systematically reviewing previous and advanced research findings, we aim to highlight the necessity of developing medical LLMs, provide a deeper understanding of their current state of development, and offer clear guidance for subsequent research. 

**Abstract (ZH)**: 人工智能技术近年来取得了 rapid进展，大型语言模型（LLMs） emerged作为重要的突破。LLMs在各个行业中 increasingly产生了影响，医学领域尤为突出。本文系统地回顾了LLMs在医学领域的最新研究进展，深入分析了大型医疗模型的训练技术、在医疗保健环境中的应用适应性、相关应用及其优势和局限性。此外，本文创新性地根据训练方法将医疗LLMs分为三类，并将评估方法分为两类。最后，研究提出了现有挑战的解决方案，并基于医疗LLMs领域识别的问题，指出了未来的研究方向。通过系统地回顾先前和最新的研究发现，本文旨在强调开发医疗LLMs的必要性，提供对其当前发展状态的深入理解，并为后续研究提供明确的指导。 

---
# TERAG: Token-Efficient Graph-Based Retrieval-Augmented Generation 

**Title (ZH)**: TERAG: 基于图的检索增强生成-token高效版 

**Authors**: Qiao Xiao, Hong Ting Tsang, Jiaxin Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.18667)  

**Abstract**: Graph-based Retrieval-augmented generation (RAG) has become a widely studied approach for improving the reasoning, accuracy, and factuality of Large Language Models. However, many existing graph-based RAG systems overlook the high cost associated with LLM token usage during graph construction, hindering large-scale adoption. To address this, we propose TERAG, a simple yet effective framework designed to build informative graphs at a significantly lower cost. Inspired by HippoRAG, we incorporate Personalized PageRank (PPR) during the retrieval phase, and we achieve at least 80% of the accuracy of widely used graph-based RAG methods while consuming only 3%-11% of the output tokens. 

**Abstract (ZH)**: 基于图的检索增强生成（RAG）已成为提高大型语言模型推理、准确性和事实性的广泛研究方法。然而，许多现有的基于图的RAG系统忽视了在构建图过程中LLM-token使用产生的高成本，阻碍了大规模应用。为此，我们提出了TERAG，这是一种简单而有效的框架，旨在以显著降低的成本构建信息性的图。受HippoRAG启发，在检索阶段引入个性化PageRank（PPR），我们在仅消耗广泛使用的基于图的RAG方法3%-11%输出token的前提下，实现了至少80%的准确率。 

---
# Solving Math Word Problems Using Estimation Verification and Equation Generation 

**Title (ZH)**: 使用估算验证和方程生成解决数学单词问题 

**Authors**: Mitchell Piehl, Dillon Wilson, Ananya Kalita, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2509.18565)  

**Abstract**: Large Language Models (LLMs) excel at various tasks, including problem-solving and question-answering. However, LLMs often find Math Word Problems (MWPs) challenging because solving them requires a range of reasoning and mathematical abilities with which LLMs seem to struggle. Recent efforts have helped LLMs solve more complex MWPs with improved prompts. This study proposes a novel method that initially prompts an LLM to create equations from a decomposition of the question, followed by using an external symbolic equation solver to produce an answer. To ensure the accuracy of the obtained answer, inspired by an established recommendation of math teachers, the LLM is instructed to solve the MWP a second time, but this time with the objective of estimating the correct answer instead of solving it exactly. The estimation is then compared to the generated answer to verify. If verification fails, an iterative rectification process is employed to ensure the correct answer is eventually found. This approach achieves new state-of-the-art results on datasets used by prior published research on numeric and algebraic MWPs, improving the previous best results by nearly two percent on average. In addition, the approach obtains satisfactory results on trigonometric MWPs, a task not previously attempted to the authors' best knowledge. This study also introduces two new datasets, SVAMPClean and Trig300, to further advance the testing of LLMs' reasoning abilities. 

**Abstract (ZH)**: 大型语言模型在求解数学文字题方面的新型方法及其应用 

---
# LLMZ+: Contextual Prompt Whitelist Principles for Agentic LLMs 

**Title (ZH)**: LLMZ+: 用于自主生成模型的上下文提示白名单原则 

**Authors**: Tom Pawelek, Raj Patel, Charlotte Crowell, Noorbakhsh Amiri, Sudip Mittal, Shahram Rahimi, Andy Perkins  

**Link**: [PDF](https://arxiv.org/pdf/2509.18557)  

**Abstract**: Compared to traditional models, agentic AI represents a highly valuable target for potential attackers as they possess privileged access to data sources and API tools, which are traditionally not incorporated into classical agents. Unlike a typical software application residing in a Demilitarized Zone (DMZ), agentic LLMs consciously rely on nondeterministic behavior of the AI (only defining a final goal, leaving the path selection to LLM). This characteristic introduces substantial security risk to both operational security and information security. Most common existing defense mechanism rely on detection of malicious intent and preventing it from reaching the LLM agent, thus protecting against jailbreak attacks such as prompt injection. In this paper, we present an alternative approach, LLMZ+, which moves beyond traditional detection-based approaches by implementing prompt whitelisting. Through this method, only contextually appropriate and safe messages are permitted to interact with the agentic LLM. By leveraging the specificity of context, LLMZ+ guarantees that all exchanges between external users and the LLM conform to predefined use cases and operational boundaries. Our approach streamlines the security framework, enhances its long-term resilience, and reduces the resources required for sustaining LLM information security. Our empirical evaluation demonstrates that LLMZ+ provides strong resilience against the most common jailbreak prompts. At the same time, legitimate business communications are not disrupted, and authorized traffic flows seamlessly between users and the agentic LLM. We measure the effectiveness of approach using false positive and false negative rates, both of which can be reduced to 0 in our experimental setting. 

**Abstract (ZH)**: 与传统模型相比，Agency AI 是潜在攻击者的一个高度有价值的攻击目标，因为它们拥有对数据源和 API 工具的特权访问权限，而这些通常未被经典代理所纳入。不同于通常驻留在非军事化区（DMZ）中的典型软件应用，Agency LLM 意识到依靠 AI 的不确定性行为（仅定义最终目标，将路径选择留给 LLM）。这一特性为操作安全和信息安全带来了重大的安全风险。现有的大多数防御机制依赖于检测恶意意图并阻止其到达 LLM 代理，从而防止诸如提示注入的监狱突破攻击。本文提出了一种替代方法 LLMZ+，通过实现提示白名单超越了传统的基于检测的方法。通过这种方法，仅允许上下文相关和安全的消息与 Agency LLM 交互。借助于上下文的特定性，LLMZ+ 保证所有外部用户与 LLM 之间的交流符合预定义的用例和操作边界。我们的方法简化了安全框架，增强了其长期韧性，并减少了维持 LLM 信息安全所需资源。实证评估表明，LLMZ+ 对最常见的监狱突破提示提供了强大的抗性。同时，合法的商业通信未受到影响，授权的流量在用户和 Agency LLM 之间顺畅流动。我们通过误报率和漏报率衡量方法的有效性，在实验设置中，这两种率都可以减少到 0。 

---
# Instruction-Following Evaluation in Function Calling for Large Language Models 

**Title (ZH)**: 大型语言模型中基于指令的函数调用评估 

**Authors**: Nikolai Skripko  

**Link**: [PDF](https://arxiv.org/pdf/2509.18420)  

**Abstract**: Function calling is a core capability of large language models, essential for AI agents. Existing benchmarks such as the Berkeley Function Calling Leaderboard (BFCL), tau^2-Bench (arXiv:2506.07982), and ACEBench (arXiv:2501.12851) evaluate argument correctness but do not test adherence to format instructions embedded in parameter descriptions, such as enclosing values in double quotes or using ISO date formats.
We introduce IFEval-FC, a benchmark inspired by IFEval (arXiv:2311.07911) that assesses precise instruction following in function calling. IFEval-FC encodes verifiable formats directly within JSON schema descriptions, for example specifying that a value must not contain punctuation. It includes 750 test cases, each consisting of a function with an embedded format for one of its input parameters and a corresponding user query. Evaluation is fully algorithmic, ensuring objectivity, reproducibility, and scalability.
Our results show that even state-of-the-art proprietary models, including GPT-5 and Claude 4.1 Opus, frequently fail to follow basic formatting rules, highlighting a practical limitation for real-world agent systems. The complete codebase and data are publicly available at this https URL. 

**Abstract (ZH)**: IFEval-FC：评估函数调用中精确指令遵循的基准 

---
# ATLAS: Benchmarking and Adapting LLMs for Global Trade via Harmonized Tariff Code Classification 

**Title (ZH)**: ATLAS: 全球贸易中的基准测试与适应大语言模型通过协调关税代码分类 

**Authors**: Pritish Yuvraj, Siva Devarakonda  

**Link**: [PDF](https://arxiv.org/pdf/2509.18400)  

**Abstract**: Accurate classification of products under the Harmonized Tariff Schedule (HTS) is a critical bottleneck in global trade, yet it has received little attention from the machine learning community. Misclassification can halt shipments entirely, with major postal operators suspending deliveries to the U.S. due to incomplete customs documentation. We introduce the first benchmark for HTS code classification, derived from the U.S. Customs Rulings Online Search System (CROSS). Evaluating leading LLMs, we find that our fine-tuned Atlas model (LLaMA-3.3-70B) achieves 40 percent fully correct 10-digit classifications and 57.5 percent correct 6-digit classifications, improvements of 15 points over GPT-5-Thinking and 27.5 points over Gemini-2.5-Pro-Thinking. Beyond accuracy, Atlas is roughly five times cheaper than GPT-5-Thinking and eight times cheaper than Gemini-2.5-Pro-Thinking, and can be self-hosted to guarantee data privacy in high-stakes trade and compliance workflows. While Atlas sets a strong baseline, the benchmark remains highly challenging, with only 40 percent 10-digit accuracy. By releasing both dataset and model, we aim to position HTS classification as a new community benchmark task and invite future work in retrieval, reasoning, and alignment. 

**Abstract (ZH)**: 准确分类海关 Harmonized Tariff Schedule (HTS) 代码是全球贸易中的关键瓶颈，但尚未得到机器学习社区的广泛关注。误分类可能导致货物被完全停止运送，尤其是由于 incomplete 的海关文件导致主要邮政运营商暂停向美国的发货。我们介绍了首个 HTS 代码分类基准，该基准源自美国海关在线裁决搜索系统 (CROSS)。评估领先的大规模语言模型 (LLM)，我们的 Fine-tuned Atlas 模型 (LLaMA-3.3-70B) 实现了 40% 的完全正确 10 位分类和 57.5% 的正确 6 位分类，分别优于 GPT-5-Thinking 15 个百分点和 Gemini-2.5-Pro-Thinking 27.5 个百分点。此外，Atlas 的成本分别是 GPT-5-Thinking 和 Gemini-2.5-Pro-Thinking 的五分之一和八分之一，并且可以自我托管以保证敏感贸易和合规工作流程中的数据隐私。尽管 Atlas 设置了强有力的基线，但基准测试仍然极具挑战性，仅实现了 40% 的 10 位准确率。通过发布数据集和模型，我们旨在将 HTS 分类定位为新的社区基准任务，并邀请未来的检索、推理和对齐工作。 

---
# Gödel Test: Can Large Language Models Solve Easy Conjectures? 

**Title (ZH)**: 哥德尔测试：大型语言模型能解决简单的猜想吗？ 

**Authors**: Moran Feldman, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2509.18383)  

**Abstract**: Recent announcements from frontier AI model labs have highlighted strong results on high-school and undergraduate math competitions. Yet it remains unclear whether large language models can solve new, simple conjectures in more advanced areas of mathematics. We propose the Gödel Test: evaluating whether a model can produce correct proofs for very simple, previously unsolved conjectures. To this end, we study the performance of GPT-5 on five conjectures in combinatorial optimization. For each problem, we provided one or two source papers from which the conjecture arose, withheld our own conjecture, and then assessed the model's reasoning in detail. On the three easier problems, GPT-5 produced nearly correct solutions; for Problem 2 it even derived a different approximation guarantee that, upon checking, refuted our conjecture while providing a valid solution. The model failed on Problem 4, which required combining results from two papers. On Problem 5, a harder case without a validated conjecture, GPT-5 proposed the same algorithm we had in mind but failed in the analysis, suggesting the proof is more challenging than expected. Although our sample is small, the results point to meaningful progress on routine reasoning, occasional flashes of originality, and clear limitations when cross-paper synthesis is required. GPT-5 may represent an early step toward frontier models eventually passing the Gödel Test. 

**Abstract (ZH)**: 戈德尔测试：大型语言模型在解决高级数学领域的新简单猜想中的性能评估 

---
# Evaluating the Safety and Skill Reasoning of Large Reasoning Models Under Compute Constraints 

**Title (ZH)**: 评估在计算约束下的大规模推理模型的安全性和技能推理能力 

**Authors**: Adarsha Balaji, Le Chen, Rajeev Thakur, Franck Cappello, Sandeep Madireddy  

**Link**: [PDF](https://arxiv.org/pdf/2509.18382)  

**Abstract**: Test-time compute scaling has demonstrated the ability to improve the performance of reasoning language models by generating longer chain-of-thought (CoT) sequences. However, this increase in performance comes with a significant increase in computational cost. In this work, we investigate two compute constraint strategies: (1) reasoning length constraint and (2) model quantization, as methods to reduce the compute demand of reasoning models and study their impact on their safety performance. Specifically, we explore two approaches to apply compute constraints to reasoning models: (1) fine-tuning reasoning models using a length controlled policy optimization (LCPO) based reinforcement learning method to satisfy a user-defined CoT reasoning length, and (2) applying quantization to maximize the generation of CoT sequences within a user-defined compute constraint. Furthermore, we study the trade-off between the computational efficiency and the safety of the model. 

**Abstract (ZH)**: 测试时计算量扩展能够通过生成更长的推理链（CoT）序列来提高语言模型的推理性能，但这一性能提升伴随着显著的计算成本增加。在本文中，我们研究了两种计算约束策略：（1）推理长度约束和（2）模型量化，以减少推理模型的计算需求，并研究这些策略对其安全性能的影响。具体地，我们探索了两种将计算约束应用于推理模型的方法：（1）使用基于强化学习的长度控制策略优化（LCPO）微调推理模型，以满足用户定义的CoT推理长度；（2）在用户定义的计算约束内最大化生成CoT序列的量化应用。此外，我们研究了计算效率与模型安全之间的权衡。 

---
# The Illusion of Readiness: Stress Testing Large Frontier Models on Multimodal Medical Benchmarks 

**Title (ZH)**: 准备就绪的错觉：在多模态医疗基准上对大前沿模型进行压力测试 

**Authors**: Yu Gu, Jingjing Fu, Xiaodong Liu, Jeya Maria Jose Valanarasu, Noel Codella, Reuben Tan, Qianchu Liu, Ying Jin, Sheng Zhang, Jinyu Wang, Rui Wang, Lei Song, Guanghui Qin, Naoto Usuyama, Cliff Wong, Cheng Hao, Hohin Lee, Praneeth Sanapathi, Sarah Hilado, Bian Jiang, Javier Alvarez-Valle, Mu Wei, Jianfeng Gao, Eric Horvitz, Matt Lungren, Hoifung Poon, Paul Vozila  

**Link**: [PDF](https://arxiv.org/pdf/2509.18234)  

**Abstract**: Large frontier models like GPT-5 now achieve top scores on medical benchmarks. But our stress tests tell a different story. Leading systems often guess correctly even when key inputs like images are removed, flip answers under trivial prompt changes, and fabricate convincing yet flawed reasoning. These aren't glitches; they expose how today's benchmarks reward test-taking tricks over medical understanding. We evaluate six flagship models across six widely used benchmarks and find that high leaderboard scores hide brittleness and shortcut learning. Through clinician-guided rubric evaluation, we show that benchmarks vary widely in what they truly measure yet are treated interchangeably, masking failure modes. We caution that medical benchmark scores do not directly reflect real-world readiness. If we want AI to earn trust in healthcare, we must demand more than leaderboard wins and must hold systems accountable for robustness, sound reasoning, and alignment with real medical demands. 

**Abstract (ZH)**: 大型前沿模型如GPT-5现在在医学基准测试中取得了顶尖成绩。但我们的压力测试却揭示了不同的故事。领先系统往往在关键输入（如图像）被移除时仍能正确作答，轻微提示变化下会逆转答案，并虚构看似合理却充满缺陷的推理。这不是错误，而是暴露了当前基准测试如何倾向于奖励测试技巧而非医学理解。我们评估了六种旗舰模型在六种广泛使用的基准测试中的表现，发现高排名成绩掩盖了模型的脆弱性和捷径学习。通过由临床医生指导的评分评价，我们展示了基准测试在真正衡量的内容上存在广泛差异，但这些差异却被视为等效，掩盖了失败模式。我们警告说，医学基准测试成绩并不能直接反映现实世界的准备情况。如果希望AI在医疗领域获得信任，我们必须不仅仅追求排行榜的胜利，还必须让系统为稳健性、合理的推理以及与实际医学需求的一致性负责。 

---
# An N-Plus-1 GPT Agency for Critical Solution of Mechanical Engineering Analysis Problems 

**Title (ZH)**: N+1 GPT机构及其在机械工程分析问题 critical 解决中的应用 

**Authors**: Anthony Patera, Rohan Abeyaratne  

**Link**: [PDF](https://arxiv.org/pdf/2509.18229)  

**Abstract**: Generative AI, and specifically GPT, can produce a remarkable solution to a mechanical engineering analysis problem - but also, on occasion, a flawed solution. For example, an elementary mechanics problem is solved flawlessly in one GPT instance and incorrectly in a subsequent GPT instance, with a success probability of only 85%. This unreliability renders "out-of-the-box" GPT unsuitable for deployment in education or engineering practice. We introduce an "N-Plus-1" GPT Agency for Initial (Low-Cost) Analysis of mechanical engineering Problem Statements. Agency first launches N instantiations of Agent Solve to yield N independent Proposed Problem Solution Realizations; Agency then invokes Agent Compare to summarize and compare the N Proposed Problem Solution Realizations and to provide a Recommended Problem Solution. We argue from Condorcet's Jury Theorem that, for a Problem Statement characterized by per-Solve success probability greater than 1/2 (and N sufficiently large), the Predominant (Agent Compare) Proposed Problem Solution will, with high probability, correspond to a Correct Proposed Problem Solution. Furthermore, Agent Compare can also incorporate aspects of Secondary (Agent Compare) Proposed Problem Solutions, in particular when the latter represent alternative Problem Statement interpretations - different Mathematical Models - or alternative Mathematical Solution Procedures. Comparisons to Grok Heavy, a commercial multi-agent model, show similarities in design and performance, but also important differences in emphasis: our Agency focuses on transparency and pedagogical value. 

**Abstract (ZH)**: 生成式AI，特别是在机械工程分析问题上的GPT，能够生成一个令人 remarkable 的解决方案，但也偶尔会产生一个不正确的解决方案。例如，一个基础的力学问题在一个GPT实例中被完美解决，而在后续的GPT实例中被错误解决，成功概率仅为85%。这种不可靠性使得“开箱即用”的GPT不适合在教育或工程实践中部署。我们提出了一个“N+1”GPT机构用于初始（低成本）分析机械工程问题陈述。该机构首先启动N个Agent Solve实例以生成N个独立的 Proposed Problem Solution 实现；然后调用Agent Compare来汇总和比较N个 Proposed Problem Solution 实现，并提供一个推荐的 Problem Solution。我们根据孔多塞悖论认为，对于具有每解决一次成功概率大于1/2（并且N足够大）的问题陈述，主导的（Agent Compare） Proposed Problem Solution 以高概率对应于一个正确的 Proposed Problem Solution。此外，Agent Compare还可以结合次要的（Agent Compare） Proposed Problem Solutions 的方面，尤其是当后者代表问题陈述的不同解释——不同的数学模型或不同的数学解题程序。与Grok Heavy这种商业多智能体模型相比，我们的机构更侧重于透明性和教学价值。 

---
# From "What to Eat?" to Perfect Recipe: ChefMind's Chain-of-Exploration for Ambiguous User Intent in Recipe Recommendation 

**Title (ZH)**: 从“吃什么？”到完美食谱：ChefMind的探索链路实现模糊用户意图的食谱推荐 

**Authors**: Yu Fu, Linyue Cai, Ruoyu Wu, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18226)  

**Abstract**: Personalized recipe recommendation faces challenges in handling fuzzy user intent, ensuring semantic accuracy, and providing sufficient detail coverage. We propose ChefMind, a hybrid architecture combining Chain of Exploration (CoE), Knowledge Graph (KG), Retrieval-Augmented Generation (RAG), and a Large Language Model (LLM). CoE refines ambiguous queries into structured conditions, KG offers semantic reasoning and interpretability, RAG supplements contextual culinary details, and LLM integrates outputs into coherent recommendations. We evaluate ChefMind on the Xiachufang dataset and manually annotated queries, comparing it with LLM-only, KG-only, and RAG-only baselines. Results show that ChefMind achieves superior performance in accuracy, relevance, completeness, and clarity, with an average score of 8.7 versus 6.4-6.7 for ablation models. Moreover, it reduces unprocessed queries to 1.6%, demonstrating robustness in handling fuzzy demands. 

**Abstract (ZH)**: 个性化菜谱推荐面临用户意图模糊、语义准确性保障和细节覆盖不足的挑战。我们提出ChefMind，一种结合链式探索(CoE)、知识图谱(KG)、检索增强生成(RAG)和大型语言模型(LLM)的混合架构。CoE将模糊查询精炼为结构化条件，KG提供语义推理和可解释性，RAG补充背景烹饪细节，而LLM将输出整合为连贯的推荐。我们使用XiaChufang数据集和手动标注查询对ChefMind进行评估，并将其与仅使用LLM、仅使用KG和仅使用RAG的基线进行比较。结果表明，ChefMind在准确度、相关性、完整性和清晰度方面表现出更优的性能，平均得分为8.7，而消融模型的得分在6.4-6.7之间。此外，它将未处理的查询减少到1.6%，展示了在处理模糊需求方面的稳健性。 

---
# Synthesizing Attitudes, Predicting Actions (SAPA): Behavioral Theory-Guided LLMs for Ridesourcing Mode Choice Modeling 

**Title (ZH)**: 基于行为理论指导的大规模语言模型合成态度、预测行为（SAPA）： ridesourcing 模式选择建模 

**Authors**: Mustafa Sameen, Xiaojian Zhang, Xilei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18181)  

**Abstract**: Accurate modeling of ridesourcing mode choices is essential for designing and implementing effective traffic management policies for reducing congestion, improving mobility, and allocating resources more efficiently. Existing models for predicting ridesourcing mode choices often suffer from limited predictive accuracy due to their inability to capture key psychological factors, and are further challenged by severe class imbalance, as ridesourcing trips comprise only a small fraction of individuals' daily travel. To address these limitations, this paper introduces the Synthesizing Attitudes, Predicting Actions (SAPA) framework, a hierarchical approach that uses Large Language Models (LLMs) to synthesize theory-grounded latent attitudes to predict ridesourcing choices. SAPA first uses an LLM to generate qualitative traveler personas from raw travel survey data and then trains a propensity-score model on demographic and behavioral features, enriched by those personas, to produce an individual-level score. Next, the LLM assigns quantitative scores to theory-driven latent variables (e.g., time and cost sensitivity), and a final classifier integrates the propensity score, latent-variable scores (with their interaction terms), and observable trip attributes to predict ridesourcing mode choice. Experiments on a large-scale, multi-year travel survey show that SAPA significantly outperforms state-of-the-art baselines, improving ridesourcing choice predictions by up to 75.9% in terms of PR-AUC on a held-out test set. This study provides a powerful tool for accurately predicting ridesourcing mode choices, and provides a methodology that is readily transferable to various applications. 

**Abstract (ZH)**: 准确建模共享出行模式选择对于设计和实施有效的交通管理政策以减少拥堵、提升 mobility 并更高效地分配资源至关重要。现有的共享出行模式选择预测模型往往因难以捕捉关键的心理因素而预测精度有限，并且受到严重类别不平衡的挑战，因为共享出行行程只占个人日常出行的一小部分。为解决这些问题，本文引入了综合态度、预测行为（SAPA）框架，这是一种层次化方法，使用大型语言模型（LLMs）合成基于理论的态度来预测共享出行选择。SAPA 首先使用 LLM 从原始旅行调查数据中生成定性的旅行者画像，然后使用这些画像丰富的人口统计和行为特征训练倾向评分模型，生成个体水平得分。接着，LLM 为理论驱动的潜在变量（例如时间和成本敏感性）分配定量评分，最终分类器将倾向评分、潜在变量评分（包括交互项）和可观察的行程特征结合起来预测共享出行模式选择。大规模、多年的旅行调查实验表明，SAPA 显著优于最先进的基准，相对于保留测试集上的 PR-AUC，共享出行选择预测提高了多达 75.9%。该研究提供了准确预测共享出行模式选择的有力工具，并提供了一种易于转移到各种应用的方法。 

---
# Large Language Models and Operations Research: A Structured Survey 

**Title (ZH)**: 大型语言模型与运筹学：一项结构化的综述 

**Authors**: Yang Wang, Kai Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18180)  

**Abstract**: Operations research (OR) provides fundamental methodologies for complex system decision-making, with established applications in transportation, supply chain management, and production scheduling. Traditional approaches, which depend on expert-based modeling and manual parameter adjustment, often face challenges in handling large-scale, dynamic, and multi-constraint problems. Recently, large language models (LLMs) have shown potential to address these limitations through semantic understanding, structured generation, and reasoning control. LLMs can translate natural language descriptions into mathematical models or executable code, generate heuristics, evolve algorithms, and directly tackle optimization tasks. This paper surveys recent progress on the integration of LLMs into OR, organizing methods into three main directions: automatic modeling, auxiliary optimization, and direct solving. It further reviews evaluation benchmarks and domain-specific applications, and summarizes key open issues such as unstable semantic-to-structure mapping, fragmented research progress, limited generalization, and insufficient evaluation systems. Finally, the survey outlines possible research avenues for advancing the role of LLMs in OR. 

**Abstract (ZH)**: 大规模语言模型在运筹学中的集成研究 

---
# SPADE: A Large Language Model Framework for Soil Moisture Pattern Recognition and Anomaly Detection in Precision Agriculture 

**Title (ZH)**: SPADE: 用于精准农业中土壤 Moisture 模式识别和异常检测的大规模语言模型框架 

**Authors**: Yeonju Lee, Rui Qi Chen, Joseph Oboamah, Po Nien Su, Wei-zhen Liang, Yeyin Shi, Lu Gan, Yongsheng Chen, Xin Qiao, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18123)  

**Abstract**: Accurate interpretation of soil moisture patterns is critical for irrigation scheduling and crop management, yet existing approaches for soil moisture time-series analysis either rely on threshold-based rules or data-hungry machine learning or deep learning models that are limited in adaptability and interpretability. In this study, we introduce SPADE (Soil moisture Pattern and Anomaly DEtection), an integrated framework that leverages large language models (LLMs) to jointly detect irrigation patterns and anomalies in soil moisture time-series data. SPADE utilizes ChatGPT-4.1 for its advanced reasoning and instruction-following capabilities, enabling zero-shot analysis without requiring task-specific annotation or fine-tuning. By converting time-series data into a textual representation and designing domain-informed prompt templates, SPADE identifies irrigation events, estimates net irrigation gains, detects, classifies anomalies, and produces structured, interpretable reports. Experiments were conducted on real-world soil moisture sensor data from commercial and experimental farms cultivating multiple crops across the United States. Results demonstrate that SPADE outperforms the existing method in anomaly detection, achieving higher recall and F1 scores and accurately classifying anomaly types. Furthermore, SPADE achieved high precision and recall in detecting irrigation events, indicating its strong capability to capture irrigation patterns accurately. SPADE's reports provide interpretability and usability of soil moisture analytics. This study highlights the potential of LLMs as scalable, adaptable tools for precision agriculture, which is capable of integrating qualitative knowledge and data-driven reasoning to produce actionable insights for accurate soil moisture monitoring and improved irrigation scheduling from soil moisture time-series data. 

**Abstract (ZH)**: 土壤水分模式和异常检测的准确解析对于灌溉调度和作物管理至关重要，现有土壤水分时间序列分析方法要么依赖于阈值规则，要么依赖于数据需求大的机器学习或深度学习模型，这些模型在适应性和可解释性方面有限。在本研究中，我们介绍了SPADE（土壤水分模式和异常检测）框架，该框架利用大型语言模型（LLMs）联合检测土壤水分时间序列数据中的灌溉模式和异常。SPADE利用ChatGPT-4.1的高级推理和指令遵循能力，实现零样本分析，无需特定任务的注释或微调。通过将时间序列数据转换为文本表示并设计领域指导的提示模板，SPADE识别灌溉事件、估算净灌溉增益、检测和分类异常，并生成结构化、可解释的报告。实验在多个作物的商业和实验农场的真实土壤水分传感器数据上进行。结果显示，SPADE在异常检测中优于现有方法，达到更高的召回率和F1分数，并准确分类异常类型。此外，SPADE在检测灌溉事件方面也表现出高精度和召回率，表明其能够准确捕捉灌溉模式。SPADE的报告提供土壤水分分析的可解释性和实用性。本研究突显了LLMs作为可扩展、适应性强的工具在精确农业中的潜力，能够集成定性知识和数据驱动推理，从土壤水分时间序列数据中产生有助于准确土壤水分监测和优化灌溉调度的可操作见解。 

---
# A Cost-Benefit Analysis of On-Premise Large Language Model Deployment: Breaking Even with Commercial LLM Services 

**Title (ZH)**: 基于 premises 的大型语言模型部署的成本效益分析：与商用 LLM 服务持平 

**Authors**: Guanzhong Pan, Haibo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18101)  

**Abstract**: Large language models (LLMs) are becoming increasingly widespread. Organizations that want to use AI for productivity now face an important decision. They can subscribe to commercial LLM services or deploy models on their own infrastructure. Cloud services from providers such as OpenAI, Anthropic, and Google are attractive because they provide easy access to state-of-the-art models and are easy to scale. However, concerns about data privacy, the difficulty of switching service providers, and long-term operating costs have driven interest in local deployment of open-source models. This paper presents a cost-benefit analysis framework to help organizations determine when on-premise LLM deployment becomes economically viable compared to commercial subscription services. We consider the hardware requirements, operational expenses, and performance benchmarks of the latest open-source models, including Qwen, Llama, Mistral, and etc. Then we compare the total cost of deploying these models locally with the major cloud providers subscription fee. Our findings provide an estimated breakeven point based on usage levels and performance needs. These results give organizations a practical framework for planning their LLM strategies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）日益普及。希望利用AI提高生产力的组织现在面临着一个重要的决策：他们是订阅商业LLM服务还是在自己的基础设施上部署模型。来自OpenAI、Anthropic和Google等提供商的云服务具有吸引力，因为它们提供了访问最新模型的便捷途径并且易于扩展。然而，关于数据隐私的担忧、切换服务提供商的难度以及长期运营成本等因素促使人们更加关注开源模型的本地部署。本文提出了一种成本效益分析框架，帮助组织确定在何种情况下本地部署LLM相较于商业订阅服务更具经济性。我们考虑了最新开源模型（包括Qwen、Llama、Mistral等）的硬件需求、运营支出和性能基准，然后将这些模型本地部署的总成本与主要云提供商的订阅费用进行了比较。我们的研究结果提供了基于使用水平和性能需求的估算盈亏平衡点。这些结果为组织制定了规划其LLM策略的实际框架。 

---
# Reinforcement Learning on Pre-Training Data 

**Title (ZH)**: 预训练数据上的强化学习 

**Authors**: Siheng Li, Kejiao Li, Zenan Xu, Guanhua Huang, Evander Yang, Kun Li, Haoyuan Wu, Jiajia Wu, Zihao Zheng, Chenchen Zhang, Kun Shi, Kyrierl Deng, Qi Yi, Ruibin Xiong, Tingqiang Xu, Yuhao Jiang, Jianfeng Yan, Yuyuan Zeng, Guanghui Xu, Jinbao Xue, Zhijiang Xu, Zheng Fang, Shuai Li, Qibin Liu, Xiaoxue Li, Zhuoyu Li, Yangyu Tao, Fei Gao, Cheng Jiang, Bo Chao Wang, Kai Liu, Jianchen Zhu, Wai Lam, Wayyt Wang, Bo Zhou, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19249)  

**Abstract**: The growing disparity between the exponential scaling of computational resources and the finite growth of high-quality text data now constrains conventional scaling approaches for large language models (LLMs). To address this challenge, we introduce Reinforcement Learning on Pre-Training data (RLPT), a new training-time scaling paradigm for optimizing LLMs. In contrast to prior approaches that scale training primarily through supervised learning, RLPT enables the policy to autonomously explore meaningful trajectories to learn from pre-training data and improve its capability through reinforcement learning (RL). While existing RL strategies such as reinforcement learning from human feedback (RLHF) and reinforcement learning with verifiable rewards (RLVR) rely on human annotation for reward construction, RLPT eliminates this dependency by deriving reward signals directly from pre-training data. Specifically, it adopts a next-segment reasoning objective, rewarding the policy for accurately predicting subsequent text segments conditioned on the preceding context. This formulation allows RL to be scaled on pre-training data, encouraging the exploration of richer trajectories across broader contexts and thereby fostering more generalizable reasoning skills. Extensive experiments on both general-domain and mathematical reasoning benchmarks across multiple models validate the effectiveness of RLPT. For example, when applied to Qwen3-4B-Base, RLPT yields absolute improvements of $3.0$, $5.1$, $8.1$, $6.0$, $6.6$, and $5.3$ on MMLU, MMLU-Pro, GPQA-Diamond, KOR-Bench, AIME24, and AIME25, respectively. The results further demonstrate favorable scaling behavior, suggesting strong potential for continued gains with more compute. In addition, RLPT provides a solid foundation, extending the reasoning boundaries of LLMs and enhancing RLVR performance. 

**Abstract (ZH)**: 基于预训练数据的强化学习训练（RLPT）：大型语言模型的新型扩展范式 

---
# Systematic Comparative Analysis of Large Pretrained Language Models on Contextualized Medication Event Extraction 

**Title (ZH)**: 大型预训练语言模型在上下文化医疗事件抽取中的系统比较分析 

**Authors**: Tariq Abdul-Quddoos, Xishuang Dong, Lijun Qian  

**Link**: [PDF](https://arxiv.org/pdf/2509.19224)  

**Abstract**: Attention-based models have become the leading approach in modeling medical language for Natural Language Processing (NLP) in clinical notes. These models outperform traditional techniques by effectively capturing contextual rep- resentations of language. In this research a comparative analysis is done amongst pre- trained attention based models namely Bert Base, BioBert, two variations of Bio+Clinical Bert, RoBerta, and Clinical Long- former on task related to Electronic Health Record (EHR) information extraction. The tasks from Track 1 of Harvard Medical School's 2022 National Clinical NLP Challenges (n2c2) are considered for this comparison, with the Contextualized Medication Event Dataset (CMED) given for these task. CMED is a dataset of unstructured EHRs and annotated notes that contain task relevant information about the EHRs. The goal of the challenge is to develop effective solutions for extracting contextual information related to patient medication events from EHRs using data driven methods. Each pre-trained model is fine-tuned and applied on CMED to perform medication extraction, medical event detection, and multi-dimensional medication event context classification. Pro- cessing methods are also detailed for breaking down EHRs for compatibility with the applied models. Performance analysis has been carried out using a script based on constructing medical terms from the evaluation portion of CMED with metrics including recall, precision, and F1-Score. The results demonstrate that models pre-trained on clinical data are more effective in detecting medication and medication events, but Bert Base, pre- trained on general domain data showed to be the most effective for classifying the context of events related to medications. 

**Abstract (ZH)**: 基于注意力的模型已在临床笔记中的医学语言建模中成为自然语言处理（NLP）的主导方法。这些模型通过有效捕捉语言的上下文表示超越了传统的技术。在本研究中，对预训练的基于注意力的模型，即Bert Base、BioBert、两种Bio+Clinical Bert变体、RoBerta和Clinical Longformer在电子健康记录（EHR）信息提取任务上的表现进行了对比分析。这些比较基于哈佛医学院2022年国家临床NLP挑战（n2c2）第一赛道的任务，使用的是Contextualized Medication Event Dataset（CMED）数据集。CMED是一个包含未结构化EHR和标注笔记的数据集，这些笔记包含与EHR相关的任务相关信息。挑战的目标是利用数据驱动的方法开发有效的解决方案，从EHR中提取与患者用药事件相关的上下文信息。每种预训练模型都被微调并应用于CMED以执行药物提取、医疗事件检测和多维度用药事件上下文分类。还详细描述了处理方法，以确保EHR与所应用的模型兼容。性能分析使用了基于CMED评估部分构建医学术语的脚本，指标包括召回率、精确率和F1分数。结果表明，基于临床数据预训练的模型在检测药物和用药事件方面更为有效，但基于通用领域数据预训练的Bert Base模型在分类与药物相关的事件上下文方面最为有效。 

---
# Steering Multimodal Large Language Models Decoding for Context-Aware Safety 

**Title (ZH)**: 面向上下文感知安全的多模态大型语言模型解码引导 

**Authors**: Zheyuan Liu, Zhangchen Xu, Guangyao Dou, Xiangchi Yuan, Zhaoxuan Tan, Radha Poovendran, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.19212)  

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in real-world applications, yet their ability to make context-aware safety decisions remains limited. Existing methods often fail to balance oversensitivity (unjustified refusals of benign queries) and undersensitivity (missed detection of visually grounded risks), leaving a persistent gap in safety alignment. To address this issue, we introduce Safety-aware Contrastive Decoding (SafeCoDe), a lightweight and model-agnostic decoding framework that dynamically adjusts token generation based on multimodal context. SafeCoDe operates in two stages: (1) a contrastive decoding mechanism that highlights tokens sensitive to visual context by contrasting real and Gaussian-noised images, and (2) a global-aware token modulation strategy that integrates scene-level reasoning with token-level adjustment to adapt refusals according to the predicted safety verdict. Extensive experiments across diverse MLLM architectures and safety benchmarks, covering undersensitivity, oversensitivity, and general safety evaluations, show that SafeCoDe consistently improves context-sensitive refusal behaviors while preserving model helpfulness. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在实际应用中日益增多，但其在做出基于情境的安全决策方面的能力仍有限。现有方法往往难以在过度敏感（对良性查询进行不必要的拒绝）和欠敏感（未能检测到基于视觉的风险）之间取得平衡，留下了持续的安全对齐缺口。为解决这一问题，我们引入了安全感知对比解码（SafeCoDe），这是一种轻量级且模型无关的解码框架，可以根据多模态上下文动态调整令牌生成。SafeCoDe 分为两个阶段：（1）对比解码机制，通过对比真实图像和高斯噪声图像来突出对视觉上下文敏感的令牌；（2）全局感知的令牌调制策略，将场景级推理与令牌级调整相结合，根据预测的安全判决调整拒绝行为。覆盖欠敏感、过度敏感和一般安全评估的广泛实验表明，SafeCoDe 在维持模型有用性的同时，一致地提升了基于情境的拒绝行为。 

---
# Soft Tokens, Hard Truths 

**Title (ZH)**: 软令牌，硬真相 

**Authors**: Natasha Butt, Ariel Kwiatkowski, Ismail Labiad, Julia Kempe, Yann Ollivier  

**Link**: [PDF](https://arxiv.org/pdf/2509.19170)  

**Abstract**: The use of continuous instead of discrete tokens during the Chain-of-Thought (CoT) phase of reasoning LLMs has garnered attention recently, based on the intuition that a continuous mixture of discrete tokens could simulate a superposition of several reasoning paths simultaneously. Theoretical results have formally proven that continuous tokens have much greater expressivity and can solve specific problems more efficiently. However, practical use of continuous tokens has been limited by strong training difficulties: previous works either just use continuous tokens at inference time on a pre-trained discrete-token model, or must distill the continuous CoT from ground-truth discrete CoTs and face computational costs that limit the CoT to very few tokens.
This is the first work introducing a scalable method to learn continuous CoTs via reinforcement learning (RL), without distilling from reference discrete CoTs. We use "soft" tokens: mixtures of tokens together with noise on the input embedding to provide RL exploration. Computational overhead is minimal, enabling us to learn continuous CoTs with hundreds of tokens. On math reasoning benchmarks with Llama and Qwen models up to 8B, training with continuous CoTs match discrete-token CoTs for pass@1 and surpass them for pass@32, showing greater CoT diversity. In systematic comparisons, the best-performing scenario is to train with continuous CoT tokens then use discrete tokens for inference, meaning the "soft" models can be deployed in a standard way. Finally, we show continuous CoT RL training better preserves the predictions of the base model on out-of-domain tasks, thus providing a softer touch to the base model. 

**Abstract (ZH)**: 连续-token代替离散-token在链式思考（CoT）推理阶段的应用：基于强化学习的方法 

---
# On the Soundness and Consistency of LLM Agents for Executing Test Cases Written in Natural Language 

**Title (ZH)**: 关于使用自然语言编写的测试用例执行中LLM代理的正确性和一致性研究 

**Authors**: Sébastien Salva, Redha Taguelmimt  

**Link**: [PDF](https://arxiv.org/pdf/2509.19136)  

**Abstract**: The use of natural language (NL) test cases for validating graphical user interface (GUI) applications is emerging as a promising direction to manually written executable test scripts, which are costly to develop and difficult to maintain. Recent advances in large language models (LLMs) have opened the possibility of the direct execution of NL test cases by LLM agents. This paper investigates this direction, focusing on the impact on NL test case unsoundness and on test case execution consistency. NL test cases are inherently unsound, as they may yield false failures due to ambiguous instructions or unpredictable agent behaviour. Furthermore, repeated executions of the same NL test case may lead to inconsistent outcomes, undermining test reliability. To address these challenges, we propose an algorithm for executing NL test cases with guardrail mechanisms and specialised agents that dynamically verify the correct execution of each test step. We introduce measures to evaluate the capabilities of LLMs in test execution and one measure to quantify execution consistency. We propose a definition of weak unsoundness to characterise contexts in which NL test case execution remains acceptable, with respect to the industrial quality levels Six Sigma. Our experimental evaluation with eight publicly available LLMs, ranging from 3B to 70B parameters, demonstrates both the potential and current limitations of current LLM agents for GUI testing. Our experiments show that Meta Llama 3.1 70B demonstrates acceptable capabilities in NL test case execution with high execution consistency (above the level 3-sigma). We provide prototype tools, test suites, and results. 

**Abstract (ZH)**: 自然语言测试用例在验证图形用户界面应用中的应用：探索大型语言模型直接执行测试用例的潜力与挑战 

---
# Analysis on distribution and clustering of weight 

**Title (ZH)**: 权重分布及聚类分析 

**Authors**: Chunming Ye, Wenquan Tian, Yalan Gao, Songzhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.19122)  

**Abstract**: The study on architecture and parameter characteristics remains the hot topic in the research of large language models. In this paper we concern with the characteristics of weight which are used to analyze the correlations and differences between models. Two kinds of vectors-standard deviation vector and clustering vector-are proposed to describe features of models. In the first case, the weights are assumed to follow normal distribution. The standard deviation values of projection matrices are normalized to form Standard-Deviation Vector, representing the distribution characteristics of models. In the second case, the singular values from each weight projection matrix are extracted and grouped by K-Means algorithm. The grouped data with the same type matrix are combined as Clustering Vector to represent the correlation characteristics of models' weights. The study reveals that these two vectors can effectively distinguish between different models and clearly show the similarities among models of the same family. Moreover, after conducting LoRA fine-tuning with different datasets and models, it is found that the distribution of weights represented by standard deviation vector is directly influenced by the dataset, but the correlations between different weights represented by clustering vector remain unaffected and maintain a high consistency with the pre-trained model. 

**Abstract (ZH)**: 大型语言模型研究中关于架构和参数特性研究仍是热点。本文关注用于分析和比较模型特征的权重特性。提出了两种向量——标准差向量和聚类向量——来描述模型特征。在第一种情况下，假设权重服从正态分布，标准化投影矩阵的标准差构成标准差向量，代表模型的分布特性。在第二种情况下，从每个权重投影矩阵中提取奇异值，并使用K-Means算法进行分组。具有相同类型矩阵的分组数据被组合为聚类向量，以表示模型权重的相关特性。研究发现这两种向量能够有效地区分不同模型，并清晰地展示相同家族模型之间的相似性。此外，在使用不同数据集和模型进行LoRA微调后发现，由标准差向量表示的权重分布直接受数据集影响，而由聚类向量表示的不同权重之间的相关性保持不变，与预训练模型保持高度一致性。 

---
# Pathways of Thoughts: Multi-Directional Thinking for Long-form Personalized Question Answering 

**Title (ZH)**: 思绪路径：多向思考在长格式个性化问答中的应用 

**Authors**: Alireza Salemi, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Zhuowan Li, Spurthi Amba Hombaiah, Weize Kong, Tao Chen, Hamed Zamani, Michael Bendersky  

**Link**: [PDF](https://arxiv.org/pdf/2509.19094)  

**Abstract**: Personalization is essential for adapting question answering (QA) systems to user-specific information needs, thereby improving both accuracy and user satisfaction. However, personalized QA remains relatively underexplored due to challenges such as inferring preferences from long, noisy, and implicit contexts, and generating responses that are simultaneously correct, contextually appropriate, and aligned with user expectations and background knowledge. To address these challenges, we propose Pathways of Thoughts (PoT), an inference-stage method that applies to any large language model (LLM) without requiring task-specific fine-tuning. The approach models the reasoning of an LLM as an iterative decision process, where the model dynamically selects among cognitive operations such as reasoning, revision, personalization, and clarification. This enables exploration of multiple reasoning trajectories, producing diverse candidate responses that capture different perspectives. PoT then aggregates and reweights these candidates according to inferred user preferences, yielding a final personalized response that benefits from the complementary strengths of diverse reasoning paths. Experiments on the LaMP-QA benchmark for personalized QA show that PoT consistently outperforms competitive baselines, achieving up to a 13.1% relative improvement. Human evaluation corroborates these results, with annotators preferring outputs from PoT in 66% of cases and reporting ties in only 15% of cases. 

**Abstract (ZH)**: 个性化对于适应用户特定的信息需求、提高问答系统准确性和用户满意度至关重要。然而，由于从长、嘈杂和隐含的上下文中推断偏好、生成同时正确、上下文适宜且与用户期望和背景知识相符的回答的挑战，个性化问答相对未被充分探索。为应对这些挑战，我们提出了一种名为Pathways of Thoughts (PoT)的方法，这是一种适用于任何大型语言模型（LLM）的推理阶段方法，无需特定任务的微调。该方法将大型语言模型的推理视为一个迭代决策过程，模型动态选择认知操作，如推理、修订、个性化和澄清。这使得可以探索多个推理轨迹，生成多样化的候选回答，捕捉不同的视角。PoT 然后根据推断出的用户偏好聚合和重新权重这些候选回答，产生一种最终的个性化回答，融合了多种推理路径的互补优势。在个人化问答基准LaMP-QA上的实验表明，PoT 一贯优于竞争性基线，相对改进率高达13.1%。人类评估进一步验证了这些结果，注释者在66%的情况下更偏好PoT的输出，在15%的情况下报告平局。 

---
# A Mega-Study of Digital Twins Reveals Strengths, Weaknesses and Opportunities for Further Improvement 

**Title (ZH)**: 大规模数字孪生研究揭示了其优势、劣势及进一步改进的机会 

**Authors**: Tiany Peng, George Gui, Daniel J. Merlau, Grace Jiarui Fan, Malek Ben Sliman, Melanie Brucks, Eric J. Johnson, Vicki Morwitz, Abdullah Althenayyan, Silvia Bellezza, Dante Donati, Hortense Fong, Elizabeth Friedman, Ariana Guevara, Mohamed Hussein, Kinshuk Jerath, Bruce Kogut, Kristen Lane, Hannah Li, Patryk Perkowski, Oded Netzer, Olivier Toubia  

**Link**: [PDF](https://arxiv.org/pdf/2509.19088)  

**Abstract**: Do "digital twins" capture individual responses in surveys and experiments? We run 19 pre-registered studies on a national U.S. panel and their LLM-powered digital twins (constructed based on previously-collected extensive individual-level data) and compare twin and human answers across 164 outcomes. The correlation between twin and human answers is modest (approximately 0.2 on average) and twin responses are less variable than human responses. While constructing digital twins based on rich individual-level data improves our ability to capture heterogeneity across participants and predict relative differences between them, it does not substantially improve our ability to predict the exact answers given by specific participants or enhance predictions of population means. Twin performance varies by domain and is higher among more educated, higher-income, and ideologically moderate participants. These results suggest current digital twins can capture some degree of relative differences but are unreliable for individual-level predictions and sample mean and variance estimation, underscoring the need for careful validation before use. Our data and code are publicly available for researchers and practitioners interested in optimizing digital twin pipelines. 

**Abstract (ZH)**: 数字孪生体能否捕捉调查和实验中的个体反应？我们针对一个国家级美国面板数据及其基于之前收集的个体层面数据构建的LLM驱动数字孪生体进行了19项事先注册的研究，并在164个结果上比较了孪生体和人类的答复。孪生体和人类答复之间的相关性适度（平均约为0.2），且孪生体的回答比人类的回答更具一致性。虽然基于丰富个体层面数据构建数字孪生体能够提高我们捕捉参与者异质性和预测他们相对差异的能力，但并未显著提升预测特定参与者确切答案的能力或改善对总体均值的预测。数字孪生体的表现因领域而异，在受过更好教育、收入更高以及政治观点更为中立的参与者中表现更佳。这些结果表明，当前的数字孪生体能够捕捉一定的相对差异，但不适用于个体水平预测和样本均值及方差估计，在使用之前需要谨慎验证。我们的数据和代码已对外公开，供对优化数字孪生体管道感兴趣的研究人员和从业人员使用。 

---
# Diversity Boosts AI-Generated Text Detection 

**Title (ZH)**: 多样性增强AI生成文本检测 

**Authors**: Advik Raj Basani, Pin-Yu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.18880)  

**Abstract**: Detecting AI-generated text is an increasing necessity to combat misuse of LLMs in education, business compliance, journalism, and social media, where synthetic fluency can mask misinformation or deception. While prior detectors often rely on token-level likelihoods or opaque black-box classifiers, these approaches struggle against high-quality generations and offer little interpretability. In this work, we propose DivEye, a novel detection framework that captures how unpredictability fluctuates across a text using surprisal-based features. Motivated by the observation that human-authored text exhibits richer variability in lexical and structural unpredictability than LLM outputs, DivEye captures this signal through a set of interpretable statistical features. Our method outperforms existing zero-shot detectors by up to 33.2% and achieves competitive performance with fine-tuned baselines across multiple benchmarks. DivEye is robust to paraphrasing and adversarial attacks, generalizes well across domains and models, and improves the performance of existing detectors by up to 18.7% when used as an auxiliary signal. Beyond detection, DivEye provides interpretable insights into why a text is flagged, pointing to rhythmic unpredictability as a powerful and underexplored signal for LLM detection. 

**Abstract (ZH)**: 检测AI生成的文字成为一个日益必要的需求，以应对教育、商业合规、新闻界和社会媒体中大规模语言模型的滥用问题，其中合成流利性可以掩盖错误信息或欺骗行为。虽然以往的方法常常依赖于令牌级别的概率或不透明的黑盒分类器，但这些方法难以应对高质量的生成文本，并且缺乏解释性。在这项工作中，我们提出DivEye，一种新颖的检测框架，使用基于 surprisal 的特征捕捉文本中不可预测性的波动。受人类撰写的文本比大规模语言模型输出展现出更丰富的词汇和结构不可预测性这一观察的启发，DivEye 通过一组可解释的统计特征捕捉这一信号。我们的方法在零样本检测器上优于现有方法最多 33.2%，并在多个基准上达到与微调基线相当的性能。DivEye 在重述和对抗攻击中表现出 robust，并能很好地跨领域和模型泛化，将其作为辅助信号使用时，可以提高现有检测器多达 18.7% 的性能。此外，DivEye 提供可解释的洞察，解释为什么文本被标记，指出节奏不可预测性是大规模语言模型检测中一个强大且未充分探索的信号。 

---
# When Ads Become Profiles: Large-Scale Audit of Algorithmic Biases and LLM Profiling Risks 

**Title (ZH)**: 当广告变成画像：大规模审计算法偏见和大语言模型画像风险 

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2509.18874)  

**Abstract**: Automated ad targeting on social media is opaque, creating risks of exploitation and invisibility to external scrutiny. Users may be steered toward harmful content while independent auditing of these processes remains blocked. Large Language Models (LLMs) raise a new concern: the potential to reverse-engineer sensitive user attributes from exposure alone. We introduce a multi-stage auditing framework to investigate these risks. First, a large-scale audit of over 435,000 ad impressions delivered to 891 Australian Facebook users reveals algorithmic biases, including disproportionate Gambling and Politics ads shown to socioeconomically vulnerable and politically aligned groups. Second, a multimodal LLM can reconstruct users' demographic profiles from ad streams, outperforming census-based baselines and matching or exceeding human performance. Our results provide the first empirical evidence that ad streams constitute rich digital footprints for public AI inference, highlighting urgent privacy risks and the need for content-level auditing and governance. 

**Abstract (ZH)**: 社交媒体上的自动化广告定向是不透明的，这创造了对外部审查的滥用和 invisibility 的风险。用户可能会被引导观看有害内容，而独立审计这些过程仍然受阻。大规模语言模型 (LLMs) 引出一个新担忧：仅从曝光中逆向工程敏感用户属性的可能性。我们介绍一个多阶段审计框架以调查这些风险。首先，对交付给891名澳大利亚Facebook用户的435,000多条广告印象进行大规模审计，揭示了算法偏见，包括不成比例地向社会经济脆弱群体和政治对齐群体展示赌博和政治广告。其次，多模态LLM可以从广告流中重建用户的 demographic 背景，优于基于人口普查的基准，并达到或超过人类性能。我们的结果提供了第一个实证证据，表明广告流构成了丰富的数字足迹，用于公共AI推理，突显出迫切的隐私风险和内容级别的审计与治理需求。 

---
# NGRPO: Negative-enhanced Group Relative Policy Optimization 

**Title (ZH)**: NGRPO：负增强组相对策略优化 

**Authors**: Gongrui Nan, Siye Chen, Jing Huang, Mengyu Lu, Dexun Wang, Chunmei Xie, Weiqi Xiong, Xianzhou Zeng, Qixuan Zhou, Yadong Li, Xingzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18851)  

**Abstract**: RLVR has enhanced the reasoning capabilities of Large Language Models (LLMs) across various tasks. However, GRPO, a representative RLVR algorithm, suffers from a critical limitation: when all responses within a group are either entirely correct or entirely incorrect, the model fails to learn from these homogeneous responses. This is particularly problematic for homogeneously incorrect groups, where GRPO's advantage function yields a value of zero, leading to null gradients and the loss of valuable learning signals. To overcome this issue, we propose NGRPO (Negative-enhanced Group Relative Policy Optimization), an algorithm designed to convert homogeneous errors into robust learning signals. First, NGRPO introduces Advantage Calibration. This mechanism hypothesizes the existence of a virtual maximum-reward sample during advantage calculation, thereby altering the mean and variance of rewards within a group and ensuring that the advantages for homogeneously incorrect samples are no longer zero. Second, NGRPO employs Asymmetric Clipping, which relaxes the update magnitude for positive samples while imposing stricter constraints on that of negative samples. This serves to stabilize the exploration pressure introduced by the advantage calibration. Our experiments on Qwen2.5-Math-7B demonstrate that NGRPO significantly outperforms baselines such as PPO, GRPO, DAPO, and PSR-NSR on mathematical benchmarks including MATH500, AMC23, and AIME2025. These results validate NGRPO's ability to learn from homogeneous errors, leading to stable and substantial improvements in mathematical reasoning. Our code is available at this https URL. 

**Abstract (ZH)**: NGRPO：通过负增强组相对策略优化将同质错误转化为稳健的学习信号 

---
# Failure Makes the Agent Stronger: Enhancing Accuracy through Structured Reflection for Reliable Tool Interactions 

**Title (ZH)**: 失败让代理更强大：通过结构化反思提升准确性的可靠工具交互 

**Authors**: Junhao Su, Yuanliang Wan, Junwei Yang, Hengyu Shi, Tianyang Han, Junfeng Luo, Yurui Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18847)  

**Abstract**: Tool-augmented large language models (LLMs) are usually trained with supervised imitation or coarse-grained reinforcement learning that optimizes single tool calls. Current self-reflection practices rely on heuristic prompts or one-way reasoning: the model is urged to 'think more' instead of learning error diagnosis and repair. This is fragile in multi-turn interactions; after a failure the model often repeats the same mistake. We propose structured reflection, which turns the path from error to repair into an explicit, controllable, and trainable action. The agent produces a short yet precise reflection: it diagnoses the failure using evidence from the previous step and then proposes a correct, executable follow-up call. For training we combine DAPO and GSPO objectives with a reward scheme tailored to tool use, optimizing the stepwise strategy Reflect, then Call, then Final. To evaluate, we introduce Tool-Reflection-Bench, a lightweight benchmark that programmatically checks structural validity, executability, parameter correctness, and result consistency. Tasks are built as mini trajectories of erroneous call, reflection, and corrected call, with disjoint train and test splits. Experiments on BFCL v3 and Tool-Reflection-Bench show large gains in multi-turn tool-call success and error recovery, and a reduction of redundant calls. These results indicate that making reflection explicit and optimizing it directly improves the reliability of tool interaction and offers a reproducible path for agents to learn from failure. 

**Abstract (ZH)**: 工具增强的大语言模型（LLMs）通常通过监督模仿或粗粒度强化学习进行训练，优化单一工具调用。当前的自省实践依赖启发式提示或单向推理：模型被敦促“思考更多”，而不是学习错误诊断和修复。在多轮交互中这是脆弱的；在失败后，模型通常会重复同样的错误。我们提出了结构化自省，将从错误到修复的过程转化为显式的、可控的和可训练的操作。代理生成简短而精确的自省：它使用上一步的证据进行故障诊断，然后提出一个正确且可执行的后续调用。在训练中，我们将DAPO和GSPO目标与针对工具使用定制的奖励方案结合，优化逐步策略“先反省，再调用，再最终确认”。为了评估，我们引入了Tool-Reflection-Bench，这是一个轻量级基准，编程检查结构有效性、可执行性、参数正确性和结果一致性。任务构建为包含错误调用、自省和修正调用的小轨迹，并具有分离的训练集和测试集。在BFCL v3和Tool-Reflection-Bench上的实验显示，在多轮工具调用成功和错误恢复方面有显著提升，并减少了冗余调用。这些结果表明，使自省明确并直接优化它可以提高工具交互的可靠性，并为代理从失败中学习提供可复制的路径。 

---
# AECBench: A Hierarchical Benchmark for Knowledge Evaluation of Large Language Models in the AEC Field 

**Title (ZH)**: AECBench: A 分层基准用于评估建筑、工程和施工领域大型语言模型的知识水平 

**Authors**: Chen Liang, Zhaoqi Huang, Haofen Wang, Fu Chai, Chunying Yu, Huanhuan Wei, Zhengjie Liu, Yanpeng Li, Hongjun Wang, Ruifeng Luo, Xianzhong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18776)  

**Abstract**: Large language models (LLMs), as a novel information technology, are seeing increasing adoption in the Architecture, Engineering, and Construction (AEC) field. They have shown their potential to streamline processes throughout the building lifecycle. However, the robustness and reliability of LLMs in such a specialized and safety-critical domain remain to be evaluated. To address this challenge, this paper establishes AECBench, a comprehensive benchmark designed to quantify the strengths and limitations of current LLMs in the AEC domain. The benchmark defines 23 representative tasks within a five-level cognition-oriented evaluation framework encompassing Knowledge Memorization, Understanding, Reasoning, Calculation, and Application. These tasks were derived from authentic AEC practice, with scope ranging from codes retrieval to specialized documents generation. Subsequently, a 4,800-question dataset encompassing diverse formats, including open-ended questions, was crafted primarily by engineers and validated through a two-round expert review. Furthermore, an LLM-as-a-Judge approach was introduced to provide a scalable and consistent methodology for evaluating complex, long-form responses leveraging expert-derived rubrics. Through the evaluation of nine LLMs, a clear performance decline across five cognitive levels was revealed. Despite demonstrating proficiency in foundational tasks at the Knowledge Memorization and Understanding levels, the models showed significant performance deficits, particularly in interpreting knowledge from tables in building codes, executing complex reasoning and calculation, and generating domain-specific documents. Consequently, this study lays the groundwork for future research and development aimed at the robust and reliable integration of LLMs into safety-critical engineering practices. 

**Abstract (ZH)**: 大型语言模型（LLMs）在建筑、工程和施工（AEC）领域的综合基准：评估当前LLMs在AEC领域的优势与限制 

---
# When Long Helps Short: How Context Length in Supervised Fine-tuning Affects Behavior of Large Language Models 

**Title (ZH)**: 长上下文帮助短文本：监督微调中的上下文长度对大型语言模型行为的影响 

**Authors**: Yingming Zheng, Hanqi Li, Kai Yu, Lu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.18762)  

**Abstract**: Large language models (LLMs) have achieved impressive performance across natural language processing (NLP) tasks. As real-world applications increasingly demand longer context windows, continued pretraining and supervised fine-tuning (SFT) on long-context data has become a common approach. While the effects of data length in continued pretraining have been extensively studied, their implications for SFT remain unclear. In this work, we systematically investigate how SFT data length influences LLM behavior on short-context tasks. Counterintuitively, we find that long-context SFT improves short-context performance, contrary to the commonly observed degradation from long-context pretraining. To uncover the underlying mechanisms of this phenomenon, we first decouple and analyze two key components, Multi-Head Attention (MHA) and Feed-Forward Network (FFN), and show that both independently benefit from long-context SFT. We further study their interaction and reveal a knowledge preference bias: long-context SFT promotes contextual knowledge, while short-context SFT favors parametric knowledge, making exclusive reliance on long-context SFT suboptimal. Finally, we demonstrate that hybrid training mitigates this bias, offering explainable guidance for fine-tuning LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理（NLP）任务中取得了令人印象深刻的表现。随着实际应用对更长上下文窗口的需求日益增加，对长上下文数据的持续预训练和监督微调（SFT）已经成为一种常见方法。尽管持续预训练中数据长度的影响已被广泛研究，但其对监督微调的影响仍不清楚。在本工作中，我们系统地探讨了监督微调数据长度如何影响LLM在短上下文任务中的行为。令人意外的是，我们发现长上下文微调能改善短上下文性能，这与从长上下文预训练中常见的性能下降现象相反。为了揭示这一现象背后的机制，我们首先将多头注意力（MHA）和前向网络（FFN）这两个关键组件分离并进行分析，表明两者均能从长上下文微调中受益。进一步地，我们研究了它们的相互作用，揭示了一种知识偏好偏差：长上下文微调促进基于上下文的知识，而短上下文微调偏向基于参数的知识，这使得单纯依赖长上下文微调是不理想的。最后，我们证明了混合训练能够缓解这种偏差，为微调LLM提供了可解释的指导。 

---
# COLT: Enhancing Video Large Language Models with Continual Tool Usage 

**Title (ZH)**: COLT: 通过持续工具使用增强视频大型语言模型 

**Authors**: Yuyang Liu, Xinyuan Shi, Bang Yang, Peilin Zhou, Jiahua Dong, Long Chen, Ian Reid, Xiaondan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18754)  

**Abstract**: The success of Large Language Models (LLMs) has significantly propelled the research of video understanding. To harvest the benefits of well-trained expert models (i.e., tools), video LLMs prioritize the exploration of tool usage capabilities. Existing methods either prompt closed-source LLMs or employ the instruction tuning paradigm for tool-use fine-tuning. These methods, however, assume an established repository of fixed tools and struggle to generalize to real-world environments where tool data is perpetually evolving and streaming in. To this end, we propose to enhance open-source video LLMs with COntinuaL Tool usage (termed COLT), which automatically acquires tool-use ability in a successive tool stream without suffering 'catastrophic forgetting' of the past learned tools. Specifically, our COLT incorporates a learnable tool codebook as a tool-specific memory system. Then relevant tools are dynamically selected based on the similarity between user instruction and tool features within the codebook. To unleash the tool usage potential of video LLMs, we collect a video-centric tool-use instruction tuning dataset VideoToolBench. Extensive experiments on both previous video LLM benchmarks and the tool-use-specific VideoToolBench dataset demonstrate the state-of-the-art performance of our proposed COLT. 

**Abstract (ZH)**: 开源视频大型语言模型的COntinuaL Tool 使用增强研究：COLT方法 

---
# MemOrb: A Plug-and-Play Verbal-Reinforcement Memory Layer for E-Commerce Customer Service 

**Title (ZH)**: MemOrb: 一键插拔基于语言强化的记忆层以提升电子商务客户服务 

**Authors**: Yizhe Huang, Yang Liu, Ruiyu Zhao, Xiaolong Zhong, Xingming Yue, Ling Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18713)  

**Abstract**: Large Language Model-based agents(LLM-based agents) are increasingly deployed in customer service, yet they often forget across sessions, repeat errors, and lack mechanisms for continual self-improvement. This makes them unreliable in dynamic settings where stability and consistency are critical. To better evaluate these properties, we emphasize two indicators: task success rate as a measure of overall effectiveness, and consistency metrics such as Pass$^k$ to capture reliability across multiple trials. To address the limitations of existing approaches, we propose MemOrb, a lightweight and plug-and-play verbal reinforcement memory layer that distills multi-turn interactions into compact strategy reflections. These reflections are stored in a shared memory bank and retrieved to guide decision-making, without requiring any fine-tuning. Experiments show that MemOrb significantly improves both success rate and stability, achieving up to a 63 percentage-point gain in multi-turn success rate and delivering more consistent performance across repeated trials. Our results demonstrate that structured reflection is a powerful mechanism for enhancing long-term reliability of frozen LLM agents in customer service scenarios. 

**Abstract (ZH)**: 基于大型语言模型的代理（LLM-based代理）在客户服务中日益普及，但它们往往会在会话之间忘记信息、重复错误，并缺乏持续自我改进的机制。这使得它们在对稳定性和一致性要求高的动态环境中不够可靠。为了更好地评估这些特性，我们强调了两个指标：任务成功率作为整体有效性的衡量标准，以及如Pass$^k$等一致性指标以捕捉多次试验中的可靠性。为了解决现有方法的局限性，我们提出了一种轻量级且即插即用的语言强化记忆层MemOrb，它可以将多轮交互精炼为紧凑的战略反思。这些反思被存储在共享记忆库中，并在决策过程中检索使用，无需任何微调。实验结果显示，MemOrb显著提高了成功率和稳定性，多轮成功率达到63个百分点的增长，并在重复试验中提供了更一致的性能表现。我们的研究结果表明，结构化的反思是一种增强冷冻LLM代理在客户服务场景中长期可靠性的强大机制。 

---
# Learning neuroimaging models from health system-scale data 

**Title (ZH)**: 从健康系统规模的数据中学习神经影像模型 

**Authors**: Yiwei Lyu, Samir Harake, Asadur Chowdury, Soumyanil Banerjee, Rachel Gologorsky, Shixuan Liu, Anna-Katharina Meissner, Akshay Rao, Chenhui Zhao, Akhil Kondepudi, Cheng Jiang, Xinhai Hou, Rushikesh S. Joshi, Volker Neuschmelting, Ashok Srinivasan, Dawn Kleindorfer, Brian Athey, Vikas Gulani, Aditya Pandey, Honglak Lee, Todd Hollon  

**Link**: [PDF](https://arxiv.org/pdf/2509.18638)  

**Abstract**: Neuroimaging is a ubiquitous tool for evaluating patients with neurological diseases. The global demand for magnetic resonance imaging (MRI) studies has risen steadily, placing significant strain on health systems, prolonging turnaround times, and intensifying physician burnout \cite{Chen2017-bt, Rula2024-qp-1}. These challenges disproportionately impact patients in low-resource and rural settings. Here, we utilized a large academic health system as a data engine to develop Prima, the first vision language model (VLM) serving as an AI foundation for neuroimaging that supports real-world, clinical MRI studies as input. Trained on over 220,000 MRI studies, Prima uses a hierarchical vision architecture that provides general and transferable MRI features. Prima was tested in a 1-year health system-wide study that included 30K MRI studies. Across 52 radiologic diagnoses from the major neurologic disorders, including neoplastic, inflammatory, infectious, and developmental lesions, Prima achieved a mean diagnostic area under the ROC curve of 92.0, outperforming other state-of-the-art general and medical AI models. Prima offers explainable differential diagnoses, worklist priority for radiologists, and clinical referral recommendations across diverse patient demographics and MRI systems. Prima demonstrates algorithmic fairness across sensitive groups and can help mitigate health system biases, such as prolonged turnaround times for low-resource populations. These findings highlight the transformative potential of health system-scale VLMs and Prima's role in advancing AI-driven healthcare. 

**Abstract (ZH)**: 神经成像是评估神经系统疾病患者的一项普遍工具。全球磁共振成像（MRI）研究的需求持续上升，给健康系统带来了巨大压力，延长了 turnaround 时间，并加剧了医生的职业倦怠 \cite{Chen2017-bt, Rula2024-qp-1}。这些挑战在低资源和农村地区尤为突出。在此基础上，我们利用一个大型学术医疗系统作为数据引擎，开发了 Prima，这是第一个作为神经成像 AI 基础的视觉语言模型（VLM），支持现实世界和临床 MRI 研究作为输入。Prima 基于超过 220,000 项 MRI 研究进行训练，采用分层视觉架构，提供通用和可转移的 MRI 特征。Prima 在为期一年的全系统研究中进行了测试，该研究包括 30,000 项 MRI 研究。在包括肿瘤性疾病、炎症性疾病、感染性疾病和发育性病变在内的 52 种主要神经系统疾病的放射学诊断中，Prima 达到了 92.0 的平均诊断 ROC 曲线下面积，优于其他最先进的通用和医学 AI 模型。Prima 提供可解释的鉴别诊断、放射科医生的工作列表优先级，以及针对不同患者群体和 MRI 系统的临床转诊推荐。Prima 在不同敏感群体中展示了算法公平性，并有助于减轻健康系统偏见，例如低资源人群的长时间周转。这些发现突显了健康系统规模的视觉语言模型的变革潜力，以及 Prima 在推动 AI 驱动的医疗服务中的作用。 

---
# FlexSED: Towards Open-Vocabulary Sound Event Detection 

**Title (ZH)**: FlexSED: 向着开放词汇集声音事件检测 

**Authors**: Jiarui Hai, Helin Wang, Weizhe Guo, Mounya Elhilali  

**Link**: [PDF](https://arxiv.org/pdf/2509.18606)  

**Abstract**: Despite recent progress in large-scale sound event detection (SED) systems capable of handling hundreds of sound classes, existing multi-class classification frameworks remain fundamentally limited. They cannot process free-text sound queries, which enable more flexible and user-friendly interaction, and they lack zero-shot capabilities and offer poor few-shot adaptability. Although text-query-based separation methods have been explored, they primarily focus on source separation and are ill-suited for SED tasks that require precise temporal localization and efficient detection across large and diverse sound vocabularies. In this paper, we propose FlexSED, an open-vocabulary sound event detection system. FlexSED builds on a pretrained audio SSL model and the CLAP text encoder, introducing an encoder-decoder composition and an adaptive fusion strategy to enable effective continuous training from pretrained weights. To ensure robust supervision, it also employs large language models (LLMs) to assist in event query selection during training, addressing challenges related to missing labels. As a result, FlexSED achieves superior performance compared to vanilla SED models on AudioSet-Strong, while demonstrating strong zero-shot and few-shot capabilities. We release the code and pretrained models to support future research and applications based on FlexSED. 

**Abstract (ZH)**: 一种开放词汇的声事件检测系统：FlexSED 

---
# The Ranking Blind Spot: Decision Hijacking in LLM-based Text Ranking 

**Title (ZH)**: LLM基于文本排名中的决策劫持盲区 

**Authors**: Yaoyao Qian, Yifan Zeng, Yuchao Jiang, Chelsi Jain, Huazheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18575)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance in information retrieval tasks like passage ranking. Our research examines how instruction-following capabilities in LLMs interact with multi-document comparison tasks, identifying what we term the "Ranking Blind Spot", a characteristic of LLM decision processes during comparative evaluation. We analyze how this ranking blind spot affects LLM evaluation systems through two approaches: Decision Objective Hijacking, which alters the evaluation goal in pairwise ranking systems, and Decision Criteria Hijacking, which modifies relevance standards across ranking schemes. These approaches demonstrate how content providers could potentially influence LLM-based ranking systems to affect document positioning. These attacks aim to force the LLM ranker to prefer a specific passage and rank it at the top. Malicious content providers can exploit this weakness, which helps them gain additional exposure by attacking the ranker. In our experiment, We empirically show that the proposed attacks are effective in various LLMs and can be generalized to multiple ranking schemes. We apply these attack to realistic examples to show their effectiveness. We also found stronger LLMs are more vulnerable to these attacks. Our code is available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在信息检索任务如段落排序中展现了强大的性能。我们的研究考察了LLMs的指令遵循能力与多文档比较任务之间的相互作用，并鉴定了我们称之为“排名盲点”的现象，这是LLMs在比较评估过程中决策过程的一个特征。我们通过两种方法分析了这种排名盲点如何影响LLM评估系统：决策目标劫持，改变成对排序系统的评估目标；决策标准劫持，修改不同排序方案的相关性标准。这些方法展示了内容提供者可能如何通过影响LLM排序系统来改变文档的位置。这些攻击旨在迫使LLM排序器偏好特定段落并将其置于首位。恶意内容提供者可以利用这一弱点，通过攻击排序器来增加自身内容的暴露。在我们的实验中，我们实证证明了所提出的攻击在多种LLMs中有效，并且可以泛化到多种排序方案中。我们应用这些攻击于真实示例以展示其有效性。我们还发现，更强的LLM更容易受到这些攻击的影响。我们的代码可在以下链接获取：this https URL。 

---
# Explore the Reinforcement Learning for the LLM based ASR and TTS system 

**Title (ZH)**: 探究基于LLM的ASR和TTS系统中的强化学习应用 

**Authors**: Changfeng Gao, Yabin Li, Keyu An, Zhifu Gao, Zhihao Du, Han Zhao, Xiangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.18569)  

**Abstract**: In recent years, large language models (LLMs) have played an important role in automatic speech recognition (ASR) and text-to-speech (TTS) systems. While reinforcement learning (RL) has significantly enhanced LLM performance in text-based tasks, its application to ASR and TTS remains underexplored due to the complexity of training audio-based models. In this study, we propose a lightweight RL framework tailored for audio-based LLMs that can process audio inputs and generate audio outputs. Based on this framework, we evaluate the effectiveness of reinforcement learning on both ASR and TTS tasks. For the ASR task, we experiment with different rule-based reward functions within the Group Relative Policy Optimization (GRPO) framework and investigate the impact of RL data construction. For the TTS task, we compare GRPO with Differentiable Reward Optimization (DiffRO) and further combine the two approaches to achieve improved performance. Our experiments demonstrate that RL can significantly enhance the performance of both ASR and TTS systems, even with limited training data and a small number of optimization steps. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在自动语音识别（ASR）和文本到语音（TTS）系统中发挥了重要作用。虽然强化学习（RL）在基于文本的任务中显著提升了LLM的性能，但在ASR和TTS中的应用仍因基于音频模型的训练复杂性而未得到充分探索。本研究提出了一种针对基于音频的LLMs的轻量级RL框架，该框架可以处理音频输入并生成音频输出。基于此框架，我们评估了强化学习在ASR和TTS任务中的有效性。对于ASR任务，我们在Group Relative Policy Optimization（GRPO）框架内尝试不同的基于规则的奖励函数，并探讨了RL数据构建的影响。对于TTS任务，我们比较了GRPO与Differentiable Reward Optimization（DiffRO），并进一步将两种方法结合起来以实现性能提升。我们的实验表明，即使在有限的训练数据和少量优化步骤的情况下，RL也能显著提升ASR和TTS系统的性能。 

---
# CCQA: Generating Question from Solution Can Improve Inference-Time Reasoning in SLMs 

**Title (ZH)**: CCQA：从解决方案生成问题可以提高SLMs的推理时间推理能力 

**Authors**: Jin Young Kim, Ji Won Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2509.18536)  

**Abstract**: Recently, inference-time reasoning strategies have further improved the accuracy of large language models (LLMs), but their effectiveness on smaller models remains unclear. Based on the observation that conventional approaches often fail to improve performance in this context, we propose \textbf{C}ycle-\textbf{C}onsistency in \textbf{Q}uestion \textbf{A}nswering (CCQA), a novel reasoning method that can be effectively applied to SLMs. Inspired by cycle consistency, CCQA generates a question from each reasoning path and answer, evaluates each by its similarity to the original question, and then selects the candidate solution with the highest similarity score as the final response. Since conventional SLMs struggle to generate accurate questions from their own reasoning paths and answers, we employ a lightweight Flan-T5 model specialized for question generation to support this process efficiently. From the experimental results, it is verified that CCQA consistently outperforms existing state-of-the-art (SOTA) methods across eight models on mathematical and commonsense reasoning benchmarks. Furthermore, our method establishes a new practical baseline for efficient reasoning in SLMs. Source code can be found at this https URL. 

**Abstract (ZH)**: Recently, Inference-Time Reasoning Strategies Have Further Improved the Accuracy of Large Language Models (LLMs), but Their Effectiveness on Smaller Models Remains Unclear: Proposal of Cycle-Consistency in Question Answering (CCQA) for Smaller Language Models 

---
# Automatic coherence-driven inference on arguments 

**Title (ZH)**: 自动连贯性驱动的论点推理 

**Authors**: Steve Huntsman  

**Link**: [PDF](https://arxiv.org/pdf/2509.18523)  

**Abstract**: Inconsistencies are ubiquitous in law, administration, and jurisprudence. Though a cure is too much to hope for, we propose a technological remedy. Large language models (LLMs) can accurately extract propositions from arguments and compile them into natural data structures that enable coherence-driven inference (CDI) via combinatorial optimization. This neurosymbolic architecture naturally separates concerns and enables meaningful judgments about the coherence of arguments that can inform legislative and policy analysis and legal reasoning. 

**Abstract (ZH)**: 法律、行政和法学中的不一致性无处不在。尽管彻底治愈这种情况期望过高，我们提出了一种技术性解决方案。大型语言模型（LLMs）能够准确提取论点中的命题，并将其编译成自然的数据结构，通过组合优化实现一致性驱动的推理（CDI）。这种神经符号架构自然地分离了关注点，并能够对论点的连贯性做出有意义的判断，从而指导立法和政策分析以及法律推理。 

---
# APRIL: Active Partial Rollouts in Reinforcement Learning to tame long-tail generation 

**Title (ZH)**: APRIL: Active 部分 rollout 在强化学习中治理长尾生成 

**Authors**: Yuzhen Zhou, Jiajun Li, Yusheng Su, Gowtham Ramesh, Zilin Zhu, Xiang Long, Chenyang Zhao, Jin Pan, Xiaodong Yu, Ze Wang, Kangrui Du, Jialian Wu, Ximeng Sun, Jiang Liu, Qiaolin Yu, Hao Chen, Zicheng Liu, Emad Barsoum  

**Link**: [PDF](https://arxiv.org/pdf/2509.18521)  

**Abstract**: Reinforcement learning (RL) has become a cornerstone in advancing large-scale pre-trained language models (LLMs). Successive generations, including GPT-o series, DeepSeek-R1, Kimi-K1.5, Grok 4, and GLM-4.5, have relied on large-scale RL training to enhance reasoning and coding capabilities. To meet the community's growing RL needs, numerous RL frameworks have been proposed. Most of these frameworks primarily rely on inference engines for rollout generation and training engines for policy updates. However, RL training remains computationally expensive, with rollout generation accounting for more than 90% of total runtime. In addition, its efficiency is often constrained by the long-tail distribution of rollout response lengths, where a few lengthy responses stall entire batches, leaving GPUs idle and underutilized. As model and rollout sizes continue to grow, this bottleneck increasingly limits scalability. To address this challenge, we propose Active Partial Rollouts in Reinforcement Learning (APRIL), which mitigates long-tail inefficiency. In the rollout phase, APRIL over-provisions rollout requests, terminates once the target number of responses is reached, and recycles incomplete responses for continuation in future steps. This strategy ensures that no rollouts are discarded while substantially reducing GPU idle time. Experiments show that APRIL improves rollout throughput by at most 44% across commonly used RL algorithms (GRPO, DAPO, GSPO), accelerates convergence, and achieves at most 8% higher final accuracy across tasks. Moreover, APRIL is both framework and hardware agnostic, already integrated into the slime RL framework, and deployable on NVIDIA and AMD GPUs alike. Taken together, this work unifies system-level and algorithmic considerations in proposing APRIL, with the aim of advancing RL training efficiency and inspiring further optimizations in RL systems. 

**Abstract (ZH)**: 主动部分回放强化学习（APRIL）：缓解长尾 inefficiency 

---
# Coherence-driven inference for cybersecurity 

**Title (ZH)**: 驱动一致性的网络安全推理 

**Authors**: Steve Huntsman  

**Link**: [PDF](https://arxiv.org/pdf/2509.18520)  

**Abstract**: Large language models (LLMs) can compile weighted graphs on natural language data to enable automatic coherence-driven inference (CDI) relevant to red and blue team operations in cybersecurity. This represents an early application of automatic CDI that holds near- to medium-term promise for decision-making in cybersecurity and eventually also for autonomous blue team operations. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以将自然语言数据编译成加权图以启用与网络安全红蓝队操作相关的自动一致性驱动推理（CDI）。这代表了自动CDI的早期应用，有望在近至中期内为网络安全决策提供支持，并最终也适用于自主蓝队操作。 

---
# CogniLoad: A Synthetic Natural Language Reasoning Benchmark With Tunable Length, Intrinsic Difficulty, and Distractor Density 

**Title (ZH)**: CogniLoad：一种可调节长度、内在难度和干扰项密度的合成自然语言推理基准测试 

**Authors**: Daniel Kaiser, Arnoldo Frigessi, Ali Ramezani-Kebrya, Benjamin Ricaud  

**Link**: [PDF](https://arxiv.org/pdf/2509.18458)  

**Abstract**: Current benchmarks for long-context reasoning in Large Language Models (LLMs) often blur critical factors like intrinsic task complexity, distractor interference, and task length. To enable more precise failure analysis, we introduce CogniLoad, a novel synthetic benchmark grounded in Cognitive Load Theory (CLT). CogniLoad generates natural-language logic puzzles with independently tunable parameters that reflect CLT's core dimensions: intrinsic difficulty ($d$) controls intrinsic load; distractor-to-signal ratio ($\rho$) regulates extraneous load; and task length ($N$) serves as an operational proxy for conditions demanding germane load. Evaluating 22 SotA reasoning LLMs, CogniLoad reveals distinct performance sensitivities, identifying task length as a dominant constraint and uncovering varied tolerances to intrinsic complexity and U-shaped responses to distractor ratios. By offering systematic, factorial control over these cognitive load dimensions, CogniLoad provides a reproducible, scalable, and diagnostically rich tool for dissecting LLM reasoning limitations and guiding future model development. 

**Abstract (ZH)**: 当前的大语言模型（LLMs）长期上下文推理基准常常模糊了内在任务复杂性、干扰项干扰和任务长度等关键因素。为了实现更精确的失败分析，我们引入了CogniLoad，这是一种基于认知负载理论（CLT）的新型合成基准。CogniLoad生成自然语言逻辑谜题，并可独立调节反映CLT核心维度的参数：内在难度（$d$）控制内在负载；干扰项与信号比（$\rho$）调节外在负载；任务长度（$N$）作为相关负载条件下操作性的代理指标。评估22种当前最先进的推理LLM后，CogniLoad揭示了不同的性能敏感性，将任务长度确定为主导限制因素，并发现了对内在复杂性的不同容忍度和干扰比率的U形反应。通过在这些认知负载维度上提供系统性和因子控制，CogniLoad提供了一种可再现、可扩展且诊断丰富的工具，用于剖析LLM推理限制并指导未来的模型开发。 

---
# FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction 

**Title (ZH)**: FastMTP：增强多 token 预测加速大语言模型推理 

**Authors**: Yuxuan Cai, Xiaozhuan Liang, Xinghua Wang, Jin Ma, Haijin Liang, Jinwen Luo, Xinyu Zuo, Lisheng Duan, Yuyang Yin, Xi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.18362)  

**Abstract**: As large language models (LLMs) become increasingly powerful, the sequential nature of autoregressive generation creates a fundamental throughput bottleneck that limits the practical deployment. While Multi-Token Prediction (MTP) has demonstrated remarkable benefits for model training efficiency and performance, its inherent potential for inference acceleration remains largely unexplored. This paper introduces FastMTP, a simple yet effective method that improves multi-step draft quality by aligning MTP training with its inference pattern, significantly enhancing speculative decoding performance. Our approach fine-tunes a single MTP head with position-shared weights on self-distilled data, enabling it to capture dependencies among consecutive future tokens and maintain high acceptance rates across multiple recursive draft steps. By integrating language-aware dynamic vocabulary compression into the MTP head, we further reduce computational overhead in the drafting process. Experimental results across seven diverse benchmarks demonstrate that FastMTP achieves an average of 2.03x speedup compared to standard next token prediction with lossless output quality, outperforming vanilla MTP by 82%. FastMTP requires only lightweight training and seamlessly integrates with existing inference frameworks, offering a practical and rapidly deployable solution for accelerating LLM inference. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）变得越来越强大，自回归生成的序贯性质形成了一个根本性的吞吐量瓶颈，限制了其实用部署。尽管多令牌预测（MTP）展示了显著的模型训练效率和性能优势，其在推断加速方面的固有潜力仍 largely unexplored。本文介绍了一种简单而有效的方法FastMTP，该方法通过将MTP训练与推断模式对齐来提高多步草稿质量，显著提升了推测性解码性能。我们的方法在自蒸馏数据上对共享位置权重的单个MTP头进行微调，使其能够捕捉连续未来令牌之间的依赖性，并在多个递归草稿步骤中保持高接受率。通过将语言感知的动态词汇压缩集成到MTP头中，我们在起草过程中进一步减少了计算开销。在七个不同基准上的实验结果表明，FastMTP相比标准下一个令牌预测实现了平均2.03倍的加速，无损输出质量，比vanilla MTP性能高出82%。FastMTP仅需要轻量级训练，并且可以无缝集成到现有的推断框架中，提供了一种实用且快速部署的解决方案，用于加速LLM推断。 

---
# Brittleness and Promise: Knowledge Graph Based Reward Modeling for Diagnostic Reasoning 

**Title (ZH)**: brittleness and promise: 基于知识图谱的诊断推理奖励建模 

**Authors**: Saksham Khatwani, He Cheng, Majid Afshar, Dmitriy Dligach, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.18316)  

**Abstract**: Large language models (LLMs) show promise for diagnostic reasoning but often lack reliable, knowledge grounded inference. Knowledge graphs (KGs), such as the Unified Medical Language System (UMLS), offer structured biomedical knowledge that can support trustworthy reasoning. Prior approaches typically integrate KGs via retrieval augmented generation or fine tuning, inserting KG content into prompts rather than enabling structured reasoning. We explore an alternative paradigm: treating the LLM as a reward model of KG reasoning paths, where the model learns to judge whether a candidate path leads to correct diagnosis for a given patient input. This approach is inspired by recent work that leverages reward training to enhance model reasoning abilities, and grounded in computational theory, which suggests that verifying a solution is often easier than generating one from scratch. It also parallels physicians' diagnostic assessment, where they judge which sequences of findings and intermediate conditions most plausibly support a diagnosis. We first systematically evaluate five task formulation for knowledge path judging and eight training paradigm. Second, we test whether the path judging abilities generalize to downstream diagnostic tasks, including diagnosis summarization and medical question answering. Experiments with three open source instruct-tuned LLMs reveal both promise and brittleness: while specific reward optimization and distillation lead to strong path-judging performance, the transferability to downstream tasks remain weak. Our finding provides the first systematic assessment of "reward model style" reasoning over clinical KGs, offering insights into how structured, reward-based supervision influences diagnostic reasoning in GenAI systems for healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）在诊断推理方面展现出潜力，但往往缺乏可靠的、基于知识的推理能力。知识图谱（KGs），如统一医学语言系统（UMLS），提供了结构化的生物医学知识，可支持可信的推理。以往的方法通常通过检索增强生成或微调将KG内容插入提示，而不是实现结构化的推理。我们探索了另一种范式：将LLM视为KG推理路径的奖励模型，其中模型学会判断候选路径是否能为给定的患者输入提供正确的诊断。这种方法受到利用奖励训练增强模型推理能力的最新工作的启发，并基于计算理论的原理，该原理表明验证一个解决方案通常比从零开始生成它要容易得多。它也与医生的诊断评估相似，医生评估哪些发现和中间条件的序列最有可能支持诊断。我们首先系统地评估了五种知识路径判断任务的表述和八种训练范式。其次，我们测试了路径判断能力是否可以泛化到下游的诊断任务，包括诊断总结和医学问答。实验证明了该方法的潜力和脆弱性：虽然特定的奖励优化和蒸馏可以实现强大的路径判断性能，但其向下游任务的迁移能力仍然较弱。我们的研究提供了对临床KG上“奖励模型风格”推理的第一个系统评估，为如何结构化的奖励监督影响医疗保健领域GenAI系统的诊断推理提供了见解。 

---
# Evaluating Large Language Models for Detecting Antisemitism 

**Title (ZH)**: 评估大型语言模型检测反犹太主义的能力 

**Authors**: Jay Patel, Hrudayangam Mehta, Jeremy Blackburn  

**Link**: [PDF](https://arxiv.org/pdf/2509.18293)  

**Abstract**: Detecting hateful content is a challenging and important problem. Automated tools, like machine-learning models, can help, but they require continuous training to adapt to the ever-changing landscape of social media. In this work, we evaluate eight open-source LLMs' capability to detect antisemitic content, specifically leveraging in-context definition as a policy guideline. We explore various prompting techniques and design a new CoT-like prompt, Guided-CoT. Guided-CoT handles the in-context policy well, increasing performance across all evaluated models, regardless of decoding configuration, model sizes, or reasoning capability. Notably, Llama 3.1 70B outperforms fine-tuned GPT-3.5. Additionally, we examine LLM errors and introduce metrics to quantify semantic divergence in model-generated rationales, revealing notable differences and paradoxical behaviors among LLMs. Our experiments highlight the differences observed across LLMs' utility, explainability, and reliability. 

**Abstract (ZH)**: 检测 hateful 内容是一个具有挑战性和重要性的问题。自动化工具，如机器学习模型，可以提供帮助，但它们需要持续训练以适应社交媒体不断变化的环境。在这项工作中，我们评估了八种开源大语言模型检测反犹太主义内容的能力，具体利用上下文内定义作为政策指南。我们探索了各种提示技术，并设计了一种新的类似 CoT 的提示——Guided-CoT。Guided-CoT 能很好地处理上下文内政策，提高了所有评估模型的表现，无论解码配置、模型大小或推理能力如何。值得注意的是，Llama 3.1 70B 的表现优于细调的 GPT-3.5。此外，我们还分析了大语言模型的错误，并引入了用于量化模型生成的论据语义差异的指标，揭示了大语言模型之间显著的差异性和悖论性行为。我们的实验突显了大语言模型在实用性、可解释性和可靠性方面的差异。 

---
# Sparse Training Scheme for Multimodal LLM 

**Title (ZH)**: 多模态大规模语言模型的稀疏训练方案 

**Authors**: Kean Shi, Liang Chen, Haozhe Zhao, Baobao Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18150)  

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated outstanding performance across a variety of domains. However, training MLLMs is often inefficient due to the significantly longer input sequences introduced by multimodal data and the low utilization of inter-layer computations. To address this challenge, we shift the focus to the training process itself and propose a novel training-efficient framework based on sparse representations, termed the Sparse Training Scheme (STS). This scheme consists of two key components: the Visual Token Compressor, which reduces the information load by compressing visual tokens, and the Layer Dynamic Skipper, which mitigates the computational overhead by dynamically skipping unnecessary layers in the language model during both forward and backward passes. Our approach is broadly applicable to diverse MLLM architectures and has been extensively evaluated on multiple benchmarks, demonstrating its effectiveness and efficiency. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在多个领域展现了出色的性能。然而，由于多模态数据引入的显著更长的输入序列以及层间计算利用效率低的问题，MLLMs的训练通常效率低下。为解决这一挑战，我们将焦点转向训练过程本身，并提出了一种基于稀疏表示的新颖训练高效框架，称为稀疏训练方案（STS）。该方案包含两个关键组件：视觉标记压缩器，通过压缩视觉标记来减少信息负载；以及层动态跳过器，在语言模型的前向和后向传递过程中动态跳过不必要的层，从而减轻计算开销。该方法适用于各种MLLM架构，并在多个基准上进行了广泛评估，展示了其有效性和效率。 

---
# From Parameters to Performance: A Data-Driven Study on LLM Structure and Development 

**Title (ZH)**: 从参数到性能：基于数据的大型语言模型结构与发展研究 

**Authors**: Suqing Wang, Zuchao Li, Luohe Shi, Bo Du, Hai Zhao, Yun Li, Qianren Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18136)  

**Abstract**: Large language models (LLMs) have achieved remarkable success across various domains, driving significant technological advancements and innovations. Despite the rapid growth in model scale and capability, systematic, data-driven research on how structural configurations affect performance remains scarce. To address this gap, we present a large-scale dataset encompassing diverse open-source LLM structures and their performance across multiple benchmarks. Leveraging this dataset, we conduct a systematic, data mining-driven analysis to validate and quantify the relationship between structural configurations and performance. Our study begins with a review of the historical development of LLMs and an exploration of potential future trends. We then analyze how various structural choices impact performance across benchmarks and further corroborate our findings using mechanistic interpretability techniques. By providing data-driven insights into LLM optimization, our work aims to guide the targeted development and application of future models. We will release our dataset at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域取得了显著的成功，推动了重要的技术创新和进步。尽管模型规模和能力迅速增长，但关于结构配置如何影响性能的系统性、数据驱动研究仍然稀缺。为弥补这一差距，我们呈现了一个涵盖多种开源LLM结构及其在多个基准上的性能的大规模数据集。利用该数据集，我们进行了一项系统性的数据挖掘分析，以验证和量化结构配置与性能之间的关系。我们的研究从LLM的历史发展回顾和未来趋势探索开始。随后，我们分析了各种结构选择如何影响不同基准上的性能，并进一步使用机制可解释性技术来验证我们的发现。通过提供有关LLM优化的数据驱动洞见，我们的工作旨在指导未来模型的精准开发和应用。我们将在此处发布我们的数据集：https://....... 

---
# Self-Evolving LLMs via Continual Instruction Tuning 

**Title (ZH)**: 通过持续指令调优实现自我进化的大语言模型 

**Authors**: Le Huang, Jiazheng Kang, Cheng Hou, Zhe Zhao, Zhenxiang Yan, Chuan Shi, Ting Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.18133)  

**Abstract**: In real-world industrial settings, large language models (LLMs) must learn continually to keep pace with diverse and evolving tasks, requiring self-evolution to refine knowledge under dynamic data distributions. However, existing continual learning (CL) approaches, such as replay and parameter isolation, often suffer from catastrophic forgetting: training on new tasks degrades performance on earlier ones by overfitting to the new distribution and weakening this http URL propose MoE-CL, a parameter-efficient adversarial mixture-of-experts framework for industrial-scale, self-evolving continual instruction tuning of LLMs. MoE-CL uses a dual-expert design: (1) a dedicated LoRA expert per task to preserve task-specific knowledge via parameter independence, mitigating forgetting; and (2) a shared LoRA expert to enable cross-task transfer. To prevent transferring task-irrelevant noise through the shared pathway, we integrate a task-aware discriminator within a GAN. The discriminator encourages the shared expert to pass only task-aligned information during sequential training. Through adversarial learning, the shared expert acquires generalized representations that mimic the discriminator, while dedicated experts retain task-specific details, balancing knowledge retention and cross-task generalization and thereby supporting this http URL experiments on the public MTL5 benchmark and an industrial Tencent3 benchmark validate the effectiveness of MoE-CL for continual instruction tuning. In real-world A/B testing for content compliance review on the Tencent Video platform, MoE-CL reduced manual review costs by 15.3%. These results demonstrate that MoE-CL is practical for large-scale industrial deployment where continual adaptation and stable transfer are critical. 

**Abstract (ZH)**: 工业规模下大语言模型的参数高效自进化持续指令调优：MoE-CL框架 

---
# Safe-SAIL: Towards a Fine-grained Safety Landscape of Large Language Models via Sparse Autoencoder Interpretation Framework 

**Title (ZH)**: Safe-SAIL：通过稀疏自编码解释框架朝着大型语言模型精细安全景观的构建 

**Authors**: Jiaqi Weng, Han Zheng, Hanyu Zhang, Qinqin He, Jialing Tao, Hui Xue, Zhixuan Chu, Xiting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.18127)  

**Abstract**: Increasing deployment of large language models (LLMs) in real-world applications raises significant safety concerns. Most existing safety research focuses on evaluating LLM outputs or specific safety tasks, limiting their ability to ad- dress broader, undefined risks. Sparse Autoencoders (SAEs) facilitate interpretability research to clarify model behavior by explaining single-meaning atomic features decomposed from entangled signals. jHowever, prior applications on SAEs do not interpret features with fine-grained safety-related con- cepts, thus inadequately addressing safety-critical behaviors, such as generating toxic responses and violating safety regu- lations. For rigorous safety analysis, we must extract a rich and diverse set of safety-relevant features that effectively capture these high-risk behaviors, yet face two challenges: identifying SAEs with the greatest potential for generating safety concept-specific neurons, and the prohibitively high cost of detailed feature explanation. In this paper, we pro- pose Safe-SAIL, a framework for interpreting SAE features within LLMs to advance mechanistic understanding in safety domains. Our approach systematically identifies SAE with best concept-specific interpretability, explains safety-related neurons, and introduces efficient strategies to scale up the in- terpretation process. We will release a comprehensive toolkit including SAE checkpoints and human-readable neuron ex- planations, which supports empirical analysis of safety risks to promote research on LLM safety. 

**Abstract (ZH)**: 增加大型语言模型在实际应用中的部署引发了显著的安全 concerns。现有大多数安全研究侧重于评估 LLM 输出或特定的安全任务，限制了它们应对更广泛、未定义的风险的能力。稀疏自编码器（SAEs）通过解释从交织信号中分解出来的单义原子特征来促进可解释性研究，以澄清模型行为。然而，先前对 SAEs 的应用没有用细粒度的安全相关概念来解释特征，因此未能充分应对关键的安全行为，如生成有害响应和违反安全规定。为了进行严格的安全分析，我们必须提取一组丰富且多样的与安全相关特征，有效地捕捉这些高风险行为，但面临两个挑战：识别最具潜力生成特定概念神经元的 SAEs，以及对详细特征解释的高成本。在本文中，我们提出 Safe-SAIL，一个在 LLM 中解释 SAE 特征的框架，以推动安全领域的机制性理解。我们的方法系统地识别具有最佳概念特定解释性的 SAE，解释安全相关的神经元，并引入高效策略以扩大解释过程的规模。我们将发布一个全面的工具包，包括 SAE 检查点和可读的神经元解释，支持对 LLM 安全风险的经验分析，促进 LLM 安全研究。 

---
# Solve it with EASE 

**Title (ZH)**: 用EASE解决它 

**Authors**: Adam Viktorin, Tomas Kadavy, Jozef Kovac, Michal Pluhacek, Roman Senkerik  

**Link**: [PDF](https://arxiv.org/pdf/2509.18108)  

**Abstract**: This paper presents EASE (Effortless Algorithmic Solution Evolution), an open-source and fully modular framework for iterative algorithmic solution generation leveraging large language models (LLMs). EASE integrates generation, testing, analysis, and evaluation into a reproducible feedback loop, giving users full control over error handling, analysis, and quality assessment. Its architecture supports the orchestration of multiple LLMs in complementary roles-such as generator, analyst, and evaluator. By abstracting the complexity of prompt design and model management, EASE provides a transparent and extensible platform for researchers and practitioners to co-design algorithms and other generative solutions across diverse domains. 

**Abstract (ZH)**: This paper presents EASE (Effortless Algorithmic Solution Evolution), 一种利用大型语言模型（LLMs）进行迭代算法解决方案生成的开源且完全可模块化的框架。EASE 将生成、测试、分析和评估整合到可重复的反馈循环中，使用户能够完全控制错误处理、分析和质量评估。其架构支持多种大型语言模型在互补角色（如生成器、分析师和评估器）下的协调工作。通过抽象提示设计和模型管理的复杂性，EASE 提供了一个透明且可扩展的平台，使研究人员和实践者能够在不同领域共同设计算法和其他生成性解决方案。 

---
