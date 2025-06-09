# PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time 

**Title (ZH)**: PersonaAgent：当大型语言模型代理在测试时遇到个性化 

**Authors**: Weizhi Zhang, Xinyang Zhang, Chenwei Zhang, Liangwei Yang, Jingbo Shang, Zhepei Wei, Henry Peng Zou, Zijie Huang, Zhengyang Wang, Yifan Gao, Xiaoman Pan, Lian Xiong, Jingguo Liu, Philip S. Yu, Xian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06254)  

**Abstract**: Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences. 

**Abstract (ZH)**: LLM赋能的个性化代理：一种解决多样化个性化任务的先进框架 

---
# CP-Bench: Evaluating Large Language Models for Constraint Modelling 

**Title (ZH)**: CP-Bench: 评估约束建模的大语言模型 

**Authors**: Kostis Michailidis, Dimos Tsouros, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2506.06052)  

**Abstract**: Combinatorial problems are present in a wide range of industries. Constraint Programming (CP) is a well-suited problem-solving paradigm, but its core process, namely constraint modelling, is a bottleneck for wider adoption. Aiming to alleviate this bottleneck, recent studies have explored using Large Language Models (LLMs) as modelling assistants, transforming combinatorial problem descriptions to executable constraint models, similar to coding assistants. However, the existing evaluation datasets for constraint modelling are often limited to small, homogeneous, or domain-specific instances, which do not capture the diversity of real-world scenarios. This work addresses this gap by introducing CP-Bench, a novel benchmark dataset that includes a diverse set of well-known combinatorial problem classes sourced from the CP community, structured explicitly for evaluating LLM-driven CP modelling. With this dataset, and given the variety of constraint modelling frameworks, we compare and evaluate the modelling capabilities of LLMs for three distinct constraint modelling systems, which vary in abstraction level and underlying syntax: the high-level MiniZinc language and Python-based CPMpy library, and the lower-level Python interface of the OR-Tools CP-SAT solver. In order to enhance the ability of LLMs to produce valid constraint models, we systematically evaluate the use of prompt-based and inference-time compute methods adapted from existing LLM-based code generation research. Our results underscore the modelling convenience provided by Python-based frameworks, as well as the effectiveness of documentation-rich system prompts, which, augmented with repeated sampling and self-verification, achieve further improvements, reaching up to 70\% accuracy on this new, highly challenging benchmark. 

**Abstract (ZH)**: 组合优化问题是许多行业的共同挑战。约束编程（CP）是一种合适的问题求解范式，但其核心过程，即约束建模，成为了更广泛采用的瓶颈。为了减轻这一瓶颈，近期研究探索了使用大规模语言模型（LLMs）作为建模助手，将组合优化问题描述转换为可执行的约束模型，类似于代码助手的功能。然而，现有的约束建模评估数据集往往局限于小规模、同质或领域特定的实例，无法捕捉现实场景的多样性。本工作通过引入CP-Bench这一新的基准数据集来填补这一空白，该数据集包含来自CP社区的多样化组合优化问题类，专门用于评估LLM驱动的CP建模。借助该数据集和不同的约束建模框架，我们比较和评估了LLM在三个具有不同抽象级别和底层语法的约束建模系统中的建模能力：MiniZinc高级语言、基于Python的CPMpy库以及OR-Tools CP-SAT求解器的Python接口。为了增强LLMs生成有效约束模型的能力，我们系统性地评估了从现有LLM基于代码生成研究中适应而来的基于提示和推理时计算方法的有效性。我们的结果强调了基于Python的框架提供的建模便利性，以及文档丰富的系统提示的有效性，这些提示在重复采样和自我验证的增强下，实现了进一步的提升，在这一新且极具挑战性的基准中达到高达70%的准确性。 

---
# CrimeMind: Simulating Urban Crime with Multi-Modal LLM Agents 

**Title (ZH)**: CrimeMind: 用多模态LLM代理模拟城市犯罪 

**Authors**: Qingbin Zeng, Ruotong Zhao, Jinzhu Mao, Haoyang Li, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05981)  

**Abstract**: Modeling urban crime is an important yet challenging task that requires understanding the subtle visual, social, and cultural cues embedded in urban environments. Previous work has predominantly focused on rule-based agent-based modeling (ABM) and deep learning methods. ABMs offer interpretability of internal mechanisms but exhibit limited predictive this http URL contrast, deep learning methods are often effective in prediction but are less interpretable and require extensive training data. Moreover, both lines of work lack the cognitive flexibility to adapt to changing environments. Leveraging the capabilities of large language models (LLMs), we propose CrimeMind, a novel LLM-driven ABM framework for simulating urban crime within a multi-modal urban context.A key innovation of our design is the integration of the Routine Activity Theory (RAT) into the agentic workflow of CrimeMind, enabling it to process rich multi-modal urban features and reason about criminal this http URL, RAT requires LLM agents to infer subtle cues in evaluating environmental safety as part of assessing guardianship, which can be challenging for LLMs. To address this, we collect a small-scale human-annotated dataset and align CrimeMind's perception with human judgment via a training-free textual gradient this http URL across four major U.S. cities demonstrate that CrimeMind outperforms both traditional ABMs and deep learning baselines in crime hotspot prediction and spatial distribution accuracy, achieving up to a 24% improvement over the strongest this http URL, we conduct counterfactual simulations of external incidents and policy interventions and it successfully captures the expected changes in crime patterns, demonstrating its ability to reflect counterfactual this http URL, CrimeMind enables fine-grained modeling of individual behaviors and facilitates evaluation of real-world interventions. 

**Abstract (ZH)**: 基于大语言模型的城市犯罪建模：一种多模态城市环境中的犯罪模拟新颖框架 

---
# Preference Learning for AI Alignment: a Causal Perspective 

**Title (ZH)**: 基于因果视角的AI对齐偏好学习 

**Authors**: Katarzyna Kobalczyk, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2506.05967)  

**Abstract**: Reward modelling from preference data is a crucial step in aligning large language models (LLMs) with human values, requiring robust generalisation to novel prompt-response pairs. In this work, we propose to frame this problem in a causal paradigm, providing the rich toolbox of causality to identify the persistent challenges, such as causal misidentification, preference heterogeneity, and confounding due to user-specific factors. Inheriting from the literature of causal inference, we identify key assumptions necessary for reliable generalisation and contrast them with common data collection practices. We illustrate failure modes of naive reward models and demonstrate how causally-inspired approaches can improve model robustness. Finally, we outline desiderata for future research and practices, advocating targeted interventions to address inherent limitations of observational data. 

**Abstract (ZH)**: 从偏好数据中构建奖励模型是使大型语言模型（LLMs）与人类价值观对齐的关键步骤，要求在新的提示-响应对上具备稳固的泛化能力。我们在本文中提出将该问题置于因果框架中，利用因果推断丰富的工具箱来识别持续的挑战，如因果误识别、偏好异质性以及由于用户特定因素引起的混杂。继承因果推断领域的文献，我们识别出可靠泛化所必需的关键假设，并将其与常见的数据收集实践进行对比。我们展示了朴素奖励模型的失败模式，并演示了如何基于因果启发的方法提高模型的鲁棒性。最后，我们概述了未来研究和实践的所需条件，提倡针对性的干预措施以解决观察数据固有的局限性。 

---
# Explainability in Context: A Multilevel Framework Aligning AI Explanations with Stakeholder with LLMs 

**Title (ZH)**: 上下文中的可解释性：一种将AI解释与利益相关者对LLM对齐的多层次框架 

**Authors**: Marilyn Bello, Rafael Bello, Maria-Matilde García, Ann Nowé, Iván Sevillano-García, Francisco Herrera  

**Link**: [PDF](https://arxiv.org/pdf/2506.05887)  

**Abstract**: The growing application of artificial intelligence in sensitive domains has intensified the demand for systems that are not only accurate but also explainable and trustworthy. Although explainable AI (XAI) methods have proliferated, many do not consider the diverse audiences that interact with AI systems: from developers and domain experts to end-users and society. This paper addresses how trust in AI is influenced by the design and delivery of explanations and proposes a multilevel framework that aligns explanations with the epistemic, contextual, and ethical expectations of different stakeholders. The framework consists of three layers: algorithmic and domain-based, human-centered, and social explainability. We highlight the emerging role of Large Language Models (LLMs) in enhancing the social layer by generating accessible, natural language explanations. Through illustrative case studies, we demonstrate how this approach facilitates technical fidelity, user engagement, and societal accountability, reframing XAI as a dynamic, trust-building process. 

**Abstract (ZH)**: 人工智能在敏感领域应用的增长加剧了对不仅准确而且可解释和可信赖的系统的需求。尽管可解释人工智能（XAI）方法层出不穷，但许多方法并未考虑到与人工智能系统互动的多元受众：从开发者和领域专家到最终用户和社会。本文探讨了设计和传递解释如何影响人们对AI的信任，并提出了一种多层框架，该框架将解释与不同利益相关者的认知、情境和伦理期望相一致。该框架由三个层次构成：算法和领域基础层、以人为本层和社会可解释性层。我们强调大型语言模型（LLMs）在增强社会层方面的作用，通过生成易于理解的自然语言解释。通过示例案例研究，我们展示了这种方法如何促进技术准确度、用户参与和社会问责，并重新定义XAI为一个动态的信任建立过程。 

---
# Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective 

**Title (ZH)**: 受限采样对于语言模型应当易于实现：从MCMC的角度看他 

**Authors**: Emmanuel Anaya Gonzalez, Sairam Vaidya, Kanghee Park, Ruyi Ji, Taylor Berg-Kirkpatrick, Loris D'Antoni  

**Link**: [PDF](https://arxiv.org/pdf/2506.05754)  

**Abstract**: Constrained decoding enables Language Models (LMs) to produce samples that provably satisfy hard constraints. However, existing constrained-decoding approaches often distort the underlying model distribution, a limitation that is especially problematic in applications like program fuzzing, where one wants to generate diverse and valid program inputs for testing purposes. We propose a new constrained sampling framework based on Markov Chain Monte Carlo (MCMC) that simultaneously satisfies three core desiderata: constraint satisfying (every sample satisfies the constraint), monotonically converging (the sampling process converges to the true conditional distribution), and efficient (high-quality samples emerge in few steps). Our method constructs a proposal distribution over valid outputs and applies a Metropolis-Hastings acceptance criterion based on the LM's likelihood, ensuring principled and efficient exploration of the constrained space. Empirically, our sampler outperforms existing methods on both synthetic benchmarks and real-world program fuzzing tasks. 

**Abstract (ZH)**: 约束解码使语言模型能够生成满足硬约束的样本。然而，现有的约束解码方法 often 常常会扭曲基础模型的概率分布，这一局限性尤其在程序 fuzzing 等应用中问题明显，因为这类应用的目标是生成多样且有效的程序输入以进行测试。我们提出了一种基于马尔可夫链蒙特卡洛（MCMC）的新约束采样框架，该框架同时满足三项核心需求：约束满足性（每个样本都满足约束）、单调收敛性（采样过程收敛到真实的条件分布）和高效性（高质量样本在少量步骤内产生）。我们的方法在有效输出上构建了建议分布，并基于语言模型的似然性应用了梅特罗波利斯-哈斯廷斯接受准则，确保了在约束空间内的原则性和高效性探索。实验证明，我们的采样器在合成基准和实际程序 fuzzing 任务中均优于现有方法。 

---
# Distillation Robustifies Unlearning 

**Title (ZH)**: 蒸馏增强正学习 

**Authors**: Bruce W. Lee, Addie Foote, Alex Infanger, Leni Shor, Harish Kamath, Jacob Goldman-Wetzler, Bryce Woodworth, Alex Cloud, Alexander Matt Turner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06278)  

**Abstract**: Current LLM unlearning methods are not robust: they can be reverted easily with a few steps of finetuning. This is true even for the idealized unlearning method of training to imitate an oracle model that was never exposed to unwanted information, suggesting that output-based finetuning is insufficient to achieve robust unlearning. In a similar vein, we find that training a randomly initialized student to imitate an unlearned model transfers desired behaviors while leaving undesired capabilities behind. In other words, distillation robustifies unlearning. Building on this insight, we propose Unlearn-Noise-Distill-on-Outputs (UNDO), a scalable method that distills an unlearned model into a partially noised copy of itself. UNDO introduces a tunable tradeoff between compute cost and robustness, establishing a new Pareto frontier on synthetic language and arithmetic tasks. At its strongest setting, UNDO matches the robustness of a model retrained from scratch with perfect data filtering while using only 60-80% of the compute and requiring only 0.01% of the pretraining data to be labeled. We also show that UNDO robustifies unlearning on the more realistic Weapons of Mass Destruction Proxy (WMDP) benchmark. Since distillation is widely used in practice, incorporating an unlearning step beforehand offers a convenient path to robust capability removal. 

**Abstract (ZH)**: 当前的LLM去学习方法不够 robust：通过 few steps 的 fine-tuning 就可以轻松恢复。即使对于从未接触过不需要信息的理想去学习方法，训练去学习模型模仿一个 oracle 模型也是如此，这表明基于输出的 fine-tuning 无法实现 robust 去学习。类似地，我们发现，初始化为随机的学生模型去模仿一个未学习模型能够传递所需的行为，同时保留不必要的能力。换句话说，知识蒸馏增强了去学习的 robust 性。基于这一见解，我们提出了去学习-加入噪声-知识蒸馏（UNDO）方法，这是一种可扩展的方法，将一个未学习模型蒸馏为一个部分噪声版本的自己。UNDO 引入了可调节的计算成本与 robust 性之间的权衡，建立了合成语言和算术任务上的新的帕累托前沿。在最强设置下，UNDO 仅需使用 60-80% 的计算量和不到 0.01% 的预训练数据标签，就能达到从头开始重新训练模型且完美数据过滤的 robust 性。我们还展示了 UNDO 在更现实的大规模毁灭性武器代理（WMDP）基准测试中增强了去学习 robust 性。由于知识蒸馏在实践中广泛应用，因此在先前引入一个去学习步骤提供了一条实现 robust 能力去除的便捷途径。 

---
# Cartridges: Lightweight and general-purpose long context representations via self-study 

**Title (ZH)**: cartridges: 自我学习驱动的轻量级通用长期上下文表示 

**Authors**: Sabri Eyuboglu, Ryan Ehrlich, Simran Arora, Neel Guha, Dylan Zinsley, Emily Liu, Will Tennien, Atri Rudra, James Zou, Azalia Mirhoseini, Christopher Re  

**Link**: [PDF](https://arxiv.org/pdf/2506.06266)  

**Abstract**: Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining. 

**Abstract (ZH)**: 一种小型缓存训练方法：通过自学习生成对话来扩展大型语言模型的有效上下文长度并降低成本 

---
# "We need to avail ourselves of GenAI to enhance knowledge distribution": Empowering Older Adults through GenAI Literacy 

**Title (ZH)**: 我们需要利用生成式AI提升知识传播能力：通过生成式AI素养赋能老年人 

**Authors**: Eunhye Grace Ko, Shaini Nanayakkara, Earl W. Huff Jr  

**Link**: [PDF](https://arxiv.org/pdf/2506.06225)  

**Abstract**: As generative AI (GenAI) becomes increasingly widespread, it is crucial to equip users, particularly vulnerable populations such as older adults (65 and older), with the knowledge to understand its benefits and potential risks. Older adults often exhibit greater reservations about adopting emerging technologies and require tailored literacy support. Using a mixed methods approach, this study examines strategies for delivering GenAI literacy to older adults through a chatbot named Litti, evaluating its impact on their AI literacy (knowledge, safety, and ethical use). The quantitative data indicated a trend toward improved AI literacy, though the results were not statistically significant. However, qualitative interviews revealed diverse levels of familiarity with generative AI and a strong desire to learn more. Findings also show that while Litti provided a positive learning experience, it did not significantly enhance participants' trust or sense of safety regarding GenAI. This exploratory case study highlights the challenges and opportunities in designing AI literacy education for the rapidly growing older adult population. 

**Abstract (ZH)**: 随着生成式人工智能（GenAI）的广泛应用，为用户，特别是老年人（65岁及以上）等脆弱群体提供相关知识以理解其益处和潜在风险变得至关重要。老年人对采用新兴技术往往表现出更大的顾虑，并需要个性化的信息素养支持。采用混合方法，本研究通过名为Litti的聊天机器人探讨向老年人传授GenAI信息素养的策略，并评估其对其AI信息素养（知识、安全和伦理使用）的影响。定量数据分析显示AI信息素养有所提高，但结果不具备统计显著性。然而，定性访谈揭示了人们对生成式AI的不同熟悉程度，并表达了强烈的学习愿望。研究结果还表明，虽然Litti提供了积极的学习体验，但并未显著增强参与者对GenAI的信任感或安全感。该探索性案例研究突显了为迅速增长的老年群体设计AI信息素养教育面临的挑战和机遇。 

---
# Can Theoretical Physics Research Benefit from Language Agents? 

**Title (ZH)**: 理论物理学研究能够从语言代理中受益吗？ 

**Authors**: Sirui Lu, Zhijing Jin, Terry Jingchen Zhang, Pavel Kos, J. Ignacio Cirac, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2506.06214)  

**Abstract**: Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics research is not yet mature. This position paper argues that LLM agents can potentially help accelerate theoretical, computational, and applied physics when properly integrated with domain knowledge and toolbox. We analyze current LLM capabilities for physics -- from mathematical reasoning to code generation -- identifying critical gaps in physical intuition, constraint satisfaction, and reliable reasoning. We envision future physics-specialized LLMs that could handle multimodal data, propose testable hypotheses, and design experiments. Realizing this vision requires addressing fundamental challenges: ensuring physical consistency, and developing robust verification methods. We call for collaborative efforts between physics and AI communities to help advance scientific discovery in physics. 

**Abstract (ZH)**: 大型语言模型在理论物理学研究中的应用尚不成熟，但正迅速迈向多元化领域。本文立场论文认为，当适当结合领域知识和工具箱时，大型语言模型代理有望加速理论物理、计算物理和应用物理的发展。我们分析了当前物理领域的大型语言模型能力——从数学推理到代码生成，并识别出物理直觉、约束满足和可靠推理方面的关键差距。我们展望未来专门化于物理学的大规模语言模型，它们能够处理多模态数据、提出可测试的假设并设计实验。要实现这一愿景，需要解决根本性的挑战：确保物理一致性和开发稳健的验证方法。我们呼吁物理学与人工智能社区之间的合作，以推动物理学中的科学发现。 

---
# Building Models of Neurological Language 

**Title (ZH)**: 构建神经语言模型 

**Authors**: Henry Watkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.06208)  

**Abstract**: This report documents the development and evaluation of domain-specific language models for neurology. Initially focused on building a bespoke model, the project adapted to rapid advances in open-source and commercial medical LLMs, shifting toward leveraging retrieval-augmented generation (RAG) and representational models for secure, local deployment. Key contributions include the creation of neurology-specific datasets (case reports, QA sets, textbook-derived data), tools for multi-word expression extraction, and graph-based analyses of medical terminology. The project also produced scripts and Docker containers for local hosting. Performance metrics and graph community results are reported, with future possible work open for multimodal models using open-source architectures like phi-4. 

**Abstract (ZH)**: 本报告记录了神经学领域特定语言模型的发展与评估。项目最初专注于构建定制模型，后根据开源和商业医疗LLM的快速进步，转向利用检索增强生成（RAG）和表示模型实现安全的本地部署。关键贡献包括创建神经学专用数据集（病例报告、问答集、教材衍生数据）、多词表达提取工具以及医学术语的图谱分析。项目还生成了本地托管的脚本和Docker容器。报告了性能指标和图社区分析结果，并开放了使用开源架构如phi-4的多模态模型未来工作。 

---
# The Lock-in Hypothesis: Stagnation by Algorithm 

**Title (ZH)**: 锁定假设：算法导致的停滞 

**Authors**: Tianyi Alex Qiu, Zhonghao He, Tejasveer Chugh, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2506.06166)  

**Abstract**: The training and deployment of large language models (LLMs) create a feedback loop with human users: models learn human beliefs from data, reinforce these beliefs with generated content, reabsorb the reinforced beliefs, and feed them back to users again and again. This dynamic resembles an echo chamber. We hypothesize that this feedback loop entrenches the existing values and beliefs of users, leading to a loss of diversity and potentially the lock-in of false beliefs. We formalize this hypothesis and test it empirically with agent-based LLM simulations and real-world GPT usage data. Analysis reveals sudden but sustained drops in diversity after the release of new GPT iterations, consistent with the hypothesized human-AI feedback loop. Code and data available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）的训练与部署与人类用户形成一个反馈循环：模型从数据中学习人类的信念，通过生成内容强化这些信念，重新吸收强化后的信念，并再次反馈给用户。这一动态类似于回音室效应。我们假设这个反馈循环巩固了用户的现有价值观和信念，导致多样性的损失，并可能锁定错误的信念。我们形式化这一假设，并通过基于代理的LLM模拟和实际使用的GPT数据进行实证测试。分析发现，在新GPT迭代发布后，多样性突然但持续地下降，与假设的人工智能-人类反馈循环一致。代码和数据可在以下链接获取：this https URL 

---
# Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems 

**Title (ZH)**: Joint-GCG：统一基于梯度的检索增强生成系统中毒攻击 

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06151)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at this https URL. 

**Abstract (ZH)**: 基于检索的生成(RAG)系统通过在生成响应之前从外部语料库检索相关文档，增强了大型语言模型(LLM)的功能。这种方法通过利用大量的最新外部知识，显著扩展了LLM的能力。然而，对外部知识的依赖使得RAG系统容易受到通过注入污染文档进行操纵的语料库污染攻击。现有的污染攻击策略通常将检索和生成阶段视为不相关的，限制了它们的效果。我们提出了一种名为Joint-GCG的新框架，它是第一个通过三项创新统一跨检索器和生成器的梯度攻击的框架：(1) 跨词汇投影以对齐嵌入空间，(2) 梯度标记同步以同步标记级别梯度信号，(3) 自适应加权融合以动态平衡攻击目标。评估结果显示，Joint-GCG在多个检索器和生成器上分别获得最高25%和平均5%更高的攻击成功率，而优化条件下生成的污染剂对未见过的模型显示出前所未有的迁移性。Joint-GCG从根本上重塑了我们对RAG系统中的潜在风险的理解。我们的代码可在以下链接获取。 

---
# Text-to-LoRA: Instant Transformer Adaption 

**Title (ZH)**: Text-to-LoRA：即时Transformer适配 

**Authors**: Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, Robert Tjarko Lange  

**Link**: [PDF](https://arxiv.org/pdf/2506.06105)  

**Abstract**: While Foundation Models provide a general tool for rapid content creation, they regularly require task-specific adaptation. Traditionally, this exercise involves careful curation of datasets and repeated fine-tuning of the underlying model. Fine-tuning techniques enable practitioners to adapt foundation models for many new applications but require expensive and lengthy training while being notably sensitive to hyper-parameter choices. To overcome these limitations, we introduce Text-to-LoRA (T2L), a model capable of adapting Large Language Models on the fly solely based on a natural language description of the target task. T2L is a hypernetwork trained to construct LoRAs in a single inexpensive forward pass. After training T2L on a suite of 9 pre-trained LoRA adapters (GSM8K, Arc, etc.), we show that the ad-hoc reconstructed LoRA instances match the performance of task-specific adapters across the corresponding test sets. Furthermore, T2L can compress hundreds of LoRA instances and zero-shot generalize to entirely unseen tasks. This approach provides a significant step towards democratizing the specialization of foundation models and enables language-based adaptation with minimal compute requirements. Our code is available at this https URL 

**Abstract (ZH)**: 基于文本的LoRA (T2L): 一种仅依赖目标任务自然语言描述即可实时适配大规模语言模型的方法 

---
# Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models 

**Title (ZH)**: 简单而有效的 federated 大型语言模型微调中跨客户端提取私人数据方法 

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.06060)  

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning. 

**Abstract (ZH)**: 联邦微调大型语言模型（FedLLMs）为在敏感领域实现强大的模型性能同时保持数据隐私提供了一种有前景的方法。然而，大型语言模型固有的记忆能力使它们容易受到训练数据提取攻击。为了研究这一风险，我们引入了一种简单有效的提取攻击算法，专门针对FedLLMs。相比之下，先前的“逐字”提取攻击假设可以访问所有训练数据的片段，而我们的方法则在攻击者只能访问单个客户端数据并试图从其他客户端提取未见过的个人可识别信息（PII）的更现实威胁模型下运作。这需要利用攻击者持有的上下文前缀来跨客户端进行泛化。为了评估我们方法的有效性，我们提出了两个严格的度量标准——覆盖率和效率，并扩展了一个符合CPIS、GDPR和CCPA标准的现实世界法律数据集，实现89.9%的人工验证精度。实验结果表明，我们的方法可以提取高达56.57%的受害者独有的PII，“地址”、“生日”和“姓名”是最容易受损的类别。我们的研究结果突显了制定 robust 防御策略的紧迫性，并为未来隐私保护联邦学习的研究提供了新的基准和评估框架。 

---
# Hey, That's My Data! Label-Only Dataset Inference in Large Language Models 

**Title (ZH)**: 嘿，这是我的数据！基于大型语言模型的标签唯一数据集推理 

**Authors**: Chen Xiong, Zihao Wang, Rui Zhu, Tsung-Yi Ho, Pin-Yu Chen, Jingwei Xiong, Haixu Tang, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2506.06057)  

**Abstract**: Large Language Models (LLMs) have revolutionized Natural Language Processing by excelling at interpreting, reasoning about, and generating human language. However, their reliance on large-scale, often proprietary datasets poses a critical challenge: unauthorized usage of such data can lead to copyright infringement and significant financial harm. Existing dataset-inference methods typically depend on log probabilities to detect suspicious training material, yet many leading LLMs have begun withholding or obfuscating these signals. This reality underscores the pressing need for label-only approaches capable of identifying dataset membership without relying on internal model logits.
We address this gap by introducing CatShift, a label-only dataset-inference framework that capitalizes on catastrophic forgetting: the tendency of an LLM to overwrite previously learned knowledge when exposed to new data. If a suspicious dataset was previously seen by the model, fine-tuning on a portion of it triggers a pronounced post-tuning shift in the model's outputs; conversely, truly novel data elicits more modest changes. By comparing the model's output shifts for a suspicious dataset against those for a known non-member validation set, we statistically determine whether the suspicious set is likely to have been part of the model's original training corpus. Extensive experiments on both open-source and API-based LLMs validate CatShift's effectiveness in logit-inaccessible settings, offering a robust and practical solution for safeguarding proprietary data. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过在解释、推理和生成人类语言方面表现出色，彻底改变了自然语言处理。然而，它们对大规模、经常是专有数据集的依赖提出了一个关键挑战：未经授权使用这些数据可能导致版权侵权和重大经济损失。现有的数据集推断方法通常依赖于对数概率来检测可疑训练材料，而许多领先的LLM已经开始隐藏或模糊这些信号。这一现实凸显了急需一种仅标签的方法，它能够在不依赖内部模型逻辑值的情况下识别数据集成员身份的需求。我们通过引入CatShift——一种利用灾难性遗忘的数据集推断框架来填补这一空白：灾难性遗忘是指当LLM接触到新数据时，会覆盖之前学习的知识的倾向。如果可疑数据集之前被模型见过，对其进行部分微调会导致模型输出显着变化；相反，真正新颖的数据只会引起适度的变化。通过将可疑数据集的模型输出变化与已知非成员验证集的变化进行统计比较，我们能够确定可疑数据集很可能是模型原始训练语料的一部分。广泛的实验在开源和基于API的LLM上验证了CatShift在对数概率不可访问环境中的有效性，提供了一种 robust 和实用的解决方案来保护专有数据。 

---
# When to Trust Context: Self-Reflective Debates for Context Reliability 

**Title (ZH)**: 何时信任背景：关于背景可靠性的自我反思辩论 

**Authors**: Zeqi Zhou, Fang Wu, Shayan Talaei, Haokai Zhao, Cheng Meixin, Tinson Xu, Amin Saberi, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.06020)  

**Abstract**: Large language models frequently encounter conflicts between their parametric knowledge and contextual input, often resulting in factual inconsistencies or hallucinations. We propose Self-Reflective Debate for Contextual Reliability (SR-DCR), a lightweight framework that integrates token-level self-confidence with an asymmetric multi-agent debate to adjudicate such conflicts. A critic, deprived of context, challenges a defender who argues from the given passage; a judge model evaluates the debate and determines the context's reliability. The final answer is selected by combining the verdict with model confidence. Experiments on the ClashEval benchmark demonstrate that SR-DCR consistently enhances robustness to misleading context while maintaining accuracy on trustworthy inputs, outperforming both classical debate and confidence-only baselines with minimal computational overhead. The code is available at this https URL. 

**Abstract (ZH)**: 自省辩论以提升上下文可靠性（SR-DCR）：一种轻量级框架 

---
# Unlocking Recursive Thinking of LLMs: Alignment via Refinement 

**Title (ZH)**: 解锁LLMs的递归思考：通过精炼实现对齐 

**Authors**: Haoke Zhang, Xiaobo Liang, Cunxiang Wang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.06009)  

**Abstract**: The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (this https URL). 

**Abstract (ZH)**: AvR：通过长形式链式思考优化的递归对齐方法 

---
# Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models 

**Title (ZH)**: .token Signature: 通过Token解码特征预测大型语言模型中的chain-of-thought收益 

**Authors**: Peijie Liu, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.06008)  

**Abstract**: Chain-of-Thought (CoT) technique has proven effective in improving the performance of large language models (LLMs) on complex reasoning tasks. However, the performance gains are inconsistent across different tasks, and the underlying mechanism remains a long-standing research question. In this work, we make a preliminary observation that the monotonicity of token probability distributions may be correlated with the gains achieved through CoT reasoning. Leveraging this insight, we propose two indicators based on the token probability distribution to assess CoT effectiveness across different tasks. By combining instance-level indicators with logistic regression model, we introduce Dynamic CoT, a method that dynamically select between CoT and direct answer. Furthermore, we extend Dynamic CoT to closed-source models by transferring decision strategies learned from open-source models. Our indicators for assessing CoT effectiveness achieve an accuracy of 89.2\%, and Dynamic CoT reduces token consumption by more than 35\% while maintaining high accuracy. Overall, our work offers a novel perspective on the underlying mechanisms of CoT reasoning and provides a framework for its more efficient deployment. 

**Abstract (ZH)**: Chain-of-Thought技术在提升大型语言模型复杂推理任务性能方面的初步观察及其动态应用研究 

---
# Leveraging Generative AI for Enhancing Automated Assessment in Programming Education Contests 

**Title (ZH)**: 利用生成式AI提升编程教育竞赛中的自动化评估 

**Authors**: Stefan Dascalescu, Adrian Marius Dumitran, Mihai Alexandru Vasiluta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05990)  

**Abstract**: Competitive programming contests play a crucial role in cultivating computational thinking and algorithmic skills among learners. However, generating comprehensive test cases to effectively assess programming solutions remains resource-intensive and challenging for educators. This paper introduces an innovative NLP-driven method leveraging generative AI (large language models) to automate the creation of high-quality test cases for competitive programming assessments. We extensively evaluated our approach on diverse datasets, including 25 years of Romanian Informatics Olympiad (OJI) data for 5th graders, recent competitions hosted on the this http URL platform, and the International Informatics Olympiad in Teams (IIOT). Our results demonstrate that AI-generated test cases substantially enhanced assessments, notably identifying previously undetected errors in 67% of the OJI 5th grade programming problems. These improvements underscore the complementary educational value of our technique in formative assessment contexts. By openly sharing our prompts, translated datasets, and methodologies, we offer practical NLP-based tools that educators and contest organizers can readily integrate to enhance assessment quality, reduce workload, and deepen insights into learner performance. 

**Abstract (ZH)**: 基于NLP的生成式AI方法在编程竞赛评估中自动创建高质量测试案例的研究 

---
# Audio-Aware Large Language Models as Judges for Speaking Styles 

**Title (ZH)**: 具有音频意识的大语言模型作为演讲风格的评判者 

**Authors**: Cheng-Han Chiang, Xiaofei Wang, Chung-Ching Lin, Kevin Lin, Linjie Li, Radu Kopetz, Yao Qian, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05984)  

**Abstract**: Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues. 

**Abstract (ZH)**: Audio-aware大型语言模型（ALLMs）能够理解音频输入中的文本和非文本信息。本文探索使用ALLMs作为自动评委评估演讲的演讲风格。我们使用ALLM评委评估由四个口语模型（SLMs）完成的两种任务（语音风格指令跟随和角色扮演）所产生的演讲，使用人类评委和ALLM评委对SLMs的响应进行评估。我们比较了两种ALLM评委，GPT-4o-audio和Gemini-2.5-pro，与人类评估结果，并展示了Gemini和人类评委之间的共识与人类评估者之间的共识相当。这些有希望的结果表明，ALLMs可以作为评委来评估SLMs。我们的结果还表明，当前的SLMs，即使包括GPT-4o-audio，仍然在控制演讲风格和生成自然对话方面存在改进空间。 

---
# Let's Put Ourselves in Sally's Shoes: Shoes-of-Others Prefixing Improves Theory of Mind in Large Language Models 

**Title (ZH)**: 让我们换位思考萨利的处境：他人鞋类前缀提高大规模语言模型的理论思维能力 

**Authors**: Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Yoshihiro Yamazaki, Keita Suzuki, Hiroaki Sugiyama, Kuniko Saito  

**Link**: [PDF](https://arxiv.org/pdf/2506.05970)  

**Abstract**: Recent studies have shown that Theory of Mind (ToM) in large language models (LLMs) has not reached human-level performance yet. Since fine-tuning LLMs on ToM datasets often degrades their generalization, several inference-time methods have been proposed to enhance ToM in LLMs. However, existing inference-time methods for ToM are specialized for inferring beliefs from contexts involving changes in the world state. In this study, we present a new inference-time method for ToM, Shoes-of-Others (SoO) prefixing, which makes fewer assumptions about contexts and is applicable to broader scenarios. SoO prefixing simply specifies the beginning of LLM outputs with ``Let's put ourselves in A's shoes.'', where A denotes the target character's name. We evaluate SoO prefixing on two benchmarks that assess ToM in conversational and narrative contexts without changes in the world state and find that it consistently improves ToM across five categories of mental states. Our analysis suggests that SoO prefixing elicits faithful thoughts, thereby improving the ToM performance. 

**Abstract (ZH)**: Recent studies have shown that大语言模型（LLM）中的心智理论（ToM）尚未达到人类水平的表现。由于在ToM数据集上微调LLM往往会损害其泛化能力，因此已经提出了几种推理时的方法来增强LLM中的ToM。然而，现有的ToM推理时方法专门用于推断涉及世界状态变化的情景下的信念。在本研究中，我们提出了一种新的ToM推理时方法——他人的鞋子前缀（Shoes-of-Others, SoO prefixing），该方法对背景的假设较少，并适用于更广泛的情景。SoO前缀简单地规定LLM输出的开始部分为“让我们站在A的立场上看问题。”，其中A表示目标人物的名称。我们在两个评估对话和叙述背景下ToM而无需世界状态变化的基准上评估了SoO前缀，发现它在五类心理状态中都能一致地提升ToM性能。我们的分析表明，SoO前缀激发了忠实的想法，从而提高了ToM性能。 

---
# IntentionESC: An Intention-Centered Framework for Enhancing Emotional Support in Dialogue Systems 

**Title (ZH)**: 意图导向的情感支持对话系统框架：一种基于意图的框架 

**Authors**: Xinjie Zhang, Wenxuan Wang, Qin Jin  

**Link**: [PDF](https://arxiv.org/pdf/2506.05947)  

**Abstract**: In emotional support conversations, unclear intentions can lead supporters to employ inappropriate strategies, inadvertently imposing their expectations or solutions on the seeker. Clearly defined intentions are essential for guiding both the supporter's motivations and the overall emotional support process. In this paper, we propose the Intention-centered Emotional Support Conversation (IntentionESC) framework, which defines the possible intentions of supporters in emotional support conversations, identifies key emotional state aspects for inferring these intentions, and maps them to appropriate support strategies. While Large Language Models (LLMs) excel in text generating, they fundamentally operate as probabilistic models trained on extensive datasets, lacking a true understanding of human thought processes and intentions. To address this limitation, we introduce the Intention Centric Chain-of-Thought (ICECoT) mechanism. ICECoT enables LLMs to mimic human reasoning by analyzing emotional states, inferring intentions, and selecting suitable support strategies, thereby generating more effective emotional support responses. To train the model with ICECoT and integrate expert knowledge, we design an automated annotation pipeline that produces high-quality training data. Furthermore, we develop a comprehensive evaluation scheme to assess emotional support efficacy and conduct extensive experiments to validate our framework. Our data and code are available at this https URL. 

**Abstract (ZH)**: 基于意图的情感支持对话框架（IntentionESC）及其生成机制（ICECoT）的研究 

---
# DynamicMind: A Tri-Mode Thinking System for Large Language Models 

**Title (ZH)**: DynamicMind: 大型语言模型的三模式思考系统 

**Authors**: Wei Li, Yanbin Wei, Qiushi Huang, Jiangyue Yan, Yang Chen, James T. Kwok, Yu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05936)  

**Abstract**: Modern large language models (LLMs) often struggle to dynamically adapt their reasoning depth to varying task complexities, leading to suboptimal performance or inefficient resource utilization. To address this, we introduce DynamicMind, a novel tri-mode thinking system. DynamicMind empowers LLMs to autonomously select between Fast, Normal, and Slow thinking modes for zero-shot question answering (ZSQA) tasks through cognitive-inspired prompt engineering. Our framework's core innovations include: (1) expanding the established dual-process framework of fast and slow thinking into a tri-mode thinking system involving a normal thinking mode to preserve the intrinsic capabilities of LLM; (2) proposing the Thinking Density metric, which aligns computational resource allocation with problem complexity; and (3) developing the Thinking Mode Capacity (TMC) dataset and a lightweight Mind Router to predict the optimal thinking mode. Extensive experiments across diverse mathematical, commonsense, and scientific QA benchmarks demonstrate that DynamicMind achieves superior ZSQA capabilities while establishing an effective trade-off between performance and computational efficiency. 

**Abstract (ZH)**: 现代大规模语言模型（LLMs）往往难以动态适应不同任务复杂性的推理深度，导致性能不佳或资源利用效率低下。为解决这一问题，我们引入了DynamicMind，一个新颖的三模思考系统。DynamicMind通过认知启发式的提示工程，使LLMs自主选择快速、正常和慢速思考模式，以应对零样本问答（ZSQA）任务。该框架的核心创新包括：（1）将快速和慢速思考的双过程框架扩展为包含正常思考模式的三模思考系统，以保留LLM的核心能力；（2）提出了思考密度度量，该度量将计算资源分配与问题复杂性对齐；（3）开发了思考模式容量（TMC）数据集和轻量级Mind Router，以预测最优思考模式。广泛的实验跨不同领域的数学、常识和科学问答基准表明，DynamicMind在提高零样本问答能力的同时，有效地在性能和计算效率之间找到了平衡。 

---
# MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models 

**Title (ZH)**: MoA：参数高效调整大型语言模型的异质适配器混合 

**Authors**: Jie Cao, Tianwei Lin, Hongyang He, Rolan Yan, Wenqiao Zhang, Juncheng Li, Dongping Zhang, Siliang Tang, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05928)  

**Abstract**: Recent studies integrate Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) to further enhance the performance of parameter-efficient fine-tuning (PEFT) methods in Large Language Model (LLM) applications. Existing methods employ \emph{homogeneous} MoE-LoRA architectures composed of LoRA experts with either similar or identical structures and capacities. However, these approaches often suffer from representation collapse and expert load imbalance, which negatively impact the potential of LLMs. To address these challenges, we propose a \emph{heterogeneous} \textbf{Mixture-of-Adapters (MoA)} approach. This method dynamically integrates PEFT adapter experts with diverse structures, leveraging their complementary representational capabilities to foster expert specialization, thereby enhancing the effective transfer of pre-trained knowledge to downstream tasks. MoA supports two variants: \textbf{(i)} \textit{Soft MoA} achieves fine-grained integration by performing a weighted fusion of all expert outputs; \textbf{(ii)} \textit{Sparse MoA} activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation. Experimental results demonstrate that heterogeneous MoA outperforms homogeneous MoE-LoRA methods in both performance and parameter efficiency. Our project is available at this https URL. 

**Abstract (ZH)**: 近期研究表明，将低秩适应（LoRA）与专家混排（MoE） integrates 与进一步增强大规模语言模型（LLM）应用中参数高效微调（PEFT）方法的性能。现有方法采用同质的MoE-LoRA架构，由具有相似或相同结构和容量的LoRA专家组成。然而，这些方法往往会导致表示坍塌和专家负载不平衡，从而负面影响LLM的潜力。为解决这些挑战，我们提出了一种异质的Mixture-of-Adapters（MoA）方法。该方法通过动态集成具有不同结构的PEFT适配器专家，利用它们互补的表示能力促进专家的专业化，从而增强预训练知识向下游任务的有效转移。MoA支持两种变体：（i）软MoA通过加权融合所有专家输出实现精细集成；（ii）稀疏MoA基于贡献稀疏激活适配器专家，实现接近无性能下降的效果。实验结果表明，异质MoA在性能和参数效率方面均优于同质MoE-LoRA方法。我们的项目可在以下链接访问：this https URL。 

---
# Small Models, Big Support: A Local LLM Framework for Teacher-Centric Content Creation and Assessment using RAG and CAG 

**Title (ZH)**: 小模型，大支持：基于RAG和CAG的以教师为中心的内容创编与评估本地LLM框架 

**Authors**: Zarreen Reza, Alexander Mazur, Michael T. Dugdale, Robin Ray-Chaudhuri  

**Link**: [PDF](https://arxiv.org/pdf/2506.05925)  

**Abstract**: While Large Language Models (LLMs) are increasingly utilized as student-facing educational aids, their potential to directly support educators, particularly through locally deployable and customizable open-source solutions, remains significantly underexplored. Many existing educational solutions rely on cloud-based infrastructure or proprietary tools, which are costly and may raise privacy concerns. Regulated industries with limited budgets require affordable, self-hosted solutions. We introduce an end-to-end, open-source framework leveraging small (3B-7B parameters), locally deployed LLMs for customized teaching material generation and assessment. Our system uniquely incorporates an interactive loop crucial for effective small-model refinement, and an auxiliary LLM verifier to mitigate jailbreaking risks, enhancing output reliability and safety. Utilizing Retrieval and Context Augmented Generation (RAG/CAG), it produces factually accurate, customized pedagogically-styled content. Deployed on-premises for data privacy and validated through an evaluation pipeline and a college physics pilot, our findings show that carefully engineered small LLM systems can offer robust, affordable, practical, and safe educator support, achieving utility comparable to larger models for targeted tasks. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs） increasingly被用作面向学生的教育辅助工具，它们直接支持教育工作者的潜力，尤其是在通过本地部署和可定制的开源解决方案方面，仍然被显著低估。许多现有的教育解决方案依赖于基于云的基础设施或专有工具，这往往成本高昂且可能引发隐私担忧。受到预算限制的受监管行业需要负担得起的、可自行托管的解决方案。我们引入了一个端到端的开源框架，利用本地部署的小型（3B-7B参数）LLM来生成和评估个性化的教学材料。我们的系统独特地集成了一个对于有效微调小型模型至关重要的互动循环，并包含一个辅助LLM验证器以减轻突破限制的风险，从而提高输出可靠性和安全性。利用检索和上下文增强生成（RAG/CAG）技术，它能够生成事实准确且符合教学风格的内容。该系统内置数据隐私保护措施，并通过评估管道和大学物理试点项目进行了验证，研究发现精心设计的小型LLM系统可以提供稳健、经济、实用且安全的教育支持，即使对于特定任务，其功能效用也与大型模型相当。 

---
# Route-and-Reason: Scaling Large Language Model Reasoning with Reinforced Model Router 

**Title (ZH)**: 路线与理由：强化模型路由扩展大型语言模型推理 

**Authors**: Chenyang Shao, Xinyang Liu, Yutang Lin, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05901)  

**Abstract**: Multi-step reasoning has proven essential for enhancing the problem-solving capabilities of Large Language Models (LLMs) by decomposing complex tasks into intermediate steps, either explicitly or implicitly. Extending the reasoning chain at test time through deeper thought processes or broader exploration, can furthur improve performance, but often incurs substantial costs due to the explosion in token usage. Yet, many reasoning steps are relatively simple and can be handled by more efficient smaller-scale language models (SLMs). This motivates hybrid approaches that allocate subtasks across models of varying capacities. However, realizing such collaboration requires accurate task decomposition and difficulty-aware subtask allocation, which is challenging. To address this, we propose R2-Reasoner, a novel framework that enables collaborative reasoning across heterogeneous LLMs by dynamically routing sub-tasks based on estimated complexity. At the core of our framework is a Reinforced Model Router, composed of a task decomposer and a subtask allocator. The task decomposer segments complex input queries into logically ordered subtasks, while the subtask allocator assigns each subtask to the most appropriate model, ranging from lightweight SLMs to powerful LLMs, balancing accuracy and efficiency. To train this router, we introduce a staged pipeline that combines supervised fine-tuning on task-specific datasets with Group Relative Policy Optimization algorithm, enabling self-supervised refinement through iterative reinforcement learning. Extensive experiments across four challenging benchmarks demonstrate that R2-Reasoner reduces API costs by 86.85% while maintaining or surpassing baseline accuracy. Our framework paves the way for more cost-effective and adaptive LLM reasoning. The code is open-source at this https URL . 

**Abstract (ZH)**: 多步推理已被证明对于通过将复杂任务分解为中间步骤（显式或隐式）来增强大型语言模型（LLMs）的问题解决能力是必不可少的。通过在测试时进行更深入的思考过程或更广泛的探索来延伸推理链，可以进一步提高性能，但由于标记使用量的爆炸性增长，往往会带来显著的成本。然而，许多推理步骤相对简单，可以由更高效的小规模语言模型（SLMs）处理。这促成了跨不同容量模型分配子任务的混合方法。然而，实现这种协作需要准确的任务分解和难度感知的子任务分配，这是具有挑战性的。为了解决这一问题，我们提出了一种新的R2-Reasoner框架，该框架允许通过基于估计复杂性的动态路由在异构LLMs之间进行协作推理。该框架的核心是一个强化模型路由器，由一个任务分解器和一个子任务分配器组成。任务分解器将复杂的输入查询分割成逻辑有序的子任务，而子任务分配器将每个子任务分配给最适合的模型，从轻量级SLMs到强大的LLMs，平衡准确性和效率。为了训练这个路由器，我们引入了一种分阶段管道，结合特定任务数据集上的监督微调和群组相对策略优化算法，通过迭代强化学习实现自我监督的精炼。在四个具有挑战性的基准上进行的广泛实验表明，R2-Reasoner将API成本降低了86.85%，同时保持或超越了基线准确性。该框架为更经济高效和适应性的LLM推理铺平了道路。代码在该网址处开源。 

---
# Research on Personalized Financial Product Recommendation by Integrating Large Language Models and Graph Neural Networks 

**Title (ZH)**: 基于大规模语言模型和图神经网络的个性化金融产品推荐研究 

**Authors**: Yushang Zhao, Yike Peng, Dannier Li, Yuxin Yang, Chengrui Zhou, Jing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2506.05873)  

**Abstract**: With the rapid growth of fintech, personalized financial product recommendations have become increasingly important. Traditional methods like collaborative filtering or content-based models often fail to capture users' latent preferences and complex relationships. We propose a hybrid framework integrating large language models (LLMs) and graph neural networks (GNNs). A pre-trained LLM encodes text data (e.g., user reviews) into rich feature vectors, while a heterogeneous user-product graph models interactions and social ties. Through a tailored message-passing mechanism, text and graph information are fused within the GNN to jointly optimize embeddings. Experiments on public and real-world financial datasets show our model outperforms standalone LLM or GNN in accuracy, recall, and NDCG, with strong interpretability. This work offers new insights for personalized financial recommendations and cross-modal fusion in broader recommendation tasks. 

**Abstract (ZH)**: 金融科技的快速成长使得个性化金融产品推荐愈发重要。传统方法如协作过滤或基于内容的模型往往难以捕捉用户的潜在偏好和复杂关系。我们提出一种结合大型语言模型（LLMs）和图神经网络（GNNs）的混合框架。预训练的LLM将文本数据（如用户评论）编码为丰富的特征向量，而异质用户-产品图则用于建模交互和社会关系。通过定制的消息传递机制，文本和图信息在GNN中融合以协同优化嵌入表示。在公共和真实世界的金融数据集上的实验表明，我们的模型在准确率、召回率和NDCG方面优于独立的LLM或GNN，并具有较强的可解释性。本工作为个性化金融推荐和更广泛的推荐任务中的跨模态融合提供了新的见解。 

---
# Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models 

**Title (ZH)**: 跨语言坍缩：语言中心基础模型如何塑造大型语言模型的推理过程 

**Authors**: Cheonbok Park, Jeonghoon Kim, Joosung Lee, Sanghwan Bae, Jaegul Choo, Kangmin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.05850)  

**Abstract**: We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning. 

**Abstract (ZH)**: 跨语言坍缩：多语言语言模型链式思考的系统性漂移 

---
# dots.llm1 Technical Report 

**Title (ZH)**: dots.llm1 技术报告 

**Authors**: Bi Huo, Bin Tu, Cheng Qin, Da Zheng, Debing Zhang, Dongjie Zhang, En Li, Fu Guo, Jian Yao, Jie Lou, Junfeng Tian, Li Hu, Ran Zhu, Shengdong Chen, Shuo Liu, Su Guang, Te Wo, Weijun Zhang, Xiaoming Shi, Xinxin Peng, Xing Wu, Yawen Liu, Yuqiu Ji, Ze Wen, Zhenhai Liu, Zichao Li, Zilong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05767)  

**Abstract**: Mixture of Experts (MoE) models have emerged as a promising paradigm for scaling language models efficiently by activating only a subset of parameters for each input token. In this report, we present dots.llm1, a large-scale MoE model that activates 14B parameters out of a total of 142B parameters, delivering performance on par with state-of-the-art models while reducing training and inference costs. Leveraging our meticulously crafted and efficient data processing pipeline, dots.llm1 achieves performance comparable to Qwen2.5-72B after pretraining on 11.2T high-quality tokens and post-training to fully unlock its capabilities. Notably, no synthetic data is used during pretraining. To foster further research, we open-source intermediate training checkpoints at every one trillion tokens, providing valuable insights into the learning dynamics of large language models. 

**Abstract (ZH)**: 混合专家模型（MoE）作为一种高效扩展语言模型的 paradigm，通过为每个输入令牌仅激活部分参数而得以发展。本报告介绍了 dots.llm1，一个大型 MoE 模型，激活了总计 142B 参数中的 14B 参数，其性能与最先进的模型相当，同时降低了训练和推理成本。借助我们精心设计且高效的數據处理管道，dots.llm1 在预训练 11.2T 高质量令牌后，经过进一步训练以充分解锁其能力，其性能与 Qwen2.5-72B 相当。值得注意的是，预训练过程中未使用合成数据。为了促进进一步研究，我们开源了每隔一万亿令牌的中间训练检查点，提供了关于大型语言模型学习动态的宝贵见解。 

---
# Efficient Online RFT with Plug-and-Play LLM Judges: Unlocking State-of-the-Art Performance 

**Title (ZH)**: 高效的在线RFT与插拔式LLM裁判：解锁最先进的性能 

**Authors**: Rudransh Agnihotri, Ananya Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2506.05748)  

**Abstract**: Reward-model training is the cost bottleneck in modern Reinforcement Learning Human Feedback (RLHF) pipelines, often requiring tens of billions of parameters and an offline preference-tuning phase. In the proposed method, a frozen, instruction-tuned 7B LLM is augmented with only a one line JSON rubric and a rank-16 LoRA adapter (affecting just 0.8% of the model's parameters), enabling it to serve as a complete substitute for the previously used heavyweight evaluation models. The plug-and-play judge achieves 96.2% accuracy on RewardBench, outperforming specialized reward networks ranging from 27B to 70B parameters. Additionally, it allows a 7B actor to outperform the top 70B DPO baseline, which scores 61.8%, by achieving 92% exact match accuracy on GSM-8K utilizing online PPO. Thorough ablations indicate that (i) six in context demonstrations deliver the majority of the zero-to-few-shot improvements (+2pp), and (ii) the LoRA effectively addresses the remaining disparity, particularly in the safety and adversarial Chat-Hard segments. The proposed model introduces HH-Rationales, a subset of 10,000 pairs from Anthropic HH-RLHF, to examine interpretability, accompanied by human generated justifications. GPT-4 scoring indicates that our LoRA judge attains approximately = 9/10 in similarity to human explanations, while zero-shot judges score around =5/10. These results indicate that the combination of prompt engineering and tiny LoRA produces a cost effective, transparent, and easily adjustable reward function, removing the offline phase while achieving new state-of-the-art outcomes for both static evaluation and online RLHF. 

**Abstract (ZH)**: 基于插拔式裁判的高效RLHF方法：结合提示工程与小型LoRA生成成本 Effective, Transparent, and Easily Adjustable Reward Function via Plug-and-Play Judge and Prompt Engineering with Small LoRA 

---
# To Protect the LLM Agent Against the Prompt Injection Attack with Polymorphic Prompt 

**Title (ZH)**: 使用多态提示来保护LLM代理免受提示注入攻击 

**Authors**: Zhilong Wang, Neha Nagaraja, Lan Zhang, Hayretdin Bahsi, Pawan Patil, Peng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05739)  

**Abstract**: LLM agents are widely used as agents for customer support, content generation, and code assistance. However, they are vulnerable to prompt injection attacks, where adversarial inputs manipulate the model's behavior. Traditional defenses like input sanitization, guard models, and guardrails are either cumbersome or ineffective. In this paper, we propose a novel, lightweight defense mechanism called Polymorphic Prompt Assembling (PPA), which protects against prompt injection with near-zero overhead. The approach is based on the insight that prompt injection requires guessing and breaking the structure of the system prompt. By dynamically varying the structure of system prompts, PPA prevents attackers from predicting the prompt structure, thereby enhancing security without compromising performance. We conducted experiments to evaluate the effectiveness of PPA against existing attacks and compared it with other defense methods. 

**Abstract (ZH)**: LLM代理广泛用于客户支持、内容生成和代码辅助。然而，它们容易受到提示注入攻击的影响，即攻击者通过操纵输入来改变模型的行为。传统的防御方法如输入 sanitization、防护模型和防护栏要么复杂难以实施，要么效果不佳。本文提出了一种新颖的轻量级防御机制——多态性提示组装（PPA），该机制能够以接近零的性能开销来抵御提示注入攻击。该方法基于这样一个洞察：提示注入需要猜测和破解系统提示的结构。通过动态变化系统提示的结构，PPA 阻止攻击者预测提示结构，从而在不牺牲性能的情况下增强安全性。我们进行了实验评估了 PPA 对现有攻击的有效性，并将其与其它防御方法进行了比较。 

---
# Large Language Models are Good Relational Learners 

**Title (ZH)**: 大型语言模型是良好的关系学习者 

**Authors**: Fang Wu, Vijay Prakash Dwivedi, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2506.05725)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various domains, yet their application to relational deep learning (RDL) remains underexplored. Existing approaches adapt LLMs by traversing relational links between entities in a database and converting the structured data into flat text documents. Still, this text-based serialization disregards critical relational structures, introduces redundancy, and often exceeds standard LLM context lengths. We introduce Rel-LLM, a novel architecture that utilizes a graph neural network (GNN)- based encoder to generate structured relational prompts for LLMs within a retrieval-augmented generation (RAG) framework. Unlike traditional text-based serialization approaches, our method preserves the inherent relational structure of databases while enabling LLMs to effectively process and reason over complex entity relationships. Specifically, the GNN encoder extracts a local subgraph around an entity to build feature representations that contain relevant entity relationships and temporal dependencies. These representations are transformed into structured prompts using a denormalization process, effectively allowing the LLM to reason over relational structures. Through extensive experiments, we demonstrate that Rel-LLM outperforms existing methods on key RDL tasks, offering a scalable and efficient approach to integrating LLMs with structured data sources. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现了卓越的能力，但其在关系深度学习（RDL）中的应用仍处于探索之中。现有方法通过遍历数据库实体间的关系链接并转换结构化数据为扁平文本文件来适应LLMs，但这种基于文本的序列化方式忽视了关键的关系结构，引入了冗余，并且常超过标准LLM的上下文长度。我们提出Rel-LLM，这是一种新颖的架构，利用基于图神经网络（GNN）的编码器在检索增强生成（RAG）框架中为LLMs生成结构化的关系提示。与传统的基于文本的序列化方法不同，我们的方法保留了数据库的固有关系结构，同时使LLMs能够有效处理和推理复杂的实体关系。具体而言，GNN编码器提取实体周围的局部子图，构建包含相关实体关系和时间依赖性的特征表示。这些表示通过反规范化过程转换为结构化提示，有效地使LLMs能够在关系结构上进行推理。通过广泛的实验，我们证明Rel-LLM在关键的RDL任务上优于现有方法，提供了一种将LLMs与结构化数据源集成的可扩展和高效的方法。代码可访问：这个链接。 

---
# RKEFino1: A Regulation Knowledge-Enhanced Large Language Model 

**Title (ZH)**: RKEFino1：一种调控知识增强的大语言模型 

**Authors**: Yan Wang, Yueru He, Ruoyu Xiang, Jeff Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.05700)  

**Abstract**: Recent advances in large language models (LLMs) hold great promise for financial applications but introduce critical accuracy and compliance challenges in Digital Regulatory Reporting (DRR). To address these issues, we propose RKEFino1, a regulation knowledge-enhanced financial reasoning model built upon Fino1, fine-tuned with domain knowledge from XBRL, CDM, and MOF. We formulate two QA tasks-knowledge-based and mathematical reasoning-and introduce a novel Numerical NER task covering financial entities in both sentences and tables. Experimental results demonstrate the effectiveness and generalization capacity of RKEFino1 in compliance-critical financial tasks. We have released our model on Hugging Face. 

**Abstract (ZH)**: Recent Advances in Large Language Models for Digital Regulatory Reporting: Introducing RKEFino1, a Regulation Knowledge-Enhanced Financial Reasoning Model 

---
# SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code 

**Title (ZH)**: SafeGenBench: 一个用于检测LLM生成代码中的安全漏洞的基准框架 

**Authors**: Xinghang Li, Jingzhe Ding, Chao Peng, Bing Zhao, Xiang Gao, Hongwan Gao, Xinchen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.05692)  

**Abstract**: The code generation capabilities of large language models(LLMs) have emerged as a critical dimension in evaluating their overall performance. However, prior research has largely overlooked the security risks inherent in the generated code. In this work, we introduce \benchmark, a benchmark specifically designed to assess the security of LLM-generated code. The dataset encompasses a wide range of common software development scenarios and vulnerability types. Building upon this benchmark, we develop an automatic evaluation framework that leverages both static application security testing(SAST) and LLM-based judging to assess the presence of security vulnerabilities in model-generated code. Through the empirical evaluation of state-of-the-art LLMs on \benchmark, we reveal notable deficiencies in their ability to produce vulnerability-free code. Our findings highlight pressing challenges and offer actionable insights for future advancements in the secure code generation performance of LLMs. The data and code will be released soon. 

**Abstract (ZH)**: 大型语言模型(LLMs)的代码生成能力已成为评估其整体性能的关键维度。然而，先前的研究大多忽视了生成代码中固有的安全风险。在本文中，我们引入了benchmark，一个专门用于评估LLM生成代码安全性基准。该数据集涵盖了广泛常见的软件开发场景和漏洞类型。基于此基准，我们开发了一种自动评估框架，该框架结合了静态应用安全测试(SAST)和基于LLM的评估，以评估模型生成代码中是否存在安全漏洞。通过对benchmark上的先进LLM进行实证评估，我们揭示了它们在产生无漏洞代码方面存在的显著缺陷。我们的研究结果突显了紧迫的挑战，并为未来提升LLM安全代码生成性能提供了可操作的见解。数据和代码将于不久后发布。 

---
# Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework 

**Title (ZH)**: 基于LLM的面向部署性的基础设施即代码生成迭代框架 

**Authors**: Tianyi Zhang, Shidong Pan, Zejun Zhang, Zhenchang Xing, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.05623)  

**Abstract**: Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research. 

**Abstract (ZH)**: 基于语言模型的基础设施即代码生成及其可部署性评估 

---
# SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs 

**Title (ZH)**: SynthesizeMe！基于人设引导提示的个性化奖励模型合成 

**Authors**: Michael J Ryan, Omar Shaikh, Aditri Bhagirath, Daniel Frees, William Held, Diyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05598)  

**Abstract**: Recent calls for pluralistic alignment of Large Language Models (LLMs) encourage adapting models to diverse user preferences. However, most prior work on personalized reward models heavily rely on additional identity information, such as demographic details or a predefined set of preference categories. To this end, we introduce SynthesizeMe, an approach to inducing synthetic user personas from user interactions for personalized reward modeling. SynthesizeMe first generates and verifies reasoning to explain user preferences, then induces synthetic user personas from that reasoning, and finally filters to informative prior user interactions in order to build personalized prompts for a particular user. We show that using SynthesizeMe induced prompts improves personalized LLM-as-a-judge accuracy by 4.4% on Chatbot Arena. Combining SynthesizeMe derived prompts with a reward model achieves top performance on PersonalRewardBench: a new curation of user-stratified interactions with chatbots collected from 854 users of Chatbot Arena and PRISM. 

**Abstract (ZH)**: Recent呼吁多样性对齐的大语言模型鼓励模型适应多元用户偏好。然而，大多数关于个性化奖励模型的先前工作严重依赖额外的身份信息，如人口统计细节或预定义的偏好类别。为此，我们提出SynthesizeMe，一种从用户互动中诱导合成用户人设的个性化奖励建模方法。SynthesizeMe首先生成并验证解释用户偏好的原因，然后从中诱导合成用户人设，并最终筛选有益的用户先前互动以构建特定用户的个性化提示。实验结果显示，使用SynthesizeMe诱导的提示在Chatbot Arena中提高了个性化LLM作为评判者的准确率4.4%。将SynthesizeMe推断出的提示与奖励模型结合，在一个新收集的854名Chatbot Arena和PRISM用户的分层交互数据集PersonalRewardBench上达到了最佳性能。 

---
# Ravan: Multi-Head Low-Rank Adaptation for Federated Fine-Tuning 

**Title (ZH)**: Ravan: 多头低秩适应的联邦微调 

**Authors**: Arian Raje, Baris Askin, Divyansh Jhunjhunwala, Gauri Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2506.05568)  

**Abstract**: Large language models (LLMs) have not yet effectively leveraged the vast amounts of edge-device data, and federated learning (FL) offers a promising paradigm to collaboratively fine-tune LLMs without transferring private edge data to the cloud. To operate within the computation and communication constraints of edge devices, recent literature on federated fine-tuning of LLMs proposes the use of low-rank adaptation (LoRA) and similar parameter-efficient methods. However, LoRA-based methods suffer from accuracy degradation in FL settings, primarily because of data and computational heterogeneity across clients. We propose \textsc{Ravan}, an adaptive multi-head LoRA method that balances parameter efficiency and model expressivity by reparameterizing the weight updates as the sum of multiple LoRA heads $s_i\textbf{B}_i\textbf{H}_i\textbf{A}_i$ in which only the core matrices $\textbf{H}_i$ and their lightweight scaling factors $s_i$ are trained. These trainable scaling factors let the optimization focus on the most useful heads, recovering a higher-rank approximation of the full update without increasing the number of communicated parameters since clients upload $s_i\textbf{H}_i$ directly. Experiments on vision and language benchmarks show that \textsc{Ravan} improves test accuracy by 2-8\% over prior parameter-efficient baselines, making it a robust and scalable solution for federated fine-tuning of LLMs. 

**Abstract (ZH)**: Ravan：一种自适应多头LoRA方法用于联邦微调大语言模型 

---
# ScaleRTL: Scaling LLMs with Reasoning Data and Test-Time Compute for Accurate RTL Code Generation 

**Title (ZH)**: ScaleRTL：使用推理数据和测试时计算进行准确RTL代码生成 

**Authors**: Chenhui Deng, Yun-Da Tsai, Guan-Ting Liu, Zhongzhi Yu, Haoxing Ren  

**Link**: [PDF](https://arxiv.org/pdf/2506.05566)  

**Abstract**: Recent advances in large language models (LLMs) have enabled near-human performance on software coding benchmarks, but their effectiveness in RTL code generation remains limited due to the scarcity of high-quality training data. While prior efforts have fine-tuned LLMs for RTL tasks, they do not fundamentally overcome the data bottleneck and lack support for test-time scaling due to their non-reasoning nature. In this work, we introduce ScaleRTL, the first reasoning LLM for RTL coding that scales up both high-quality reasoning data and test-time compute. Specifically, we curate a diverse set of long chain-of-thought reasoning traces averaging 56K tokens each, resulting in a dataset of 3.5B tokens that captures rich RTL knowledge. Fine-tuning a general-purpose reasoning model on this corpus yields ScaleRTL that is capable of deep RTL reasoning. Subsequently, we further enhance the performance of ScaleRTL through a novel test-time scaling strategy that extends the reasoning process via iteratively reflecting on and self-correcting previous reasoning steps. Experimental results show that ScaleRTL achieves state-of-the-art performance on VerilogEval and RTLLM, outperforming 18 competitive baselines by up to 18.4% on VerilogEval and 12.7% on RTLLM. 

**Abstract (ZH)**: recent 进展在大规模语言模型（LLMs）已在软件编码基准上实现了接近人类的表现，但在 RTL 代码生成方面的有效性受限于高质量训练数据的稀缺性。尽管先前的努力已经在 RTL 任务上微调了 LLMs，但它们仍然无法从根本上克服数据瓶颈，并且由于其非推理特性，缺乏测试时缩放的支持。在本文中，我们介绍了 ScaleRTL，这是首个能够扩展高质量推理数据和测试时计算的用于 RTL 编码的推理 LLM。具体地，我们精心编纂了一组长度平均为 56K 令牌的多样化长推理链记录，最终形成一个包含 3.5B 令牌的数据集，捕捉了丰富的 RTL 知识。在该语料库上微调一个通用推理模型产生了具备深度 RTL 推理能力的 ScaleRTL。随后，我们通过一种新颖的测试时缩放策略进一步增强了 ScaleRTL 的性能，该策略通过迭代地反思和自我纠正先前的推理步骤来扩展推理过程。实验结果表明，ScaleRTL 在 VerilogEval 和 RTLLM 上实现了最先进的性能，在 VerilogEval 上优于 18 个竞争基线最高达 18.4%，在 RTLLM 上优于 12.7%。 

---
# StealthInk: A Multi-bit and Stealthy Watermark for Large Language Models 

**Title (ZH)**: StealthInk: 一种针对大规模语言模型的多比特隐形水印 

**Authors**: Ya Jiang, Chuxiong Wu, Massieh Kordi Boroujeny, Brian Mark, Kai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05502)  

**Abstract**: Watermarking for large language models (LLMs) offers a promising approach to identifying AI-generated text. Existing approaches, however, either compromise the distribution of original generated text by LLMs or are limited to embedding zero-bit information that only allows for watermark detection but ignores identification. We present StealthInk, a stealthy multi-bit watermarking scheme that preserves the original text distribution while enabling the embedding of provenance data, such as userID, TimeStamp, and modelID, within LLM-generated text. This enhances fast traceability without requiring access to the language model's API or prompts. We derive a lower bound on the number of tokens necessary for watermark detection at a fixed equal error rate, which provides insights on how to enhance the capacity. Comprehensive empirical evaluations across diverse tasks highlight the stealthiness, detectability, and resilience of StealthInk, establishing it as an effective solution for LLM watermarking applications. 

**Abstract (ZH)**: 面向大规模语言模型的隐蔽多比特水印方案：保持原始文本分布的同时嵌入来源数据 

---
# Conformal Prediction Beyond the Seen: A Missing Mass Perspective for Uncertainty Quantification in Generative Models 

**Title (ZH)**: 超越已见：生成模型中不确定性量化的一种缺失质量视角 

**Authors**: Sima Noorani, Shayan Kiyani, George Pappas, Hamed Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2506.05497)  

**Abstract**: Uncertainty quantification (UQ) is essential for safe deployment of generative AI models such as large language models (LLMs), especially in high stakes applications. Conformal prediction (CP) offers a principled uncertainty quantification framework, but classical methods focus on regression and classification, relying on geometric distances or softmax scores: tools that presuppose structured outputs. We depart from this paradigm by studying CP in a query only setting, where prediction sets must be constructed solely from finite queries to a black box generative model, introducing a new trade off between coverage, test time query budget, and informativeness. We introduce Conformal Prediction with Query Oracle (CPQ), a framework characterizing the optimal interplay between these objectives. Our finite sample algorithm is built on two core principles: one governs the optimal query policy, and the other defines the optimal mapping from queried samples to prediction sets. Remarkably, both are rooted in the classical missing mass problem in statistics. Specifically, the optimal query policy depends on the rate of decay, or the derivative, of the missing mass, for which we develop a novel estimator. Meanwhile, the optimal mapping hinges on the missing mass itself, which we estimate using Good Turing estimators. We then turn our focus to implementing our method for language models, where outputs are vast, variable, and often under specified. Fine grained experiments on three real world open ended tasks and two LLMs, show CPQ applicability to any black box LLM and highlight: (1) individual contribution of each principle to CPQ performance, and (2) CPQ ability to yield significantly more informative prediction sets than existing conformal methods for language uncertainty quantification. 

**Abstract (ZH)**: 生成人工智能模型如大型语言模型（LLMs）的安全部署需要不确定性量化（UQ），尤其是在高风险应用中。形式预测（CP）提供了一种原则性的不确定性量化框架，但经典方法主要关注回归和分类，依赖几何距离或softmax分数：这些工具假设了结构化的输出。我们摒弃这一范式，在仅查询设置中研究CP，其中预测集必须仅从对黑盒生成模型的有限查询中构建，引入了覆盖、测试时间查询预算和信息量之间的新权衡。我们引入了基于查询 oracle 的形式预测（CPQ）框架，该框架刻画了这些目标之间最佳交互。我们的有限样本算法基于两个核心原则：一个管理最优查询策略，另一个定义从查询样本到预测集的最优映射。令人惊讶的是，两者都根植于统计中的经典缺失质量问题。具体而言，最优查询策略取决于缺失质量衰减率，或其导数，为此我们开发了一种新的估计器。同时，最优映射依赖于缺失质量本身，我们使用Good Turing估计器进行估计。然后，我们将注意力转向在语言模型中实施该方法，其中输出量大、变化且往往不够明确。针对三个实际开放任务和两个LLMs的精细实验表明，CPQ在任何黑盒LLMs中的适用性，并突出显示：（1）每个原则对CPQ性能的独立贡献，以及（2）CPQ在语言不确定性量化中比现有形式预测方法生成显著更具信息量的预测集的能力。 

---
# MLLM-CL: Continual Learning for Multimodal Large Language Models 

**Title (ZH)**: MLLM-CL: 多模态大型语言模型的持续学习 

**Authors**: Hongbo Zhao, Fei Zhu, Rundong Wang, Gaofeng Meng, Zhaoxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05453)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods. 

**Abstract (ZH)**: 最近的多模态大型语言模型在视觉-语言理解方面表现出色，但在应对需要持续集成新知识和技能的动态现实场景方面面临挑战。尽管持续学习提供了一种潜在的解决方案，但现有基准和方法存在重大局限。本文介绍了一种新的基准MLLM-CL，涵盖了领域和能力的持续学习，前者关注独立同分布（IID）评估在不断演变的主要领域的独立性，后者则评估在新兴模型能力非IID场景下的表现。方法上，我们提出了通过参数隔离防止灾难性干扰，并设计了一种基于多模态大型语言模型的路由机制。广泛实验证明，我们的方法能够以最小的遗忘整合领域特定知识和功能能力，显著优于现有方法。 

---
# Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety 

**Title (ZH)**: 解读与安全并重：改进大型语言模型安全性的解释方法与工具综述 

**Authors**: Seongmin Lee, Aeree Cho, Grace C. Kim, ShengYun Peng, Mansi Phute, Duen Horng Chau  

**Link**: [PDF](https://arxiv.org/pdf/2506.05451)  

**Abstract**: As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在更广泛的实际应用中被采用，理解和缓解其不安全行为至关重要。解释技术可以揭示不安全输出的原因并指导安全措施，但在之前的综述中，这种与安全的联系往往被忽视。我们首次提出了填补这一空白的综述，介绍了一个统一框架，将安全导向的解释方法、它们所指导的安全增强措施以及实施这些措施的工具联系起来。我们按照LLM工作流程阶段组织的创新分类法总结了近70项相关工作。最后，我们提出了开放性挑战和未来方向。这项及时的综述有助于研究人员和实践者导航更安全、更具可解释性的LLM的关键进展。 

---
# Training Dynamics Underlying Language Model Scaling Laws: Loss Deceleration and Zero-Sum Learning 

**Title (ZH)**: 语言模型规模规律背后的训练动力学：损失减速与零和学习 

**Authors**: Andrei Mircea, Supriyo Chakraborty, Nima Chitsazan, Irina Rish, Ekaterina Lobacheva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05447)  

**Abstract**: This work aims to understand how scaling improves language models, specifically in terms of training dynamics. We find that language models undergo loss deceleration early in training; an abrupt slowdown in the rate of loss improvement, resulting in piecewise linear behaviour of the loss curve in log-log space. Scaling up the model mitigates this transition by (1) decreasing the loss at which deceleration occurs, and (2) improving the log-log rate of loss improvement after deceleration. We attribute loss deceleration to a type of degenerate training dynamics we term zero-sum learning (ZSL). In ZSL, per-example gradients become systematically opposed, leading to destructive interference in per-example changes in loss. As a result, improving loss on one subset of examples degrades it on another, bottlenecking overall progress. Loss deceleration and ZSL provide new insights into the training dynamics underlying language model scaling laws, and could potentially be targeted directly to improve language models independent of scale. We make our code and artefacts available at: this https URL 

**Abstract (ZH)**: 本项工作旨在理解扩增如何改善语言模型，特别是在训练动力学方面的表现。我们发现语言模型在训练初期会出现损失减速现象；损失改进速率的突然减缓，在对数-对数空间中导致损失曲线呈现分段线性行为。通过扩增模型，这种过渡得以缓解，具体表现为：（1）损失减速发生的损失值降低，（2）在损失减速后，损失改进的对数-对数速率得到提升。我们将损失减速归因于一种我们称之为零和学习（ZSL）的退化训练动态。在ZSL中，每个样本的梯度变得系统性地对立，导致损失的样本变化出现破坏性干涉。因此，在一个样本子集上改进损失会损害另一个子集上的损失，从而阻碍整体进展。损失减速和ZSL为语言模型缩放定律背后的训练动力学提供了新的见解，并有可能直接针对这些现象来独立于规模地改进语言模型。我们已在以下链接提供了我们的代码和 artefacts: this https URL。 

---
# Sentinel: SOTA model to protect against prompt injections 

**Title (ZH)**: 哨兵：当前最佳模型，用于抵御提示注入攻击 

**Authors**: Dror Ivry, Oran Nahum  

**Link**: [PDF](https://arxiv.org/pdf/2506.05446)  

**Abstract**: Large Language Models (LLMs) are increasingly powerful but remain vulnerable to prompt injection attacks, where malicious inputs cause the model to deviate from its intended instructions. This paper introduces Sentinel, a novel detection model, qualifire/prompt-injection-sentinel, based on the \answerdotai/ModernBERT-large architecture. By leveraging ModernBERT's advanced features and fine-tuning on an extensive and diverse dataset comprising a few open-source and private collections, Sentinel achieves state-of-the-art performance. This dataset amalgamates varied attack types, from role-playing and instruction hijacking to attempts to generate biased content, alongside a broad spectrum of benign instructions, with private datasets specifically targeting nuanced error correction and real-world misclassifications. On a comprehensive, unseen internal test set, Sentinel demonstrates an average accuracy of 0.987 and an F1-score of 0.980. Furthermore, when evaluated on public benchmarks, it consistently outperforms strong baselines like protectai/deberta-v3-base-prompt-injection-v2. This work details Sentinel's architecture, its meticulous dataset curation, its training methodology, and a thorough evaluation, highlighting its superior detection capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益强大但仍易受提示注入攻击的影响，恶意输入会导致模型偏离其预期指令。本文引入了基于\answerdotai/ModernBERT-large架构的Sentinel，一种新颖的检测模型，qualifire/prompt-injection-sentinel。通过利用ModernBERT的高级特性并在涵盖多种开源和私有数据集合的广泛且多样化的数据集上进行微调，Sentinel实现了最先进的性能。该数据集整合了各种攻击类型，包括角色扮演、指令劫持以及生成偏见内容的尝试，同时还包括广泛的良性指令，其中包含针对细微错误修正和现实世界误分类的私有数据集。在全面的未见过的内部测试集上，Sentinel的平均准确率为0.987，F1分为0.980。此外，在公开基准测试中，Sentinel一致地优于强大的基线模型，如protectai/deberta-v3-base-prompt-injection-v2。本文详细介绍了Sentinel的架构、精心的数据集编纂、训练方法以及全面评估，突显了其卓越的检测能力。 

---
# LLMs Can Compensate for Deficiencies in Visual Representations 

**Title (ZH)**: LLMs可以在视觉表示的不足之处进行补偿。 

**Authors**: Sho Takishita, Jay Gala, Abdelrahman Mohamed, Kentaro Inui, Yova Kementchedjhieva  

**Link**: [PDF](https://arxiv.org/pdf/2506.05439)  

**Abstract**: Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder. 

**Abstract (ZH)**: 许多证明在多种多模态任务中非常有效的vision-language模型（VLMs）基于CLIP的视觉编码器，但这些编码器已知存在各种局限。我们研究了VLMs中的强大语言骨干能够通过上下文化或丰富视觉特征来补偿可能较弱的视觉特征这一假设。使用三种基于CLIP的VLMs，我们在精心设计的探针任务上进行了受控的自注意力消融实验。我们的研究结果表明，尽管CLIP的视觉表示存在已知的局限性，但它们仍为语言解码器提供了易于读取的语义信息。然而，在视觉表示中上下文化减弱的情况下，语言解码器可以大量补偿不足并恢复性能。这表明VLMs中存在动态的工作分工，并激发了未来将更多视觉处理卸载到语言解码器的架构设计。 

---
# PCDVQ: Enhancing Vector Quantization for Large Language Models via Polar Coordinate Decoupling 

**Title (ZH)**: PCDVQ：通过极坐标解耦提升大规模语言模型的向量量化 

**Authors**: Yuxuan Yue, Zukang Xu, Zhihang Yuan, Dawei Yang, Jianglong Wu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2506.05432)  

**Abstract**: Large Language Models (LLMs) face significant challenges in edge deployment due to their massive parameter scale. Vector Quantization (VQ), a clustering-based quantization method, serves as a prevalent solution to this issue for its extremely low-bit (even at 2-bit) and considerable accuracy. Since a vector is a quantity in mathematics and physics that has both direction and magnitude, existing VQ works typically quantize them in a coupled manner. However, we find that direction exhibits significantly greater sensitivity to quantization compared to the magnitude. For instance, when separately clustering the directions and magnitudes of weight vectors in LLaMA-2-7B, the accuracy drop of zero-shot tasks are 46.5\% and 2.3\%, respectively. This gap even increases with the reduction of clustering centers. Further, Euclidean distance, a common metric to access vector similarities in current VQ works, places greater emphasis on reducing the magnitude error. This property is contrary to the above finding, unavoidably leading to larger quantization errors. To these ends, this paper proposes Polar Coordinate Decoupled Vector Quantization (PCDVQ), an effective and efficient VQ framework consisting of two key modules: 1) Polar Coordinate Decoupling (PCD), which transforms vectors into their polar coordinate representations and perform independent quantization of the direction and magnitude parameters.2) Distribution Aligned Codebook Construction (DACC), which optimizes the direction and magnitude codebooks in accordance with the source distribution. Experimental results show that PCDVQ outperforms baseline methods at 2-bit level by at least 1.5\% zero-shot accuracy, establishing a novel paradigm for accurate and highly compressed LLMs. 

**Abstract (ZH)**: 基于极坐标解耦的矢量量化（Polar Coordinate Decoupled Vector Quantization） 

---
# SIV-Bench: A Video Benchmark for Social Interaction Understanding and Reasoning 

**Title (ZH)**: SIV-Bench: 用于社会互动理解与推理的视频基准 

**Authors**: Fanqi Kong, Weiqin Zu, Xinyu Chen, Yaodong Yang, Song-Chun Zhu, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.05425)  

**Abstract**: The rich and multifaceted nature of human social interaction, encompassing multimodal cues, unobservable relations and mental states, and dynamical behavior, presents a formidable challenge for artificial intelligence. To advance research in this area, we introduce SIV-Bench, a novel video benchmark for rigorously evaluating the capabilities of Multimodal Large Language Models (MLLMs) across Social Scene Understanding (SSU), Social State Reasoning (SSR), and Social Dynamics Prediction (SDP). SIV-Bench features 2,792 video clips and 8,792 meticulously generated question-answer pairs derived from a human-LLM collaborative pipeline. It is originally collected from TikTok and YouTube, covering a wide range of video genres, presentation styles, and linguistic and cultural backgrounds. It also includes a dedicated setup for analyzing the impact of different textual cues-original on-screen text, added dialogue, or no text. Our comprehensive experiments on leading MLLMs reveal that while models adeptly handle SSU, they significantly struggle with SSR and SDP, where Relation Inference (RI) is an acute bottleneck, as further examined in our analysis. Our study also confirms the critical role of transcribed dialogue in aiding comprehension of complex social interactions. By systematically identifying current MLLMs' strengths and limitations, SIV-Bench offers crucial insights to steer the development of more socially intelligent AI. The dataset and code are available at this https URL. 

**Abstract (ZH)**: 人类社会互动的丰富多样性，涵盖多模态线索、不可观察的关系和心理状态以及动态行为，给人工智能带来了巨大的挑战。为了推进这一领域的研究，我们引入了SIV-Bench，这是一个新的视频基准，用于严格评估多模态大型语言模型（MLLMs）在社会场景理解（SSU）、社会状态推理（SSR）和社会动力预测（SDP）方面的能力。SIV-Bench 包含 2,792 个视频片段和 8,792 个精心生成的问题-答案对，这些数据源自人-LLM 合作管道。这些数据最初是从 TikTok 和 YouTube 收集的，涵盖了广泛的视频类型、呈现风格以及语言和文化背景。此外，它还包括一种专门设置，用于分析不同文本线索（原始屏幕文本、添加对话或无文本）的影响。我们在全面实验中发现，虽然模型在社会场景理解方面表现良好，但在社会状态推理和社会动力预测方面却面临巨大挑战，其中关系推理（RI）是一个关键瓶颈，我们在分析中进一步进行了探讨。我们的研究还证实，转录的对话在理解复杂社会互动中起着关键作用。通过系统地识别当前 MLLMs 的优势和局限性，SIV-Bench 提供了重要的见解，以引导更加社交智能的 AI 的开发。该数据集和代码可通过此链接访问。 

---
# SAVVY: Spatial Awareness via Audio-Visual LLMs through Seeing and Hearing 

**Title (ZH)**: SAVVY: 通过视听LLM观察与聆听的空间意识 

**Authors**: Mingfei Chen, Zijun Cui, Xiulong Liu, Jinlin Xiang, Caleb Zheng, Jingyuan Li, Eli Shlizerman  

**Link**: [PDF](https://arxiv.org/pdf/2506.05414)  

**Abstract**: 3D spatial reasoning in dynamic, audio-visual environments is a cornerstone of human cognition yet remains largely unexplored by existing Audio-Visual Large Language Models (AV-LLMs) and benchmarks, which predominantly focus on static or 2D scenes. We introduce SAVVY-Bench, the first benchmark for 3D spatial reasoning in dynamic scenes with synchronized spatial audio. SAVVY-Bench is comprised of thousands of relationships involving static and moving objects, and requires fine-grained temporal grounding, consistent 3D localization, and multi-modal annotation. To tackle this challenge, we propose SAVVY, a novel training-free reasoning pipeline that consists of two stages: (i) Egocentric Spatial Tracks Estimation, which leverages AV-LLMs as well as other audio-visual methods to track the trajectories of key objects related to the query using both visual and spatial audio cues, and (ii) Dynamic Global Map Construction, which aggregates multi-modal queried object trajectories and converts them into a unified global dynamic map. Using the constructed map, a final QA answer is obtained through a coordinate transformation that aligns the global map with the queried viewpoint. Empirical evaluation demonstrates that SAVVY substantially enhances performance of state-of-the-art AV-LLMs, setting a new standard and stage for approaching dynamic 3D spatial reasoning in AV-LLMs. 

**Abstract (ZH)**: 三维动态环境中的视听空间推理是人类认知的一个基石，但现有视听大型语言模型（AV-LLMs）及其基准测试大多集中于静态或二维场景，尚未充分探索。我们引入了SAVVY-Bench，这是首个用于动态场景中同步空间音频的精细三维空间推理基准测试，包括成千上万涉及静止和移动物体的关系，要求精细的时间对接、一致的三维定位和多模态注释。为应对这一挑战，我们提出了SAVVY，一种无需训练的新型推理管道，包含两个阶段：(i) 自我中心空间轨迹估计，利用AV-LLMs及其他视听方法，结合视觉和空间音频线索跟踪与查询相关的关键物体的轨迹，以及(ii) 动态全局地图构建，聚合多模态查询物体轨迹并转化为统一的全球动态地图。通过构建的地图，最终的问答答案通过坐标转换获得，使全球地图与查询视角对齐。实证评估表明，SAVVY显著提升了现有最佳视听大型语言模型的性能，为AV-LLMs中的动态三维空间推理设定了新的标准和起点。 

---
# SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs 

**Title (ZH)**: SmoothRot: 结合通道级缩放和旋转以实现量化友好的大语言模型 

**Authors**: Patrik Czakó, Gábor Kertész, Sándor Szénási  

**Link**: [PDF](https://arxiv.org/pdf/2506.05413)  

**Abstract**: We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at this https URL. 

**Abstract (ZH)**: SmoothRot：一种改进大型语言模型4比特量化效率的新颖后训练量化技术 

---
# Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs 

**Title (ZH)**: 改进解码策略：增强局部典型采样方法在大语言模型中的应用 

**Authors**: Jaydip Sen, Saptarshi Sengupta. Subhasis Dasgupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.05387)  

**Abstract**: This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency. 

**Abstract (ZH)**: 本章探讨了大规模语言模型（LLM）解码策略的进展，重点关注增强局部典型采样（LTS）算法。传统的解码方法，如top-k和nucleus采样，往往在文本生成中难以平衡流畅性、多样性和一致性。为此，提出了一种改进的自适应语义感知典型性采样（ASTS）算法，该算法结合了动态熵阈值、多目标评分和奖惩调整。ASTS能够保证上下文一致性和多样性的同时保持计算效率。其性能通过故事生成和抽象总结等基准测试进行评估，采用困惑度、MAUVE和多样性评分等指标。实验结果表明，ASTS在减少重复、增强语义对齐和改善流畅性方面优于现有采样技术。 

---
# Beyond RAG: Reinforced Reasoning Augmented Generation for Clinical Notes 

**Title (ZH)**: 超越RAG：强化推理增强生成在临床笔记中的应用 

**Authors**: Lo Pang-Yun Ting, Chengshuai Zhao, Yu-Hua Zeng, Yuan Jee Lim, Kun-Ta Chuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.05386)  

**Abstract**: Clinical note generation aims to automatically produce free-text summaries of a patient's condition and diagnostic process, with discharge instructions being a representative long-form example. While recent large language model (LLM)-based methods pre-trained on general clinical corpora show promise in clinical text generation, they fall short in producing long-form notes from limited patient information. In this paper, we propose R2AG, the first reinforced retriever for long-form discharge instruction generation based on pre-admission data. R2AG is trained with reinforcement learning to retrieve reasoning paths from a medical knowledge graph, providing explicit semantic guidance to the LLM. To bridge the information gap, we propose Group-Based Retriever Optimization (GRO) which improves retrieval quality with group-relative rewards, encouraging reasoning leaps for deeper inference by the LLM. Comprehensive experiments on the MIMIC-IV-Note dataset show that R2AG outperforms baselines in both clinical efficacy and natural language generation metrics. Further analysis reveals that R2AG fills semantic gaps in sparse input scenarios, and retrieved reasoning paths help LLMs avoid clinical misinterpretation by focusing on key evidence and following coherent reasoning. 

**Abstract (ZH)**: 临床笔记生成旨在自动生成患者状况和诊断过程的自由文本总结，出院指示是典型的长文例证。虽然基于通用临床语料库预训练的大规模语言模型 (LLM) 在临床文本生成方面展现出希望，但在有限的患者信息下生成长文笔记方面仍然不足。本文提出 R2AG，这是一种基于预住院数据的第一个强化检索器，用于长文出院指示生成。R2AG 通过强化学习训练，从医学知识图谱中检索推理路径，为LLM 提供显式的语义指导。为弥补信息缺口，我们提出基于组别优化的检索器（GRO），通过组内相对奖励提高检索质量，鼓励LLM 进行更深层次的推理。在 MIMIC-IV-Note 数据集上的全面实验表明，R2AG 在临床效果和自然语言生成指标上均优于基线方法。进一步的分析表明，R2AG 在稀疏输入场景中填补了语义空缺，并检索到的推理路径有助于LLM 避免临床误读，通过关注关键证据并遵循连贯的推理过程。 

---
# Q-Ponder: A Unified Training Pipeline for Reasoning-based Visual Quality Assessment 

**Title (ZH)**: Q-Ponder：一种基于推理的视觉质量评估统一训练流程 

**Authors**: Zhuoxuan Cai, Jian Zhang, Xinbin Yuan, Pengtao Jiang, Wenxiang Chen, Bowen Tang, Lujian Yao, Qiyuan Wang, Jinwen Chen, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.05384)  

**Abstract**: Recent studies demonstrate that multimodal large language models (MLLMs) can proficiently evaluate visual quality through interpretable assessments. However, existing approaches typically treat quality scoring and reasoning descriptions as separate tasks with disjoint optimization objectives, leading to a trade-off: models adept at quality reasoning descriptions struggle with precise score regression, while score-focused models lack interpretability. This limitation hinders the full potential of MLLMs in visual quality assessment, where accuracy and interpretability should be mutually reinforcing. To address this, we propose a unified two-stage training framework comprising a cold-start stage and a reinforcement learning-based fine-tuning stage. Specifically, in the first stage, we distill high-quality data from a teacher model through expert-designed prompts, initializing reasoning capabilities via cross-entropy loss supervision. In the second stage, we introduce a novel reward with Group Relative Policy Optimization (GRPO) to jointly optimize scoring accuracy and reasoning consistency. We designate the models derived from these two stages as Q-Ponder-CI and Q-Ponder. Extensive experiments show that Q-Ponder achieves state-of-the-art (SOTA) performance on quality score regression benchmarks, delivering up to 6.5% higher SRCC on cross-domain datasets. Furthermore, Q-Ponder significantly outperforms description-based SOTA models, including its teacher model Qwen-2.5-VL-72B, particularly in description accuracy and reasonableness, demonstrating the generalization potential over diverse tasks. 

**Abstract (ZH)**: 近期研究显示，多模态大型语言模型（MLLMs）能够通过可解释的评估来熟练地评价图像质量。然而，现有方法通常将质量评分和理由描述视为分离的任务，各自优化目标不一致，导致权衡：擅长理由描述的模型在精确评分回归方面表现不佳，而注重评分的模型缺乏解释性。这一限制阻碍了MLLMs在图像质量评估中的全部潜力，其中准确性与解释性应该是相互促进的关系。为了解决这一问题，我们提出了一种统一的两阶段训练框架，包括一个冷启动阶段和一个基于强化学习的微调阶段。在第一阶段，我们通过专家设计的提示从教师模型中提取高质量数据，并通过交叉熵损失监督初始化推理能力。在第二阶段，我们引入了组相对策略优化（GRPO）的新型奖励，以联合优化评分准确性和理由一致性。我们从这两个阶段得出的模型分别命名为Q-Ponder-CI和Q-Ponder。 extensive实验表明，Q-Ponder在质量评分回归基准测试中达到了现有最佳性能（SOTA），在跨领域数据集上SRCC可提高6.5%。此外，Q-Ponder显著优于基于描述的SOTA模型，包括其教师模型Qwen-2.5-VL-72B，在描述准确性和合理性方面表现尤为出色，展示了其在多样化任务中的泛化潜力。 

---
# Designing DSIC Mechanisms for Data Sharing in the Era of Large Language Models 

**Title (ZH)**: 大型语言模型时代的数据共享DSIC机制设计 

**Authors**: Seyed Moein Ayyoubzadeh, Kourosh Shahnazari, Mohammmadali Keshtparvar, MohammadAmin Fazli  

**Link**: [PDF](https://arxiv.org/pdf/2506.05379)  

**Abstract**: Training large language models (LLMs) requires vast amounts of high-quality data from institutions that face legal, privacy, and strategic constraints. Existing data procurement methods often rely on unverifiable trust or ignore heterogeneous provider costs. We introduce a mechanism-design framework for truthful, trust-minimized data sharing that ensures dominant-strategy incentive compatibility (DSIC), individual rationality, and weak budget balance, while rewarding data based on both quality and learning utility. We formalize a model where providers privately know their data cost and quality, and value arises solely from the data's contribution to model performance. Based on this, we propose the Quality-Weighted Marginal-Incentive Auction (Q-MIA), which ranks providers using a virtual cost metric and uses Myerson-style payments to ensure DSIC and budget feasibility. To support settings with limited liquidity or long-term incentives, we introduce the Marginal Utility Token (MUT), which allocates future rights based on marginal contributions. We unify these in Mixed-MIA, a hybrid mechanism balancing upfront payments and deferred rewards. All mechanisms support verifiable, privacy-preserving implementation. Theoretically and empirically, they outperform volume-based and trust-based baselines, eliciting higher-quality data under budget constraints while remaining robust to misreporting and collusion. This establishes a principled foundation for sustainable and fair data markets for future LLMs. 

**Abstract (ZH)**: 一种面向大规模语言模型的机制设计框架：最小化信任的数据共享机制 

---
# A Red Teaming Roadmap Towards System-Level Safety 

**Title (ZH)**: 面向系统级安全的红队演练路线图 

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael  

**Link**: [PDF](https://arxiv.org/pdf/2506.05376)  

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future. 

**Abstract (ZH)**: 大型语言模型（LLM）防护措施，通过实施请求拒绝，已成为应对滥用的广泛应用的缓解策略。在对抗机器学习和人工智能安全的交叉点上，防护红队活动有效地识别了先进拒绝训练的LLM中的关键漏洞。然而，我们认为，关于LLM红队的众多会议投稿在整体上没有优先解决正确的研究问题。首先，针对清晰的产品安全规范进行测试应高于抽象的社会偏见或伦理原则。其次，红队活动应优先考虑现实威胁模型，以代表不断扩大的风险景观和实际攻击者可能的行为。最后，我们认为，系统级安全是推动红队研究向前发展的必要步骤，因为一旦在部署环境中存在，AI模型既带来了新的威胁，也提供了威胁缓解的可能性（如检测和封禁恶意用户）。为了使红队研究能够充分应对当前及非常近未来快速发展的AI技术带来的新威胁，采纳这些优先事项是必要的。 

---
# Can ChatGPT Perform Image Splicing Detection? A Preliminary Study 

**Title (ZH)**: ChatGPT能否进行图像拼接检测？一项初步研究 

**Authors**: Souradip Nath  

**Link**: [PDF](https://arxiv.org/pdf/2506.05358)  

**Abstract**: Multimodal Large Language Models (MLLMs) like GPT-4V are capable of reasoning across text and image modalities, showing promise in a variety of complex vision-language tasks. In this preliminary study, we investigate the out-of-the-box capabilities of GPT-4V in the domain of image forensics, specifically, in detecting image splicing manipulations. Without any task-specific fine-tuning, we evaluate GPT-4V using three prompting strategies: Zero-Shot (ZS), Few-Shot (FS), and Chain-of-Thought (CoT), applied over a curated subset of the CASIA v2.0 splicing dataset.
Our results show that GPT-4V achieves competitive detection performance in zero-shot settings (more than 85% accuracy), with CoT prompting yielding the most balanced trade-off across authentic and spliced images. Qualitative analysis further reveals that the model not only detects low-level visual artifacts but also draws upon real-world contextual knowledge such as object scale, semantic consistency, and architectural facts, to identify implausible composites. While GPT-4V lags behind specialized state-of-the-art splicing detection models, its generalizability, interpretability, and encyclopedic reasoning highlight its potential as a flexible tool in image forensics. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）如GPT-4V能够在文本和图像模态间进行推理，在复杂视觉语言任务中展现出巨大潜力。在这一初步研究中，我们探讨了GPT-4V在图像取证领域的通用能力，特别是在检测图像拼接篡改方面的能力。未经任何任务特定微调，我们通过三种提示策略（零样本、少量样本和逐步推理）评估了GPT-4V在CASIA v2.0拼接数据集子集上的表现。结果显示，GPT-4V在零样本设置下实现了竞争力的检测性能（超过85%的准确率），逐步推理提示策略提供了在真伪图像间最为均衡的权衡。进一步的定性分析表明，该模型不仅检测低级别视觉伪影，还利用现实世界上下文知识如物体尺度、语义一致性及建筑事实来识别不合逻辑的复合图像。虽然GPT-4V在专门的拼接检测模型面前略逊一筹，但其泛化能力、可解释性和百科全书式推理展示了其在图像取证领域作为灵活工具的潜力。 

---
