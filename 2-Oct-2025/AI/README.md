# Generalized Parallel Scaling with Interdependent Generations 

**Title (ZH)**: 广义并行缩放与相互依赖代际 

**Authors**: Harry Dong, David Brandfonbrener, Eryk Helenowski, Yun He, Mrinal Kumar, Han Fang, Yuejie Chi, Karthik Abinav Sankararaman  

**Link**: [PDF](https://arxiv.org/pdf/2510.01143)  

**Abstract**: Parallel LLM inference scaling involves sampling a set of $N>1$ responses for a single input prompt. However, these $N$ parallel responses tend to be generated independently from each other, partitioning compute resources and leaving potentially useful information in one generation untapped by others. This is in contrast to response length scaling where past computation is used in all future steps. For higher quality responses and response sets, we propose Bridge to generate interdependent responses in parallel by rethinking batched LLM hidden states as holistic tensors rather than independent slices. With only a small amount (2.8%-5.1%) of new parameters, Bridge improves the relative mean accuracy gains from reinforcement learning with verifiable rewards by up to 50% and boosts consistency of correct responses. Trained once, Bridge scales to any generation width, all with greater performance than independent generations, unlocking a more general mode of parallel scaling that effectively leverages information between sequences, compatible with any post-generation aggregation technique. 

**Abstract (ZH)**: 并行LLM推理扩展涉及对单个输入提示采样一组$N>1$个响应。然而，这$N$个并行响应通常是从彼此独立生成的，导致计算资源被分割，可能会导致一代中的有用信息被其他代所未利用。这与响应长度扩展不同，在响应长度扩展中，之前的计算会在所有后续步骤中被利用。为了生成更高质量的响应和响应集，我们提出Bridge通过将批量LLM隐藏状态重新构想为整体张量而非独立切片，以并行生成相互依赖的响应。通过引入少量的新参数（仅2.8%-5.1%），Bridge通过验证奖励的方式提高了强化学习的相对平均准确度 gain至多50%，并且提高了正确响应的一致性。Bridge经过一次训练即可扩展到任何生成宽度，性能优于独立生成，并解锁了一种更通用的并行扩展模式，这种模式有效地利用了序列之间的信息，与任何后生成聚合技术兼容。 

---
# Apriel-1.5-15b-Thinker 

**Title (ZH)**: April-1.5-15B-思辨者 

**Authors**: Shruthan Radhakrishna, Aman Tiwari, Aanjaneya Shukla, Masoud Hashemi, Rishabh Maheshwary, Shiva Krishna Reddy Malay, Jash Mehta, Pulkit Pattnaik, Saloni Mittal, Khalil Slimi, Kelechi Ogueji, Akintunde Oladipo, Soham Parikh, Oluwanifemi Bamgbose, Toby Liang, Ahmed Masry, Khyati Mahajan, Sai Rajeswar Mudumba, Vikas Yadav, Sathwik Tejaswi Madhusudhan, Torsten Scholak, Sagar Davasam, Srinivas Sunkara, Nicholas Chapados  

**Link**: [PDF](https://arxiv.org/pdf/2510.01141)  

**Abstract**: We present Apriel-1.5-15B-Thinker, a 15-billion parameter open-weights multimodal reasoning model that achieves frontier-level performance through training design rather than sheer scale. Starting from Pixtral-12B, we apply a progressive three-stage methodology: (1) depth upscaling to expand reasoning capacity without pretraining from scratch, (2) staged continual pre-training that first develops foundational text and vision understanding, then enhances visual reasoning through targeted synthetic data generation addressing spatial structure, compositional understanding, and fine-grained perception, and (3) high-quality text-only supervised fine-tuning on curated instruction-response pairs with explicit reasoning traces spanning mathematics, coding, science, and tool use. Notably, our model achieves competitive results without reinforcement learning or preference optimization, isolating the contribution of our data-centric continual pre-training approach. On the Artificial Analysis Intelligence Index, Apriel-1.5-15B-Thinker attains a score of 52, matching DeepSeek-R1-0528 despite requiring significantly fewer computational resources. Across ten image benchmarks, its performance is on average within five points of Gemini-2.5-Flash and Claude Sonnet-3.7, a key achievement for a model operating within single-GPU deployment constraints. Our results demonstrate that thoughtful mid-training 2 design can close substantial capability gaps without massive scale, making frontier-level multimodal reasoning accessible to organizations with limited infrastructure. We release the model checkpoint, all training recipes, and evaluation protocols under the MIT license to to advance open-source research. 

**Abstract (ZH)**: 我们呈现了Apriel-1.5-15B-Thinker，这是一个通过训练设计而非单纯规模来实现前沿性能的150亿参数开放权重多模态推理模型。从Pixtral-12B出发，我们应用了渐进的三阶段方法：(1) 深度扩展以扩大推理能力而无需从头预训练，(2) 阶段性持续预训练，首先发展基础的文字和视觉理解，然后通过针对空间结构、组合理解及细粒度感知的目标合成数据生成来增强视觉推理，以及(3) 在精心挑选的指令-响应对上进行高质量的仅文本监督微调，这些对跨越了数学、编程、科学和工具使用领域，并含有显式的推理痕迹。值得注意的是，我们的模型在没有应用强化学习或偏好优化的情况下达到了竞争力的结果，从而隔离了我们以数据为中心的持续预训练方法的贡献。Apriel-1.5-15B-Thinker在人工分析人工智能指数上的得分为52，尽管计算资源需求显著较少，仍与DeepSeek-R1-0528相当。在整个十项图像基准测试中，其平均性能与Gemini-2.5-Flash和Claude Sonnet-3.7相差不超过五个点，这是一个在单GPU部署限制下模型的重要成就。我们的结果表明，经过深思熟虑的中训练设计可以在无需大规模扩展的情况下填补重要的能力差距，使前沿水平的多模态推理对基础设施有限的组织来说变得可行。我们以MIT许可证发布模型检查点、所有训练食谱和评估协议，以促进开源研究。 

---
# Exploring Network-Knowledge Graph Duality: A Case Study in Agentic Supply Chain Risk Analysis 

**Title (ZH)**: 探索网络-知识图谱二元性：代理供应链风险分析案例研究 

**Authors**: Evan Heus, Rick Bookstaber, Dhruv Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.01115)  

**Abstract**: Large Language Models (LLMs) struggle with the complex, multi-modal, and network-native data underlying financial risk. Standard Retrieval-Augmented Generation (RAG) oversimplifies relationships, while specialist models are costly and static. We address this gap with an LLM-centric agent framework for supply chain risk analysis. Our core contribution is to exploit the inherent duality between networks and knowledge graphs (KG). We treat the supply chain network as a KG, allowing us to use structural network science principles for retrieval. A graph traverser, guided by network centrality scores, efficiently extracts the most economically salient risk paths. An agentic architecture orchestrates this graph retrieval alongside data from numerical factor tables and news streams. Crucially, it employs novel ``context shells'' -- descriptive templates that embed raw figures in natural language -- to make quantitative data fully intelligible to the LLM. This lightweight approach enables the model to generate concise, explainable, and context-rich risk narratives in real-time without costly fine-tuning or a dedicated graph database. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理金融风险背后的复杂、多模态和网络原生数据时面临挑战。标准的检索增强生成（RAG）过分简化了关系，而专业模型则成本高昂且静态。我们通过一个以LLM为中心的代理框架来解决这一问题，用于供应链风险分析。我们核心的贡献在于利用网络和知识图谱（KG）之间的内在二元性。我们将供应链网络视为一个KG，从而可以利用结构网络科学的原则来进行检索。图遍历器在网络中心性分数的引导下，高效地提取出最具经济意义的风险路径。一种代理架构协调这一图检索过程，同时整合数值因子表数据和新闻流数据。关键的是，该架构采用了新型的“上下文壳”——描述性模板，将原始数字信息嵌入自然语言中，从而使定量数据对LLM完全可理解。这一轻量级的方法使模型能够实时生成简洁、可解释且富含上下文的风险叙事，而无需昂贵的微调或专用图数据库。 

---
# PRISM-Consult: A Panel-of-Experts Architecture for Clinician-Aligned Diagnosis 

**Title (ZH)**: PRISM-Consult: 专家面板架构以实现临床导向的诊断 

**Authors**: Lionel Levine, John Santerre, Alexander S. Young, T. Barry Levine, Francis Campion, Majid Sarrafzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2510.01114)  

**Abstract**: We present PRISM-Consult, a clinician-aligned panel-of-experts architecture that extends the compact PRISM sequence model into a routed family of domain specialists. Episodes are tokenized as structured clinical events; a light-weight router reads the first few tokens and dispatches to specialist models (Cardiac-Vascular, Pulmonary, Gastro-Oesophageal, Musculoskeletal, Psychogenic). Each specialist inherits PRISM's small transformer backbone and token template, enabling parameter efficiency and interpretability. On real-world Emergency Department cohorts, specialists exhibit smooth convergence with low development perplexities across domains, while the router achieves high routing quality and large compute savings versus consult-all under a safety-first policy. We detail the data methodology (initial vs. conclusive ICD-9 families), routing thresholds and calibration, and report per-domain results to avoid dominance by common events. The framework provides a practical path to safe, auditable, and low-latency consult at scale, and we outline validation steps-external/temporal replication, asymmetric life-threat thresholds, and multi-label arbitration-to meet prospective clinical deployment standards. 

**Abstract (ZH)**: PRISM-Consult：一种符合临床医生需求的专科专家架构 

---
# Optimizing Fairness in Production Planning: A Human-Centric Approach to Machine and Workforce Allocation 

**Title (ZH)**: 基于以人为本的方法优化生产规划中的公平性：机器与劳动力分配的公平性优化方法 

**Authors**: Alexander Nasuta, Alessandro Cisi, Sylwia Olbrych, Gustavo Vieira, Rui Fernandes, Lucas Paletta, Marlene Mayr, Rishyank Chevuri, Robert Woitsch, Hans Aoyang Zhou, Anas Abdelrazeq, Robert H. Schmitt  

**Link**: [PDF](https://arxiv.org/pdf/2510.01094)  

**Abstract**: This work presents a two-layer, human-centric production planning framework designed to optimize both operational efficiency and workforce fairness in industrial manufacturing. The first layer formulates the Order-Line allocation as a Constraint Programming (CP) problem, generating high-utilization production schedules that respect machine capacities, processing times, and due dates. The second layer models Worker-Line allocation as a Markov Decision Process (MDP), integrating human factors such as worker preference, experience, resilience, and medical constraints into the assignment process. Three solution strategies, greedy allocation, MCTS, and RL, are implemented and compared across multiple evaluation scenarios. The proposed system is validated through 16 test sessions with domain experts from the automotive industry, combining quantitative key performance indicators (KPIs) with expert ratings. Results indicate that the CP-based scheduling approach produces compact, feasible production plans with low tardiness, while the MDP-based worker allocation significantly improves fairness and preference alignment compared to baseline approaches. Domain experts rated both the Order-Line and Worker-Line components as effective and highlighted opportunities to further refine the objective function to penalize excessive earliness and improve continuity in worker assignments. Overall, the findings demonstrate that combining CP with learning-based decision-making provides a robust approach for human-centric production planning. The approach enables simultaneous optimization of throughput and workforce well-being, offering a practical foundation for fair and efficient manufacturing scheduling in industrial settings. 

**Abstract (ZH)**: 一种兼顾运营效率与 workforce 公平性的两层人类中心生产规划框架 

---
# Safety Instincts: LLMs Learn to Trust Their Internal Compass for Self-Defense 

**Title (ZH)**: 安全直觉：大语言模型学会依靠内部指南针进行自我保护 

**Authors**: Guobin Shen, Dongcheng Zhao, Haibo Tong, Jindong Li, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01088)  

**Abstract**: Ensuring Large Language Model (LLM) safety remains challenging due to the absence of universal standards and reliable content validators, making it difficult to obtain effective training signals. We discover that aligned models already possess robust internal safety beliefs: they consistently produce high-confidence refusals to harmful requests while exhibiting high entropy when generating potentially dangerous content. This entropy gap reveals an untapped signal--models intrinsically "know" when to refuse. We introduce Safety Instincts Reinforcement Learning (SIRL), which transforms this internal confidence into a self-generated reward signal, eliminating dependence on external validators or human annotations. SIRL teaches models to trust their safety instincts by reinforcing low-entropy refusal behaviors. Evaluated on Llama and Qwen models, SIRL maintains 89%+ Defense Success Rates (DSRs) against 20+ jailbreak methods, from static prompts to adaptive attacks. Using only 15,000 unlabeled prompts, SIRL surpasses resource-intensive supervised methods while preserving performance on mathematics, coding, and conversation benchmarks. Our work demonstrates that effective alignment can emerge from within, paving the way for more autonomous and robust AI safety mechanisms that scale without extensive human oversight. 

**Abstract (ZH)**: 确保大型语言模型（LLM）的安全性仍具有挑战性，由于缺乏通用标准和可靠的內容验证器，使得获得有效的训练信号变得困难。我们发现对齐的模型已经具备稳健的内部安全信念：它们在面对有害请求时始终产生高置信度的拒绝，而在生成潜在危险内容时则表现出高随机性。这种随机性 gap 表明了一个未开发的信号——模型内在地“知道”何时拒绝。我们引入了 Safety Instincts Reinforcement Learning (SIRL)，将其内部信心转换为自我生成的奖励信号，从而消除对外部验证器或人工注释的依赖。SIRL 通过强化低随机性拒绝行为来教导模型信任其内在的安全直觉。在 Llama 和 Qwen 模型上进行评估，SIRL 在针对 20 多种不同的 jailbreak 方法（从静态提示到自适应攻击）中保持了 89% 以上的防御成功率 (DSRs)。仅使用 15,000 个未标记的提示，SIRL 超过了资源密集型的监督方法，同时在数学、编程和对话基准上保持了性能。我们的工作证明，有效的对齐可以从内部产生，为无需大量人工监督即可扩展的更自主和稳健的 AI 安全机制铺平了道路。 

---
# Typed Chain-of-Thought: A Curry-Howard Framework for Verifying LLM Reasoning 

**Title (ZH)**: 类型化的链式思考：一种 Curry-Howard 框架用于验证大规模语言模型推理 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2510.01069)  

**Abstract**: While Chain-of-Thought (CoT) prompting enhances the reasoning capabilities of large language models, the faithfulness of the generated rationales remains an open problem for model interpretability. We propose a novel theoretical lens for this problem grounded in the Curry-Howard correspondence, which posits a direct relationship between formal proofs and computer programs. Under this paradigm, a faithful reasoning trace is analogous to a well-typed program, where each intermediate step corresponds to a typed logical inference. We operationalise this analogy, presenting methods to extract and map the informal, natural language steps of CoT into a formal, typed proof structure. Successfully converting a CoT trace into a well-typed proof serves as a strong, verifiable certificate of its computational faithfulness, moving beyond heuristic interpretability towards formal verification. Our framework provides a methodology to transform plausible narrative explanations into formally verifiable programs, offering a path towards building more reliable and trustworthy AI systems. 

**Abstract (ZH)**: 基于 Curry-Howard 对应的链式思考推理的忠实性理论分析：从启发式可解释性到形式验证 

---
# Activation-Deactivation: A General Framework for Robust Post-hoc Explainable AI 

**Title (ZH)**: 激活-去激活：稳健的后验可解释人工智能 geral 框架 

**Authors**: Akchunya Chanchal, David A. Kelly, Hana Chockler  

**Link**: [PDF](https://arxiv.org/pdf/2510.01038)  

**Abstract**: Black-box explainability methods are popular tools for explaining the decisions of image classifiers. A major drawback of these tools is their reliance on mutants obtained by occluding parts of the input, leading to out-of-distribution images. This raises doubts about the quality of the explanations. Moreover, choosing an appropriate occlusion value often requires domain knowledge. In this paper we introduce a novel forward-pass paradigm Activation-Deactivation (AD), which removes the effects of occluded input features from the model's decision-making by switching off the parts of the model that correspond to the occlusions. We introduce ConvAD, a drop-in mechanism that can be easily added to any trained Convolutional Neural Network (CNN), and which implements the AD paradigm. This leads to more robust explanations without any additional training or fine-tuning. We prove that the ConvAD mechanism does not change the decision-making process of the network. We provide experimental evaluation across several datasets and model architectures. We compare the quality of AD-explanations with explanations achieved using a set of masking values, using the proxies of robustness, size, and confidence drop-off. We observe a consistent improvement in robustness of AD explanations (up to 62.5%) compared to explanations obtained with occlusions, demonstrating that ConvAD extracts more robust explanations without the need for domain knowledge. 

**Abstract (ZH)**: 黑盒可解释性方法是解释图像分类器决策的流行工具。这些工具的主要缺点是依赖于通过遮蔽输入部分获得的变异体，导致生成离分布图像。这引发了对解释质量的怀疑。此外，选择合适的遮蔽值通常需要领域知识。在本文中，我们引入了一种新的前向传递范式：激活-去激活（AD），通过关闭与遮蔽部分对应的模型部分来消除遮蔽输入特征对模型决策的影响。我们引入了ConvAD，这是一种可轻松添加到任何训练好的卷积神经网络（CNN）中的机制，并实现了AD范式。这带来了更稳健的解释，而无需额外的训练或微调。我们证明ConvAD机制不会改变网络的决策过程。我们在多个数据集和模型架构上提供了实验评估。我们用鲁棒性、大小和置信度下降作为代理，将AD解释的质量与使用一系列遮蔽值获得的解释进行比较。我们观察到AD解释的鲁棒性在某些情况下提高了62.5%（与遮蔽获得的解释相比），证明ConvAD在不需要领域知识的情况下提取了更稳健的解释。 

---
# Uncovering the Computational Ingredients of Human-Like Representations in LLMs 

**Title (ZH)**: 揭示人类like表示在大语言模型中计算成分的原理 

**Authors**: Zach Studdiford, Timothy T. Rogers, Kushin Mukherjee, Siddharth Suresh  

**Link**: [PDF](https://arxiv.org/pdf/2510.01030)  

**Abstract**: The ability to translate diverse patterns of inputs into structured patterns of behavior has been thought to rest on both humans' and machines' ability to learn robust representations of relevant concepts. The rapid advancement of transformer-based large language models (LLMs) has led to a diversity of computational ingredients -- architectures, fine tuning methods, and training datasets among others -- but it remains unclear which of these ingredients are most crucial for building models that develop human-like representations. Further, most current LLM benchmarks are not suited to measuring representational alignment between humans and models, making benchmark scores unreliable for assessing if current LLMs are making progress towards becoming useful cognitive models. We address these limitations by first evaluating a set of over 70 models that widely vary in their computational ingredients on a triplet similarity task, a method well established in the cognitive sciences for measuring human conceptual representations, using concepts from the THINGS database. Comparing human and model representations, we find that models that undergo instruction-finetuning and which have larger dimensionality of attention heads are among the most human aligned, while multimodal pretraining and parameter size have limited bearing on alignment. Correlations between alignment scores and scores on existing benchmarks reveal that while some benchmarks (e.g., MMLU) are better suited than others (e.g., MUSR) for capturing representational alignment, no existing benchmark is capable of fully accounting for the variance of alignment scores, demonstrating their insufficiency in capturing human-AI alignment. Taken together, our findings help highlight the computational ingredients most essential for advancing LLMs towards models of human conceptual representation and address a key benchmarking gap in LLM evaluation. 

**Abstract (ZH)**: 大型语言模型的计算成分对于构建具备人类类似表示的能力模型至关重要：一种基于三重体相似性任务的评估方法及其启示 

---
# Shape Happens: Automatic Feature Manifold Discovery in LLMs via Supervised Multi-Dimensional Scaling 

**Title (ZH)**: 形状存在：通过监督多维标度学习在大型语言模型中自动发现特征流形 

**Authors**: Federico Tiblias, Irina Bigoulaeva, Jingcheng Niu, Simone Balloccu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2510.01025)  

**Abstract**: The linear representation hypothesis states that language models (LMs) encode concepts as directions in their latent space, forming organized, multidimensional manifolds. Prior efforts focus on discovering specific geometries for specific features, and thus lack generalization. We introduce Supervised Multi-Dimensional Scaling (SMDS), a model-agnostic method to automatically discover feature manifolds. We apply SMDS to temporal reasoning as a case study, finding that different features form various geometric structures such as circles, lines, and clusters. SMDS reveals many insights on these structures: they consistently reflect the properties of the concepts they represent; are stable across model families and sizes; actively support reasoning in models; and dynamically reshape in response to context changes. Together, our findings shed light on the functional role of feature manifolds, supporting a model of entity-based reasoning in which LMs encode and transform structured representations. 

**Abstract (ZH)**: 线性表示假设表明语言模型（LMs）将概念编码为潜在空间中的方向，形成有组织的多维流形。先前的努力专注于发现特定特征的具体几何结构，因此缺乏普适性。我们引入了监督多维标度（SMDS）方法，这是一种模型无关的方法，用于自动发现特征流形。我们将SMDS应用于时间推理作为案例研究，发现不同的特征形成了各种几何结构，如圆、线和聚类。SMDS揭示了这些结构的许多见解：它们一致地反映了所代表的概念的性质；在不同模型族和规模上保持稳定；积极支持模型中的推理；并在上下文变化时动态重塑。我们的发现共同揭示了特征流形的功能作用，支持一种基于实体的推理模型，其中LMs编码和变换结构化的表示。 

---
# Integrating AI and Ensemble Forecasting: Explainable Materials Planning with Scorecards and Trend Insights for a Large-Scale Manufacturer 

**Title (ZH)**: 将AI与 ensemble 预测集成：具有评分卡和趋势洞察的可解释材料规划 

**Authors**: Saravanan Venkatachalam  

**Link**: [PDF](https://arxiv.org/pdf/2510.01006)  

**Abstract**: This paper presents a practical architecture for after-sales demand forecasting and monitoring that unifies a revenue- and cluster-aware ensemble of statistical, machine-learning, and deep-learning models with a role-driven analytics layer for scorecards and trend diagnostics. The framework ingests exogenous signals (installed base, pricing, macro indicators, life cycle, seasonality) and treats COVID-19 as a distinct regime, producing country-part forecasts with calibrated intervals. A Pareto-aware segmentation forecasts high-revenue items individually and pools the long tail via clusters, while horizon-aware ensembling aligns weights with business-relevant losses (e.g., WMAPE). Beyond forecasts, a performance scorecard delivers decision-focused insights: accuracy within tolerance thresholds by revenue share and count, bias decomposition (over- vs under-forecast), geographic and product-family hotspots, and ranked root causes tied to high-impact part-country pairs. A trend module tracks trajectories of MAPE/WMAPE and bias across recent months, flags entities that are improving or deteriorating, detects change points aligned with known regimes, and attributes movements to lifecycle and seasonal factors. LLMs are embedded in the analytics layer to generate role-aware narratives and enforce reporting contracts. They standardize business definitions, automate quality checks and reconciliations, and translate quantitative results into concise, explainable summaries for planners and executives. The system exposes a reproducible workflow - request specification, model execution, database-backed artifacts, and AI-generated narratives - so planners can move from "How accurate are we now?" to "Where is accuracy heading and which levers should we pull?", closing the loop between forecasting, monitoring, and inventory decisions across more than 90 countries and about 6,000 parts. 

**Abstract (ZH)**: 一种基于角色的统计、机器学习与深度学习模型集成的售后需求预测和监控实用架构及其性能分析与趋势诊断框架 

---
# Adaptive Federated Few-Shot Rare-Disease Diagnosis with Energy-Aware Secure Aggregation 

**Title (ZH)**: 自适应联邦少量示例罕见疾病诊断的能源感知安全聚合 

**Authors**: Aueaphum Aueawatthanaphisut  

**Link**: [PDF](https://arxiv.org/pdf/2510.00976)  

**Abstract**: Rare-disease diagnosis remains one of the most pressing challenges in digital health, hindered by extreme data scarcity, privacy concerns, and the limited resources of edge devices. This paper proposes the Adaptive Federated Few-Shot Rare-Disease Diagnosis (AFFR) framework, which integrates three pillars: (i) few-shot federated optimization with meta-learning to generalize from limited patient samples, (ii) energy-aware client scheduling to mitigate device dropouts and ensure balanced participation, and (iii) secure aggregation with calibrated differential privacy to safeguard sensitive model updates. Unlike prior work that addresses these aspects in isolation, AFFR unifies them into a modular pipeline deployable on real-world clinical networks. Experimental evaluation on simulated rare-disease detection datasets demonstrates up to 10% improvement in accuracy compared with baseline FL, while reducing client dropouts by over 50% without degrading convergence. Furthermore, privacy-utility trade-offs remain within clinically acceptable bounds. These findings highlight AFFR as a practical pathway for equitable and trustworthy federated diagnosis of rare conditions. 

**Abstract (ZH)**: Rare疾病的诊断仍然是数字健康领域最具挑战性的问题之一，受到数据极度稀缺、隐私担忧以及边缘设备资源限制的阻碍。本文提出了一种适配 federated 少样本罕见疾病诊断框架（Adaptive Federated Few-Shot Rare-Disease Diagnosis, AFFR），该框架整合了三个支柱：(i) 结合元学习的少样本 federated 优化，以从有限的患者样本中泛化；(ii) 能量感知客户端调度，以减少设备掉线并确保参与的均衡；(iii) 校准差分隐私的安全聚合，以保护敏感模型更新。与以往各自解决这些方面的工作不同，AFFR 将它们统一到一个模块化的管道中，可在实际临床网络中部署。在模拟的罕见疾病检测数据集上的实验评估显示，与基线 federated 学习相比，准确性提高了多达 10%，同时降低了超过 50% 的客户端掉线率，而不会影响收敛性。此外，隐私-效用权衡仍然在临床可接受的范围内。这些发现突显了 AFFR 作为公平和可信赖的 federated 罕见疾病诊断实践路径的重要性。 

---
# QUASAR: Quantum Assembly Code Generation Using Tool-Augmented LLMs via Agentic RL 

**Title (ZH)**: QUASAR: 用增强型RL的工具辅助LLM生成量子装配代码 

**Authors**: Cong Yu, Valter Uotila, Shilong Deng, Qingyuan Wu, Tuo Shi, Songlin Jiang, Lei You, Bo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00967)  

**Abstract**: Designing and optimizing task-specific quantum circuits are crucial to leverage the advantage of quantum computing. Recent large language model (LLM)-based quantum circuit generation has emerged as a promising automatic solution. However, the fundamental challenges remain unaddressed: (i) parameterized quantum gates require precise numerical values for optimal performance, which also depend on multiple aspects, including the number of quantum gates, their parameters, and the layout/depth of the circuits. (ii) LLMs often generate low-quality or incorrect quantum circuits due to the lack of quantum domain-specific knowledge. We propose QUASAR, an agentic reinforcement learning (RL) framework for quantum circuits generation and optimization based on tool-augmented LLMs. To align the LLM with quantum-specific knowledge and improve the generated quantum circuits, QUASAR designs (i) a quantum circuit verification approach with external quantum simulators and (ii) a sophisticated hierarchical reward mechanism in RL training. Extensive evaluation shows improvements in both syntax and semantic performance of the generated quantum circuits. When augmenting a 4B LLM, QUASAR has achieved the validity of 99.31% in Pass@1 and 100% in Pass@10, outperforming industrial LLMs of GPT-4o, GPT-5 and DeepSeek-V3 and several supervised-fine-tuning (SFT)-only and RL-only baselines. 

**Abstract (ZH)**: 基于工具增强的量子电路生成与优化的代理强化学习框架QUASAR 

---
# A Neuro-Fuzzy System for Interpretable Long-Term Stock Market Forecasting 

**Title (ZH)**: 一种用于可解释的长期股市 Forecasting 的神经模糊系统 

**Authors**: Miha Ožbot, Igor Škrjanc, Vitomir Štruc  

**Link**: [PDF](https://arxiv.org/pdf/2510.00960)  

**Abstract**: In the complex landscape of multivariate time series forecasting, achieving both accuracy and interpretability remains a significant challenge. This paper introduces the Fuzzy Transformer (Fuzzformer), a novel recurrent neural network architecture combined with multi-head self-attention and fuzzy inference systems to analyze multivariate stock market data and conduct long-term time series forecasting. The method leverages LSTM networks and temporal attention to condense multivariate data into interpretable features suitable for fuzzy inference systems. The resulting architecture offers comparable forecasting performance to conventional models such as ARIMA and LSTM while providing meaningful information flow within the network. The method was examined on the real world stock market index S\&P500. Initial results show potential for interpretable forecasting and identify current performance tradeoffs, suggesting practical application in understanding and forecasting stock market behavior. 

**Abstract (ZH)**: 在多变量时间序列预测的复杂景观中，同时实现准确性和可解释性仍是一项重大挑战。本文介绍了Fuzzy Transformer（Fuzzformer）这一新颖的递归神经网络架构，该架构结合了多头自注意力机制和模糊推理系统，用于分析多变量股市数据并进行长期时间序列预测。该方法利用LSTM网络和时间注意力机制将多变量数据压缩成适用于模糊推理系统的可解释特征。所提出的方法在与ARIMA和LSTM等传统模型相当的预测性能的同时，提供了网络内部有意义的信息流。该方法在实际的标普500指数股市上进行了测试。初步结果表明，其在可解释预测方面具有潜力，并识别出当前的表现权衡，暗示了其在理解和预测股市行为中的实际应用价值。 

---
# Test-Time Search in Neural Graph Coarsening Procedures for the Capacitated Vehicle Routing Problem 

**Title (ZH)**: 容量约束车辆 routing 问题中神经图粗化过程的测试时搜索 

**Authors**: Yoonju Sim, Hyeonah Kim, Changhyun Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2510.00958)  

**Abstract**: The identification of valid inequalities, such as the rounded capacity inequalities (RCIs), is a key component of cutting plane methods for the Capacitated Vehicle Routing Problem (CVRP). While a deep learning-based separation method can learn to find high-quality cuts, our analysis reveals that the model produces fewer cuts than expected because it is insufficiently sensitive to generate a diverse set of generated subsets. This paper proposes an alternative: enhancing the performance of a trained model at inference time through a new test-time search with stochasticity. First, we introduce stochastic edge selection into the graph coarsening procedure, replacing the previously proposed greedy approach. Second, we propose the Graph Coarsening History-based Partitioning (GraphCHiP) algorithm, which leverages coarsening history to identify not only RCIs but also, for the first time, the Framed capacity inequalities (FCIs). Experiments on randomly generated CVRP instances demonstrate the effectiveness of our approach in reducing the dual gap compared to the existing neural separation method. Additionally, our method discovers effective FCIs on a specific instance, despite the challenging nature of identifying such cuts. 

**Abstract (ZH)**: 基于深度学习的比例容量不等式分离方法在容量约束车辆路径问题中的改进研究 

---
# On Discovering Algorithms for Adversarial Imitation Learning 

**Title (ZH)**: Discovering 算法以应对对抗性模仿学习 

**Authors**: Shashank Reddy Chirra, Jayden Teoh, Praveen Paruchuri, Pradeep Varakantham  

**Link**: [PDF](https://arxiv.org/pdf/2510.00922)  

**Abstract**: Adversarial Imitation Learning (AIL) methods, while effective in settings with limited expert demonstrations, are often considered unstable. These approaches typically decompose into two components: Density Ratio (DR) estimation $\frac{\rho_E}{\rho_{\pi}}$, where a discriminator estimates the relative occupancy of state-action pairs under the policy versus the expert; and Reward Assignment (RA), where this ratio is transformed into a reward signal used to train the policy. While significant research has focused on improving density estimation, the role of reward assignment in influencing training dynamics and final policy performance has been largely overlooked. RA functions in AIL are typically derived from divergence minimization objectives, relying heavily on human design and ingenuity. In this work, we take a different approach: we investigate the discovery of data-driven RA functions, i.e, based directly on the performance of the resulting imitation policy. To this end, we leverage an LLM-guided evolutionary framework that efficiently explores the space of RA functions, yielding \emph{Discovered Adversarial Imitation Learning} (DAIL), the first meta-learnt AIL algorithm. Remarkably, DAIL generalises across unseen environments and policy optimization algorithms, outperforming the current state-of-the-art of \emph{human-designed} baselines. Finally, we analyse why DAIL leads to more stable training, offering novel insights into the role of RA functions in the stability of AIL. Code is publicly available: this https URL. 

**Abstract (ZH)**: 基于敌对模仿学习的发现驱动奖励分配方法：一种元学习算法 

---
# FusionAdapter for Few-Shot Relation Learning in Multimodal Knowledge Graphs 

**Title (ZH)**: Few-Shot Relation Learning in Multimodal Knowledge Graphs 的融合适配器 

**Authors**: Ran Liu, Yuan Fang, Xiaoli Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00894)  

**Abstract**: Multimodal Knowledge Graphs (MMKGs) incorporate various modalities, including text and images, to enhance entity and relation representations. Notably, different modalities for the same entity often present complementary and diverse information. However, existing MMKG methods primarily align modalities into a shared space, which tends to overlook the distinct contributions of specific modalities, limiting their performance particularly in low-resource settings. To address this challenge, we propose FusionAdapter for the learning of few-shot relationships (FSRL) in MMKG. FusionAdapter introduces (1) an adapter module that enables efficient adaptation of each modality to unseen relations and (2) a fusion strategy that integrates multimodal entity representations while preserving diverse modality-specific characteristics. By effectively adapting and fusing information from diverse modalities, FusionAdapter improves generalization to novel relations with minimal supervision. Extensive experiments on two benchmark MMKG datasets demonstrate that FusionAdapter achieves superior performance over state-of-the-art methods. 

**Abstract (ZH)**: 多模态知识图谱中的少量样本关系学习：FusionAdapter方法 

---
# Unveiling Interesting Insights: Monte Carlo Tree Search for Knowledge Discovery 

**Title (ZH)**: 揭开有趣洞察的面纱：蒙特卡洛树搜索在知识发现中的应用 

**Authors**: Pietro Totis, Alberto Pozanco, Daniel Borrajo  

**Link**: [PDF](https://arxiv.org/pdf/2510.00876)  

**Abstract**: Organizations are increasingly focused on leveraging data from their processes to gain insights and drive decision-making. However, converting this data into actionable knowledge remains a difficult and time-consuming task. There is often a gap between the volume of data collected and the ability to process and understand it, which automated knowledge discovery aims to fill. Automated knowledge discovery involves complex open problems, including effectively navigating data, building models to extract implicit relationships, and considering subjective goals and knowledge. In this paper, we introduce a novel method for Automated Insights and Data Exploration (AIDE), that serves as a robust foundation for tackling these challenges through the use of Monte Carlo Tree Search (MCTS). We evaluate AIDE using both real-world and synthetic data, demonstrating its effectiveness in identifying data transformations and models that uncover interesting data patterns. Among its strengths, AIDE's MCTS-based framework offers significant extensibility, allowing for future integration of additional pattern extraction strategies and domain knowledge. This makes AIDE a valuable step towards developing a comprehensive solution for automated knowledge discovery. 

**Abstract (ZH)**: 组织越来越注重利用其过程中的数据以获得洞察并推动决策。然而，将这些数据转化为可操作的知识仍然是一个困难且耗时的过程。往往存在收集的数据量与处理和理解数据的能力之间的差距，自动化知识发现旨在填补这一差距。自动化知识发现涉及复杂的开放性问题，包括有效地导航数据、构建模型以提取隐含关系、以及考虑主观目标和知识。在本文中，我们介绍了一种新型的自动化洞察与数据探索（AIDE）方法，通过蒙特卡洛树搜索（MCTS）的应用为其提供了坚实的基础，以应对这些挑战。我们使用真实世界和合成数据评估了AIDE，展示了其在识别数据转换和模型方面的有效性，这些模型能够揭示有趣的数据模式。AIDE的基于MCTS的框架具有显著的扩展性，未来可以集成其他模式提取策略和领域知识，使其成为开发全面的自动化知识发现解决方案的重要一步。 

---
# Learning Compact Representations of LLM Abilities via Item Response Theory 

**Title (ZH)**: 通过项目反应理论学习大语言模型能力的紧凑表示 

**Authors**: Jianhao Chen, Chenxu Wang, Gengrui Zhang, Peng Ye, Lei Bai, Wei Hu, Yuzhong Qu, Shuyue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00844)  

**Abstract**: Recent years have witnessed a surge in the number of large language models (LLMs), yet efficiently managing and utilizing these vast resources remains a significant challenge. In this work, we explore how to learn compact representations of LLM abilities that can facilitate downstream tasks, such as model routing and performance prediction on new benchmarks. We frame this problem as estimating the probability that a given model will correctly answer a specific query. Inspired by the item response theory (IRT) in psychometrics, we model this probability as a function of three key factors: (i) the model's multi-skill ability vector, (2) the query's discrimination vector that separates models of differing skills, and (3) the query's difficulty scalar. To learn these parameters jointly, we introduce a Mixture-of-Experts (MoE) network that couples model- and query-level embeddings. Extensive experiments demonstrate that our approach leads to state-of-the-art performance in both model routing and benchmark accuracy prediction. Moreover, analysis validates that the learned parameters encode meaningful, interpretable information about model capabilities and query characteristics. 

**Abstract (ZH)**: 最近几年，大型语言模型（LLMs）的数量急剧增加，但有效地管理并充分利用这些庞大的资源仍是一项重大挑战。本文探讨了如何学习紧凑的LLM能力表示，以促进下游任务，如模型路由和新基准上的性能预测。我们将这个问题框架化为估计给定模型正确回答特定查询的概率。受心理测量学中的项目反应理论（IRT）的启发，我们将这个概率建模为三个关键因素的函数：（i）模型的多技能能力向量，（ii）区分不同技能模型的查询区分向量，以及（iii）查询的难度标量。为了联合学习这些参数，我们引入了一个专家混合（MoE）网络，将模型级和查询级嵌入相结合。广泛的经验研究表明，我们的方法在模型路由和基准准确率预测方面达到了最先进的性能。此外，分析验证了学习到的参数编码了关于模型能力和查询特征的有意义且可解释的信息。 

---
# Improving Cryptocurrency Pump-and-Dump Detection through Ensemble-Based Models and Synthetic Oversampling Techniques 

**Title (ZH)**: 基于集成模型和合成过采样技术的加密货币 Pump-and-Dump 检测改进方法 

**Authors**: Jieun Yu, Minjung Park, Sangmi Chai  

**Link**: [PDF](https://arxiv.org/pdf/2510.00836)  

**Abstract**: This study aims to detect pump and dump (P&D) manipulation in cryptocurrency markets, where the scarcity of such events causes severe class imbalance and hinders accurate detection. To address this issue, the Synthetic Minority Oversampling Technique (SMOTE) was applied, and advanced ensemble learning models were evaluated to distinguish manipulative trading behavior from normal market activity. The experimental results show that applying SMOTE greatly enhanced the ability of all models to detect P&D events by increasing recall and improving the overall balance between precision and recall. In particular, XGBoost and LightGBM achieved high recall rates (94.87% and 93.59%, respectively) with strong F1-scores and demonstrated fast computational performance, making them suitable for near real time surveillance. These findings indicate that integrating data balancing techniques with ensemble methods significantly improves the early detection of manipulative activities, contributing to a fairer, more transparent, and more stable cryptocurrency market. 

**Abstract (ZH)**: 本研究旨在检测加密货币市场中的泵抬饯卖（P&D）操纵，由于此类事件的稀有性导致严重的类别不平衡，阻碍了准确检测。为解决这一问题，本研究应用了合成少数类过采样技术（SMOTE），并评估了先进的集成学习模型，以区分操纵性交易行为和正常市场活动。实验结果表明，应用SMOTE显著增强了所有模型检测P&D事件的能力，通过提高召回率并改善精确率和召回率之间的整体平衡。特别是XGBoost和LightGBM实现了高召回率（分别为94.87%和93.59%），具有强大的F1分数和快速的计算性能，使其适合近实时监控。这些发现表明，将数据平衡技术与集成方法结合显著提高了操纵性活动的早期检测能力，有助于构建更加公平、透明和稳定的加密货币市场。 

---
# Benchmarking Machine Learning Models for Fault Classification and Localization in Power System Protection 

**Title (ZH)**: 基于电力系统保护中故障分类与定位的机器学习模型 benchmark研究 

**Authors**: Julian Oelhaf, Georg Kordowich, Changhun Kim, Paula Andrea Pérez-Toro, Christian Bergler, Andreas Maier, Johann Jäger, Siming Bayer  

**Link**: [PDF](https://arxiv.org/pdf/2510.00831)  

**Abstract**: The increasing integration of distributed energy resources (DERs), particularly renewables, poses significant challenges for power system protection, with fault classification (FC) and fault localization (FL) being among the most critical tasks. Conventional protection schemes, based on fixed thresholds, cannot reliably identify and localize short circuits with the increasing complexity of the grid under dynamic conditions. Machine learning (ML) offers a promising alternative; however, systematic benchmarks across models and settings remain limited. This work presents, for the first time, a comparative benchmarking study of classical ML models for FC and FL in power system protection based on EMT data. Using voltage and current waveforms segmented into sliding windows of 10 ms to 50 ms, we evaluate models under realistic real-time constraints. Performance is assessed in terms of accuracy, robustness to window size, and runtime efficiency. The best-performing FC model achieved an F1 score of 0.992$\pm$0.001, while the top FL model reached an R2 of 0.806$\pm$0.008 with a mean processing time of 0.563 ms. 

**Abstract (ZH)**: 分布式能源资源（DERs）特别是可再生能源 increasing 连接对电力系统保护带来的挑战：短路分类（FC）和短路定位（FL）的关键任务分析及基于EMT数据的经典机器学习模型比较研究 

---
# Logical Consistency Between Disagreeing Experts and Its Role in AI Safety 

**Title (ZH)**: 分歧专家之间的逻辑一致性及其在AI安全中的作用 

**Authors**: Andrés Corrada-Emmanuel  

**Link**: [PDF](https://arxiv.org/pdf/2510.00821)  

**Abstract**: If two experts disagree on a test, we may conclude both cannot be 100 per cent correct. But if they completely agree, no possible evaluation can be excluded. This asymmetry in the utility of agreements versus disagreements is explored here by formalizing a logic of unsupervised evaluation for classifiers. Its core problem is computing the set of group evaluations that are logically consistent with how we observe them agreeing and disagreeing in their decisions. Statistical summaries of their aligned decisions are inputs into a Linear Programming problem in the integer space of possible correct or incorrect responses given true labels. Obvious logical constraints, such as, the number of correct responses cannot exceed the number of observed responses, are inequalities. But in addition, there are axioms, universally applicable linear equalities that apply to all finite tests. The practical and immediate utility of this approach to unsupervised evaluation using only logical consistency is demonstrated by building no-knowledge alarms that can detect when one or more LLMs-as-Judges are violating a minimum grading threshold specified by the user. 

**Abstract (ZH)**: 探讨一致性和分歧在评估中的不对称性：基于无监督评价的逻辑 formalization及其应用 

---
# Semantic Bridges Between First Order c-Representations and Cost-Based Semantics: An Initial Perspective 

**Title (ZH)**: 一阶 c-表示与成本基础语义之间的语义桥梁：初步视角 

**Authors**: Nicholas Leisegang, Giovanni Casini, Thomas Meyer  

**Link**: [PDF](https://arxiv.org/pdf/2510.00817)  

**Abstract**: Weighted-knowledge bases and cost-based semantics represent a recent formalism introduced by Bienvenu et al. for Ontology Mediated Data Querying in the case where a given knowledge base is inconsistent. This is done by adding a weight to each statement in the knowledge base (KB), and then giving each DL interpretation a cost based on how often it breaks rules in the KB. In this paper we compare this approach with c-representations, a form of non-monotonic reasoning originally introduced by Kern-Isberner. c-Representations describe a means to interpret defeasible concept inclusions in the first-order case. This is done by assigning a numerical ranking to each interpretations via penalties for each violated conditional. We compare these two approaches on a semantic level. In particular, we show that under certain conditions a weighted knowledge base and a set of defeasible conditionals can generate the same ordering on interpretations, and therefore an equivalence of semantic structures up to relative cost. Moreover, we compare entailment described in both cases, where certain notions are equivalently expressible in both formalisms. Our results have the potential to benefit further work on both cost-based semantics and c-representations 

**Abstract (ZH)**: 基于权重的知识本体和成本导向语义在本体介导数据查询中的应用：与Kern-Isberner提出c-表示的比较 

---
# Benchmarking Agentic Systems in Automated Scientific Information Extraction with ChemX 

**Title (ZH)**: 基于ChemX的自动化科学研究信息抽取中代理系统基准测试 

**Authors**: Anastasia Vepreva, Julia Razlivina, Maria Eremeeva, Nina Gubina, Anastasia Orlova, Aleksei Dmitrenko, Ksenya Kapranova, Susan Jyakhwo, Nikita Vasilev, Arsen Sarkisyan, Ivan Yu. Chernyshov, Vladimir Vinogradov, Andrei Dmitrenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.00795)  

**Abstract**: The emergence of agent-based systems represents a significant advancement in artificial intelligence, with growing applications in automated data extraction. However, chemical information extraction remains a formidable challenge due to the inherent heterogeneity of chemical data. Current agent-based approaches, both general-purpose and domain-specific, exhibit limited performance in this domain. To address this gap, we present ChemX, a comprehensive collection of 10 manually curated and domain-expert-validated datasets focusing on nanomaterials and small molecules. These datasets are designed to rigorously evaluate and enhance automated extraction methodologies in chemistry. To demonstrate their utility, we conduct an extensive benchmarking study comparing existing state-of-the-art agentic systems such as ChatGPT Agent and chemical-specific data extraction agents. Additionally, we introduce our own single-agent approach that enables precise control over document preprocessing prior to extraction. We further evaluate the performance of modern baselines, such as GPT-5 and GPT-5 Thinking, to compare their capabilities with agentic approaches. Our empirical findings reveal persistent challenges in chemical information extraction, particularly in processing domain-specific terminology, complex tabular and schematic representations, and context-dependent ambiguities. The ChemX benchmark serves as a critical resource for advancing automated information extraction in chemistry, challenging the generalization capabilities of existing methods, and providing valuable insights into effective evaluation strategies. 

**Abstract (ZH)**: 基于代理的系统 emergence 代表了人工智能的重要进展，并在自动化数据提取方面有着日益增长的应用。然而，由于化学数据的固有异质性，化学信息提取仍然是一个严峻的挑战。现有的通用和领域特定的基于代理的方法在这个领域表现有限。为了解决这一差距，我们提出了 ChemX，这是一个包含 10 个手工精选和领域专家验证的数据集的综合集合，专注于纳米材料和小分子。这些数据集旨在严格评估和提升化学领域的自动化提取方法。为了展示其用途，我们进行了广泛的标准测试，比较了现有的基于代理的系统（如 ChatGPT Agent 和化学特定数据提取代理）。此外，我们还引入了一种自己的单代理方法，使其能够在提取前对文档预处理进行精确控制。我们进一步评估了现代基线方法（如 GPT-5 和 GPT-5 Thinking），以比较它们与基于代理的方法的能力差异。我们的实证发现揭示了化学信息提取中持续存在的挑战，特别是在处理领域特定术语、复杂表格和示意图表示以及上下文依赖性歧义方面。ChemX 基准测试作为推动化学领域自动化信息提取进展的宝贵资源，挑战现有方法的泛化能力，并提供了关于有效评估策略的有价值的见解。 

---
# AI in data science education: experiences from the classroom 

**Title (ZH)**: AI在数据科学教育中的应用：课堂教学经验 

**Authors**: J.A. Hageman, C.F.W. Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2510.00793)  

**Abstract**: This study explores the integration of AI, particularly large language models (LLMs) like ChatGPT, into educational settings, focusing on the implications for teaching and learning. Through interviews with course coordinators from data science courses at Wageningen University, this research identifies both the benefits and challenges associated with AI in the classroom. While AI tools can streamline tasks and enhance learning, concerns arise regarding students' overreliance on these technologies, potentially hindering the development of essential cognitive and problem solving skills. The study highlights the importance of responsible AI usage, ethical considerations, and the need for adapting assessment methods to ensure educational outcomes are met. With careful integration, AI can be a valuable asset in education, provided it is used to complement rather than replace fundamental learning processes. 

**Abstract (ZH)**: 本研究探讨了人工智能，特别是大型语言模型（LLMs）如ChatGPT，融入教育环境中的方式，重点关注其对教学和学习的影响。通过访谈瓦格宁根大学数据科学课程的课程协调者，本研究识别了人工智能在学校环境中使用的优势与挑战。尽管人工智能工具可以简化任务并增强学习，但学生过度依赖这些技术可能导致其认知和解决问题能力的发展受阻。研究强调负责任地使用人工智能、伦理考虑以及适应性评估方法调整的重要性，以确保教育成果的实现。通过谨慎融合，人工智能可以成为教育领域的有价值的资产，前提是将其用于补充而不是替代基本的学习过程。 

---
# DIA: The Adversarial Exposure of Deterministic Inversion in Diffusion Models 

**Title (ZH)**: DIA：确定性反演在扩散模型中的对抗曝光 

**Authors**: Seunghoo Hong, Geonho Son, Juhun Lee, Simon S. Woo  

**Link**: [PDF](https://arxiv.org/pdf/2510.00778)  

**Abstract**: Diffusion models have shown to be strong representation learners, showcasing state-of-the-art performance across multiple domains. Aside from accelerated sampling, DDIM also enables the inversion of real images back to their latent codes. A direct inheriting application of this inversion operation is real image editing, where the inversion yields latent trajectories to be utilized during the synthesis of the edited image. Unfortunately, this practical tool has enabled malicious users to freely synthesize misinformative or deepfake contents with greater ease, which promotes the spread of unethical and abusive, as well as privacy-, and copyright-infringing contents. While defensive algorithms such as AdvDM and Photoguard have been shown to disrupt the diffusion process on these images, the misalignment between their objectives and the iterative denoising trajectory at test time results in weak disruptive this http URL this work, we present the DDIM Inversion Attack (DIA) that attacks the integrated DDIM trajectory path. Our results support the effective disruption, surpassing previous defensive methods across various editing methods. We believe that our frameworks and results can provide practical defense methods against the malicious use of AI for both the industry and the research community. Our code is available here: this https URL. 

**Abstract (ZH)**: 基于DDIM的逆过程攻击：有效抵御AI滥用的防御方法 

---
# EvolProver: Advancing Automated Theorem Proving by Evolving Formalized Problems via Symmetry and Difficulty 

**Title (ZH)**: EvolProver: 通过对称性和难度演化正式化问题以推动自动定理证明的发展 

**Authors**: Yuchen Tian, Ruiyuan Huang, Xuanwu Wang, Jing Ma, Zengfeng Huang, Ziyang Luo, Hongzhan Lin, Da Zheng, Lun Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.00732)  

**Abstract**: Large Language Models (LLMs) for formal theorem proving have shown significant promise, yet they often lack generalizability and are fragile to even minor transformations of problem statements. To address this limitation, we introduce a novel data augmentation pipeline designed to enhance model robustness from two perspectives: symmetry and difficulty. From the symmetry perspective, we propose two complementary methods: EvolAST, an Abstract Syntax Tree (AST) based approach that targets syntactic symmetry to generate semantically equivalent problem variants, and EvolDomain, which leverages LLMs to address semantic symmetry by translating theorems across mathematical domains. From the difficulty perspective, we propose EvolDifficulty, which uses carefully designed evolutionary instructions to guide LLMs in generating new theorems with a wider range of difficulty. We then use the evolved data to train EvolProver, a 7B-parameter non-reasoning theorem prover. EvolProver establishes a new state-of-the-art (SOTA) on FormalMATH-Lite with a 53.8% pass@32 rate, surpassing all models of comparable size, including reasoning-based models. It also sets new SOTA records for non-reasoning models on MiniF2F-Test (69.8% pass@32), Ineq-Comp-Seed (52.2% pass@32), and Ineq-Comp-Transformed (34.0% pass@32). Ablation studies further confirm our data augmentation pipeline's effectiveness across multiple benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在形式定理证明中的应用显示出了显著的潜力，但它们往往缺乏通用性，并且容易受到问题陈述轻微变化的影响。为了解决这一限制，我们提出了一种新颖的数据增强管道，从对称性和难度两个方面增强模型的稳健性。从对称性的角度来看，我们提出了两种互补的方法：EvolAST，一种基于抽象语法树（AST）的方法，针对语法规则的对称性来生成语义等价的问题变体；EvolDomain，利用LLMs通过在不同数学领域之间翻译定理来解决语义对称性问题。从难度角度来看，我们提出了EvolDifficulty，它使用精心设计的进化指令来引导LLMs生成具有更广泛难度范围的新定理。然后，我们使用增强后的数据来训练EvolProver，这是一个7B参数的非推理定理证明器。EvolProver在FormalMATH-Lite上达到了新的最佳性能，通过率为53.8%，超越了所有可比规模的模型，包括基于推理的模型。它还在MiniF2F-Test、Ineq-Comp-Seed 和 Ineq-Comp-Transformed 上分别达到了新的最佳性能记录，通过率分别为69.8%、52.2%和34.0%。消融研究表明，我们的数据增强管道在多个基准上的有效性。 

---
# AttentionDep: Domain-Aware Attention for Explainable Depression Severity Assessment 

**Title (ZH)**: AttentionDep: 域aware注意力机制用于抑郁严重程度可解释评估 

**Authors**: Yusif Ibrahimov, Tarique Anwar, Tommy Yuan, Turan Mutallimov, Elgun Hasanov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00706)  

**Abstract**: In today's interconnected society, social media platforms provide a window into individuals' thoughts, emotions, and mental states. This paper explores the use of platforms like Facebook, X (formerly Twitter), and Reddit for depression severity detection. We propose AttentionDep, a domain-aware attention model that drives explainable depression severity estimation by fusing contextual and domain knowledge. Posts are encoded hierarchically using unigrams and bigrams, with attention mechanisms highlighting clinically relevant tokens. Domain knowledge from a curated mental health knowledge graph is incorporated through a cross-attention mechanism, enriching the contextual features. Finally, depression severity is predicted using an ordinal regression framework that respects the clinical-relevance and natural ordering of severity levels. Our experiments demonstrate that AttentionDep outperforms state-of-the-art baselines by over 5% in graded F1 score across datasets, while providing interpretable insights into its predictions. This work advances the development of trustworthy and transparent AI systems for mental health assessment from social media. 

**Abstract (ZH)**: 今天相互连接的社会中，社交媒体平台为了解个体的思想、情感和心理状态提供了一个窗口。本文探讨了在Facebook、X（原Twitter）和Reddit等平台中检测抑郁严重程度的方法。我们提出了一种AttentionDep模型，该模型通过融合上下文和领域知识来驱动可解释的抑郁严重程度估计。帖子通过单克隆和双克隆逐层编码，并使用注意力机制突出显示临床相关的单词。通过交叉注意力机制整合精心构建的心理健康知识图谱领域的知识，丰富了上下文特征。最后，使用一个尊重临床相关性和自然严重程度级别顺序的序数回归框架来预测抑郁严重程度。我们的实验表明，AttentionDep在多个数据集上的分级F1分数上比最先进的基线方法高5%以上，并提供了对其预测的可解释洞见。这项工作促进了从社交媒体进行心理健康评估的可信和透明AI系统的开发。 

---
# ACPO: Adaptive Curriculum Policy Optimization for Aligning Vision-Language Models in Complex Reasoning 

**Title (ZH)**: 自适应课程策略优化：面向复杂推理中视觉语言模型对齐 

**Authors**: Yunhao Wang, Ziting Li, Shuai Chen, Tao Liu, Chao Song, Junjie Jiang, Jian Zhu, Peng Gao, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.00690)  

**Abstract**: Aligning large-scale vision-language models (VLMs) for complex reasoning via reinforcement learning is often hampered by the limitations of existing policy optimization algorithms, such as static training schedules and the rigid, uniform clipping mechanism in Proximal Policy Optimization (PPO). In this work, we introduce Adaptive Curriculum Policy Optimization (ACPO), a novel framework that addresses these challenges through a dual-component adaptive learning strategy. First, ACPO employs a dynamic curriculum that orchestrates a principled transition from a stable, near on-policy exploration phase to an efficient, off-policy exploitation phase by progressively increasing sample reuse. Second, we propose an Advantage-Aware Adaptive Clipping (AAAC) mechanism that replaces the fixed clipping hyperparameter with dynamic, sample-wise bounds modulated by the normalized advantage of each token. This allows for more granular and robust policy updates, enabling larger gradients for high-potential samples while safeguarding against destructive ones. We conduct extensive experiments on a suite of challenging multimodal reasoning benchmarks, including MathVista, LogicVista, and MMMU-Pro. Results demonstrate that ACPO consistently outperforms strong baselines such as DAPO and PAPO, achieving state-of-the-art performance, accelerated convergence, and superior training stability. 

**Abstract (ZH)**: 通过强化学习调整大规模视觉-语言模型进行复杂推理：Adaptive Curriculum Policy Optimization (ACPO) 通过双组件自适应学习策略解决现有挑战 

---
# Relevance-Zone Reduction in Game Solving 

**Title (ZH)**: 游戏求解中的相关性区域缩减 

**Authors**: Chi-Huang Lin, Ting Han Wei, Chun-Jui Wang, Hung Guei, Chung-Chin Shih, Yun-Jui Tsai, I-Chen Wu, Ti-Rong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00689)  

**Abstract**: Game solving aims to find the optimal strategies for all players and determine the theoretical outcome of a game. However, due to the exponential growth of game trees, many games remain unsolved, even though methods like AlphaZero have demonstrated super-human level in game playing. The Relevance-Zone (RZ) is a local strategy reuse technique that restricts the search to only the regions relevant to the outcome, significantly reducing the search space. However, RZs are not unique. Different solutions may result in RZs of varying sizes. Smaller RZs are generally more favorable, as they increase the chance of reuse and improve pruning efficiency. To this end, we propose an iterative RZ reduction method that repeatedly solves the same position while gradually restricting the region involved, guiding the solver toward smaller RZs. We design three constraint generation strategies and integrate an RZ Pattern Table to fully leverage past solutions. In experiments on 7x7 Killall-Go, our method reduces the average RZ size to 85.95% of the original. Furthermore, the reduced RZs can be permanently stored as reusable knowledge for future solving tasks, especially for larger board sizes or different openings. 

**Abstract (ZH)**: 游戏求解旨在找到所有玩家的最优策略并确定游戏的理论结果。然而，由于游戏树的指数增长，很多游戏仍处于未解状态，尽管像AlphaZero这样的方法已经在游戏对弈中展现了超人类水平。相关性区域（RZ）是一种局部策略重用技术，它将搜索限制在对结果相关的区域，显著减少了搜索空间。然而，RZ并不唯一。不同的解决方案可能导致不同大小的RZ。较小的RZ通常更为有利，因为它们增加了重用的机会并提高了剪枝效率。为此，我们提出了一种迭代的RZ缩减方法，该方法在每次解决相同位置时逐步限制涉及的区域，引导求解器趋向于更小的RZ。我们设计了三种约束生成策略并集成了一个RZ模式表，以充分利用以往的解决方案。在7x7 Killall-Go的实验中，我们的方法将平均RZ大小减少了85.95%。此外，缩减后的RZ可以被永久存储为可重复使用的知识，特别适用于更大棋盘大小或不同开局的解题任务。 

---
# Batch-CAM: Introduction to better reasoning in convolutional deep learning models 

**Title (ZH)**: 批量CAM：介绍在卷积深度学习模型中更好地进行推理的方法 

**Authors**: Giacomo Ignesti, Davide Moroni, Massimo Martinelli  

**Link**: [PDF](https://arxiv.org/pdf/2510.00664)  

**Abstract**: Understanding the inner workings of deep learning models is crucial for advancing artificial intelligence, particularly in high-stakes fields such as healthcare, where accurate explanations are as vital as precision. This paper introduces Batch-CAM, a novel training paradigm that fuses a batch implementation of the Grad-CAM algorithm with a prototypical reconstruction loss. This combination guides the model to focus on salient image features, thereby enhancing its performance across classification tasks. Our results demonstrate that Batch-CAM achieves a simultaneous improvement in accuracy and image reconstruction quality while reducing training and inference times. By ensuring models learn from evidence-relevant information,this approach makes a relevant contribution to building more transparent, explainable, and trustworthy AI systems. 

**Abstract (ZH)**: 理解深度学习模型的内部工作机制对于促进人工智能的发展至关重要，尤其是在如医疗健康这样高 stakes 的领域，准确的解释与精确性同样重要。本文提出了一种新的训练范式 Batch-CAM，它将批处理实现的 Grad-CAM 算法与原型重建损失相结合。这种结合引导模型关注关键图像特征，从而在分类任务中提升其性能。我们的结果表明，Batch-CAM 同时提高了分类准确率和图像重建质量，并缩短了训练和推理时间。通过确保模型从相关证据中学习，这种方法为构建更透明、可解释和值得信赖的 AI 系统做出了相关贡献。 

---
# Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution 

**Title (ZH)**: 预期注意力：通过估计未来查询分布来压缩KV缓存 

**Authors**: Alessio Devoto, Maximilian Jeblick, Simon Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00636)  

**Abstract**: Memory consumption of the Key-Value (KV) cache represents a major bottleneck for efficient large language model inference. While attention-score-based KV cache pruning shows promise, it faces critical practical limitations: attention scores from future tokens are unavailable during compression, and modern implementations like Flash Attention do not materialize the full attention matrix, making past scores inaccessible. To overcome these challenges, we introduce $\textbf{Expected Attention, a training-free compression method}$ that estimates KV pairs importance by predicting how future queries will attend to them. Our approach leverages the distributional properties of LLM activations to compute expected attention scores in closed form for each KV pair. These scores enable principled ranking and pruning of KV pairs with minimal impact on the residual stream, achieving effective compression without performance degradation. Importantly, our method operates seamlessly across both prefilling and decoding phases, consistently outperforming state-of-the-art baselines in both scenarios. Finally, $\textbf{we release KVPress, a comprehensive library to enable researchers to implement and benchmark KV cache compression methods, already including more than 20 techniques}$. 

**Abstract (ZH)**: 基于期望注意的KV缓存压缩方法：一种无需训练的压缩方法，以及KVPress库的发布 

---
# Collaborative-Distilled Diffusion Models (CDDM) for Accelerated and Lightweight Trajectory Prediction 

**Title (ZH)**: 协作精简扩散模型（CDDM）用于加速和轻量级轨迹预测 

**Authors**: Bingzhang Wang, Kehua Chen, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00627)  

**Abstract**: Trajectory prediction is a fundamental task in Autonomous Vehicles (AVs) and Intelligent Transportation Systems (ITS), supporting efficient motion planning and real-time traffic safety management. Diffusion models have recently demonstrated strong performance in probabilistic trajectory prediction, but their large model size and slow sampling process hinder real-world deployment. This paper proposes Collaborative-Distilled Diffusion Models (CDDM), a novel method for real-time and lightweight trajectory prediction. Built upon Collaborative Progressive Distillation (CPD), CDDM progressively transfers knowledge from a high-capacity teacher diffusion model to a lightweight student model, jointly reducing both the number of sampling steps and the model size across distillation iterations. A dual-signal regularized distillation loss is further introduced to incorporate guidance from both the teacher and ground-truth data, mitigating potential overfitting and ensuring robust performance. Extensive experiments on the ETH-UCY pedestrian benchmark and the nuScenes vehicle benchmark demonstrate that CDDM achieves state-of-the-art prediction accuracy. The well-distilled CDDM retains 96.2% and 95.5% of the baseline model's ADE and FDE performance on pedestrian trajectories, while requiring only 231K parameters and 4 or 2 sampling steps, corresponding to 161x compression, 31x acceleration, and 9 ms latency. Qualitative results further show that CDDM generates diverse and accurate trajectories under dynamic agent behaviors and complex social interactions. By bridging high-performing generative models with practical deployment constraints, CDDM enables resource-efficient probabilistic prediction for AVs and ITS. Code is available at this https URL. 

**Abstract (ZH)**: 协作精简扩散模型（CDDM）：面向实时轻量级轨迹预测的新型方法 

---
# Is Model Editing Built on Sand? Revealing Its Illusory Success and Fragile Foundation 

**Title (ZH)**: 模型编辑建立在沙子之上？揭示其虚幻的成功与脆弱的基础 

**Authors**: Wei Liu, Haomei Xu, Bingqing Liu, Zhiying Deng, Haozhao Wang, Jun Wang, Ruixuan Li, Yee Whye Teh, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00625)  

**Abstract**: Large language models (LLMs) inevitably encode outdated or incorrect knowledge. Updating, deleting, and forgetting such knowledge is important for alignment, safety, and other issues. To address this issue, model editing has emerged as a promising paradigm: by precisely editing a small subset of parameters such that a specific fact is updated while preserving other knowledge. Despite its great success reported in previous papers, we find the apparent reliability of editing rests on a fragile foundation and the current literature is largely driven by illusory success. The fundamental goal of steering the model's output toward a target with minimal modification would encourage exploiting hidden shortcuts, rather than utilizing real semantics. This problem directly challenges the feasibility of the current model editing literature at its very foundation, as shortcuts are inherently at odds with robust knowledge integration. Coincidentally, this issue has long been obscured by evaluation frameworks that lack the design of negative examples. To uncover it, we systematically develop a suite of new evaluation methods. Strikingly, we find that state-of-the-art approaches collapse even under the simplest negation queries. Our empirical evidence shows that editing is likely to be based on shortcuts rather than full semantics, calling for an urgent reconsideration of the very basis of model editing before further advancements can be meaningfully pursued. 

**Abstract (ZH)**: 大型语言模型（LLMs）不可避免地包含过时或错误的知识。更新、删除和遗忘这些知识对于对齐、安全性及其他问题至关重要。为解决这一问题，模型编辑已 emerges 作为一种有前景的范式：通过精确编辑一小部分参数，使特定事实得到更新同时保留其他知识。尽管 previous 论文报道了其显著的成功，但我们发现编辑的显著可靠性建立在一个脆弱的基础之上，而当前文献很大程度上是由虚假的成功驱动的。引导模型输出朝向目标的同时最小化修改的根本目标会促进利用隐藏的捷径，而非利用真实的语义。这一问题直接挑战了当前模型编辑文献的基础，因为捷径与稳健的知识整合本质上是矛盾的。巧合的是，这一问题长期以来一直被缺乏负面样例设计的评估框架所掩盖。为了揭示这一点，我们系统地开发了一系列新的评估方法。令人惊讶的是，我们发现最先进的方法在最简单的否定查询下就会崩溃。我们的实证证据表明，编辑很可能基于捷径而非完整语义，这要求在进一步取得实质性进展之前，亟需重新审视模型编辑的基础。 

---
# HARPA: A Testability-Driven, Literature-Grounded Framework for Research Ideation 

**Title (ZH)**: HARPA：一种以测试性驱动、文献为基础的研究构想框架 

**Authors**: Rosni Vasu, Peter Jansen, Pao Siangliulue, Cristina Sarasua, Abraham Bernstein, Peter Clark, Bhavana Dalvi Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2510.00620)  

**Abstract**: While there has been a surge of interest in automated scientific discovery (ASD), especially with the emergence of LLMs, it remains challenging for tools to generate hypotheses that are both testable and grounded in the scientific literature. Additionally, existing ideation tools are not adaptive to prior experimental outcomes. We developed HARPA to address these challenges by incorporating the ideation workflow inspired by human researchers. HARPA first identifies emerging research trends through literature mining, then explores hypothesis design spaces, and finally converges on precise, testable hypotheses by pinpointing research gaps and justifying design choices. Our evaluations show that HARPA-generated hypothesis-driven research proposals perform comparably to a strong baseline AI-researcher across most qualitative dimensions (e.g., specificity, novelty, overall quality), but achieve significant gains in feasibility(+0.78, p$<0.05$, bootstrap) and groundedness (+0.85, p$<0.01$, bootstrap) on a 10-point Likert scale. When tested with the ASD agent (CodeScientist), HARPA produced more successful executions (20 vs. 11 out of 40) and fewer failures (16 vs. 21 out of 40), showing that expert feasibility judgments track with actual execution success. Furthermore, to simulate how researchers continuously refine their understanding of what hypotheses are both testable and potentially interesting from experience, HARPA learns a reward model that scores new hypotheses based on prior experimental outcomes, achieving approx. a 28\% absolute gain over HARPA's untrained baseline scorer. Together, these methods represent a step forward in the field of AI-driven scientific discovery. 

**Abstract (ZH)**: 自动化科学发现中的HARPA：应对假设生成挑战的新型工具 

---
# ACON: Optimizing Context Compression for Long-horizon LLM Agents 

**Title (ZH)**: ACON: 优化长_horizon LLM代理的情境压缩 

**Authors**: Minki Kang, Wei-Ning Chen, Dongge Han, Huseyin A. Inan, Lukas Wutschitz, Yanzhi Chen, Robert Sim, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00615)  

**Abstract**: Large language models (LLMs) are increasingly deployed as agents in dynamic, real-world environments, where success requires both reasoning and effective tool use. A central challenge for agentic tasks is the growing context length, as agents must accumulate long histories of actions and observations. This expansion raises costs and reduces efficiency in long-horizon tasks, yet prior work on context compression has mostly focused on single-step tasks or narrow applications. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both environment observations and interaction histories into concise yet informative condensations. ACON leverages compression guideline optimization in natural language space: given paired trajectories where full context succeeds but compressed context fails, capable LLMs analyze the causes of failure, and the compression guideline is updated accordingly. Furthermore, we propose distilling the optimized LLM compressor into smaller models to reduce the overhead of the additional module. Experiments on AppWorld, OfficeBench, and Multi-objective QA show that ACON reduces memory usage by 26-54% (peak tokens) while largely preserving task performance, preserves over 95% of accuracy when distilled into smaller compressors, and enhances smaller LMs as long-horizon agents with up to 46% performance improvement. 

**Abstract (ZH)**: 大型语言模型（LLMs） increasingly deployed as代理 在动态现实环境中的代理任务，其中成功需要推理和有效的工具使用。代理任务的核心挑战是在增长的上下文长度下积累长时间序列的动作和观察。这种扩展在长期任务中增加了成本并降低了效率，而前期的工作主要集中在单步任务或窄应用上的上下文压缩。我们引入了代理上下文优化（ACON），这是一种统一框架，可以最优地压缩环境观察和交互历史，使其简洁但富有信息量。ACON利用自然语言空间的压缩指导原则优化：给定全上下文成功但压缩上下文失败的配对轨迹，有能力的LLM分析失败原因，并相应地更新压缩指导原则。此外，我们提出将优化的LLM压缩器提炼为更小的模型以减少附加模块的开销。在AppWorld、OfficeBench和多目标QA上的实验表明，ACON在内存使用上减少26-54%（峰值标记数量）的同时，几乎可以保持任务性能，提炼到更小的压缩器后保持超过95%的准确性，并且可以提升更小的LLM作为长期代理，性能提高高达46%。 

---
# Toward Safer Diffusion Language Models: Discovery and Mitigation of Priming Vulnerability 

**Title (ZH)**: 朝向更安全的扩散语言模型：发现和缓解提示漏洞 

**Authors**: Shojiro Yamabe, Jun Sakuma  

**Link**: [PDF](https://arxiv.org/pdf/2510.00565)  

**Abstract**: Diffusion language models (DLMs) generate tokens in parallel through iterative denoising, which can reduce latency and enable bidirectional conditioning. However, the safety risks posed by jailbreak attacks that exploit this inference mechanism are not well understood. In this paper, we reveal that DLMs have a critical vulnerability stemming from their iterative denoising process and propose a countermeasure. Specifically, our investigation shows that if an affirmative token for a harmful query appears at an intermediate step, subsequent denoising can be steered toward a harmful response even in aligned models. As a result, simply injecting such affirmative tokens can readily bypass the safety guardrails. Furthermore, we demonstrate that the vulnerability allows existing optimization-based jailbreak attacks to succeed on DLMs. Building on this analysis, we propose a novel safety alignment method tailored to DLMs that trains models to generate safe responses from contaminated intermediate states that contain affirmative tokens. Our experiments indicate that the proposed method significantly mitigates the vulnerability with minimal impact on task performance. Furthermore, our method improves robustness against conventional jailbreak attacks. Our work underscores the need for DLM-specific safety research. 

**Abstract (ZH)**: 基于迭代去噪的语言扩散模型存在关键脆弱性及其对策研究 

---
# Data Quality Challenges in Retrieval-Augmented Generation 

**Title (ZH)**: 检索增强生成中的数据质量挑战 

**Authors**: Leopold Müller, Joshua Holstein, Sarah Bause, Gerhard Satzger, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2510.00552)  

**Abstract**: Organizations increasingly adopt Retrieval-Augmented Generation (RAG) to enhance Large Language Models with enterprise-specific knowledge. However, current data quality (DQ) frameworks have been primarily developed for static datasets, and only inadequately address the dynamic, multi-stage nature of RAG systems. This study aims to develop DQ dimensions for this new type of AI-based systems. We conduct 16 semi-structured interviews with practitioners of leading IT service companies. Through a qualitative content analysis, we inductively derive 15 distinct DQ dimensions across the four processing stages of RAG systems: data extraction, data transformation, prompt & search, and generation. Our findings reveal that (1) new dimensions have to be added to traditional DQ frameworks to also cover RAG contexts; (2) these new dimensions are concentrated in early RAG steps, suggesting the need for front-loaded quality management strategies, and (3) DQ issues transform and propagate through the RAG pipeline, necessitating a dynamic, step-aware approach to quality management. 

**Abstract (ZH)**: 组织越来越多地采用检索增强生成（RAG）来增强大型语言模型的企业特定知识。然而，当前的数据质量（DQ）框架主要为静态数据集开发，仅不充分地解决了RAG系统的动态、多阶段性质。本研究旨在为这种新型的基于AI的系统开发DQ维度。我们对主要IT服务公司的从业人员进行了16次半结构化访谈，并通过定性内容分析，归纳出15个跨RAG系统四个处理阶段的数据质量维度：数据提取、数据转换、提示与搜索以及生成。我们的发现表明：（1）需要向传统的DQ框架添加新的维度以涵盖RAG背景；（2）这些新维度集中在RAG的早期步骤，表明需要前置的数据质量管理策略；（3）DQ问题在整个RAG管道中变换和传播，需要采用动态的、阶段意识的数据质量管理方法。 

---
# VIRTUE: Visual-Interactive Text-Image Universal Embedder 

**Title (ZH)**: 视觉互动文本图像统一嵌入模型：VIRTUE 

**Authors**: Wei-Yao Wang, Kazuya Tateishi, Qiyu Wu, Shusuke Takahashi, Yuki Mitsufuji  

**Link**: [PDF](https://arxiv.org/pdf/2510.00523)  

**Abstract**: Multimodal representation learning models have demonstrated successful operation across complex tasks, and the integration of vision-language models (VLMs) has further enabled embedding models with instruction-following capabilities. However, existing embedding models lack visual-interactive capabilities to specify regions of interest from users (e.g., point, bounding box, mask), which have been explored in generative models to broaden their human-interactive applicability. Equipping embedding models with visual interactions not only would unlock new applications with localized grounding of user intent, which remains unexplored, but also enable the models to learn entity-level information within images to complement their global representations for conventional embedding tasks. In this paper, we propose a novel Visual-InteRactive Text-Image Universal Embedder (VIRTUE) that extends the capabilities of the segmentation model and the vision-language model to the realm of representation learning. In VIRTUE, the segmentation model can process visual prompts that pinpoint specific regions within an image, thereby enabling the embedder to handle complex and ambiguous scenarios more precisely. To evaluate the visual-interaction ability of VIRTUE, we introduce a large-scale Segmentation-and-Scene Caption Retrieval (SCaR) benchmark comprising 1M samples that aims to retrieve the text caption by jointly considering the entity with a specific object and image scene. VIRTUE consistently achieves a state-of-the-art performance with significant improvements across 36 universal MMEB (3.1%-8.5%) and five visual-interactive SCaR (15.2%-20.3%) tasks. 

**Abstract (ZH)**: 具有视觉互动能力的文本-图像通用嵌入器（VIRTUE） 

---
# Rethinking Reward Models for Multi-Domain Test-Time Scaling 

**Title (ZH)**: 重新思考多域测试时奖励模型的扩展方法 

**Authors**: Dong Bok Lee, Seanie Lee, Sangwoo Park, Minki Kang, Jinheon Baek, Dongki Kim, Dominik Wagner, Jiongdao Jin, Heejun Lee, Tobias Bocklet, Jinyu Wang, Jingjing Fu, Sung Ju Hwang, Jiang Bia, Lei Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.00492)  

**Abstract**: The reliability of large language models (LLMs) during test-time scaling is often assessed with \emph{external verifiers} or \emph{reward models} that distinguish correct reasoning from flawed logic. Prior work generally assumes that process reward models (PRMs), which score every intermediate reasoning step, outperform outcome reward models (ORMs) that assess only the final answer. This view is based mainly on evidence from narrow, math-adjacent domains. We present the first unified evaluation of four reward model variants, discriminative ORM and PRM (\DisORM, \DisPRM) and generative ORM and PRM (\GenORM, \GenPRM), across 14 diverse domains. Contrary to conventional wisdom, we find that (i) \DisORM performs on par with \DisPRM, (ii) \GenPRM is not competitive, and (iii) overall, \GenORM is the most robust, yielding significant and consistent gains across every tested domain. We attribute this to PRM-style stepwise scoring, which inherits label noise from LLM auto-labeling and has difficulty evaluating long reasoning trajectories, including those involving self-correcting reasoning. Our theoretical analysis shows that step-wise aggregation compounds errors as reasoning length grows, and our empirical observations confirm this effect. These findings challenge the prevailing assumption that fine-grained supervision is always better and support generative outcome verification for multi-domain deployment. We publicly release our code, datasets, and checkpoints at \href{this https URL}{\underline{\small\texttt{this https URL}}} to facilitate future research in multi-domain settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）在测试时缩放过程中的可靠性通常通过外部验证器或奖励模型进行评估，这些模型能够区分正确的推理和有缺陷的逻辑。先前的工作通常假设过程奖励模型（PRMs），它可以对每个中间推理步骤进行评分，优于仅评估最终答案的结果奖励模型（ORMs）。这种观点主要基于狭窄、数学邻近领域的证据。我们首次在14个不同领域中统一评估了四种奖励模型变体：区分性ORM和PRM（\DisORM、\DisPRM）和生成性ORM和PRM（\GenORM、\GenPRM）。与传统的观点相反，我们发现（i）\DisORM与\DisPRM相当，（ii）\GenPRM缺乏竞争力，（iii）总体而言，\GenORM最为稳健，能够在每个测试领域中实现显著且一致的改进。我们认为这归因于PRM风格的分步评分方式，这种评分方式继承了LLM自动标签化的标签噪声，并且在评估长推理轨迹（包括自我纠正推理）时存在困难。我们的理论分析表明，随着推理长度的增加，分步聚合错误会累积，而我们的实证观察结果也证实了这一效果。这些发现挑战了精细监督总是更好的假设，并支持生成性结果验证在多领域部署中的应用。我们已在\href{this https URL}{\underline{\small\texttt{this https URL}}} 公开发布我们的代码、数据集和检查点，以促进在多领域设置中的未来研究。 

---
# Expandable Decision-Making States for Multi-Agent Deep Reinforcement Learning in Soccer Tactical Analysis 

**Title (ZH)**: 扩展决策状态的多Agent深度强化学习在足球战术分析中的应用 

**Authors**: Kenjiro Ide, Taiga Someya, Kohei Kawaguchi, Keisuke Fujii  

**Link**: [PDF](https://arxiv.org/pdf/2510.00480)  

**Abstract**: Invasion team sports such as soccer produce a high-dimensional, strongly coupled state space as many players continuously interact on a shared field, challenging quantitative tactical analysis. Traditional rule-based analyses are intuitive, while modern predictive machine learning models often perform pattern-matching without explicit agent representations. The problem we address is how to build player-level agent models from data, whose learned values and policies are both tactically interpretable and robust across heterogeneous data sources. Here, we propose Expandable Decision-Making States (EDMS), a semantically enriched state representation that augments raw positions and velocities with relational variables (e.g., scoring of space, pass, and score), combined with an action-masking scheme that gives on-ball and off-ball agents distinct decision sets. Compared to prior work, EDMS maps learned value functions and action policies to human-interpretable tactical concepts (e.g., marking pressure, passing lanes, ball accessibility) instead of raw coordinate features, and aligns agent choices with the rules of play. In the experiments, EDMS with action masking consistently reduced both action-prediction loss and temporal-difference (TD) error compared to the baseline. Qualitative case studies and Q-value visualizations further indicate that EDMS highlights high-risk, high-reward tactical patterns (e.g., fast counterattacks and defensive breakthroughs). We also integrated our approach into an open-source library and demonstrated compatibility with multiple commercial and open datasets, enabling cross-provider evaluation and reproducible experiments. 

**Abstract (ZH)**: 可扩展决策状态表示（EDMS）在足球等侵入性团队运动中的战术分析应用 

---
# Automated Evaluation can Distinguish the Good and Bad AI Responses to Patient Questions about Hospitalization 

**Title (ZH)**: 自动评估可以区分AI对患者关于住院问题的回答好坏 

**Authors**: Sarvesh Soni, Dina Demner-Fushman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00436)  

**Abstract**: Automated approaches to answer patient-posed health questions are rising, but selecting among systems requires reliable evaluation. The current gold standard for evaluating the free-text artificial intelligence (AI) responses--human expert review--is labor-intensive and slow, limiting scalability. Automated metrics are promising yet variably aligned with human judgments and often context-dependent. To address the feasibility of automating the evaluation of AI responses to hospitalization-related questions posed by patients, we conducted a large systematic study of evaluation approaches. Across 100 patient cases, we collected responses from 28 AI systems (2800 total) and assessed them along three dimensions: whether a system response (1) answers the question, (2) appropriately uses clinical note evidence, and (3) uses general medical knowledge. Using clinician-authored reference answers to anchor metrics, automated rankings closely matched expert ratings. Our findings suggest that carefully designed automated evaluation can scale comparative assessment of AI systems and support patient-clinician communication. 

**Abstract (ZH)**: 自动回答患者提出健康问题的方法正在兴起，但在多种系统中进行选择需要可靠的评估。目前评估自由文本人工智能（AI）响应的标准——人类专家审查——耗费人力且速度慢，限制了其可扩展性。自动化指标前景广阔但与人类判断的契合度不一，且常依赖于具体情境。为解决自动化评估医院化相关患者提问的AI响应可行性问题，我们进行了大规模的评估方法系统研究。在100个患者案例中，我们收集了28个AI系统（总计2800个响应）的回复，并从三个维度对其进行评估：（1）系统响应是否回答了问题，（2）是否恰当地使用了临床笔记证据，以及（3）是否运用了通用医学知识。利用临床专家撰写的参考答案作为指标的基础，自动排名与专家评分高度一致。我们的发现表明，精心设计的自动化评估可以实现AI系统的比较评估，并支持患者与临床医生之间的沟通。 

---
# Towards Self-Evolving Benchmarks: Synthesizing Agent Trajectories via Test-Time Exploration under Validate-by-Reproduce Paradigm 

**Title (ZH)**: 基于验证再现 paradigm 下的代理轨迹合成：迈向自主进化的基准测试 

**Authors**: Dadi Guo, Tianyi Zhou, Dongrui Liu, Chen Qian, Qihan Ren, Shuai Shao, Zhiyuan Fan, Yi R. Fung, Kun Wang, Linfeng Zhang, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00415)  

**Abstract**: Recent advances in large language models (LLMs) and agent system designs have empowered agents with unprecedented levels of capability. However, existing agent benchmarks are showing a trend of rapid ceiling-hitting by newly developed agents, making it difficult to meet the demands for evaluating agent abilities. To address this problem, we propose the Trajectory-based Validated-by-Reproducing Agent-benchmark Complexity Evolution (TRACE) framework. This framework takes an original task from an existing benchmark and encourages agents to freely explore and evolve it into a new task with higher difficulty while recording validatable agent trajectories. The framework proceeds in three stages: (1) evolutionary proposal mining, which provides task evolution proposals through preliminary exploration and divergent thinking; (2) problem formation and free exploration, where proposals are conceptualized into feasible problem candidates and the agents then explore them freely while recording their execution trajectories; and (3) multi-level validation, which ensures that the evolved tasks are accompanied by validatable and reproducible trajectories. Experiments on the GAIA benchmark demonstrate that the TRACE framework consistently enhances task complexity while improving the reliability of correctness through validatable execution trajectories. This work marks a paradigm shift from static, manually curated benchmarks to dynamic, self-evolving evaluation systems, providing a sustainable and challenging runway for agent development. 

**Abstract (ZH)**: 基于轨迹验证的代理评估复杂性演变框架（TRACE） 

---
# Semantic-Driven AI Agent Communications: Challenges and Solutions 

**Title (ZH)**: 基于语义的AI代理通信：挑战与解决方案 

**Authors**: Kaiwen Yu, Mengying Sun, Zhijin Qin, Xiaodong Xu, Ping Yang, Yue Xiao, Gang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00381)  

**Abstract**: With the rapid growth of intelligent services, communication targets are shifting from humans to artificial intelligent (AI) agents, which require new paradigms to enable real-time perception, decision-making, and collaboration. Semantic communication, which conveys task-relevant meaning rather than raw data, offers a promising solution. However, its practical deployment remains constrained by dynamic environments and limited resources. To address these issues, this article proposes a semantic-driven AI agent communication framework and develops three enabling techniques. First, semantic adaptation transmission applies fine-tuning with real or generative samples to efficiently adapt models to varying environments. Second, semantic lightweight transmission incorporates pruning, quantization, and perception-aware sampling to reduce model complexity and alleviate computational burden on edge agents. Third, semantic self-evolution control employs distributed hierarchical decision-making to optimize multi-dimensional resources, enabling robust multi-agent collaboration in dynamic environments. Simulation results show that the proposed solutions achieve faster convergence and stronger robustness, while the proposed distributed hierarchical optimization method significantly outperforms conventional decision-making schemes, highlighting its potential for AI agent communication networks. 

**Abstract (ZH)**: 智能服务快速发展背景下，通信目标从人类转向人工智能（AI）代理，这需要新的 paradigms 以支持实时感知、决策和协作。基于语义的通信通过传达任务相关的意义而非原始数据，提供了有希望的解决方案。然而，其实用部署仍受到动态环境和有限资源的限制。为解决这些问题，本文提出了一种基于语义的 AI 代理通信框架，并开发了三种使能技术。首先，语义自适应传输通过使用真实或生成样本进行微调，以高效适应不同环境。其次，语义轻量级传输结合剪枝、量化和感知自适应采样来降低模型复杂度并缓解边缘代理的计算负担。第三，语义自我进化控制采用分布式分层决策制定以优化多维资源，在动态环境中实现稳健的多代理协作。仿真结果表明，所提出的方法实现了更快的收敛和更强的鲁棒性，而提出的分布式分层优化方法显著优于传统决策方案，突显了其在 AI 代理通信网络中的潜在优势。 

---
# Hierarchical Reasoning Model: A Critical Supplementary Material 

**Title (ZH)**: 层次推理模型：关键补充材料 

**Authors**: Renee Ge, Qianli Liao, Tomaso Poggio  

**Link**: [PDF](https://arxiv.org/pdf/2510.00355)  

**Abstract**: Transformers have demonstrated remarkable performance in natural language processing and related domains, as they largely focus on sequential, autoregressive next-token prediction tasks. Yet, they struggle in logical reasoning, not necessarily because of a fundamental limitation of these models, but possibly due to the lack of exploration of more creative uses, such as latent space and recurrent reasoning. An emerging exploration in this direction is the Hierarchical Reasoning Model (Wang et al., 2025), which introduces a novel type of recurrent reasoning in the latent space of transformers, achieving remarkable performance on a wide range of 2D reasoning tasks. Despite the promising results, this line of models is still at an early stage and calls for in-depth investigation. In this work, we perform a critical review on this class of models, examine key design choices and present intriguing variants that achieve significantly better performance on the Sudoku-Extreme and Maze-Hard tasks than previously reported. Our results also raise surprising observations and intriguing directions for further research. 

**Abstract (ZH)**: 变换器在自然语言处理及相关领域展示了卓越性能，主要是因为它们主要关注序贯的自回归下一个词预测任务。然而，在逻辑推理方面，它们面临挑战，这并不是因为这些模型本身存在根本限制，而是可能由于缺乏更具创意的应用探索，如潜在空间和递归推理。这一方向上的一项新兴探索是层次推理模型（Wang et al., 2025），它在变换器的潜在空间中引入了一种新颖的递归推理方式，实现了广泛2D推理任务的卓越性能。尽管取得了一定的成果，但这类模型仍处于早期阶段，需要深入研究。在这项工作中，我们对这类模型进行了关键性回顾，检查了关键设计选择，并展示了在数独极难和迷宫极难任务上显著优于先前报告的有趣变体。我们的结果还提出了进一步研究中令人惊讶的发现和有趣的方向。 

---
# When Hallucination Costs Millions: Benchmarking AI Agents in High-Stakes Adversarial Financial Markets 

**Title (ZH)**: 当幻觉成本以百万计： adversarial 财经市场中 AI 代理的基准测试 

**Authors**: Zeshi Dai, Zimo Peng, Zerui Cheng, Ryan Yihe Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00332)  

**Abstract**: We present CAIA, a benchmark exposing a critical blind spot in AI evaluation: the inability of state-of-the-art models to operate in adversarial, high-stakes environments where misinformation is weaponized and errors are irreversible. While existing benchmarks measure task completion in controlled settings, real-world deployment demands resilience against active deception. Using crypto markets as a testbed where $30 billion was lost to exploits in 2024, we evaluate 17 models on 178 time-anchored tasks requiring agents to distinguish truth from manipulation, navigate fragmented information landscapes, and make irreversible financial decisions under adversarial pressure.
Our results reveal a fundamental capability gap: without tools, even frontier models achieve only 28% accuracy on tasks junior analysts routinely handle. Tool augmentation improves performance but plateaus at 67.4% versus 80% human baseline, despite unlimited access to professional resources. Most critically, we uncover a systematic tool selection catastrophe: models preferentially choose unreliable web search over authoritative data, falling for SEO-optimized misinformation and social media manipulation. This behavior persists even when correct answers are directly accessible through specialized tools, suggesting foundational limitations rather than knowledge gaps. We also find that Pass@k metrics mask dangerous trial-and-error behavior for autonomous deployment.
The implications extend beyond crypto to any domain with active adversaries, e.g. cybersecurity, content moderation, etc. We release CAIA with contamination controls and continuous updates, establishing adversarial robustness as a necessary condition for trustworthy AI autonomy. The benchmark reveals that current models, despite impressive reasoning scores, remain fundamentally unprepared for environments where intelligence must survive active opposition. 

**Abstract (ZH)**: CAIA：揭示AI评估关键盲点的基准测试 

---
# BiasBusters: Uncovering and Mitigating Tool Selection Bias in Large Language Models 

**Title (ZH)**: BiasBusters: 揭示并缓解大型语言模型工具选择偏见 

**Authors**: Thierry Blankenstein, Jialin Yu, Zixuan Li, Vassilis Plachouras, Sunando Sengupta, Philip Torr, Yarin Gal, Alasdair Paren, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00307)  

**Abstract**: Agents backed by large language models (LLMs) often rely on external tools drawn from marketplaces where multiple providers offer functionally equivalent options. This raises a critical point concerning fairness: if selection is systematically biased, it can degrade user experience and distort competition by privileging some providers over others. We introduce a benchmark of diverse tool categories, each containing multiple functionally equivalent tools, to evaluate tool-selection bias. Using this benchmark, we test seven models and show that unfairness exists with models either fixating on a single provider or disproportionately preferring earlier-listed tools in context. To investigate the origins of this bias, we conduct controlled experiments examining tool features, metadata (name, description, parameters), and pre-training exposure. We find that: (1) semantic alignment between queries and metadata is the strongest predictor of choice; (2) perturbing descriptions significantly shifts selections; and (3) repeated pre-training exposure to a single endpoint amplifies bias. Finally, we propose a lightweight mitigation that first filters the candidate tools to a relevant subset and then samples uniformly, reducing bias while preserving good task coverage. Our findings highlight tool-selection bias as a key obstacle for the fair deployment of tool-augmented LLMs. 

**Abstract (ZH)**: 大型语言模型支持的智能代理往往依赖于来自包含多个提供商可替代选项的市场交易平台的外部工具。这引起了一个关键的公平性问题：如果选择过程存在系统性的偏差，可能会损害用户体验并扭曲竞争，通过偏好某些提供商。我们引入了一个多样化的工具类别基准，每个类别包含多个功能等效工具，以评估工具选择偏见。利用此基准，我们测试了七种模型，并发现不公平性存在于模型过度依赖单一提供商或过度偏好上下文中列出的早期工具的情况中。为了探究这种偏见的来源，我们进行了控制实验，检查工具特征、元数据（名称、描述、参数）以及预训练暴露。我们发现：（1）查询与元数据之间的语义对齐是选择决策最强的预测因素；（2）扰动描述显著改变了选择；（3）重复对单个端点的预训练暴露加剧了偏见。最后，我们提出了一种轻量级的解决方案，首先筛选候选工具到相关子集，然后均匀抽样，从而减少偏见同时保持良好的任务覆盖。我们的研究结果强调工具选择偏见是公平部署工具增强的大语言模型的关键障碍。 

---
# ICL Optimized Fragility 

**Title (ZH)**: ICL优化脆弱性 

**Authors**: Serena Gomez Wannaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00300)  

**Abstract**: ICL guides are known to improve task-specific performance, but their impact on cross-domain cognitive abilities remains unexplored. This study examines how ICL guides affect reasoning across different knowledge domains using six variants of the GPT-OSS:20b model: one baseline model and five ICL configurations (simple, chain-of-thought, random, appended text, and symbolic language). The models were subjected to 840 tests spanning general knowledge questions, logic riddles, and a mathematical olympiad problem. Statistical analysis (ANOVA) revealed significant behavioral modifications (p less than 0.001) across ICL variants, demonstrating a phenomenon termed "optimized fragility." ICL models achieved 91%-99% accuracy on general knowledge tasks while showing degraded performance on complex reasoning problems, with accuracy dropping to 10-43% on riddles compared to 43% for the baseline model. Notably, no significant differences emerged on the olympiad problem (p=0.2173), suggesting that complex mathematical reasoning remains unaffected by ICL optimization. These findings indicate that ICL guides create systematic trade-offs between efficiency and reasoning flexibility, with important implications for LLM deployment and AI safety. 

**Abstract (ZH)**: ICL引导对跨领域认知能力的影响尚未探究：基于六种GPT-OSS:20b模型变体的推理研究 

---
# MAGIC-MASK: Multi-Agent Guided Inter-Agent Collaboration with Mask-Based Explainability for Reinforcement Learning 

**Title (ZH)**: MAGIC-MASK：基于掩码解释的多Agent引导跨Agent协作强化学习 

**Authors**: Maisha Maliha, Dean Hougen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00274)  

**Abstract**: Understanding the decision-making process of Deep Reinforcement Learning agents remains a key challenge for deploying these systems in safety-critical and multi-agent environments. While prior explainability methods like StateMask, have advanced the identification of critical states, they remain limited by computational cost, exploration coverage, and lack of adaptation to multi-agent settings. To overcome these limitations, we propose a mathematically grounded framework, MAGIC-MASK (Multi-Agent Guided Inter-agent Collaboration with Mask-Based Explainability for Reinforcement Learning), that extends perturbation-based explanation to Multi-Agent Reinforcement Learning. Our method integrates Proximal Policy Optimization, adaptive epsilon-greedy exploration, and lightweight inter-agent collaboration to share masked state information and peer experience. This collaboration enables each agent to perform saliency-guided masking and share reward-based insights with peers, reducing the time required for critical state discovery, improving explanation fidelity, and leading to faster and more robust learning. The core novelty of our approach lies in generalizing explainability from single-agent to multi-agent systems through a unified mathematical formalism built on trajectory perturbation, reward fidelity analysis, and Kullback-Leibler divergence regularization. This framework yields localized, interpretable explanations grounded in probabilistic modeling and multi-agent Markov decision processes. We validate our framework on both single-agent and multi-agent benchmarks, including a multi-agent highway driving environment and Google Research Football, demonstrating that MAGIC-MASK consistently outperforms state-of-the-art baselines in fidelity, learning efficiency, and policy robustness while offering interpretable and transferable explanations. 

**Abstract (ZH)**: 理解深度强化学习代理的决策过程仍然是将这些系统应用于关键安全和多代理环境中的一个关键挑战。为了克服先前方法如StateMask所面临的计算成本、探索覆盖范围以及不适应多代理设置的局限，我们提出了一种基于数学原理的框架MAGIC-MASK（多代理引导的跨代理协作及基于掩码的强化学习解释方法），该框架将扰动解释扩展到了多代理强化学习中。该方法结合了接近策略优化、自适应ε-贪婪探索以及轻量级的代理间协作，以共享掩码状态信息和同伴体验。这种协作使每个代理能够进行显著性指导的掩码，并与同伴分享基于奖励的见解，从而减少关键状态的发现时间，提高解释的准确度，并加快和增强学习过程。我们的方法的核心创新之处在于通过基于轨迹扰动、奖励保真分析和Kullback-Leibler散度正则化的统一数学形式，将解释性从单代理系统推广到了多代理系统。该框架提供了基于概率建模和多代理马尔可夫决策过程的局部可解释性解释。我们在单代理和多代理基准测试上验证了该框架，包括多代理高速公路驾驶环境和Google Research Football，结果显示MAGIC-MASK在准确度、学习效率和策略稳健性方面均优于现有最先进的基线方法，同时提供了可解释性和可传递性解释。 

---
# DualTune: Decoupled Fine-Tuning for On-Device Agentic Systems 

**Title (ZH)**: DualTune: 解耦细调用于设备端智能代理系统 

**Authors**: Rohan Kadekodi, Zhan Jin, Keisuke Kamahori, Yile Gu, Sean Khatiri, Noah H. Bayindirli, Sergey Gorbunov, Baris Kasikci  

**Link**: [PDF](https://arxiv.org/pdf/2510.00229)  

**Abstract**: The deployment of Large Language Models (LLMs) as agentic orchestrators has revolutionized task automation, but the need for privacy-preserving, cost-effective solutions demands on-device inference capabilities. However, local LLMs consistently underperform compared to frontier models in tool calling scenarios, struggling with both tool selection from large tool sets and accurate argument generation for complex parameter structures. We introduce a methodology that disaggregates a tool-calling task into two distinct subtasks: tool selection and argument generation. We propose "decoupled fine-tuning", a novel post-training approach that employs LoRA fine-tuning to create dedicated LoRA adapters for tool selection and tool-specific argument generation using separate loss masking for each of the subtasks. Furthermore, we present DualTune, an inference framework that leverages the LoRA adapters created using decoupled fine-tuning to perform efficient agent orchestration with the help of local models on end-user devices. DualTune decomposes the tool-call generation step into tool selection and argument generation, and dynamically loads the corresponding LoRA adapters to generate tool calls. Additionally, DualTune implements hierarchical orchestration to restrict the number of tools required for tool selection. Our experiments on the MCP-Bench benchmark demonstrate that the Qwen-2.5-7B model trained using decoupled fine-tuning improves the tool calling accuracy of the base model by 46%, and outperforms other local reasoning, non-reasoning and fine-tuned models of similar size in all cases, and models that are 2x larger, in most cases. 

**Abstract (ZH)**: 基于分解微调的方法实现工具调用任务的局部推理能力改进：DualTune方法 

---
# Thinkquel: A Model Dedicated to Text-to-dbt Using Synthetic Data and a Span-Aware Objective 

**Title (ZH)**: Thinkquel：一个专用的文本到对话模型，采用合成数据和区间感知目标 

**Authors**: Anni Li, Aria Attar, Paul Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00186)  

**Abstract**: Transforming natural-language requests into reliable, production-ready data transformations remains challenging: correctness depends on precise schema linking and warehouse-specific SQL dialects, while the strongest supervision available during training--execution success and result matching--are provided only at the sequence level. At the same time, assembling large, execution-validated corpora is costly, and token-level objectives misalign with these global signals, yielding unstable optimization and limited portability. We introduce Thinkquel, a fine-tuned model for producing robust, portable, and execution-validated database queries. Methodologies in Thinkquel integrates a novel synthetic data pipeline, TS-SQL, that leverages dbt as a portable intermediate representation with a span-aware reinforcement learning objective, and Token-Sequence GRPO (TS-GRPO), specifically designed to bridge the gap between token-level training signals and sequence-level execution rewards when finetuning LLMs. On the 500-example TS-SQL test set, Thinkquel (32B) reaches 93.2\% execution success and 61.8\% exact-result match with a two-stage SFT curriculum, improving over the base model by 67.2\% (exec.) and 44.4\% (match). In Spider (14B) experiments, TS-GRPO increases training stability and speeds convergence of the execution-match reward relative to GRPO and GSPO. 

**Abstract (ZH)**: 将自然语言请求转化为可靠且生产-ready的数据转换仍然具有挑战性：正确性依赖于精确的模式链接和特定于数据仓库的SQL方言，而可用的最强监督——执行成功和结果匹配——仅提供在序列级别。同时，构建大型、执行验证的语料库成本高昂，而基于token级别的目标与这些全局信号不一致，导致优化不稳定且适用性有限。我们引入Thinkquel，一种细调模型，用于生成稳健、可移植且执行验证的数据库查询。Thinkquel中的方法结合了新型合成数据管道TS-SQL，利用dbt作为可移植的中间表示，并采用基于跨度感知的强化学习目标，以及专门设计的Token-Sequence GRPO（TS-GRPO），用于在微调LLM时弥合基于token级别的训练信号和基于序列级别的执行奖励之间的差距。在500个示例TS-SQL测试集上，Thinkquel（32B）通过两阶段SFT课程达到93.2%的执行成功率和61.8%的确切结果匹配率，分别比基础模型提高了67.2%（执行）和44.4%（匹配）。在Spider（14B）实验中，TS-GRPO相对于GRPO和GSPO提高了训练的稳定性并加快了执行匹配奖励的收敛速度。 

---
# Object-Centric Case-Based Reasoning via Argumentation 

**Title (ZH)**: 基于论据的对象中心案例推理 

**Authors**: Gabriel de Olim Gaul, Adam Gould, Avinash Kori, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2510.00185)  

**Abstract**: We introduce Slot Attention Argumentation for Case-Based Reasoning (SAA-CBR), a novel neuro-symbolic pipeline for image classification that integrates object-centric learning via a neural Slot Attention (SA) component with symbolic reasoning conducted by Abstract Argumentation for Case-Based Reasoning (AA-CBR). We explore novel integrations of AA-CBR with the neural component, including feature combination strategies, casebase reduction via representative samples, novel count-based partial orders, a One-Vs-Rest strategy for extending AA-CBR to multi-class classification, and an application of Supported AA-CBR, a bipolar variant of AA-CBR. We demonstrate that SAA-CBR is an effective classifier on the CLEVR-Hans datasets, showing competitive performance against baseline models. 

**Abstract (ZH)**: Slot 注意论辩案例基推理 (SAA-CBR) 用于图像分类的新型神经符号管道 

---
# Drones that Think on their Feet: Sudden Landing Decisions with Embodied AI 

**Title (ZH)**: 能够在瞬间做出着陆决定的无人机：基于身体化AI的突发着陆决策 

**Authors**: Diego Ortiz Barbosa, Mohit Agrawal, Yash Malegaonkar, Luis Burbano, Axel Andersson, György Dán, Henrik Sandberg, Alvaro A. Cardenas  

**Link**: [PDF](https://arxiv.org/pdf/2510.00167)  

**Abstract**: Autonomous drones must often respond to sudden events, such as alarms, faults, or unexpected changes in their environment, that require immediate and adaptive decision-making. Traditional approaches rely on safety engineers hand-coding large sets of recovery rules, but this strategy cannot anticipate the vast range of real-world contingencies and quickly becomes incomplete. Recent advances in embodied AI, powered by large visual language models, provide commonsense reasoning to assess context and generate appropriate actions in real time. We demonstrate this capability in a simulated urban benchmark in the Unreal Engine, where drones dynamically interpret their surroundings and decide on sudden maneuvers for safe landings. Our results show that embodied AI makes possible a new class of adaptive recovery and decision-making pipelines that were previously infeasible to design by hand, advancing resilience and safety in autonomous aerial systems. 

**Abstract (ZH)**: 自主无人机必须经常应对警报、故障或环境中的意外变化等突然事件，这些事件要求即时且适应性的决策。传统方法依靠安全工程师手工编码大量恢复规则，但这种方法无法预见各种实际 contingency 并且很快就会变得不完整。近期以大规模视觉语言模型为动力的嵌入式人工智能进展提供了常识推理能力，能够在实时环境中评估情境并生成适当的动作。我们在 Unreal Engine 中的一个模拟城市基准测试中展示了这种能力，无人机动态解读其周围环境并决定突然的机动以确保安全着陆。我们的结果表明，嵌入式人工智能使设计以往通过手工方式无法实现的新类别的自适应恢复和决策流水线成为可能，从而推动自主空中系统的鲁棒性和安全性发展。 

---
# AuditAgent: Expert-Guided Multi-Agent Reasoning for Cross-Document Fraudulent Evidence Discovery 

**Title (ZH)**: AuditAgent: 专家引导的多代理跨文档欺诈证据推理 

**Authors**: Songran Bai, Bingzhe Wu, Yiwei Zhang, Chengke Wu, Xiaolong Zheng, Yaze Yuan, Ke Wu, Jianqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00156)  

**Abstract**: Financial fraud detection in real-world scenarios presents significant challenges due to the subtlety and dispersion of evidence across complex, multi-year financial disclosures. In this work, we introduce a novel multi-agent reasoning framework AuditAgent, enhanced with auditing domain expertise, for fine-grained evidence chain localization in financial fraud cases. Leveraging an expert-annotated dataset constructed from enforcement documents and financial reports released by the China Securities Regulatory Commission, our approach integrates subject-level risk priors, a hybrid retrieval strategy, and specialized agent modules to efficiently identify and aggregate cross-report evidence. Extensive experiments demonstrate that our method substantially outperforms General-Purpose Agent paradigm in both recall and interpretability, establishing a new benchmark for automated, transparent financial forensics. Our results highlight the value of domain-specific reasoning and dataset construction for advancing robust financial fraud detection in practical, real-world regulatory applications. 

**Abstract (ZH)**: 现实世界场景中的财务欺诈检测由于证据在复杂多年的财务披露中具有细微性和分散性而面临重大挑战。本文介绍了一个增强审计专业知识的新型多智能体推理框架AuditAgent，用于财务欺诈案件中的细粒度证据链定位。通过利用由中国证券监督管理委员会发布的执法文件和财务报告构建的专家标注数据集，我们的方法结合了主题级风险先验、混合检索策略和专门的智能体模块，以高效地识别和聚合跨报告证据。大量实验表明，我们的方法在召回率和可解释性上显著优于通用智能体 paradigm，建立了自动化、透明财务 forensic 的新基准。我们的结果突显了在实际现实世界监管应用中推进稳健财务欺诈检测的价值，特别是在专业领域推理和数据集构建方面。 

---
# Judging by Appearances? Auditing and Intervening Vision-Language Models for Bail Prediction 

**Title (ZH)**: 凭表象判断？审计与干预视觉语言模型的保释预测 

**Authors**: Sagnik Basu, Shubham Prakash, Ashish Maruti Barge, Siddharth D Jaiswal, Abhisek Dash, Saptarshi Ghosh, Animesh Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00088)  

**Abstract**: Large language models (LLMs) have been extensively used for legal judgment prediction tasks based on case reports and crime history. However, with a surge in the availability of large vision language models (VLMs), legal judgment prediction systems can now be made to leverage the images of the criminals in addition to the textual case reports/crime history. Applications built in this way could lead to inadvertent consequences and be used with malicious intent. In this work, we run an audit to investigate the efficiency of standalone VLMs in the bail decision prediction task. We observe that the performance is poor across multiple intersectional groups and models \textit{wrongly deny bail to deserving individuals with very high confidence}. We design different intervention algorithms by first including legal precedents through a RAG pipeline and then fine-tuning the VLMs using innovative schemes. We demonstrate that these interventions substantially improve the performance of bail prediction. Our work paves the way for the design of smarter interventions on VLMs in the future, before they can be deployed for real-world legal judgment prediction. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经广泛用于基于案例报告和犯罪记录的法律判决预测任务。然而，随着大型视觉语言模型（VLMs）的可用性激增，现在可以将犯罪分子的图像和文本案例报告/犯罪记录结合起来用于法律判决预测系统。这样构建的应用可能会导致无意的后果，并可能被恶意使用。在此工作中，我们进行了一项审计，调查单个VLM在保释决定预测任务中的效率。我们观察到，在多个交叉群体中性能较差，并且模型错误地以极高置信度拒绝了有资格获得保释的个体。我们通过首先通过RAG管道纳入法律 precedents，然后使用创新方案微调VLMs，设计了不同的干预算法。我们证明了这些干预措施显著提高了保释预测的性能。我们的工作为未来针对VLMs设计更智能的干预措施奠定了基础，在它们被部署用于实际法律判决预测之前。 

---
# Towards a Framework for Supporting the Ethical and Regulatory Certification of AI Systems 

**Title (ZH)**: 面向支持人工智能系统伦理与监管认证的框架 

**Authors**: Fabian Kovac, Sebastian Neumaier, Timea Pahi, Torsten Priebe, Rafael Rodrigues, Dimitrios Christodoulou, Maxime Cordy, Sylvain Kubler, Ali Kordia, Georgios Pitsiladis, John Soldatos, Petros Zervoudakis  

**Link**: [PDF](https://arxiv.org/pdf/2510.00084)  

**Abstract**: Artificial Intelligence has rapidly become a cornerstone technology, significantly influencing Europe's societal and economic landscapes. However, the proliferation of AI also raises critical ethical, legal, and regulatory challenges. The CERTAIN (Certification for Ethical and Regulatory Transparency in Artificial Intelligence) project addresses these issues by developing a comprehensive framework that integrates regulatory compliance, ethical standards, and transparency into AI systems. In this position paper, we outline the methodological steps for building the core components of this framework. Specifically, we present: (i) semantic Machine Learning Operations (MLOps) for structured AI lifecycle management, (ii) ontology-driven data lineage tracking to ensure traceability and accountability, and (iii) regulatory operations (RegOps) workflows to operationalize compliance requirements. By implementing and validating its solutions across diverse pilots, CERTAIN aims to advance regulatory compliance and to promote responsible AI innovation aligned with European standards. 

**Abstract (ZH)**: 人工智能已成为核心技术，显著影响欧洲的社会和经济格局。然而，人工智能的广泛应用也引发了关键的伦理、法律和监管挑战。CERTAIN（伦理与监管透明度认证的人工智能）项目通过开发综合框架来应对这些问题，该框架将监管合规性、伦理标准和透明性整合到人工智能系统中。在本文中，我们概述了构建该框架核心组件的方法步骤。具体而言，我们介绍了：(i) 语义机器学习运营 (MLOps) 以实现结构化的人工智能生命周期管理，(ii) 基于本体的数据血缘追踪以确保可追溯性和问责制，以及(iii) 监管运营 (RegOps) 工作流以实现合规要求的操作化。通过在多样化的试点项目中实施和验证其解决方案，CERTAIN 目标在于推动符合欧洲标准的责任型人工智能创新。 

---
# NeurIPS should lead scientific consensus on AI policy 

**Title (ZH)**: NeURIPS 应引领AI政策的科学共识 

**Authors**: Rishi Bommasani  

**Link**: [PDF](https://arxiv.org/pdf/2510.00075)  

**Abstract**: Designing wise AI policy is a grand challenge for society. To design such policy, policymakers should place a premium on rigorous evidence and scientific consensus. While several mechanisms exist for evidence generation, and nascent mechanisms tackle evidence synthesis, we identify a complete void on consensus formation. In this position paper, we argue NeurIPS should actively catalyze scientific consensus on AI policy. Beyond identifying the current deficit in consensus formation mechanisms, we argue that NeurIPS is the best option due its strengths and the paucity of compelling alternatives. To make progress, we recommend initial pilots for NeurIPS by distilling lessons from the IPCC's leadership to build scientific consensus on climate policy. We dispel predictable counters that AI researchers disagree too much to achieve consensus and that policy engagement is not the business of NeurIPS. NeurIPS leads AI on many fronts, and it should champion scientific consensus to create higher quality AI policy. 

**Abstract (ZH)**: 设计明智的AI政策是社会面临的重大挑战。为了设计这样的政策，政策制定者应高度重视严谨的证据和科学共识。尽管存在多种证据生成机制，且初步机制致力于证据综合，但我们发现共识形成机制存在重大缺失。在这一立场论文中，我们主张NeurIPS应积极催化AI政策的科学共识。除了指出当前共识形成机制的不足，我们认为NeurIPS是最合适的选择，因其优势显著且缺乏更具说服力的替代方案。为了取得进展，我们建议NeurIPS通过借鉴IPCC领导力建设气候变化政策科学共识的经验，开展初步试点。我们驳斥了AI研究人员分歧过大难以达成共识以及政策参与不属于NeurIPS职责范围的预期反对意见。NeurIPS在AI领域引领多项进展，它应当倡导科学共识，以创造更高质量的AI政策。 

---
# ARS: Adaptive Reasoning Suppression for Efficient Large Reasoning Language Models 

**Title (ZH)**: ARS: 自适应推理抑制以提升大型推理语言模型的效率 

**Authors**: Dongqi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00071)  

**Abstract**: Large Reasoning Language Models (LRLMs or LRMs) demonstrate remarkable capabilities in complex reasoning tasks, but suffer from significant computational inefficiencies due to overthinking phenomena. Existing efficient reasoning methods face the challenge of balancing reasoning quality with inference cost reduction. We propose \textbf{Adaptive Reasoning Suppression (ARS)}, a novel training-free approach that dynamically suppresses redundant reasoning steps while preserving accuracy through adaptive certainty monitoring. ARS introduces a multi-checkpoint certainty estimation mechanism with progressive suppression thresholds, achieving superior efficiency compared to static suppression methods. Our extensive evaluation across mathematical reasoning benchmarks using multiple model architectures demonstrates that ARS achieves up to 53%, 46.1%, and 57.9% in token, latency and energy reduction, while maintaining or improving accuracy. 

**Abstract (ZH)**: 具有自适应推理抑制的大型推理语言模型（LRLMs或LRMs）在复杂推理任务中表现出色，但由于过度推理现象导致了显著的计算效率低下。现有的高效推理方法面临着在推理质量与推理成本降低之间取得平衡的挑战。我们提出了自适应推理抑制（ARS），这是一种无需训练的新颖方法，通过自适应的确定性监控动态抑制冗余推理步骤，同时保持准确性。ARS 引入了多检查点确定性估计机制，并采用逐步抑制阈值，其效率优于静态抑制方法。我们在多个模型架构下对数学推理基准测试的广泛评估表明，与静态抑制方法相比，ARS 在保持或提高准确性的基础上分别实现了高达 53%、46.1% 和 57.9% 的 tokens、延迟和能耗降低。 

---
# ToolBrain: A Flexible Reinforcement Learning Framework for Agentic Tools 

**Title (ZH)**: ToolBrain：一种灵活的智能体工具强化学习框架 

**Authors**: Quy Minh Le, Minh Sao Khue Luu, Khanh-Tung Tran, Duc-Hai Nguyen, Hoang-Quoc-Viet Pham, Quan Le, Hoang Thanh Lam, Hoang D. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00023)  

**Abstract**: Effective tool use is essential for agentic AI, yet training agents to utilize tools remains challenging due to manually designed rewards, limited training data, and poor multi-tool selection, resulting in slow adaptation, wasted computational resources, and suboptimal performance. We introduce ToolBrain, a lightweight and user-friendly framework for coaching tool use in agentic models with flexible reinforcement learning (RL), easing the barriers for researchers and practitioners to adapt LLM-based agents to specific domains. It supports a wide range of training strategies, including RL algorithms such as GRPO and DPO, as well as supervised learning. ToolBrain enables custom reward callables directly on an agent's execution traces or simply utilizes an automated LLM-as-a-judge system for reward generation. It is packed with useful capabilities, including knowledge distillation from large to small models for efficient development, automatic task generation from tool descriptions, seamless tool retrieval, efficient fine-tuning pipelines with QLoRA through Unsloth, and quantized inference via bitsandbytes. We demonstrate ToolBrain through diverse use cases, such as training a CodeAct agent to autonomously execute email search tasks, showing fast, targeted improvements (up to 30.0%) in tool-use skills while keeping the codebase simple and extensible in Agentic AI. Our framework is publicly available at this https URL. 

**Abstract (ZH)**: 有效工具使用对于自主人工智能至关重要，然而训练代理利用工具仍然由于手动设计的奖励、有限的训练数据和糟糕的多工具选择而充满挑战，导致适应速度慢、浪费计算资源和性能不佳。我们引入了ToolBrain——一个轻量级且用户友好的框架，用于灵活的强化学习（RL）辅助自主模型的工具使用训练，降低了研究人员和实践者将基于LLM的代理适应到特定领域的门槛。它支持广泛的训练策略，包括如GRPO和DPO等RL算法以及监督学习。ToolBrain允许在代理执行轨迹上直接自定义奖励函数，或简单地使用自动化LLM作为裁判系统进行奖励生成。它还集成了从大到小模型的知识蒸馏以提高开发效率、从工具描述自动生成任务、无缝工具检索、通过Unsloth和QLoRA进行高效细调管道，以及利用bitsandbytes进行量化推理。我们通过多种应用场景演示了ToolBrain，例如训练CodeAct代理自主执行电子邮件搜索任务，显示出工具使用技能快速、针对性改进（最多提高30.0%）的同时保持代码库简洁和可扩展性。我们的框架已在GitHub上公开：this https URL。 

---
# Learning to Lead Themselves: Agentic AI in MAS using MARL 

**Title (ZH)**: 自我驱动：使用多智能体强化学习的自主智能代理 

**Authors**: Ansh Kamthan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00022)  

**Abstract**: As autonomous systems move from prototypes to real deployments, the ability of multiple agents to make decentralized, cooperative decisions becomes a core requirement. This paper examines how agentic artificial intelligence, agents that act independently, adaptively and proactively can improve task allocation and coordination in multi-agent systems, with primary emphasis on drone delivery and secondary relevance to warehouse automation. We formulate the problem in a cooperative multi-agent reinforcement learning setting and implement a lightweight multi-agent Proximal Policy Optimization, called IPPO, approach in PyTorch under a centralized-training, decentralized-execution paradigm. Experiments are conducted in PettingZoo environment, where multiple homogeneous drones or agents must self-organize to cover distinct targets without explicit communication. 

**Abstract (ZH)**: 自主系统从原型走向实际部署时，多个代理进行去中心化、合作决策的能力成为关键要求。本文探讨了自主人工智能代理，即独立、适应性和主动地行动的代理，如何在多代理系统中改善任务分配和协调，主要集中在无人机交付领域，次级相关性在于仓库自动化。我们将问题形式化为合作多代理强化学习环境，并在集中训练、去中心化执行的框架下，使用轻量级多代理 proximal 策略优化方法（称作IPPO）进行实施。实验在PettingZoo环境中进行，多个同质无人机或代理需要自我组织覆盖不同的目标而无需显式通信。 

---
# TOUCAN: Synthesizing 1.5M Tool-Agentic Data from Real-World MCP Environments 

**Title (ZH)**: TOUCAN: 从实际 MCP 环境中合成 1.5M 工具-代理数据 

**Authors**: Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, Rameswar Panda  

**Link**: [PDF](https://arxiv.org/pdf/2510.01179)  

**Abstract**: Large Language Model (LLM) agents are rapidly emerging as powerful systems for automating tasks across domains. Yet progress in the open-source community is constrained by the lack of high quality permissively licensed tool-agentic training data. Existing datasets are often limited in diversity, realism, and complexity, particularly regarding multi-tool and multi-turn interactions. To address this gap, we introduce Toucan, the largest publicly available tool-agentic dataset to date, containing 1.5 million trajectories synthesized from nearly 500 real-world Model Context Protocols (MCPs). Unlike prior work, Toucan leverages authentic MCP environments to generate diverse, realistic, and challenging tasks with trajectories involving real tool execution. Our pipeline first produces a broad spectrum of tool-use queries using five distinct models, applies model-based quality filtering, and then generates agentic trajectories with three teacher models using two agentic frameworks. Rigorous rule-based and model-based validation ensures high-quality outputs. We also introduce three extension mechanisms to further diversify tasks and simulate multi-turn conversations. Models fine-tuned on Toucan outperform larger closed-source counterparts on the BFCL V3 benchmark and push the Pareto frontier forward on MCP-Universe Bench. 

**Abstract (ZH)**: 大型语言模型（LLM）代理正在迅速崛起为跨领域自动化任务的强大系统。然而，开源社区的进步受限于高质量许可开源的工具-代理训练数据的缺乏。现有数据集往往在多样性、逼真性和复杂性方面有限，尤其是在多工具和多轮交互方面。为解决这一问题，我们引入了Toucan，这是迄今为止最大的公开可用工具-代理数据集，包含近500个真实世界模型上下文协议（MCPs）合成的150万条轨迹。与以往工作不同，Toucan利用真实的MCP环境生成多样、真实且具有挑战性的任务轨迹，涉及实际工具执行。我们的流水线首先使用五种不同模型生成工具使用查询的广泛谱系，应用基于模型的质量过滤，然后使用两种代理框架生成具有三个教师模型的代理轨迹。严格的基于规则和基于模型的验证确保了高质量的输出。我们还引入了三种扩展机制进一步增加任务多样性并模拟多轮对话。在Toucan上 fine-tune 的模型在BFCL V3基准测试中优于大型封闭源代码对应模型，并且在MCP-Universe Bench上推动了帕累托前沿。 

---
# COM-BOM: Bayesian Exemplar Search for Efficiently Exploring the Accuracy-Calibration Pareto Frontier 

**Title (ZH)**: COM-BOM: 基于贝叶斯原型搜索的高效探索准确率校准帕多瓦前沿方法 

**Authors**: Gaoxiang Luo, Aryan Deshwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01178)  

**Abstract**: Selecting an optimal set of exemplars is critical for good performance of in-context learning. However, prior exemplar search methods narrowly optimize for predictive accuracy, critically neglecting model calibration--a key determinant of trustworthiness and safe deployment. In this paper, we formulate exemplar selection as a multi-objective optimization problem, explicitly targeting both the maximization of predictive accuracy and the minimization of expected calibration error. We solve this problem with a sample-efficient Combinatorial Bayesian Optimization algorithm (COM-BOM) to find the Pareto front that optimally trades off the two objectives of accuracy and calibration. We evaluate COM-BOM on multiple tasks from unsaturated MMLU-Pro benchmark and find that COM-BOM beats or matches the baselines at jointly optimizing the two objectives, while requiring a minimal number of LLM API calls. 

**Abstract (ZH)**: 选择最优示例集对于上下文学习的良好表现至关重要。然而，先前的示例搜索方法仅狭隘地优化预测准确性，严重忽视了模型校准——这是信任度和安全部署的关键决定因素。本文将示例选择形式化为一个多目标优化问题，明确针对预测准确性的最大化和期望校准误差的最小化。我们使用一种高效的组合贝叶斯优化算法（COM-BOM）来解决该问题，以找到在准确性和校准之间最优权衡的帕累托前沿。我们在未饱和MMLU-Pro基准上的多个任务上评估了COM-BOM，发现它能够在同时优化两个目标方面优于或匹配基线模型，同时仅需最少的LLM API调用。 

---
# Code2Video: A Code-centric Paradigm for Educational Video Generation 

**Title (ZH)**: 代码为中心的教育视频生成范式 

**Authors**: Yanzhe Chen, Kevin Qinghong Lin, Mike Zheng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2510.01174)  

**Abstract**: While recent generative models advance pixel-space video synthesis, they remain limited in producing professional educational videos, which demand disciplinary knowledge, precise visual structures, and coherent transitions, limiting their applicability in educational scenarios. Intuitively, such requirements are better addressed through the manipulation of a renderable environment, which can be explicitly controlled via logical commands (e.g., code). In this work, we propose Code2Video, a code-centric agent framework for generating educational videos via executable Python code. The framework comprises three collaborative agents: (i) Planner, which structures lecture content into temporally coherent flows and prepares corresponding visual assets; (ii) Coder, which converts structured instructions into executable Python codes while incorporating scope-guided auto-fix to enhance efficiency; and (iii) Critic, which leverages vision-language models (VLM) with visual anchor prompts to refine spatial layout and ensure clarity. To support systematic evaluation, we build MMMC, a benchmark of professionally produced, discipline-specific educational videos. We evaluate MMMC across diverse dimensions, including VLM-as-a-Judge aesthetic scores, code efficiency, and particularly, TeachQuiz, a novel end-to-end metric that quantifies how well a VLM, after unlearning, can recover knowledge by watching the generated videos. Our results demonstrate the potential of Code2Video as a scalable, interpretable, and controllable approach, achieving 40% improvement over direct code generation and producing videos comparable to human-crafted tutorials. The code and datasets are available at this https URL. 

**Abstract (ZH)**: 尽管近期生成模型在像素空间视频合成方面取得了进展，它们在生成专业教育视频方面的应用仍然受限，因为教育视频需要学科知识、精确的视觉结构和连贯的过渡，限制了其在教育场景中的应用。直观上， 이러한要求可以通过可渲染环境的操控来更好地解决，该环境可以通过逻辑命令（例如代码）明确控制。在本文中，我们提出了Code2Video，这是一种通过可执行Python代码生成教育视频的代码为中心的代理框架。该框架包括三个协作代理：（i）规划者，将讲座内容结构化为时间上连贯的流程，并准备相应的视觉资产；（ii）编程员，将结构化指令转换为可执行的Python代码，同时结合范围引导的自动修复以提高效率；以及（iii）评论员，利用带有视觉锚点提示的多模态视觉语言模型（VLM）来优化空间布局并确保清晰度。为了支持系统的评估，我们构建了MMMC基准，这是一个由专业制作、学科特定的教育视频组成的基准。我们从多个维度评估MMMC，包括使用VLM作为评判者的美学分数、代码效率，特别是TeachQuiz，这是一个新颖的端到端度量标准，量化了一个VLM在忘记之后通过观看生成的视频恢复知识的程度。我们的结果显示，Code2Video作为一种可扩展、可解释且可控的方法，其性能比直接代码生成提高了40%，并与人工制作的教程生成的视频相当。代码和数据集可在以下链接获取。 

---
# EditTrack: Detecting and Attributing AI-assisted Image Editing 

**Title (ZH)**: EditTrack: 检测与归因AI辅助图像编辑 

**Authors**: Zhengyuan Jiang, Yuyang Zhang, Moyang Guo, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2510.01173)  

**Abstract**: In this work, we formulate and study the problem of image-editing detection and attribution: given a base image and a suspicious image, detection seeks to determine whether the suspicious image was derived from the base image using an AI editing model, while attribution further identifies the specific editing model responsible. Existing methods for detecting and attributing AI-generated images are insufficient for this problem, as they focus on determining whether an image was AI-generated/edited rather than whether it was edited from a particular base image. To bridge this gap, we propose EditTrack, the first framework for this image-editing detection and attribution problem. Building on four key observations about the editing process, EditTrack introduces a novel re-editing strategy and leverages carefully designed similarity metrics to determine whether a suspicious image originates from a base image and, if so, by which model. We evaluate EditTrack on five state-of-the-art editing models across six datasets, demonstrating that it consistently achieves accurate detection and attribution, significantly outperforming five baselines. 

**Abstract (ZH)**: 本研究提出了图像编辑检测与归因的问题建模与研究：给定一个基图像和一个可疑图像，检测旨在确定可疑图像是否是基于基图像使用AI编辑模型生成的，而归因进一步识别具体的编辑模型。现有的检测和归因AI生成图像的方法不足以解决这个问题，因为它们侧重于确定图像是否是AI生成或编辑，而非是否从特定基图像生成。为弥补这一差距，我们提出了EditTrack，这是第一个针对此图像编辑检测与归因问题的框架。基于编辑过程的四个关键观察，EditTrack引入了一种新颖的再编辑策略，并利用精心设计的相似性度量来确定可疑图像是否源自基图像，以及如果是，则是由哪个模型生成的。我们在六大数据集上对EditTrack进行了五种最先进的编辑模型的评估，展示了它在检测和归因方面的一致准确性，并且显著优于五个基线方法。 

---
# Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity 

**Title (ZH)**: 口头化采样：如何缓解模式崩溃并释放大规模语言模型多样性 

**Authors**: Jiayi Zhang, Simon Yu, Derek Chong, Anthony Sicilia, Michael R. Tomz, Christopher D. Manning, Weiyan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01171)  

**Abstract**: Post-training alignment often reduces LLM diversity, leading to a phenomenon known as mode collapse. Unlike prior work that attributes this effect to algorithmic limitations, we identify a fundamental, pervasive data-level driver: typicality bias in preference data, whereby annotators systematically favor familiar text as a result of well-established findings in cognitive psychology. We formalize this bias theoretically, verify it on preference datasets empirically, and show that it plays a central role in mode collapse. Motivated by this analysis, we introduce Verbalized Sampling, a simple, training-free prompting strategy to circumvent mode collapse. VS prompts the model to verbalize a probability distribution over a set of responses (e.g., ``Generate 5 jokes about coffee and their corresponding probabilities''). Comprehensive experiments show that VS significantly improves performance across creative writing (poems, stories, jokes), dialogue simulation, open-ended QA, and synthetic data generation, without sacrificing factual accuracy and safety. For instance, in creative writing, VS increases diversity by 1.6-2.1x over direct prompting. We further observe an emergent trend that more capable models benefit more from VS. In sum, our work provides a new data-centric perspective on mode collapse and a practical inference-time remedy that helps unlock pre-trained generative diversity. 

**Abstract (ZH)**: Post-training Alignment Often Reduces LLM Diversity, Leading to Mode Collapse: A Data-Level Driver and a Training-free Solution Through Verbalized Sampling 

---
# Fiaingen: A financial time series generative method matching real-world data quality 

**Title (ZH)**: Fiaingen: 一种匹配实际数据质量的金融时间序列生成方法 

**Authors**: Jože M. Rožanec, Tina Žezlin, Laurentiu Vasiliu, Dunja Mladenić, Radu Prodan, Dumitru Roman  

**Link**: [PDF](https://arxiv.org/pdf/2510.01169)  

**Abstract**: Data is vital in enabling machine learning models to advance research and practical applications in finance, where accurate and robust models are essential for investment and trading decision-making. However, real-world data is limited despite its quantity, quality, and variety. The data shortage of various financial assets directly hinders the performance of machine learning models designed to trade and invest in these assets. Generative methods can mitigate this shortage. In this paper, we introduce a set of novel techniques for time series data generation (we name them Fiaingen) and assess their performance across three criteria: (a) overlap of real-world and synthetic data on a reduced dimensionality space, (b) performance on downstream machine learning tasks, and (c) runtime performance. Our experiments demonstrate that the methods achieve state-of-the-art performance across the three criteria listed above. Synthetic data generated with Fiaingen methods more closely mirrors the original time series data while keeping data generation time close to seconds - ensuring the scalability of the proposed approach. Furthermore, models trained on it achieve performance close to those trained with real-world data. 

**Abstract (ZH)**: 金融领域中基于时间序列数据的生成方法在增强机器学习模型性能方面的研究与评估 

---
# Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards 

**Title (ZH)**: 跨可验证性和非可验证性奖励的多目标同时对齐 

**Authors**: Yiran Shen, Yu Xia, Jonathan Chang, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01167)  

**Abstract**: Aligning large language models to human preferences is inherently multidimensional, yet most pipelines collapse heterogeneous signals into a single optimizeable objective. We seek to answer what it would take to simultaneously align a model across various domains spanning those with: verifiable rewards (mathematical accuracy), non-verifiable subjective preferences (human values), and complex interactive scenarios (multi-turn AI tutoring dialogues). Such multi-objective reinforcement learning setups are often plagued by the individual objectives being at odds with each other, resulting in inefficient training and little user control during inference. We propose a unified framework that: (i) standardizes {process reward model} (PRM) training across both verifiable and non-verifiable settings to better supervise models' chain-of-thought reasoning; (ii) performs {multi-objective alignment} by training the LLM with our $\textbf{M}$ulti-$\textbf{A}$ction-$\textbf{H}$ead $\textbf{DPO}$ (MAH-DPO) and a vectorized reward where the dimensions of the vector correspond to the various objectives instead of a single scalar; and (iii) demonstrates how such a system provides fine-grained inference-time user control. Experiments across math reasoning, value alignment, and multi-turn dialogue show that our framework improves performance across multiple objectives simultaneously, while minimizing cross-objective trade-offs and enabling flexible inference time user control. The code can be found at this https URL. 

**Abstract (ZH)**: 将大型语言模型与人类偏好对齐本质上是多维度的，然而大多数流程将异质信号简化为单一可优化目标。我们旨在回答如何在数学准确性、非验证性主观偏好和复杂交互场景等各类领域中同时对模型进行对齐。此类多目标强化学习设置往往因个体目标相互冲突而导致训练效率低下，且在推理过程中缺乏用户控制。我们提出了一种统一框架，该框架：(i) 在可验证和非可验证设置中标准化过程奖励模型 (PRM) 训练，以更好地监督模型的推理过程；(ii) 通过使用我们提出的多行动头DPO (MAH-DPO) 和向量奖励进行多目标对齐，其中向量的维度对应于各种目标，而不是单一标量；和(iii) 展示了这样一种系统在推理时为用户提供精细控制的可能性。跨数学推理、价值对齐和多轮对话的实验结果显示，我们的框架能够同时在多个目标上提高性能，最大限度地减少跨目标权衡，从而使推理时的用户控制更加灵活。代码见此链接。 

---
# GRAD: Generative Retrieval-Aligned Demonstration Sampler for Efficient Few-Shot Reasoning 

**Title (ZH)**: GRAD: 生成检索对齐演示样本器以实现高效的少样本推理 

**Authors**: Oussama Gabouj, Kamel Charaf, Ivan Zakazov, Nicolas Baldwin, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2510.01165)  

**Abstract**: Large Language Models (LLMs) achieve strong performance across diverse tasks, but their effectiveness often depends on the quality of the provided context. Retrieval-Augmented Generation (RAG) enriches prompts with external information, but its reliance on static databases constrains adaptability and can result in irrelevant demonstrations. In this work, we propose a Generative Retrieval-Aligned Demonstrator (GRAD), a dynamic demonstration-based approach where an LLM model is trained to generate input-specific concise demonstrations. By tailoring demonstrations to each input, our method offers better contextual support than traditional RAG approaches. We demonstrate the superiority of GRAD under budget constraints, where we limit both the number of tokens used per demonstration and the number of tokens used for the final output. Trained solely on a math dataset, GRAD consistently outperforms strong baselines on Qwen2.5-14B across mathematical reasoning and advanced STEM questions, highlighting GRAD's robust generalization to out-of-distribution (OOD) domains such as physics, chemistry, and computer science. Furthermore, we show that demonstrations generated by trained smaller models can effectively guide larger target models, reducing training costs while maintaining competitive accuracy. Overall, this work introduces a scalable demonstration generator model presenting the first step toward a dynamic few-shot learning paradigm in resource-constrained settings. We release the code used for the project. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多样化的任务中表现出色，但其效果往往依赖于提供的上下文质量。检索增强生成（RAG）通过外部信息丰富提示，但其对静态数据库的依赖限制了其适应性，并可能导致不相关的表现。在这项工作中，我们提出了一种生成检索对齐示范者（GRAD），这是一种动态的示范导向方法，其中LLM模型被训练生成针对每个输入的简洁示范。通过针对每个输入定制示范，我们的方法提供了比传统RAG方法更好的上下文支持。在预算限制条件下，我们限制每个示范和最终输出使用的token数，展示了GRAD的优越性。仅基于数学数据集训练的GRAD在Qwen2.5-14B上在数学推理和高级STEM问题上始终优于强大的基线模型，突显了GRAD在物理学、化学和计算机科学等分布外（OOD）领域中的稳健泛化能力。此外，我们展示了由训练较小模型生成的示范可以有效地指导更大目标模型，从而降低成本同时保持竞争力。整体而言，这项工作介绍了一种可扩展的示范生成模型，展示了在资源受限环境下动态少数样本学习范式的初步步骤。我们释放了该项目所使用的代码。 

---
# Social Welfare Function Leaderboard: When LLM Agents Allocate Social Welfare 

**Title (ZH)**: 社会福利函数排行榜：当LLM代理分配社会福利 

**Authors**: Zhengliang Shi, Ruotian Ma, Jen-tse Huang, Xinbei Ma, Xingyu Chen, Mengru Wang, Qu Yang, Yue Wang, Fanghua Ye, Ziyang Chen, Shanyi Wang, Cixing Li, Wenxuan Wang, Zhaopeng Tu, Xiaolong Li, Zhaochun Ren, Linus  

**Link**: [PDF](https://arxiv.org/pdf/2510.01164)  

**Abstract**: Large language models (LLMs) are increasingly entrusted with high-stakes decisions that affect human welfare. However, the principles and values that guide these models when distributing scarce societal resources remain largely unexamined. To address this, we introduce the Social Welfare Function (SWF) Benchmark, a dynamic simulation environment where an LLM acts as a sovereign allocator, distributing tasks to a heterogeneous community of recipients. The benchmark is designed to create a persistent trade-off between maximizing collective efficiency (measured by Return on Investment) and ensuring distributive fairness (measured by the Gini coefficient). We evaluate 20 state-of-the-art LLMs and present the first leaderboard for social welfare allocation. Our findings reveal three key insights: (i) A model's general conversational ability, as measured by popular leaderboards, is a poor predictor of its allocation skill. (ii) Most LLMs exhibit a strong default utilitarian orientation, prioritizing group productivity at the expense of severe inequality. (iii) Allocation strategies are highly vulnerable, easily perturbed by output-length constraints and social-influence framing. These results highlight the risks of deploying current LLMs as societal decision-makers and underscore the need for specialized benchmarks and targeted alignment for AI governance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越多地被委托作出高风险决策，这些决策影响人类福祉。然而，当分配稀缺的社会资源时，指导这些模型的原则和价值观尚未得到充分研究。为了解决这一问题，我们引入了社会福利函数（SWF）基准，这是一种动态模拟环境，在此环境中，一个LLM充当主权分配者，将任务分配给一群异质的接收者。该基准旨在在最大化集体效率（通过投资回报率衡量）和确保分配公平性（通过基尼系数衡量）之间创建持久的权衡。我们评估了20个最先进的LLM，并提供了社会福利分配的第一个排行榜。我们的发现揭示了三个关键见解：（i）通过流行排行榜衡量的一般对话能力并不是其分配技能的可靠预测指标。（ii）大多数LLM表现出强烈的功利主义倾向，优先考虑团体 productivity 至于严重的不平等。（iii）分配策略高度脆弱，容易受输出长度约束和社会影响力框架的影响。这些结果突显了当前将LLM部署为社会决策制定者的风险，并强调了需要专门的基准和针对性对齐以实现AI治理的紧迫性。 

---
# Prosperity before Collapse: How Far Can Off-Policy RL Reach with Stale Data on LLMs? 

**Title (ZH)**: 繁荣之前 decline之前：基于陈旧数据的离策略RL能达到多远——在大规模语言模型上的探索 

**Authors**: Haizhong Zheng, Jiawei Zhao, Bedi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01161)  

**Abstract**: Reinforcement learning has been central to recent advances in large language model reasoning, but most algorithms rely on on-policy training that demands fresh rollouts at every update, limiting efficiency and scalability. Asynchronous RL systems alleviate this by decoupling rollout generation from training, yet their effectiveness hinges on tolerating large staleness in rollout data, a setting where existing methods either degrade in performance or collapse. We revisit this challenge and uncover a prosperity-before-collapse phenomenon: stale data can be as informative as on-policy data if exploited properly. Building on this insight, we introduce M2PO (Second-Moment Trust Policy Optimization), which constrains the second moment of importance weights to suppress only extreme outliers while preserving informative updates. Notably, M2PO sharply reduces the fraction of clipped tokens under high staleness (from 1.22% to 0.06% over training), precisely masking high-variance tokens while maintaining stable optimization. Extensive evaluation across six models (from 1.7B to 32B) and eight benchmarks shows that M2PO delivers stable off-policy training even with data stale by at least 256 model updates and matches on-policy performance. 

**Abstract (ZH)**: 增强学习近年来在大规模语言模型推理中发挥了核心作用，但大多数算法依赖于在线策略训练，要求每次更新都进行新鲜 rollout，这限制了效率和可扩展性。异步 RL 系统通过将 rollout 生成与训练解耦来解决这一问题，但在 rollout 数据有较大 staleness 的情况下，其效果依赖于容忍这种 staleness，而现有方法在这种情况下要么性能下降，要么失效。我们重新审视这一挑战，并发现一个繁荣胜于崩溃的现象：如果充分利用，stale 数据可以与在线策略数据一样有信息价值。基于这一洞察，我们引入了 M2PO（第二矩信任策略优化），它通过约束重要权重的第二矩来抑制只有极端离群值，同时保留信息更新。值得注意的是，M2PO 在高 staleness 下显著减少了被截断的 token 比例（从训练时的 1.22% 降低到 0.06%），精确地掩蔽了高方差 token，同时保持了稳定的优化。在六种不同规模（从 1.7B 到 32B）的模型和八个基准上的广泛评估表明，即使数据在至少 256 模型更新后仍然有效，M2PO 仍能提供稳定的 off-policy 训练，并匹配在线策略性能。 

---
# mR3: Multilingual Rubric-Agnostic Reward Reasoning Models 

**Title (ZH)**: 多语种无评分标准奖励推理模型 

**Authors**: David Anugraha, Shou-Yi Hung, Zilu Tang, Annie En-Shiun Lee, Derry Tanti Wijaya, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2510.01146)  

**Abstract**: Evaluation using Large Language Model (LLM) judges has been widely adopted in English and shown to be effective for automatic evaluation. However, their performance does not generalize well to non-English settings, and it remains unclear what constitutes effective multilingual training for such judges. In this paper, we introduce mR3, a massively multilingual, rubric-agnostic reward reasoning model trained on 72 languages, achieving the broadest language coverage in reward modeling to date. We present a comprehensive study of data and curriculum selection for training to identify effective strategies and data sources for building high-quality reward models, including the integration of target-language reasoning datasets. Our approach attains state-of-the-art performance on multilingual reward model benchmarks, surpassing much larger models (i.e., GPT-OSS-120B) while being up to 9x smaller, and its effectiveness is further confirmed through extensive ablation studies. Our models, data, and code are available as open source at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLM）裁判的评价研究已被广泛应用于英语中并显示出了自动评价的有效性。然而，它们的表现并不适用于非英语环境，目前尚不清楚什么是有效的多语言训练。本文介绍了一种基于72种语言训练的mR3模型，这是一种广泛多语言且不受评分标准约束的奖励推理模型，实现了迄今为止最广泛的奖励建模语言覆盖范围。本文对训练数据和课程选择进行了全面研究，以确定构建高质量奖励模型的有效策略和数据来源，包括目标语言推理数据集的整合。我们的方法在多语言奖励模型基准测试中达到了最先进的性能，并且在比其大得多的模型（如GPT-OSS-120B）的基础上小了9倍，其有效性通过广泛的消融研究得到了进一步确认。我们的模型、数据和代码已作为开源发布。 

---
# TabINR: An Implicit Neural Representation Framework for Tabular Data Imputation 

**Title (ZH)**: TabINR: 一种用于表格式数据插补的隐式神经表示框架 

**Authors**: Vincent Ochs, Florentin Bieder, Sidaty el Hadramy, Paul Friedrich, Stephanie Taha-Mehlitz, Anas Taha, Philippe C. Cattin  

**Link**: [PDF](https://arxiv.org/pdf/2510.01136)  

**Abstract**: Tabular data builds the basis for a wide range of applications, yet real-world datasets are frequently incomplete due to collection errors, privacy restrictions, or sensor failures. As missing values degrade the performance or hinder the applicability of downstream models, and while simple imputing strategies tend to introduce bias or distort the underlying data distribution, we require imputers that provide high-quality imputations, are robust across dataset sizes and yield fast inference. We therefore introduce TabINR, an auto-decoder based Implicit Neural Representation (INR) framework that models tables as neural functions. Building on recent advances in generalizable INRs, we introduce learnable row and feature embeddings that effectively deal with the discrete structure of tabular data and can be inferred from partial observations, enabling instance adaptive imputations without modifying the trained model. We evaluate our framework across a diverse range of twelve real-world datasets and multiple missingness mechanisms, demonstrating consistently strong imputation accuracy, mostly matching or outperforming classical (KNN, MICE, MissForest) and deep learning based models (GAIN, ReMasker), with the clearest gains on high-dimensional datasets. 

**Abstract (ZH)**: 基于自动解码器的隐式神经表示（INR）框架TabINR：建模表格数据及其实例自适应插补方法 

---
# A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning 

**Title (ZH)**: 一个从业者指南：多轮自主强化学习 

**Authors**: Ruiyi Wang, Prithviraj Ammanabrolu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01132)  

**Abstract**: We study what actually works and what doesn't for training large language models as agents via multi-turn reinforcement learning. Despite rapid progress, existing frameworks and definitions are fragmented, and there is no systematic formulation or analysis of which design choices matter across tasks. We address this gap by first breaking down the design space into three inter-related pillars -- environment, reward, and policy -- and empirically derive a recipe for training LLM agents in situated textual domains. In particular, we test TextWorld and ALFWorld, popular domains for testing situated embodied reasoning, as well as SWE-Gym for more software engineering style tasks. (i) For the environment, we analyze the impacts of task complexity in terms of sizes of the state and action spaces as well as optimal solution length, finding that even simple environments within a domain can provide signal on how well an agent can generalize to more complex tasks. (ii) For the reward, we ablate relative reward sparsity, observing that while dense turn-level rewards accelerate training, performance and stability is highly dependent on the choice of RL algorithm. (iii) And for the agent's policy, we explore the interplay between reward sparsity and biased (PPO, GRPO) and unbiased (RLOO) policy gradient methods in addition to showing how to find the optimal Supervised Fine-tuning (SFT) to RL training ratio given a fixed budget. We distill these findings into a training recipe that guides co-design across the three pillars, facilitating research and practical efforts in multi-turn agentic RL. Code: this https URL 

**Abstract (ZH)**: 我们研究通过多轮强化学习训练大规模语言模型作为代理的有效性和无效方法。尽管进展迅速，现有框架和定义仍然支离破碎，且缺乏对不同任务中哪些设计选择重要的系统化公式和分析。我们首先将设计空间分解为三个相互关联的支柱——环境、奖励和策略，并通过实证方法推导出在基于文本的情境领域训练LLM代理的配方。特别是，我们测试了TextWorld和ALFWorld，这两个流行的领域用于测试情境化体态推理，以及SWE-Gym，用于更符合软件工程风格的任务。(i) 对于环境，我们分析了状态空间和动作空间大小以及最优解长度的任务复杂性的影响，发现即使在一个领域内的简单环境中也能够提供代理能否泛化到更复杂任务的信号。(ii) 对于奖励，我们剥离相对稀疏的奖励，发现虽然密集的轮次级奖励可以加速训练，但性能和稳定性高度依赖于所选的RL算法。(iii) 对于代理的策略，我们探索了稀疏奖励与有偏（PPO、GRPO）和无偏（RLOO）策略梯度方法之间的相互作用，并展示了如何在固定预算下找到监督微调（SFT）与RL训练的最佳比例。我们将这些发现总结成一个训练配方，以指导三个支柱之间的协同设计，促进多轮代理强化学习的研究和实践。代码：这个 https:// эта ссылка 

---
# Rethinking Thinking Tokens: LLMs as Improvement Operators 

**Title (ZH)**: 重思思考令牌：LLMs作为改进操作符 

**Authors**: Lovish Madaan, Aniket Didolkar, Suchin Gururangan, John Quan, Ruan Silva, Ruslan Salakhutdinov, Manzil Zaheer, Sanjeev Arora, Anirudh Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01123)  

**Abstract**: Reasoning training incentivizes LLMs to produce long chains of thought (long CoT), which among other things, allows them to explore solution strategies with self-checking. This results in higher accuracy, but inflates context length, token/compute cost, and answer latency. We ask: Can current models leverage their metacognition to provide other combinations on this Pareto frontier, e.g., better accuracy with lower context length and/or latency? Abstractly, we view the model as an improvement operator on its own "thoughts" with a continuum of possible strategies. We identify an interesting inference family Parallel-Distill-Refine (PDR), which performs the following: (i) generate diverse drafts in parallel; (ii) distill them into a bounded, textual workspace; and (iii) refine conditioned on this workspace, producing an output that seeds the next round. Importantly, context length (hence compute cost) is controllable via degree of parallelism, and is no longer conflated with the total number of generated tokens. We report PDR instantiations of current models that give better accuracy than long CoT while incurring lower latency. Setting degree of parallelism to 1 yields an interesting subcase, Sequential Refinement (SR) (iteratively improve a single candidate answer) which provides performance superior to long CoT. Success of such model orchestrations raises the question whether further training could shift the Pareto frontier. To this end, we train an 8B thinking model with Reinforcement Learning (RL) to make it consistent with PDR as the inference method. On math tasks with verifiable answers, iterative pipelines surpass single-pass baselines at matched sequential budgets, with PDR delivering the largest gains (e.g., +11% on AIME 2024 and +9% on AIME 2025). 

**Abstract (ZH)**: LLM推理训练激励生成长链条的思考（长CoT），这有助于探索带自我检查的解题策略，从而提高准确性，但增加了上下文长度、标记/计算成本和答案延迟。我们询问：当前模型是否能利用其元认知提供帕累托前沿上的其他组合，例如在更低的上下文长度和/或延迟下获得更好的准确性？ 

---
# CodeGenLink: A Tool to Find the Likely Origin and License of Automatically Generated Code 

**Title (ZH)**: CodeGenLink: 一个查找自动生成代码可能来源及其许可证的工具 

**Authors**: Daniele Bifolco, Guido Annicchiarico, Pierluigi Barbiero, Massimiliano Di Penta, Fiorella Zampetti  

**Link**: [PDF](https://arxiv.org/pdf/2510.01077)  

**Abstract**: Large Language Models (LLMs) are widely used in software development tasks nowadays. Unlike reusing code taken from the Web, for LLMs' generated code, developers are concerned about its lack of trustworthiness and possible copyright or licensing violations, due to the lack of code provenance information. This paper proposes CodeGenLink, a GitHub CoPilot extension for Visual Studio Code aimed at (i) suggesting links containing code very similar to automatically generated code, and (ii) whenever possible, indicating the license of the likely origin of the code. CodeGenLink retrieves candidate links by combining LLMs with their web search features and then performs similarity analysis between the generated and retrieved code. Preliminary results show that CodeGenLink effectively filters unrelated links via similarity analysis and provides licensing information when available. Tool URL: this https URL Tool Video: this https URL 

**Abstract (ZH)**: 大型语言模型(LLMs)在当今的软件开发任务中广泛应用。对于LLMs生成的代码，开发者对其缺乏可信度以及可能的版权或许可违规表示担忧，这主要是因为缺乏代码的来源信息。本文提出CodeGenLink，这是一个旨在（i）建议包含与自动生成代码非常相似的链接，以及（ii）尽可能指示代码可能来源的许可信息的GitHub CoPilot扩展程序。CodeGenLink通过结合LLMs的网络搜索功能来检索候选链接，然后在生成的代码和检索的代码之间进行相似性分析。初步结果显示，CodeGenLink通过相似性分析有效过滤了无关链接，并在可用时提供了许可信息。工具URL: 这里是链接。工具视频: 这里是链接。 

---
# Hybrid Dialogue State Tracking for Persian Chatbots: A Language Model-Based Approach 

**Title (ZH)**: 基于语言模型的混合对话状态跟踪方法：以波斯聊天机器人为例 

**Authors**: Samin Mahdipour Aghabagher, Saeedeh Momtazi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01052)  

**Abstract**: Dialogue State Tracking (DST) is an essential element of conversational AI with the objective of deeply understanding the conversation context and leading it toward answering user requests. Due to high demands for open-domain and multi-turn chatbots, the traditional rule-based DST is not efficient enough, since it cannot provide the required adaptability and coherence for human-like experiences in complex conversations. This study proposes a hybrid DST model that utilizes rule-based methods along with language models, including BERT for slot filling and intent detection, XGBoost for intent validation, GPT for DST, and online agents for real-time answer generation. This model is uniquely designed to be evaluated on a comprehensive Persian multi-turn dialogue dataset and demonstrated significantly improved accuracy and coherence over existing methods in Persian-based chatbots. The results demonstrate how effectively a hybrid approach may improve DST capabilities, paving the way for conversational AI systems that are more customized, adaptable, and human-like. 

**Abstract (ZH)**: 对话状态跟踪(DST)是会话AI的核心组成部分，旨在深入理解对话背景并引导对话以回答用户请求。由于对开放式和多轮聊天机器人的高需求，传统的基于规则的DST不足以提供复杂对话中所需的适应性和连贯性，以实现类似人类的体验。本研究提出了一种混合DST模型，该模型结合了基于规则的方法和语言模型，包括BERT用于槽填充和意图检测、XGBoost用于意图验证、GPT用于DST，以及在线代理用于实时答案生成。该模型专门设计用于在全面的波斯语多轮对话数据集上进行评估，并证明在波斯语基于聊天机器人中具有显著更高的准确性和连贯性。结果展示了混合方法如何有效提升DST能力，为更个性化、适应性强且类似人类的会话AI系统铺平了道路。 

---
# GEM: A Gym for Agentic LLMs 

**Title (ZH)**: GEM：代理LLM的 Gym 环境 

**Authors**: Zichen Liu, Anya Sims, Keyu Duan, Changyu Chen, Simon Yu, Xiangxin Zhou, Haotian Xu, Shaopan Xiong, Bo Liu, Chenmien Tan, Chuen Yang Beh, Weixun Wang, Hao Zhu, Weiyan Shi, Diyi Yang, Michael Shieh, Yee Whye Teh, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.01051)  

**Abstract**: The training paradigm for large language models (LLMs) is moving from static datasets to experience-based learning, where agents acquire skills via interacting with complex environments. To facilitate this transition we introduce GEM (General Experience Maker), an open-source environment simulator designed for the age of LLMs. Analogous to OpenAI-Gym for traditional reinforcement learning (RL), GEM provides a standardized framework for the environment-agent interface, including asynchronous vectorized execution for high throughput, and flexible wrappers for easy extensibility. GEM also features a diverse suite of environments, robust integrated tools, and single-file example scripts demonstrating using GEM with five popular RL training frameworks. Along with this, we also provide a set of baselines across 24 environments using REINFORCE with Return Batch Normalization (ReBN), which -- unlike GRPO -- is compatible with the full RL setting of dense per-turn rewards and offers better credit assignment. We further conduct apple-to-apple benchmarking of PPO, GRPO and REINFORCE in both single- and multi-turn settings using GEM to shed light on the algorithmic designs. Lastly, GEM also functions as a convenient evaluation toolkit besides a training environment. We hope this framework can help accelerate future agentic LLM research. 

**Abstract (ZH)**: 大型语言模型（LLMs）的训练范式从静态数据集转向基于经验的学习，其中代理通过与复杂环境交互来获取技能。为促进这一过渡，我们引入了GEM（通用经验制造者），这是一种开源的环境模拟器，适用于LLM时代。类似于传统强化学习（RL）中的OpenAI-Gym，GEM提供了一个标准化的环境-代理接口框架，包括异步向量执行以实现高吞吐量，以及灵活的封装以实现便捷的扩展。GEM还配备了多样化的环境、 robust的集成工具，以及使用GEM与五种流行的RL训练框架示例脚本的单文件示例。此外，我们还在24个环境中提供了基于ReINFORCE带回报批归一化（ReBN）的一系列基线算法，这些算法与传统的GRPO不同，不仅能够兼容密集的每轮奖励设置，还能更有效地归因。我们还在GEM中对PPO、GRPO和ReINFORCE在单轮和多轮设置下的算法设计进行了逐点基准测试。最后，GEM还充当了评估工具而不仅仅是训练环境。我们希望这一框架能帮助加速未来基于代理的LLM研究。 

---
# Interpreting Language Models Through Concept Descriptions: A Survey 

**Title (ZH)**: 通过概念描述解释语言模型：一个综述 

**Authors**: Nils Feldhus, Laura Kopf  

**Link**: [PDF](https://arxiv.org/pdf/2510.01048)  

**Abstract**: Understanding the decision-making processes of neural networks is a central goal of mechanistic interpretability. In the context of Large Language Models (LLMs), this involves uncovering the underlying mechanisms and identifying the roles of individual model components such as neurons and attention heads, as well as model abstractions such as the learned sparse features extracted by Sparse Autoencoders (SAEs). A rapidly growing line of work tackles this challenge by using powerful generator models to produce open-vocabulary, natural language concept descriptions for these components. In this paper, we provide the first survey of the emerging field of concept descriptions for model components and abstractions. We chart the key methods for generating these descriptions, the evolving landscape of automated and human metrics for evaluating them, and the datasets that underpin this research. Our synthesis reveals a growing demand for more rigorous, causal evaluation. By outlining the state of the art and identifying key challenges, this survey provides a roadmap for future research toward making models more transparent. 

**Abstract (ZH)**: 理解神经网络的决策过程是机械可解释性中的一个核心目标。在大规模语言模型（LLMs）的背景下，这涉及发现其背后的机制并识别单个模型组件（如神经元和注意力头）以及模型抽象（如稀疏自动编码器SAE提取的稀疏特征）的作用。越来越多的研究通过使用强大的生成模型来生成开放词汇的自然语言概念描述来应对这一挑战。在本文中，我们提供了关于模型组件和抽象概念描述新兴领域的首次综述，探讨了生成这些描述的关键方法、自动化和人工评估指标的发展景观以及支撑这项研究的数据集。我们的综述揭示了对更加严谨且因果关系的评估方法的需求。通过概述最新的研究状况并识别关键挑战，本文为未来使模型更加透明的研究提供了路线图。 

---
# Authentic Discrete Diffusion Model 

**Title (ZH)**: 真实离散扩散模型 

**Authors**: Xiao Li, Jiaqi Zhang, Shuxiang Zhang, Tianshui Chen, Liang Lin, Guangrun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01047)  

**Abstract**: We propose an Authentic Discrete Diffusion (ADD) framework that fundamentally redefines prior pseudo-discrete approaches by preserving core diffusion characteristics directly in the one-hot space through a suite of coordinated mechanisms. Unlike conventional "pseudo" discrete diffusion (PDD) methods, ADD reformulates the diffusion input by directly using float-encoded one-hot class data, without relying on diffusing in the continuous latent spaces or masking policies. At its core, a timestep-conditioned cross-entropy loss is introduced between the diffusion model's outputs and the original one-hot labels. This synergistic design establishes a bridge between discriminative and generative learning. Our experiments demonstrate that ADD not only achieves superior performance on classification tasks compared to the baseline, but also exhibits excellent text generation capabilities on Image captioning. Extensive ablations validate the measurable gains of each component. 

**Abstract (ZH)**: 我们提出了一种真实性离散扩散（ADD）框架，该框架通过一系列协调机制，直接在独热空间中保留核心扩散特性，从根本上重新定义了先前三伪离散方法。与传统的“伪”离散扩散（PDD）方法不同，ADD 通过直接使用浮点编码的独热类别数据重新形式化扩散输入，而不依赖于在连续潜在空间中扩散或使用遮罩策略。核心上，引入了条件时间步长交叉熵损失，该损失在扩散模型的输出和原始独热标签之间建立桥梁。这种协同设计建立了辨别学习与生成学习之间的联系。实验结果表明，ADD 不仅在分类任务上的性能优于基线，还具备出色的图像 Caption 生成能力。广泛的消融实验验证了每个组件的可衡量改进。 

---
# CurES: From Gradient Analysis to Efficient Curriculum Learning for Reasoning LLMs 

**Title (ZH)**: CurES: 从梯度分析到高效的逻辑语言模型 Curriculum 学习 

**Authors**: Yongcheng Zeng, Zexu Sun, Bokai Ji, Erxue Min, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Haifeng Zhang, Xu Chen, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01037)  

**Abstract**: Curriculum learning plays a crucial role in enhancing the training efficiency of large language models (LLMs) on reasoning tasks. However, existing methods often fail to adequately account for variations in prompt difficulty or rely on simplistic filtering mechanisms to select prompt datasets within a narrow criterion range, resulting in significant computational waste. In this work, we approach the problem from the perspective of reinforcement learning gradient optimization, offering a systematic and theoretical investigation into how to improve the training efficiency of LLMs. We identify two key factors influencing training efficiency: the selection of training prompts and the allocation of rollout quantities across different prompts. Our theoretical analysis reveals that the sampling distribution of prompts dictates the convergence rate of gradient descent, while the allocation of the rollout quantity influences the consistency and stability of overall gradient updates. Based on these insights, we propose CurES, an efficient training method that accelerates convergence and employs Bayesian posterior estimation to minimize computational overhead. Experiments demonstrate that our CurES outperforms Group Relative Policy Optimization (GRPO) by \textbf{+3.30} points and \textbf{+4.82} points with 1.5B and 7B models, respectively. Additionally, CurES exhibits faster convergence compared to baselines, including GRPO. 

**Abstract (ZH)**: Curriculum 学习在增强大型语言模型在推理任务训练效率方面扮演着 crucial 角色。然而，现有方法往往未能充分考虑提示难度的变化，或依赖于简单的过滤机制来选择提示数据集，导致大量计算资源浪费。在本工作中，我们从强化学习梯度优化的角度出发，对如何提高大型语言模型训练效率进行了系统性和理论性的研究。我们识别出影响训练效率的两个关键因素：训练提示的选择以及在不同提示之间分配展开量。我们的理论分析表明，提示的采样分布决定了梯度下降的收敛速度，而展开量的分配影响了整个梯度更新的一致性和稳定性。基于这些见解，我们提出了 CurES，一种高效的训练方法，能够加速收敛并利用贝叶斯后验估计来最小化计算开销。实验表明，与 Group Relative Policy Optimization (GRPO) 相比，我们的 CurES 分别在 1.5B 和 7B 模型上性能提升了 \textbf{+3.30} 分和 \textbf{+4.82} 分。此外，CurES 在收敛速度上也优于基准方法，包括 GRPO。 

---
# The Good, the Bad, and the Sampled: a No-Regret Approach to Safe Online Classification 

**Title (ZH)**: 好的、坏的和抽样的：一种无悔的在线分类安全方法 

**Authors**: Tavor Z. Baharav, Spyros Dragazis, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2510.01020)  

**Abstract**: We study the problem of sequentially testing individuals for a binary disease outcome whose true risk is governed by an unknown logistic model. At each round, a patient arrives with feature vector $x_t$, and the decision maker may either pay to administer a (noiseless) diagnostic test--revealing the true label--or skip testing and predict the patient's disease status based on their feature vector and prior history. Our goal is to minimize the total number of costly tests required while guaranteeing that the fraction of misclassifications does not exceed a prespecified error tolerance $\alpha$, with probability at least $1-\delta$. To address this, we develop a novel algorithm that interleaves label-collection and distribution estimation to estimate both $\theta^{*}$ and the context distribution $P$, and computes a conservative, data-driven threshold $\tau_t$ on the logistic score $|x_t^\top\theta|$ to decide when testing is necessary. We prove that, with probability at least $1-\delta$, our procedure does not exceed the target misclassification rate, and requires only $O(\sqrt{T})$ excess tests compared to the oracle baseline that knows both $\theta^{*}$ and the patient feature distribution $P$. This establishes the first no-regret guarantees for error-constrained logistic testing, with direct applications to cost-sensitive medical screening. Simulations corroborate our theoretical guarantees, showing that in practice our procedure efficiently estimates $\theta^{*}$ while retaining safety guarantees, and does not require too many excess tests. 

**Abstract (ZH)**: 我们研究了一个二元疾病结果个体的序贯测试问题，其真实风险由未知的逻辑模型支配。在每一轮中，一名患者带着特征向量 $x_t$ 到达，决策者可以选择支付进行无噪声诊断测试（揭示真实标签）或跳过测试并基于特征向量和过往历史预测患者的疾病状态。我们的目标是在最小化昂贵测试总数的同时，确保误分类的比例不超过事先指定的误差容忍度 $\alpha$，且概率不低于 $1-\delta$。为此，我们提出了一种新颖的算法，交替进行标签收集和分布估计，以估计 $\theta^{*}$ 和上下文分布 $P$，并基于逻辑评分 $|x_t^\top\theta|$ 计算保守的数据驱动阈值 $\tau_t$ 以决定何时进行测试。我们证明，以概率不低于 $1-\delta$ 的概率，我们的流程不会超过目标误分类率，并且相比于知道 $\theta^{*}$ 和患者特征分布 $P$ 的理想基准，仅需要 $O(\sqrt{T})$ 的额外测试。这首次为误差约束逻辑测试建立了无遗憾保证，直接应用于成本敏感医疗筛查。模拟结果验证了我们的理论保证，显示我们的流程在实践中能够有效地估计 $\theta^{*}$ 同时保留安全保证，并不需要太多额外的测试。 

---
# TextCAM: Explaining Class Activation Map with Text 

**Title (ZH)**: TextCAM：基于文本的类激活图解释 

**Authors**: Qiming Zhao, Xingjian Li, Xiaoyu Cao, Xiaolong Wu, Min Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01004)  

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success across domains but remain difficult to interpret, limiting their trustworthiness in high-stakes applications. This paper focuses on deep vision models, for which a dominant line of explainability methods are Class Activation Mapping (CAM) and its variants working by highlighting spatial regions that drive predictions. We figure out that CAM provides little semantic insight into what attributes underlie these activations. To address this limitation, we propose TextCAM, a novel explanation framework that enriches CAM with natural languages. TextCAM combines the precise spatial localization of CAM with the semantic alignment of vision-language models (VLMs). Specifically, we derive channel-level semantic representations using CLIP embeddings and linear discriminant analysis, and aggregate them with CAM weights to produce textual descriptions of salient visual evidence. This yields explanations that jointly specify where the model attends and what visual attributes likely support its decision. We further extend TextCAM to generate feature channels into semantically coherent groups, enabling more fine-grained visual-textual explanations. Experiments on ImageNet, CLEVR, and CUB demonstrate that TextCAM produces faithful and interpretable rationales that improve human understanding, detect spurious correlations, and preserve model fidelity. 

**Abstract (ZH)**: 深度神经网络（DNNs）在各个领域取得了显著成功，但仍然难以解释，限制了其在高风险应用中的可信度。本文专注于深度视觉模型，对于这类模型，主要的可解释性方法包括基于激活映射（Class Activation Mapping, CAM）及其变种方法，这些方法通过突出显示驱动预测的空间区域来工作。我们发现在这些激活中，CAM提供的语义洞察能力有限。为了解决这一局限性，我们提出了一种名为TextCAM的新颖解释框架，该框架将自然语言与CAM结合在一起。TextCAM结合了CAM精确的空间定位与视觉语言模型（VLMs）的语义对齐。具体来说，我们使用CLIP嵌入和线性判别分析从通道级别提取语义表示，并将其与CAM权重聚合以生成显著视觉证据的文字描述。这提供了既能说明模型关注的位置又能说明支持其决策的视觉属性的解释。我们进一步将TextCAM扩展为生成语义连贯的特征通道组，从而实现更精细的视觉-文本解释。在ImageNet、CLEVR和CUB上的实验表明，TextCAM生成了忠于事实且可解释的合理性描述，有助于提高人类理解、检测伪相关，并保持模型的准确性。 

---
# Deep Learning-Based Approach for Improving Relational Aggregated Search 

**Title (ZH)**: 基于深度学习的方法以提升关系聚合搜索 

**Authors**: Sara Saad Soliman, Ahmed Younes, Islam Elkabani, Ashraf Elsayed  

**Link**: [PDF](https://arxiv.org/pdf/2510.00966)  

**Abstract**: Due to an information explosion on the internet, there is a need for the development of aggregated search systems that can boost the retrieval and management of content in various formats. To further improve the clustering of Arabic text data in aggregated search environments, this research investigates the application of advanced natural language processing techniques, namely stacked autoencoders and AraBERT embeddings. By transcending the limitations of traditional search engines, which are imprecise, not contextually relevant, and not personalized, we offer more enriched, context-aware characterizations of search results, so we used a K-means clustering algorithm to discover distinctive features and relationships in these results, we then used our approach on different Arabic queries to evaluate its effectiveness. Our model illustrates that using stacked autoencoders in representation learning suits clustering tasks and can significantly improve clustering search results. It also demonstrates improved accuracy and relevance of search results. 

**Abstract (ZH)**: 由于互联网上的信息爆炸，需要开发聚合搜索系统以提升各类格式内容的检索和管理。为进一步提高聚合搜索环境中阿拉伯文本数据的聚类效果，本研究探讨了高级自然语言处理技术，即堆叠自动编码器和AraBERT嵌入的应用。通过超越传统搜索引擎的精度不足、缺乏上下文相关性和个性化的问题，我们提供了更加丰富和上下文相关的检索结果表征，利用K-means聚类算法发现这些结果的独特特征和关系，并在不同阿拉伯查询上评估了该方法的有效性。我们的模型说明了在表示学习中使用堆叠自动编码器适合聚类任务，并能显著提高聚类搜索结果的质量，同时也展示了检索结果的准确性和相关性得到了提升。 

---
# Bridging the Gap Between Simulated and Real Network Data Using Transfer Learning 

**Title (ZH)**: 使用迁移学习弥合模拟网络数据与真实网络数据之间的差距 

**Authors**: Carlos Güemes-Palau, Miquel Ferriol-Galmés, Jordi Paillisse-Vilanova, Albert López-Brescó, Pere Barlet-Ros, Albert Cabellos-Aparicio  

**Link**: [PDF](https://arxiv.org/pdf/2510.00956)  

**Abstract**: Machine Learning (ML)-based network models provide fast and accurate predictions for complex network behaviors but require substantial training data. Collecting such data from real networks is often costly and limited, especially for critical scenarios like failures. As a result, researchers commonly rely on simulated data, which reduces accuracy when models are deployed in real environments. We propose a hybrid approach leveraging transfer learning to combine simulated and real-world data. Using RouteNet-Fermi, we show that fine-tuning a pre-trained model with a small real dataset significantly improves performance. Our experiments with OMNeT++ and a custom testbed reduce the Mean Absolute Percentage Error (MAPE) in packet delay prediction by up to 88%. With just 10 real scenarios, MAPE drops by 37%, and with 50 scenarios, by 48%. 

**Abstract (ZH)**: 基于迁移学习的混合数据方法在混合仿真与实网数据中的应用：提高网络行为预测性能 

---
# Benchmarking Foundation Models with Retrieval-Augmented Generation in Olympic-Level Physics Problem Solving 

**Title (ZH)**: 基于检索增强生成方法的奥林匹克级别物理问题求解基础模型基准测试 

**Authors**: Shunfeng Zheng, Yudi Zhang, Meng Fang, Zihan Zhang, Zhitan Wu, Mykola Pechenizkiy, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00919)  

**Abstract**: Retrieval-augmented generation (RAG) with foundation models has achieved strong performance across diverse tasks, but their capacity for expert-level reasoning-such as solving Olympiad-level physics problems-remains largely unexplored. Inspired by the way students prepare for competitions by reviewing past problems, we investigate the potential of RAG to enhance physics reasoning in foundation models. We introduce PhoPile, a high-quality multimodal dataset specifically designed for Olympiad-level physics, enabling systematic study of retrieval-based reasoning. PhoPile includes diagrams, graphs, and equations, capturing the inherently multimodal nature of physics problem solving. Using PhoPile, we benchmark RAG-augmented foundation models, covering both large language models (LLMs) and large multimodal models (LMMs) with multiple retrievers. Our results demonstrate that integrating retrieval with physics corpora can improve model performance, while also highlighting challenges that motivate further research in retrieval-augmented physics reasoning. 

**Abstract (ZH)**: 基于检索的生成（RAG）与基础模型在多种任务中取得了 strong 表现，但其在专家级推理方面的能力——例如解决奥林匹克级别物理问题——仍待探索。受学生为竞赛复习过往问题的启发，我们研究了 RAG 在增强基础模型物理推理方面的潜力。我们引入了 PhoPile，一个专门用于奥林匹克级别物理的高度质量多模态数据集，使检索基础的推理研究得以系统化。PhoPile 包含图表、图形和方程，捕捉了物理问题解决的固有多模态性质。使用 PhoPile，我们对 RAG 增强的基础模型进行了基准测试，涵盖大型语言模型（LLMs）和具有多个检索器的大型多模态模型（LMMs）。我们的结果表明，将检索与物理语料库结合可以提高模型性能，同时也指出了挑战，进一步推动了检索增强物理推理的研究。 

---
# Reinforcement Learning with Verifiable yet Noisy Rewards under Imperfect Verifiers 

**Title (ZH)**: 可验证但有噪声的奖励强化学习在不完美的验证者下 

**Authors**: Xin-Qiang Cai, Wei Wang, Feng Liu, Tongliang Liu, Gang Niu, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.00915)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) trains policies against automated verifiers to avoid costly human labeling. To reduce vulnerability to verifier hacking, many RLVR systems collapse rewards to binary $\{0,1\}$ during training. This choice carries a cost: it introduces \textit{false negatives} (rejecting correct answers, FNs) and \textit{false positives} (accepting incorrect ones, FPs). For instance, a rule-based checker may mark the correct fraction $\frac{12}{36}$ as wrong when compared against the canonical $\frac{1}{3}$ due to brittle parsing/equivalence rules (FN), while a large language model (LLM) judges can be gamed by superficial cues or even a single adversarial token, yielding inflated correctness for wrong solutions (FP). We formalize verifier unreliability by modeling the verifier as a stochastic reward channel with asymmetric noise rates. From this abstraction, we derive two correction algorithms for verifier errors. The first is a \textit{backward} correction that de-biases the observed binary reward to recover an \textit{unbiased} estimator of the clean policy gradient. The second is a \textit{forward} correction that reweights score-function terms so that the expected update direction aligns with the \textit{clean gradient}; notably, it requires only the FN rate. We implement both as lightweight hooks in a group relative policy optimization (GRPO)-based RLVR pipeline and evaluate them on math-reasoning models and benchmarks. Across models and datasets, both corrections improve over uncorrected training; the forward variant converges faster and remains stable under heavier noise. Finally, we show a practical appeal mechanism in which a lightweight LLM verifier estimates the FN rate online by rechecking rule-based negatives, obtaining outperformance compared with other state-of-the-art contenders. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）：训练策略以对抗自动化验证器避免昂贵的人工标注。通过在训练中将奖励坍缩为二元值{0,1}来减少验证器攻击的脆弱性。这种选择带来了成本：引入了假阴性（拒绝正确答案，FNs）和假阳性（接受错误答案，FPs）。例如，基于规则的检查器可能由于脆弱的解析/等价规则，将正确分数 $\frac{12}{36}$ 错误地标记为错误（FN），而大规模语言模型（LLM）评判者可能通过表象线索甚至单一对抗性标记被操控，导致错误解答被高估为正确（FP）。我们将验证器不可靠性形式化为具有非对称噪声率的随机奖励信道。从这一抽象出发，我们推导出两种校正算法来修正验证器错误。第一个是反向校正，它通过去偏见观察到的二元奖励来恢复干净策略梯度的无偏估计。第二个是正向校正，它重新加权得分函数项，使得期望的更新方向与干净梯度对齐；特别地，它只需要假阴性率。我们将两者实现为基于组相对策略优化（GRPO）的RLVR管道中的轻量级钩子，并在数学推理模型和基准上进行评估。在不同的模型和数据集上，这两种校正都优于未经校正的训练；正向版本收敛更快，并且在更大噪声下保持稳定。最后，我们展示了一个实际的申诉机制，在该机制中，一个轻量级的LLM验证器通过重新检查基于规则的负例估计假阴性率，并取得了优于其他最新竞品的表现。 

---
# RiskPO: Risk-based Policy Optimization via Verifiable Reward for LLM Post-Training 

**Title (ZH)**: RiskPO: 基于风险的策略优化方法通过可验证奖励进行LLM后训练 

**Authors**: Tao Ren, Jinyang Jiang, Hui Yang, Wan Tian, Minhao Zou, Guanghao Li, Zishi Zhang, Qinghao Wang, Shentao Qin, Yanjun Zhao, Rui Tao, Hui Shao, Yijie Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00911)  

**Abstract**: Reinforcement learning with verifiable reward has recently emerged as a central paradigm for post-training large language models (LLMs); however, prevailing mean-based methods, such as Group Relative Policy Optimization (GRPO), suffer from entropy collapse and limited reasoning gains. We argue that these issues stem from overemphasizing high-probability output sequences while neglecting rare but informative reasoning paths. To address these challenges, we propose Risk-based Policy Optimization (RiskPO), which substitutes classical mean-based objectives with principled risk measures. Specifically, we introduce a Mixed Value-at-Risk objective that integrates weighted attention over multiple regions of the reward distribution, thereby amplifying gradient signals on challenging instances and preventing overconfident convergence. We further design a bundling scheme that aggregates multiple questions into bundles, thus enriching the feedback signal and yielding more stable and informative training dynamics. Theoretically, we prove that the risk-averse update alleviates entropy collapse and promotes exploration. Numerically, RiskPO achieves consistent and significant improvements in mathematical reasoning, multi-modal reasoning, and code generation benchmarks, surpassing GRPO and its variants on both Pass@1 and Pass@k metrics. Our results demonstrate that risk-based optimization provides a rigorous and effective paradigm for enhancing LLM reasoning capabilities. 

**Abstract (ZH)**: 具有可验证奖励的强化学习 recently emerged as a central paradigm for post-training large language models (LLMs)；然而，现有的基于均值的方法，如组相对策略优化（GRPO），存在熵坍塌和有限的推理增益问题。我们argue这些问题源于过度强调高概率输出序列而忽视了罕见但有信息量的推理路径。为此，我们提出了一种基于风险的策略优化（RiskPO），它用原则性的风险度量取代了传统的基于均值的目标。具体来说，我们引入了一个混合VaR目标，该目标在奖励分布的多个区域上引入加权注意力，从而增强具有挑战性实例的梯度信号并防止过自信的收敛。我们还设计了一种捆绑方案，将多个问题捆绑在一起，从而丰富反馈信号并产生更稳定和有信息量的训练动力学。从理论上讲，我们证明了风险规避的更新可以缓解熵坍塌并促进探索。从数值上讲，RiskPO 在数学推理、多模态推理和代码生成基准测试中实现了持续且显著的改进，其在Pass@1和Pass@k指标上优于GRPO及其变体。我们的结果表明，基于风险的优化为增强LLM推理能力提供了一个严格而有效的范式。 

---
# "We are not Future-ready": Understanding AI Privacy Risks and Existing Mitigation Strategies from the Perspective of AI Developers in Europe 

**Title (ZH)**: “我们还未准备好迎接未来”:从欧洲AI开发者视角理解AI隐私风险及现有缓解策略 

**Authors**: Alexandra Klymenko, Stephen Meisenbacher, Patrick Gage Kelley, Sai Teja Peddinti, Kurt Thomas, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2510.00909)  

**Abstract**: The proliferation of AI has sparked privacy concerns related to training data, model interfaces, downstream applications, and more. We interviewed 25 AI developers based in Europe to understand which privacy threats they believe pose the greatest risk to users, developers, and businesses and what protective strategies, if any, would help to mitigate them. We find that there is little consensus among AI developers on the relative ranking of privacy risks. These differences stem from salient reasoning patterns that often relate to human rather than purely technical factors. Furthermore, while AI developers are aware of proposed mitigation strategies for addressing these risks, they reported minimal real-world adoption. Our findings highlight both gaps and opportunities for empowering AI developers to better address privacy risks in AI. 

**Abstract (ZH)**: AI的发展引发了对训练数据、模型接口、下游应用等方面的隐私担忧：我们采访了25名基于欧洲的AI开发者，以了解他们认为对用户、开发者和企业构成最大风险的隐私威胁，并探讨是否有防护策略能够缓解这些风险。我们发现，AI开发者在隐私风险的相对排名上缺乏共识。这些差异源于显著的原因模式，往往与人为因素而非纯粹的技术因素相关。此外，尽管AI开发者意识到了应对这些风险的防护策略，但他们报告的实际应用程度很低。我们的研究结果突显了赋能AI开发者更好地应对AI隐私风险的空白与机遇。 

---
# Bridging Language Gaps: Advances in Cross-Lingual Information Retrieval with Multilingual LLMs 

**Title (ZH)**: 跨越语言障碍：多语言大语言模型在跨语言信息检索方面的进展 

**Authors**: Roksana Goworek, Olivia Macmillan-Scott, Eda B. Özyiğit  

**Link**: [PDF](https://arxiv.org/pdf/2510.00908)  

**Abstract**: Cross-lingual information retrieval (CLIR) addresses the challenge of retrieving relevant documents written in languages different from that of the original query. Research in this area has typically framed the task as monolingual retrieval augmented by translation, treating retrieval methods and cross-lingual capabilities in isolation. Both monolingual and cross-lingual retrieval usually follow a pipeline of query expansion, ranking, re-ranking and, increasingly, question answering. Recent advances, however, have shifted from translation-based methods toward embedding-based approaches and leverage multilingual large language models (LLMs), for which aligning representations across languages remains a central challenge. The emergence of cross-lingual embeddings and multilingual LLMs has introduced a new paradigm, offering improved retrieval performance and enabling answer generation. This survey provides a comprehensive overview of developments from early translation-based methods to state-of-the-art embedding-driven and generative techniques. It presents a structured account of core CLIR components, evaluation practices, and available resources. Persistent challenges such as data imbalance and linguistic variation are identified, while promising directions are suggested for advancing equitable and effective cross-lingual information retrieval. By situating CLIR within the broader landscape of information retrieval and multilingual language processing, this work not only reviews current capabilities but also outlines future directions for building retrieval systems that are robust, inclusive, and adaptable. 

**Abstract (ZH)**: 跨语言信息检索（CLIR）解决了使用与原始查询不同语言的文档进行检索的挑战。该领域的研究通常将任务建模为通过翻译增强的单语言检索，并将检索方法和跨语言能力视为独立的部分。单语言和跨语言检索通常遵循查询扩展、排序、再排序以及越来越多的问答的流水线。然而，近期的进步已从基于翻译的方法转向基于嵌入的方法，并利用多语言大规模语言模型（LLMs），其中跨语言表示的对齐仍然是一个核心挑战。跨语言嵌入和多语言LLMs的出现引入了一种新的范式，提高了检索性能并实现了生成答案能力。本文综述了从早期基于翻译的方法到最新的基于嵌入和生成的技术的发展。本文提供了一个结构化的CLIR核心组件、评估实践和可用资源的综述。指出了持续存在的数据不平衡和语言变异等挑战，同时提出了促进公平和有效的跨语言信息检索的发展方向。通过将CLIR置于更广泛的检索和多语言语言处理的背景下，本文不仅回顾了当前的能力，还指出了构建健壮、包容和灵活的检索系统的未来方向。 

---
# TubeDAgger: Reducing the Number of Expert Interventions with Stochastic Reach-Tubes 

**Title (ZH)**: TubeDAgger: 减少专家干预次数的随机可达管方法 

**Authors**: Julian Lemmel, Manuel Kranzl, Adam Lamine, Philipp Neubauer, Radu Grosu, Sophie A. Neubauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.00906)  

**Abstract**: Interactive Imitation Learning deals with training a novice policy from expert demonstrations in an online fashion. The established DAgger algorithm trains a robust novice policy by alternating between interacting with the environment and retraining of the network. Many variants thereof exist, that differ in the method of discerning whether to allow the novice to act or return control to the expert. We propose the use of stochastic reachtubes - common in verification of dynamical systems - as a novel method for estimating the necessity of expert intervention. Our approach does not require fine-tuning of decision thresholds per environment and effectively reduces the number of expert interventions, especially when compared with related approaches that make use of a doubt classification model. 

**Abstract (ZH)**: 基于交互模仿学习的动态系统验证中随机可达管的专家干预估计方法 

---
# Span-level Detection of AI-generated Scientific Text via Contrastive Learning and Structural Calibration 

**Title (ZH)**: 基于对比学习和结构校准的跨句级检测生成科学文本 

**Authors**: Zhen Yin, Shenghua Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00890)  

**Abstract**: The rapid adoption of large language models (LLMs) in scientific writing raises serious concerns regarding authorship integrity and the reliability of scholarly publications. Existing detection approaches mainly rely on document-level classification or surface-level statistical cues; however, they neglect fine-grained span localization, exhibit weak calibration, and often fail to generalize across disciplines and generators. To address these limitations, we present Sci-SpanDet, a structure-aware framework for detecting AI-generated scholarly texts. The proposed method combines section-conditioned stylistic modeling with multi-level contrastive learning to capture nuanced human-AI differences while mitigating topic dependence, thereby enhancing cross-domain robustness. In addition, it integrates BIO-CRF sequence labeling with pointer-based boundary decoding and confidence calibration to enable precise span-level detection and reliable probability estimates. Extensive experiments on a newly constructed cross-disciplinary dataset of 100,000 annotated samples generated by multiple LLM families (GPT, Qwen, DeepSeek, LLaMA) demonstrate that Sci-SpanDet achieves state-of-the-art performance, with F1(AI) of 80.17, AUROC of 92.63, and Span-F1 of 74.36. Furthermore, it shows strong resilience under adversarial rewriting and maintains balanced accuracy across IMRaD sections and diverse disciplines, substantially surpassing existing baselines. To ensure reproducibility and to foster further research on AI-generated text detection in scholarly documents, the curated dataset and source code will be publicly released upon publication. 

**Abstract (ZH)**: 大语言模型在科研写作中的快速应用引发了关于作者身份完整性和学术出版可靠性的严重关切。现有的检测方法主要依赖于文档级别的分类或表面统计线索；然而，它们忽视了细粒度的跨度定位，表现出较差的校准性能，并且往往无法跨领域和生成器推广。为解决这些局限性，我们提出了Sci-SpanDet，这是一种结构感知框架，用于检测AI生成的学术文本。该方法结合了节条件风格建模与多级对比学习，以捕获细腻的人机差异并减轻主题依赖性，从而增强跨域魯棒性。此外，它结合了BIO-CRF序列标注与基于指针的边界解码和置信校准，以实现精确的跨度级别检测和可靠的概率估计。在包含100,000个标注样本的新建跨学科数据集中进行的广泛实验（这些样本由多个LLM家族（GPT、Qwen、DeepSeek、LLaMA）生成），证明Sci-SpanDet达到了最先进的性能，F1(AI)为80.17，AUROC为92.63，Span-F1为74.36。此外，它在对抗性重写下显示出强大的鲁棒性，并在IMRaD节和多种学科中保持了均衡的准确性，大幅优于现有基线。为了确保可重复性并促进对学术文档中AI生成文本检测的进一步研究，精心构建的数据集和源代码将在发表后公开。 

---
# GLAI: GreenLightningAI for Accelerated Training through Knowledge Decoupling 

**Title (ZH)**: GLAI：通过知识解耦加速训练的GreenLightningAI 

**Authors**: Jose I. Mestre, Alberto Fernández-Hernández, Cristian Pérez-Corral, Manuel F. Dolz, Jose Duato, Enrique S. Quintana-Ortí  

**Link**: [PDF](https://arxiv.org/pdf/2510.00883)  

**Abstract**: In this work we introduce GreenLightningAI (GLAI), a new architectural block designed as an alternative to conventional MLPs. The central idea is to separate two types of knowledge that are usually entangled during training: (i) *structural knowledge*, encoded by the stable activation patterns induced by ReLU activations; and (ii) *quantitative knowledge*, carried by the numerical weights and biases. By fixing the structure once stabilized, GLAI reformulates the MLP as a combination of paths, where only the quantitative component is optimized. This reformulation retains the universal approximation capabilities of MLPs, yet achieves a more efficient training process, reducing training time by ~40% on average across the cases examined in this study. Crucially, GLAI is not just another classifier, but a generic block that can replace MLPs wherever they are used, from supervised heads with frozen backbones to projection layers in self-supervised learning or few-shot classifiers. Across diverse experimental setups, GLAI consistently matches or exceeds the accuracy of MLPs with an equivalent number of parameters, while converging faster. Overall, GLAI establishes a new design principle that opens a direction for future integration into large-scale architectures such as Transformers, where MLP blocks dominate the computational footprint. 

**Abstract (ZH)**: GreenLightningAI：一种替代常规MLP的新架构模块 

---
# Advancing Automated Ethical Profiling in SE: a Zero-Shot Evaluation of LLM Reasoning 

**Title (ZH)**: SE中自动化伦理画像的发展：LLM推理的零样本评估 

**Authors**: Patrizio Migliarini, Mashal Afzal Memon, Marco Autili, Paola Inverardi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00881)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into software engineering (SE) tools for tasks that extend beyond code synthesis, including judgment under uncertainty and reasoning in ethically significant contexts. We present a fully automated framework for assessing ethical reasoning capabilities across 16 LLMs in a zero-shot setting, using 30 real-world ethically charged scenarios. Each model is prompted to identify the most applicable ethical theory to an action, assess its moral acceptability, and explain the reasoning behind their choice. Responses are compared against expert ethicists' choices using inter-model agreement metrics. Our results show that LLMs achieve an average Theory Consistency Rate (TCR) of 73.3% and Binary Agreement Rate (BAR) on moral acceptability of 86.7%, with interpretable divergences concentrated in ethically ambiguous cases. A qualitative analysis of free-text explanations reveals strong conceptual convergence across models despite surface-level lexical diversity. These findings support the potential viability of LLMs as ethical inference engines within SE pipelines, enabling scalable, auditable, and adaptive integration of user-aligned ethical reasoning. Our focus is the Ethical Interpreter component of a broader profiling pipeline: we evaluate whether current LLMs exhibit sufficient interpretive stability and theory-consistent reasoning to support automated profiling. 

**Abstract (ZH)**: 大型语言模型（LLMs）在软件工程（SE）工具中的伦理推理能力评估：基于30个真实伦理情境的零样本设置 

---
# A Technique Based on Trade-off Maps to Visualise and Analyse Relationships Between Objectives in Optimisation Problems 

**Title (ZH)**: 基于权衡图的技术：用于优化问题中目标间关系的可视化与分析方法 

**Authors**: Rodrigo Lankaites Pinheiro, Dario Landa-Silva, Jason Atkin  

**Link**: [PDF](https://arxiv.org/pdf/2510.00877)  

**Abstract**: Understanding the relationships between objectives in a multiobjective optimisation problem is important for developing tailored and efficient solving techniques. In particular, when tackling combinatorial optimisation problems with many objectives, that arise in real-world logistic scenarios, better support for the decision maker can be achieved through better understanding of the often complex fitness landscape. This paper makes a contribution in this direction by presenting a technique that allows a visualisation and analysis of the local and global relationships between objectives in optimisation problems with many objectives. The proposed technique uses four steps: First, the global pairwise relationships are analysed using the Kendall correlation method; then, the ranges of the values found on the given Pareto front are estimated and assessed; next, these ranges are used to plot a map using Gray code, similar to Karnaugh maps, that has the ability to highlight the trade-offs between multiple objectives; and finally, local relationships are identified using scatter plots. Experiments are presented for three combinatorial optimisation problems: multiobjective multidimensional knapsack problem, multiobjective nurse scheduling problem, and multiobjective vehicle routing problem with time windows . Results show that the proposed technique helps in the gaining of insights into the problem difficulty arising from the relationships between objectives. 

**Abstract (ZH)**: 多目标优化问题中目标间关系的理解对于开发定制化和高效的求解技术至关重要。特别是在处理多个目标的组合优化问题时，通过对通常复杂的fitness景观的更好理解，可以为决策者提供更好的支持。本文在此方向上作出贡献，通过提出一种技术来可视化和分析多目标优化问题中局部和全局的目标关系。提出的该技术通过四个步骤进行：首先，使用Kendall相关性方法分析全局成对关系；然后，估计并评估给定Pareto前沿上找到的值的范围；接着，使用Gray代码绘制类似Karnaugh图的地图，能够突出多目标之间的权衡；最后，使用散点图识别局部关系。实验结果表明，所提出的技术有助于理解由目标间关系引起的问题难度。 

---
# Gather-Scatter Mamba: Accelerating Propagation with Efficient State Space Model 

**Title (ZH)**: Gather-Scatter Mamba: 加速状态空间模型下的传播计算 

**Authors**: Hyun-kyu Ko, Youbin Kim, Jihyeon Park, Dongheok Park, Gyeongjin Kang, Wonjun Cho, Hyung Yi, Eunbyung Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.00862)  

**Abstract**: State Space Models (SSMs)-most notably RNNs-have historically played a central role in sequential modeling. Although attention mechanisms such as Transformers have since dominated due to their ability to model global context, their quadratic complexity and limited scalability make them less suited for long sequences. Video super-resolution (VSR) methods have traditionally relied on recurrent architectures to propagate features across frames. However, such approaches suffer from well-known issues including vanishing gradients, lack of parallelism, and slow inference speed. Recent advances in selective SSMs like Mamba offer a compelling alternative: by enabling input-dependent state transitions with linear-time complexity, Mamba mitigates these issues while maintaining strong long-range modeling capabilities. Despite this potential, Mamba alone struggles to capture fine-grained spatial dependencies due to its causal nature and lack of explicit context aggregation. To address this, we propose a hybrid architecture that combines shifted window self-attention for spatial context aggregation with Mamba-based selective scanning for efficient temporal propagation. Furthermore, we introduce Gather-Scatter Mamba (GSM), an alignment-aware mechanism that warps features toward a center anchor frame within the temporal window before Mamba propagation and scatters them back afterward, effectively reducing occlusion artifacts and ensuring effective redistribution of aggregated information across all frames. The official implementation is provided at: this https URL. 

**Abstract (ZH)**: 基于状态空间模型（SSMs）如RNNs的方法在序列建模中历来扮演着中心角色。尽管Transformer等注意力机制因其能够建模全局上下文而占据主导地位，但由于其二次复杂性和有限的可扩展性，它们并不适合长序列。视频超分辨率（VSR）方法传统上依赖递归架构在帧间传播特征。然而，此类方法面临着vanishing gradients、缺乏并行性和慢速推理速度等已知问题。近来，基于选择性SSMs的方法如Mamba提出了具有吸引力的替代方案：通过使用线性时间复杂度实现输入依赖的状态转换，Mamba缓解了这些问题同时保持了强大的长距离建模能力。尽管如此，Mamba单独使用时难以捕捉细粒度的空间依赖性，这主要是由于其因果性质和显式上下文聚合的缺失。为此，我们提出了一种混合架构，结合了移窗自注意力机制进行空间上下文聚合，并利用基于Mamba的选择性扫描进行高效的时序传播。此外，我们引入了一种对齐感知机制Gather-Scatter Mamba（GSM），该机制在Mamba传播前将特征向时间窗内的中心锚帧进行扭曲，并在传播后将它们散射回原位，从而有效减少了遮挡伪影并确保所有帧间聚集信息的有效再分配。官方实现代码可在以下链接获取：this https URL。 

---
# Erase to Improve: Erasable Reinforcement Learning for Search-Augmented LLMs 

**Title (ZH)**: 擦除以提升：可擦除强化学习在搜索增强的大语言模型中的应用 

**Authors**: Ziliang Wang, Kang An, Xuhui Zheng, Faqiang Qian, Weikun Zhang, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00861)  

**Abstract**: While search-augmented large language models (LLMs) exhibit impressive capabilities, their reliability in complex multi-hop reasoning remains limited. This limitation arises from three fundamental challenges: decomposition errors, where tasks are incorrectly broken down; retrieval missing, where key evidence fails to be retrieved; and reasoning errors, where flawed logic propagates through the reasoning chain. A single failure in any of these stages can derail the final answer. We propose Erasable Reinforcement Learning (ERL), a novel framework that transforms fragile reasoning into a robust process. ERL explicitly identifies faulty steps, erases them, and regenerates reasoning in place, preventing defective logic from propagating through the reasoning chain. This targeted correction mechanism turns brittle reasoning into a more resilient process. Models trained with ERL, termed ESearch, achieve substantial improvements on HotpotQA, MuSiQue, 2Wiki, and Bamboogle, with the 3B model achieving +8.48% EM and +11.56% F1, and the 7B model achieving +5.38% EM and +7.22% F1 over previous state-of-the-art(SOTA) results. These findings suggest that erasable reinforcement learning provides a powerful paradigm shift for robust multi-step reasoning in LLMs. 

**Abstract (ZH)**: 增强搜索的大语言模型虽然表现出色，但在复杂多跳推理方面的可靠性仍有限。Erasable Reinforcement Learning (ERL)：增强多步推理的新型框架及其应用 

---
# Can World Models Benefit VLMs for World Dynamics? 

**Title (ZH)**: 世界模型能为世界动力学提供益处吗？ 

**Authors**: Kevin Zhang, Kuangzhi Ge, Xiaowei Chi, Renrui Zhang, Shaojun Shi, Zhen Dong, Sirui Han, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00855)  

**Abstract**: Trained on internet-scale video data, generative world models are increasingly recognized as powerful world simulators that can generate consistent and plausible dynamics over structure, motion, and physics. This raises a natural question: with the advent of strong video foundational models, might they supplant conventional vision encoder paradigms for general-purpose multimodal understanding? While recent studies have begun to explore the potential of world models on common vision tasks, these explorations typically lack a systematic investigation of generic, multimodal tasks. In this work, we strive to investigate the capabilities when world model priors are transferred into Vision-Language Models: we re-purpose a video diffusion model as a generative encoder to perform a single denoising step and treat the resulting latents as a set of visual embedding. We empirically investigate this class of models, which we refer to as World-Language Models (WorldLMs), and we find that generative encoders can capture latents useful for downstream understanding that show distinctions from conventional encoders. Naming our best-performing variant Dynamic Vision Aligner (DyVA), we further discover that this method significantly enhances spatial reasoning abilities and enables single-image models to perform multi-frame reasoning. Through the curation of a suite of visual reasoning tasks, we find DyVA to surpass both open-source and proprietary baselines, achieving state-of-the-art or comparable performance. We attribute these gains to WorldLM's inherited motion-consistency internalization from video pre-training. Finally, we systematically explore extensive model designs to highlight promising directions for future work. We hope our study can pave the way for a new family of VLMs that leverage priors from world models and are on a promising path towards generalist vision learners. 

**Abstract (ZH)**: 基于互联网规模视频数据训练的生成世界模型被越来越多地视为强大的世界模拟器，能够生成一致且合理的结构、运动和物理的动力学。这引发了一个自然的问题：随着强大的视频基础模型的出现，它们是否会取代通用多模态理解中的传统视觉编码范式？虽然近年来的研究开始探索世界模型在通用视觉任务上的潜力，但这些探索通常缺乏对通用多模态任务的系统性研究。在本项工作中，我们致力于研究当世界模型先验知识转移到视觉语言模型中的能力：我们重新利用一个视频扩散模型作为生成编码器进行单步去噪，并将得到的潜在变量视为一组视觉嵌入。我们实证研究了这类模型，称之为世界语言模型（WorldLMs），并发现生成编码器可以捕捉到对下游理解有用的潜在变量，这些潜在变量与传统编码器存在区别。我们将其性能最佳的版本命名为动态视觉对齐器（DyVA），进一步发现这种方法显著增强了空间推理能力，并使单帧模型能够进行多帧推理。通过精心设计一组视觉推理任务，我们发现DyVA超越了开源和专有基准模型，实现了最先进的或可比的性能。我们将这些收益归因于世界语言模型继承的视频预训练中运动一致性的内在化。最后，我们系统地探索了广泛的模型设计，指出了未来工作的有希望的方向。我们希望我们的研究能够为利用世界模型先验知识的新型视觉语言模型铺平道路，并且这些模型正朝着通用视觉学习者的方向前进。 

---
# Mechanistic Interpretability as Statistical Estimation: A Variance Analysis of EAP-IG 

**Title (ZH)**: 机制可解释性作为一种统计估计：EAP-IG的方差分析 

**Authors**: Maxime Méloux, Maxime Peyrard, François Portet  

**Link**: [PDF](https://arxiv.org/pdf/2510.00845)  

**Abstract**: The development of trustworthy artificial intelligence requires moving beyond black-box performance metrics toward an understanding of models' internal computations. Mechanistic Interpretability (MI) aims to meet this need by identifying the algorithmic mechanisms underlying model behaviors. Yet, the scientific rigor of MI critically depends on the reliability of its findings. In this work, we argue that interpretability methods, such as circuit discovery, should be viewed as statistical estimators, subject to questions of variance and robustness. To illustrate this statistical framing, we present a systematic stability analysis of a state-of-the-art circuit discovery method: EAP-IG. We evaluate its variance and robustness through a comprehensive suite of controlled perturbations, including input resampling, prompt paraphrasing, hyperparameter variation, and injected noise within the causal analysis itself. Across a diverse set of models and tasks, our results demonstrate that EAP-IG exhibits high structural variance and sensitivity to hyperparameters, questioning the stability of its findings. Based on these results, we offer a set of best-practice recommendations for the field, advocating for the routine reporting of stability metrics to promote a more rigorous and statistically grounded science of interpretability. 

**Abstract (ZH)**: 可信人工智能的发展需要超越黑箱性能指标，转向理解模型内部计算。机制可解释性（MI）旨在通过识别模型行为背后的算法机制来满足这一需求。然而，MI的科学严谨性关键取决于其发现的可靠性。在本文中，我们argue that解释方法，如电路发现，应被视为统计估计量，受方差和稳健性问题的影响。为了说明这种统计框架，我们对一种最先进的电路发现方法EAP-IG进行了系统的稳定性分析。我们通过一系列受控扰动进行全面评估，包括输入重新采样、指令改写、超参数变化以及因果分析中的注入噪声。在多种模型和任务上，我们的结果表明，EAP-IG表现出高度的结构方差和对超参数的敏感性，质疑其发现的稳定性。基于这些结果，我们为该领域提供了一套最佳实践建议，提倡常规报告稳定性指标，促进更严谨和统计基础的解释性科学。 

---
# Feature Identification for Hierarchical Contrastive Learning 

**Title (ZH)**: 层次对比学习中的特征识别 

**Authors**: Julius Ott, Nastassia Vysotskaya, Huawei Sun, Lorenzo Servadei, Robert Wille  

**Link**: [PDF](https://arxiv.org/pdf/2510.00837)  

**Abstract**: Hierarchical classification is a crucial task in many applications, where objects are organized into multiple levels of categories. However, conventional classification approaches often neglect inherent inter-class relationships at different hierarchy levels, thus missing important supervisory signals. Thus, we propose two novel hierarchical contrastive learning (HMLC) methods. The first, leverages a Gaussian Mixture Model (G-HMLC) and the second uses an attention mechanism to capture hierarchy-specific features (A-HMLC), imitating human processing. Our approach explicitly models inter-class relationships and imbalanced class distribution at higher hierarchy levels, enabling fine-grained clustering across all hierarchy levels. On the competitive CIFAR100 and ModelNet40 datasets, our method achieves state-of-the-art performance in linear evaluation, outperforming existing hierarchical contrastive learning methods by 2 percentage points in terms of accuracy. The effectiveness of our approach is backed by both quantitative and qualitative results, highlighting its potential for applications in computer vision and beyond. 

**Abstract (ZH)**: 层次分类是许多应用中的关键任务，其中对象被组织成多个层次的类别。然而，传统的分类方法往往忽略了不同层次间的固有类间关系，从而错过了重要的监督信号。因此，我们提出了两种新颖的层次对比学习（HMLC）方法。第一种方法利用高斯混合模型（G-HMLC），第二种方法使用注意力机制捕获层次特定特征（A-HMLC），模拟人类处理方式。我们的方法明确建模了高层次类别间的相互关系和类别不平衡分布，从而在所有层次上实现了精细聚类。在竞争性的CIFAR100和ModelNet40数据集上，我们的方法在线性评估中取得了最先进的性能，在准确率方面比现有层次对比学习方法高出2个百分点。我们的方法的有效性由定量和定性结果支持，突显了其在计算机视觉等领域应用的潜力。 

---
# Towards Verifiable Federated Unlearning: Framework, Challenges, and The Road Ahead 

**Title (ZH)**: 可验证联邦卸载: 框架、挑战及未来之路 

**Authors**: Thanh Linh Nguyen, Marcela Tuler de Oliveira, An Braeken, Aaron Yi Ding, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2510.00833)  

**Abstract**: Federated unlearning (FUL) enables removing the data influence from the model trained across distributed clients, upholding the right to be forgotten as mandated by privacy regulations. FUL facilitates a value exchange where clients gain privacy-preserving control over their data contributions, while service providers leverage decentralized computing and data freshness. However, this entire proposition is undermined because clients have no reliable way to verify that their data influence has been provably removed, as current metrics and simple notifications offer insufficient assurance. We envision unlearning verification becoming a pivotal and trust-by-design part of the FUL life-cycle development, essential for highly regulated and data-sensitive services and applications like healthcare. This article introduces veriFUL, a reference framework for verifiable FUL that formalizes verification entities, goals, approaches, and metrics. Specifically, we consolidate existing efforts and contribute new insights, concepts, and metrics to this domain. Finally, we highlight research challenges and identify potential applications and developments for verifiable FUL and veriFUL. 

**Abstract (ZH)**: 联邦遗忘（FUL）使客户端能够从分布式训练模型中去除其数据的影响，遵守隐私法规规定的数据被遗忘的权利。FUL 促进了数据贡献方在保护隐私的情况下对数据贡献的控制权，同时也使服务提供商能够利用分布式计算和数据新鲜度。然而，这一主张因客户端缺乏可靠的方法来验证其数据影响是否已被确凿地去除而受到削弱，当前的度量标准和简单通知提供的保障不足。我们设想可验证性遗忘验证将成为FUL生命周期发展中的关键且设计即信任的一部分，对于如医疗保健等高度监管和数据敏感的服务与应用至关重要。本文介绍了一种名为veriFUL的可验证FUL参考框架，正式化了验证实体、目标、方法和度量。我们具体整合了现有努力，并在此领域贡献了新的见解、概念和度量标准。最后，我们指出了研究挑战，并确定了可验证FUL和veriFUL的潜在应用和发展方向。 

---
# Stabilizing Policy Gradients for Sample-Efficient Reinforcement Learning in LLM Reasoning 

**Title (ZH)**: 稳定策略梯度以实现高效样例在大规模语言模型推理中的强化学习 

**Authors**: Luckeciano C. Melo, Alessandro Abate, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2510.00819)  

**Abstract**: Reinforcement Learning, particularly through policy gradient methods, has played a central role in enabling reasoning capabilities of Large Language Models. However, the optimization stability of policy gradients in this setting remains understudied. As a result, existing implementations often resort to conservative hyperparameter choices to ensure stability, which requires more training samples and increases computational costs. Hence, developing models for reliably tracking the underlying optimization dynamics and leveraging them into training enables more sample-efficient regimes and further unleashes scalable post-training. We address this gap by formalizing the stochastic optimization problem of policy gradients with explicit consideration of second-order geometry. We propose a tractable computational framework that tracks and leverages curvature information during policy updates. We further employ this framework to design interventions in the optimization process through data selection. The resultant algorithm, Curvature-Aware Policy Optimization (CAPO), identifies samples that contribute to unstable updates and masks them out. Theoretically, we establish monotonic improvement guarantees under realistic assumptions. On standard math reasoning benchmarks, we empirically show that CAPO ensures stable updates under aggressive learning regimes where baselines catastrophically fail. With minimal intervention (rejecting fewer than 8% of tokens), CAPO achieves up to 30x improvement in sample efficiency over standard GRPO for LLM reasoning. 

**Abstract (ZH)**: 强化学习，特别是通过策略梯度方法，已经在使大型语言模型具备推理能力方面发挥了核心作用。然而，该设置下策略梯度的优化稳定性研究仍然不足。因此，现有的实现往往依赖保守的超参数选择以确保稳定，这需要更多的训练样本并增加计算成本。因此，开发能够可靠跟踪其下的优化动力学并利用它们进行训练的模型能够实现更高效的样本利用并进一步提高可扩展性。我们通过明确考虑二阶几何来形式化策略梯度的随机优化问题，提出了一个可计算的框架，在策略更新过程中跟踪并利用曲率信息。进一步地，我们利用此框架通过数据选择在优化过程中设计干预措施。所提出的算法，曲率感知策略优化（CAPO），识别出导致不稳定更新的样本并将其屏蔽。理论上，我们在现实假设下建立了单调改进保证。在标准数学推理基准上，我们实验证明，在基线算法可能灾难性失败的积极学习环境中，CAPO 能确保稳定更新。通过最小的干预（拒绝不到 8% 的标记），CAPO 在大型语言模型推理中的样本效率上相比标准 GRPO 提高了多达 30 倍。 

---
# What You See is What You Ask: Evaluating Audio Descriptions 

**Title (ZH)**: 你所听即你所问：评估音频描述 

**Authors**: Divy Kala, Eshika Khandelwal, Makarand Tapaswi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00808)  

**Abstract**: Audio descriptions (ADs) narrate important visual details in movies, enabling Blind and Low Vision (BLV) users to understand narratives and appreciate visual details. Existing works in automatic AD generation mostly focus on few-second trimmed clips, and evaluate them by comparing against a single ground-truth reference AD. However, writing ADs is inherently subjective. Through alignment and analysis of two independent AD tracks for the same movies, we quantify the subjectivity in when and whether to describe, and what and how to highlight. Thus, we show that working with trimmed clips is inadequate. We propose ADQA, a QA benchmark that evaluates ADs at the level of few-minute long, coherent video segments, testing whether they would help BLV users understand the story and appreciate visual details. ADQA features visual appreciation (VA) questions about visual facts and narrative understanding (NU) questions based on the plot. Through ADQA, we show that current AD generation methods lag far behind human-authored ADs. We conclude with several recommendations for future work and introduce a public leaderboard for benchmarking. 

**Abstract (ZH)**: 基于音频描述的回答质量评估（ADQA）：评估电影中多分钟连续视频段的视觉细节理解和叙事理解 

---
# MG2FlowNet: Accelerating High-Reward Sample Generation via Enhanced MCTS and Greediness Control 

**Title (ZH)**: MG2FlowNet: 通过增强的MCTS和贪婪性控制加速高奖励样本生成 

**Authors**: Rui Zhu, Xuan Yu, Yudong Zhang, Chen Zhang, Xu Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00805)  

**Abstract**: Generative Flow Networks (GFlowNets) have emerged as a powerful tool for generating diverse and high-reward structured objects by learning to sample from a distribution proportional to a given reward function. Unlike conventional reinforcement learning (RL) approaches that prioritize optimization of a single trajectory, GFlowNets seek to balance diversity and reward by modeling the entire trajectory distribution. This capability makes them especially suitable for domains such as molecular design and combinatorial optimization. However, existing GFlowNets sampling strategies tend to overexplore and struggle to consistently generate high-reward samples, particularly in large search spaces with sparse high-reward regions. Therefore, improving the probability of generating high-reward samples without sacrificing diversity remains a key challenge under this premise. In this work, we integrate an enhanced Monte Carlo Tree Search (MCTS) into the GFlowNets sampling process, using MCTS-based policy evaluation to guide the generation toward high-reward trajectories and Polynomial Upper Confidence Trees (PUCT) to balance exploration and exploitation adaptively, and we introduce a controllable mechanism to regulate the degree of greediness. Our method enhances exploitation without sacrificing diversity by dynamically balancing exploration and reward-driven guidance. The experimental results show that our method can not only accelerate the speed of discovering high-reward regions but also continuously generate high-reward samples, while preserving the diversity of the generative distribution. All implementations are available at this https URL. 

**Abstract (ZH)**: 生成流网络（GFlowNets）作为一种通过学习从给定奖励函数成比例的分布中采样来生成多样化和高奖励结构化对象的强大工具而 emerge，尤其适用于分子设计和组合优化等领域。然而，现有的 GFlowNets 抽样策略倾向于过度探索，并且在稀疏高奖励区域广泛存在的大型搜索空间中难以一致地生成高奖励样本。因此，在不牺牲多样性的前提下提高生成高奖励样本的概率仍然是一个关键挑战。本文将增强的蒙特卡罗树搜索（MCTS）集成到 GFlowNets 的抽样过程中，使用基于 MCTS 的策略评估引导生成向高奖励轨迹，并通过多项式上置信树（PUCT）自适应地平衡探索与利用，同时引入可控机制调节贪婪程度。通过动态平衡探索和奖励驱动的引导，我们的方法能够在不牺牲多样性的情况下增强利用。实验证明，我们的方法不仅能加速高奖励区域的发现速度，还能连续生成高奖励样本，同时保持生成分布的多样性。所有实现均可在以下链接获取：this https URL。 

---
# Fast, Secure, and High-Capacity Image Watermarking with Autoencoded Text Vectors 

**Title (ZH)**: 基于自编码文本向量的快速、安全且高容量图像水印算法 

**Authors**: Gautier Evennou, Vivien Chappelier, Ewa Kijak  

**Link**: [PDF](https://arxiv.org/pdf/2510.00799)  

**Abstract**: Most image watermarking systems focus on robustness, capacity, and imperceptibility while treating the embedded payload as meaningless bits. This bit-centric view imposes a hard ceiling on capacity and prevents watermarks from carrying useful information. We propose LatentSeal, which reframes watermarking as semantic communication: a lightweight text autoencoder maps full-sentence messages into a compact 256-dimensional unit-norm latent vector, which is robustly embedded by a finetuned watermark model and secured through a secret, invertible rotation. The resulting system hides full-sentence messages, decodes in real time, and survives valuemetric and geometric attacks. It surpasses prior state of the art in BLEU-4 and Exact Match on several benchmarks, while breaking through the long-standing 256-bit payload ceiling. It also introduces a statistically calibrated score that yields a ROC AUC score of 0.97-0.99, and practical operating points for deployment. By shifting from bit payloads to semantic latent vectors, LatentSeal enables watermarking that is not only robust and high-capacity, but also secure and interpretable, providing a concrete path toward provenance, tamper explanation, and trustworthy AI governance. Models, training and inference code, and data splits will be available upon publication. 

**Abstract (ZH)**: LatentSeal: Semantic Watermarking for Robust and Secure Full-Sentence Embedding 

---
# Solar PV Installation Potential Assessment on Building Facades Based on Vision and Language Foundation Models 

**Title (ZH)**: 基于视觉与语言基础模型的建筑外墙太阳能光伏安装潜力评估 

**Authors**: Ruyu Liu, Dongxu Zhuang, Jianhua Zhang, Arega Getaneh Abate, Per Sieverts Nielsen, Ben Wang, Xiufeng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00797)  

**Abstract**: Building facades represent a significant untapped resource for solar energy generation in dense urban environments, yet assessing their photovoltaic (PV) potential remains challenging due to complex geometries and semantic com ponents. This study introduces SF-SPA (Semantic Facade Solar-PV Assessment), an automated framework that transforms street-view photographs into quantitative PV deployment assessments. The approach combines com puter vision and artificial intelligence techniques to address three key challenges: perspective distortion correction, semantic understanding of facade elements, and spatial reasoning for PV layout optimization. Our four-stage pipeline processes images through geometric rectification, zero-shot semantic segmentation, Large Language Model (LLM) guided spatial reasoning, and energy simulation. Validation across 80 buildings in four countries demonstrates ro bust performance with mean area estimation errors of 6.2% &#177; 2.8% compared to expert annotations. The auto mated assessment requires approximately 100 seconds per building, a substantial gain in efficiency over manual methods. Simulated energy yield predictions confirm the method's reliability and applicability for regional poten tial studies, urban energy planning, and building-integrated photovoltaic (BIPV) deployment. Code is available at: https:github.com/CodeAXu/Solar-PV-Installation 

**Abstract (ZH)**: 基于语义的外墙光伏评估：SF-SPA框架研究 

---
# MetaLogic: Robustness Evaluation of Text-to-Image Models via Logically Equivalent Prompts 

**Title (ZH)**: MetaLogic：通过逻辑等价提示评估文本到图像模型的稳健性 

**Authors**: Yifan Shen, Yangyang Shu, Hye-young Paik, Yulei Sui  

**Link**: [PDF](https://arxiv.org/pdf/2510.00796)  

**Abstract**: Recent advances in text-to-image (T2I) models, especially diffusion-based architectures, have significantly improved the visual quality of generated images. However, these models continue to struggle with a critical limitation: maintaining semantic consistency when input prompts undergo minor linguistic variations. Despite being logically equivalent, such prompt pairs often yield misaligned or semantically inconsistent images, exposing a lack of robustness in reasoning and generalisation. To address this, we propose MetaLogic, a novel evaluation framework that detects T2I misalignment without relying on ground truth images. MetaLogic leverages metamorphic testing, generating image pairs from prompts that differ grammatically but are semantically identical. By directly comparing these image pairs, the framework identifies inconsistencies that signal failures in preserving the intended meaning, effectively diagnosing robustness issues in the model's logic understanding. Unlike existing evaluation methods that compare a generated image to a single prompt, MetaLogic evaluates semantic equivalence between paired images, offering a scalable, ground-truth-free approach to identifying alignment failures. It categorises these alignment errors (e.g., entity omission, duplication, positional misalignment) and surfaces counterexamples that can be used for model debugging and refinement. We evaluate MetaLogic across multiple state-of-the-art T2I models and reveal consistent robustness failures across a range of logical constructs. We find that even the SOTA text-to-image models like this http URL and DALLE-3 demonstrate a 59 percent and 71 percent misalignment rate, respectively. Our results show that MetaLogic is not only efficient and scalable, but also effective in uncovering fine-grained logical inconsistencies that are overlooked by existing evaluation metrics. 

**Abstract (ZH)**: Recent Advances in Text-to-Image Models: Addressing Semantic Consistency Challenges with MetaLogic 

---
# Uncertainty-Aware Concept Bottleneck Models with Enhanced Interpretability 

**Title (ZH)**: 具有增强可解释性的不确定性意识概念瓶颈模型 

**Authors**: Haifei Zhang, Patrick Barry, Eduardo Brandao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00773)  

**Abstract**: In the context of image classification, Concept Bottleneck Models (CBMs) first embed images into a set of human-understandable concepts, followed by an intrinsically interpretable classifier that predicts labels based on these intermediate representations. While CBMs offer a semantically meaningful and interpretable classification pipeline, they often sacrifice predictive performance compared to end-to-end convolutional neural networks. Moreover, the propagation of uncertainty from concept predictions to final label decisions remains underexplored. In this paper, we propose a novel uncertainty-aware and interpretable classifier for the second stage of CBMs. Our method learns a set of binary class-level concept prototypes and uses the distances between predicted concept vectors and each class prototype as both a classification score and a measure of uncertainty. These prototypes also serve as interpretable classification rules, indicating which concepts should be present in an image to justify a specific class prediction. The proposed framework enhances both interpretability and robustness by enabling conformal prediction for uncertain or outlier inputs based on their deviation from the learned binary class-level concept prototypes. 

**Abstract (ZH)**: 基于图像分类的不确定性感知和可解释分类器研究 

---
# UniverSR: Unified and Versatile Audio Super-Resolution via Vocoder-Free Flow Matching 

**Title (ZH)**: UniverSR: 无需 vocoder 的统一高效音频超分辨率方法 

**Authors**: Woongjib Choi, Sangmin Lee, Hyungseob Lim, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00771)  

**Abstract**: In this paper, we present a vocoder-free framework for audio super-resolution that employs a flow matching generative model to capture the conditional distribution of complex-valued spectral coefficients. Unlike conventional two-stage diffusion-based approaches that predict a mel-spectrogram and then rely on a pre-trained neural vocoder to synthesize waveforms, our method directly reconstructs waveforms via the inverse Short-Time Fourier Transform (iSTFT), thereby eliminating the dependence on a separate vocoder. This design not only simplifies end-to-end optimization but also overcomes a critical bottleneck of two-stage pipelines, where the final audio quality is fundamentally constrained by vocoder performance. Experiments show that our model consistently produces high-fidelity 48 kHz audio across diverse upsampling factors, achieving state-of-the-art performance on both speech and general audio datasets. 

**Abstract (ZH)**: 基于流匹配生成模型的无 vocoder 音频超分辨率框架 

---
# Multi-Objective Task-Aware Predictor for Image-Text Alignment 

**Title (ZH)**: 面向多目标任务的图像-文本对齐预测器 

**Authors**: Eunki Kim, Na Min An, James Thorne, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2510.00766)  

**Abstract**: Evaluating image-text alignment while reflecting human preferences across multiple aspects is a significant issue for the development of reliable vision-language applications. It becomes especially crucial in real-world scenarios where multiple valid descriptions exist depending on contexts or user needs. However, research progress is hindered by the lack of comprehensive benchmarks and existing evaluation predictors lacking at least one of these key properties: (1) Alignment with human judgments, (2) Long-sequence processing, (3) Inference efficiency, and (4) Applicability to multi-objective scoring. To address these challenges, we propose a plug-and-play architecture to build a robust predictor, MULTI-TAP (Multi-Objective Task-Aware Predictor), capable of both multi and single-objective scoring. MULTI-TAP can produce a single overall score, utilizing a reward head built on top of a large vision-language model (LVLMs). We show that MULTI-TAP is robust in terms of application to different LVLM architectures, achieving significantly higher performance than existing metrics and even on par with the GPT-4o-based predictor, G-VEval, with a smaller size (7-8B). By training a lightweight ridge regression layer on the frozen hidden states of a pre-trained LVLM, MULTI-TAP can produce fine-grained scores for multiple human-interpretable objectives. MULTI-TAP performs better than VisionREWARD, a high-performing multi-objective reward model, in both performance and efficiency on multi-objective benchmarks and our newly released text-image-to-text dataset, EYE4ALL. Our new dataset, consisting of chosen/rejected human preferences (EYE4ALLPref) and human-annotated fine-grained scores across seven dimensions (EYE4ALLMulti), can serve as a foundation for developing more accessible AI systems by capturing the underlying preferences of users, including blind and low-vision (BLV) individuals. 

**Abstract (ZH)**: 评估多方面反映人类偏好的图像-文本对齐对于可靠视觉语言应用的发展是一项重要问题。在多种有效描述依赖于上下文或用户需求的现实场景中，这一问题变得尤为重要。然而，研究进展受限于缺乏全面基准和现有评价预测器至少缺乏一种关键属性：（1）与人类判断的一致性，（2）长序列处理能力，（3）推理效率，以及（4）多目标评分的适用性。为应对这些挑战，我们提出了一种插件式架构来构建一个稳健的预测器MULTI-TAP（多目标任务感知预测器），该预测器能够进行多目标和单目标评分。MULTI-TAP可以生成一个综合评分，利用大型视觉语言模型（LVLMs）顶部构建的奖励头实现。我们展示了MULTI-TAP在不同LVLM架构中的稳健性，其性能显著高于现有指标，甚至在规模较小（7-8B）的情况下与基于GPT-4o的预测器G-VEval相当。通过在预训练LVLM的冻结隐藏状态上训练一个轻量级岭回归层，MULTI-TAP能够为多个可解释的人类目标生成细粒度评分。在多目标基准和我们新发布的文本-图像到文本数据集EYE4ALL上，MULTI-TAP在性能和效率方面均优于高性能的多目标奖励模型VisionREWARD。我们的新数据集EYE4ALLPref和EYE4ALLMulti分别由选择/拒绝的人类偏好和七个维度上的细粒度人工注释评分组成，可以作为开发更易用的AI系统的基础，以捕捉用户的潜在偏好，包括盲人和低视力个体。 

---
# From Scores to Preferences: Redefining MOS Benchmarking for Speech Quality Reward Modeling 

**Title (ZH)**: 从评分到偏好：重新定义MOS基准以适应语音质量奖励建模 

**Authors**: Yifei Cao, Changhao Jiang, Jiabao Zhuang, Jiajun Sun, Ming Zhang, Zhiheng Xi, Hui Li, Shihan Dou, Yuran Wang, Yunke Zhang, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00743)  

**Abstract**: Assessing the perceptual quality of synthetic speech is crucial for guiding the development and refinement of speech generation models. However, it has traditionally relied on human subjective ratings such as the Mean Opinion Score (MOS), which depend on manual annotations and often suffer from inconsistent rating standards and poor reproducibility. To address these limitations, we introduce MOS-RMBench, a unified benchmark that reformulates diverse MOS datasets into a preference-comparison setting, enabling rigorous evaluation across different datasets. Building on MOS-RMBench, we systematically construct and evaluate three paradigms for reward modeling: scalar reward models, semi-scalar reward models, and generative reward models (GRMs). Our experiments reveal three key findings: (1) scalar models achieve the strongest overall performance, consistently exceeding 74% accuracy; (2) most models perform considerably worse on synthetic speech than on human speech; and (3) all models struggle on pairs with very small MOS differences. To improve performance on these challenging pairs, we propose a MOS-aware GRM that incorporates an MOS-difference-based reward function, enabling the model to adaptively scale rewards according to the difficulty of each sample pair. Experimental results show that the MOS-aware GRM significantly improves fine-grained quality discrimination and narrows the gap with scalar models on the most challenging cases. We hope this work will establish both a benchmark and a methodological framework to foster more rigorous and scalable research in automatic speech quality assessment. 

**Abstract (ZH)**: 评估合成语音的感知质量对于指导语音生成模型的发展和 refinement 至关重要。然而，这一直依赖于人力主观评分，如平均意见分 (MOS)，这需要手工注释且通常存在评分标准不一致和重现性差的问题。为解决这些局限性，我们引入了 MOS-RMBench，一种统一的基准，将多样化的 MOS 数据集重新阐述为偏好比较设置，从而在不同数据集上实现严格的评估。基于 MOS-RMBench，我们系统地构建和评估了三种奖励建模范式：标量奖励模型、半标量奖励模型和生成奖励模型（GRMs）。实验揭示了三个关键发现：(1) 标量模型的整体性能最佳，持续超过 74% 的准确率；(2) 大多数模型在合成语音上的表现远不如在人类语音上的表现；(3) 所有模型在 MOS 差异非常小的配对上表现都很差。为了提高在这些具有挑战性配对上的性能，我们提出了一种 Awareness MOS 的 GRM，其中包含基于 MOS 差异的奖励函数，使模型能够根据每个样本配对的难度自适应地调整奖励。实验结果表明，Awareness MOS 的 GRM 显著提高了细致质量区分，并在最具有挑战性的案例上缩小了与标量模型之间的差距。我们希望这项工作能够建立一个基准和方法论框架，以促进自动语音质量评估研究的更严格和可扩展。 

---
# Neural Diffusion Processes for Physically Interpretable Survival Prediction 

**Title (ZH)**: 物理可解释的生存预测的神经扩散过程 

**Authors**: Alessio Cristofoletto, Cesare Rollo, Giovanni Birolo, Piero Fariselli  

**Link**: [PDF](https://arxiv.org/pdf/2510.00733)  

**Abstract**: We introduce DeepFHT, a survival-analysis framework that couples deep neural networks with first hitting time (FHT) distributions from stochastic process theory. Time to event is represented as the first passage of a latent diffusion process to an absorbing boundary. A neural network maps input variables to physically meaningful parameters including initial condition, drift, and diffusion, within a chosen FHT process such as Brownian motion, both with drift and driftless. This yields closed-form survival and hazard functions and captures time-varying risk without assuming proportional-hazards.
We compare DeepFHT with Cox regression and other existing parametric survival models, using synthetic and real-world datasets. The method achieves predictive accuracy on par with state-of-the-art approaches, while maintaining a physics-based interpretable parameterization that elucidates the relation between input features and risk. This combination of stochastic process theory and deep learning provides a principled avenue for modeling survival phenomena in complex systems. 

**Abstract (ZH)**: DeepFHT：一种将深度神经网络与随机过程理论中的首次触碰时间分布相结合的生存分析框架 

---
# Extreme Blind Image Restoration via Prompt-Conditioned Information Bottleneck 

**Title (ZH)**: 极值盲图像恢复 via 提示条件信息瓶颈 

**Authors**: Hongeun Kim, Bryan Sangwoo Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.00728)  

**Abstract**: Blind Image Restoration (BIR) methods have achieved remarkable success but falter when faced with Extreme Blind Image Restoration (EBIR), where inputs suffer from severe, compounded degradations beyond their training scope. Directly learning a mapping from extremely low-quality (ELQ) to high-quality (HQ) images is challenging due to the massive domain gap, often leading to unnatural artifacts and loss of detail. To address this, we propose a novel framework that decomposes the intractable ELQ-to-HQ restoration process. We first learn a projector that maps an ELQ image onto an intermediate, less-degraded LQ manifold. This intermediate image is then restored to HQ using a frozen, off-the-shelf BIR model. Our approach is grounded in information theory; we provide a novel perspective of image restoration as an Information Bottleneck problem and derive a theoretically-driven objective to train our projector. This loss function effectively stabilizes training by balancing a low-quality reconstruction term with a high-quality prior-matching term. Our framework enables Look Forward Once (LFO) for inference-time prompt refinement, and supports plug-and-play strengthening of existing image restoration models without need for finetuning. Extensive experiments under severe degradation regimes provide a thorough analysis of the effectiveness of our work. 

**Abstract (ZH)**: 盲图像恢复方法在应对极端盲图像恢复任务时表现出色但在输入遭受超出现有训练范围的严重复合退化时会遇到困难。直接学习从极度低质量到高质量图像的映射因领域差距巨大而极具挑战性，常导致不自然的伪影和细节丢失。为解决这一问题，我们提出了一种新颖的框架，将难以处理的极度低质量到高质量的恢复过程分解为两个步骤。首先，学习一个投影器将极度低质量图像映射到一个中间的、退化较轻的低质量流形。然后，使用一个冻结的现成盲图像恢复模型恢复该中间图像为高质量图像。我们的方法基于信息论；我们提供了一种新的视角将图像恢复问题视为信息瓶颈问题，并推导出一个理论上驱动的目标来训练我们的投影器。该损失函数通过平衡低质量重建项和高质量先验匹配项来有效稳定训练。我们的框架允许在推理时进行一次前瞻性的提示细化，并支持无需微调即可增强现有图像恢复模型。在严重的退化条件下进行的广泛实验提供了我们工作的有效性分析。 

---
# CroSTAta: Cross-State Transition Attention Transformer for Robotic Manipulation 

**Title (ZH)**: 跨状态转换注意变换器：用于机器人操作的跨状态转换注意力变换器 

**Authors**: Giovanni Minelli, Giulio Turrisi, Victor Barasuol, Claudio Semini  

**Link**: [PDF](https://arxiv.org/pdf/2510.00726)  

**Abstract**: Learning robotic manipulation policies through supervised learning from demonstrations remains challenging when policies encounter execution variations not explicitly covered during training. While incorporating historical context through attention mechanisms can improve robustness, standard approaches process all past states in a sequence without explicitly modeling the temporal structure that demonstrations may include, such as failure and recovery patterns. We propose a Cross-State Transition Attention Transformer that employs a novel State Transition Attention (STA) mechanism to modulate standard attention weights based on learned state evolution patterns, enabling policies to better adapt their behavior based on execution history. Our approach combines this structured attention with temporal masking during training, where visual information is randomly removed from recent timesteps to encourage temporal reasoning from historical context. Evaluation in simulation shows that STA consistently outperforms standard cross-attention and temporal modeling approaches like TCN and LSTM networks across all tasks, achieving more than 2x improvement over cross-attention on precision-critical tasks. 

**Abstract (ZH)**: 通过监督学习从演示中学习机器人操作策略，在政策遇到未在训练中明确涵盖的执行变异性时仍然具有挑战性。虽然通过注意力机制融入历史上下文可以提高鲁棒性，但标准方法会在序列中处理所有过去的状态，而不明确建模演示中可能包含的时间结构，例如故障和恢复模式。我们提出了一种跨状态转换注意力变换器（Cross-State Transition Attention Transformer），采用了一种新的状态转换注意力（STA）机制，根据学习到的状态演变模式调整标准注意力权重，从而使策略能够更好地根据执行历史适应其行为。我们的方法在训练中结合了这种结构化注意力和时间掩码，其中随机从最近的时间步去除视觉信息，以促进从历史上下文中进行时间推理。在模拟中的评估表明，STA在所有任务中始终优于标准交叉注意力和时间建模方法（如TCN和LSTM网络），在关键精度任务上取得了超过2倍的性能提升。 

---
# ALARB: An Arabic Legal Argument Reasoning Benchmark 

**Title (ZH)**: ALARB: 阿拉伯法律论证推理基准 

**Authors**: Harethah Abu Shairah, Somayah AlHarbi, Abdulaziz AlHussein, Sameer Alsabea, Omar Shaqaqi, Hebah AlShamlan, Omar Knio, George Turkiyyah  

**Link**: [PDF](https://arxiv.org/pdf/2510.00694)  

**Abstract**: We introduce ALARB, a dataset and suite of tasks designed to evaluate the reasoning capabilities of large language models (LLMs) within the Arabic legal domain. While existing Arabic benchmarks cover some knowledge-intensive tasks such as retrieval and understanding, substantial datasets focusing specifically on multistep reasoning for Arabic LLMs, especially in open-ended contexts, are lacking. The dataset comprises over 13K commercial court cases from Saudi Arabia, with each case including the facts presented, the reasoning of the court, the verdict, as well as the cited clauses extracted from the regulatory documents. We define a set of challenging tasks leveraging this dataset and reflecting the complexity of real-world legal reasoning, including verdict prediction, completion of reasoning chains in multistep legal arguments, and identification of relevant regulations based on case facts. We benchmark a representative selection of current open and closed Arabic LLMs on these tasks and demonstrate the dataset's utility for instruction tuning. Notably, we show that instruction-tuning a modest 12B parameter model using ALARB significantly enhances its performance in verdict prediction and Arabic verdict generation, reaching a level comparable to that of GPT-4o. 

**Abstract (ZH)**: ALARB：阿拉伯法律领域的大语言模型推理能力评估数据集及任务套件 

---
# Inclusive Easy-to-Read Generation for Individuals with Cognitive Impairments 

**Title (ZH)**: 认知 impairment 个体的包容性易读生成 

**Authors**: François Ledoyen, Gaël Dias, Alexis Lechervy, Jeremie Pantin, Fabrice Maurel, Youssef Chahir, Elisa Gouzonnat, Mélanie Berthelot, Stanislas Moravac, Armony Altinier, Amy Khairalla  

**Link**: [PDF](https://arxiv.org/pdf/2510.00691)  

**Abstract**: Ensuring accessibility for individuals with cognitive impairments is essential for autonomy, self-determination, and full citizenship. However, manual Easy-to-Read (ETR) text adaptations are slow, costly, and difficult to scale, limiting access to crucial information in healthcare, education, and civic life. AI-driven ETR generation offers a scalable solution but faces key challenges, including dataset scarcity, domain adaptation, and balancing lightweight learning of Large Language Models (LLMs). In this paper, we introduce ETR-fr, the first dataset for ETR text generation fully compliant with European ETR guidelines. We implement parameter-efficient fine-tuning on PLMs and LLMs to establish generative baselines. To ensure high-quality and accessible outputs, we introduce an evaluation framework based on automatic metrics supplemented by human assessments. The latter is conducted using a 36-question evaluation form that is aligned with the guidelines. Overall results show that PLMs perform comparably to LLMs and adapt effectively to out-of-domain texts. 

**Abstract (ZH)**: 确保认知障碍个体的无障碍访问对于自主权、自我决定权和完整公民身份是必不可少的。然而，手动易读文本（ETR）改编速度慢、成本高且难以扩展，限制了医疗、教育和公民生活中的关键信息访问。基于AI的ETR生成提供了可扩展的解决方案，但面临关键挑战，包括数据集稀缺、领域适配以及平衡大型语言模型（LLMs）的轻量级学习。本文介绍了ETR-fr，这是首个完全符合欧洲ETR指南要求的ETR文本生成数据集。我们进行了参数高效的微调，以在预训练语言模型（PLMs）和大型语言模型（LLMs）上建立生成基准。为确保高质量和可访问的输出，我们提出了一种基于自动评估指标结合人工评估的评估框架。后者使用了36个问题的评估表单，该表单与指南保持一致。总体结果表明，PLMs在性能上与LLMs相当，并能有效适应领域外文本。 

---
# Facilitating Cognitive Accessibility with LLMs: A Multi-Task Approach to Easy-to-Read Text Generation 

**Title (ZH)**: 利用大型语言模型促进认知 accessibility：一种易读文本生成的多任务方法 

**Authors**: François Ledoyen, Gaël Dias, Jeremie Pantin, Alexis Lechervy, Fabrice Maurel, Youssef Chahir  

**Link**: [PDF](https://arxiv.org/pdf/2510.00662)  

**Abstract**: Simplifying complex texts is essential for ensuring equitable access to information, especially for individuals with cognitive impairments. The Easy-to-Read (ETR) initiative offers a framework for making content accessible to the neurodivergent population, but the manual creation of such texts remains time-consuming and resource-intensive. In this work, we investigate the potential of large language models (LLMs) to automate the generation of ETR content. To address the scarcity of aligned corpora and the specificity of ETR constraints, we propose a multi-task learning (MTL) approach that trains models jointly on text summarization, text simplification, and ETR generation. We explore two different strategies: multi-task retrieval-augmented generation (RAG) for in-context learning, and MTL-LoRA for parameter-efficient fine-tuning. Our experiments with Mistral-7B and LLaMA-3-8B, based on ETR-fr, a new high-quality dataset, demonstrate the benefits of multi-task setups over single-task baselines across all configurations. Moreover, results show that the RAG-based strategy enables generalization in out-of-domain settings, while MTL-LoRA outperforms all learning strategies within in-domain configurations. 

**Abstract (ZH)**: 简化复杂文本对于确保认知障碍个体公平获取信息至关重要。Easy-to-Read (ETR)倡议提供了一种框架，使其内容能够被神经多样性人群访问，但此类文本的手动创建仍耗时且资源密集。在本文中，我们探讨了大规模语言模型（LLMs）在自动化生成ETR内容方面的潜在应用。为应对对齐数据集稀少和ETR约束特定性问题，我们提出了一种多任务学习（MTL）方法，该方法联合训练文本摘要、文本简化和ETR生成模型。我们探索了两种不同的策略：基于多任务检索增强生成（RAG）的上下文学习方法和参数高效微调的MTL-LoRA方法。基于ETR-fr，一个新高质量数据集，针对Mistral-7B和LLaMA-3-8B的实验表明，多任务设置在所有配置中都优于单任务基线。此外，结果表明，基于RAG的方法在领域外场景中表现出良好的泛化能力，而MTL-LoRA在领域内配置中优于所有学习策略。 

---
# Align Your Tangent: Training Better Consistency Models via Manifold-Aligned Tangents 

**Title (ZH)**: 对齐你的切线：通过流形对齐切线训练更好的一致性模型 

**Authors**: Beomsu Kim, Byunghee Cha, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.00658)  

**Abstract**: With diffusion and flow matching models achieving state-of-the-art generating performance, the interest of the community now turned to reducing the inference time without sacrificing sample quality. Consistency Models (CMs), which are trained to be consistent on diffusion or probability flow ordinary differential equation (PF-ODE) trajectories, enable one or two-step flow or diffusion sampling. However, CMs typically require prolonged training with large batch sizes to obtain competitive sample quality. In this paper, we examine the training dynamics of CMs near convergence and discover that CM tangents -- CM output update directions -- are quite oscillatory, in the sense that they move parallel to the data manifold, not towards the manifold. To mitigate oscillatory tangents, we propose a new loss function, called the manifold feature distance (MFD), which provides manifold-aligned tangents that point toward the data manifold. Consequently, our method -- dubbed Align Your Tangent (AYT) -- can accelerate CM training by orders of magnitude and even out-perform the learned perceptual image patch similarity metric (LPIPS). Furthermore, we find that our loss enables training with extremely small batch sizes without compromising sample quality. Code: this https URL 

**Abstract (ZH)**: 一致模型训练动态分析及调和切线方向以加速训练和提升性能：Align Your Tangent (AYT) 

---
# Tenyidie Syllabification corpus creation and deep learning applications 

**Title (ZH)**: Tenhyidie 音节分割语料库构建与深度学习应用 

**Authors**: Teisovi Angami, Kevisino Khate  

**Link**: [PDF](https://arxiv.org/pdf/2510.00629)  

**Abstract**: The Tenyidie language is a low-resource language of the Tibeto-Burman family spoken by the Tenyimia Community of Nagaland in the north-eastern part of India and is considered a major language in Nagaland. It is tonal, Subject-Object-Verb, and highly agglutinative in nature. Being a low-resource language, very limited research on Natural Language Processing (NLP) has been conducted. To the best of our knowledge, no work on syllabification has been reported for this language. Among the many NLP tasks, syllabification or syllabication is an important task in which the given word syllables are identified. The contribution of this work is the creation of 10,120 syllabified Tenyidie words and the application of the Deep Learning techniques on the created corpus. In this paper, we have applied LSTM, BLSTM, BLSTM+CRF, and Encoder-decoder deep learning architectures on our created dataset. In our dataset split of 80:10:10 (train:validation:test) set, we achieved the highest accuracy of 99.21% with BLSTM model on the test set. This work will find its application in numerous other NLP applications, such as morphological analysis, part-of-speech tagging, machine translation, etc, for the Tenyidie Language.
Keywords: Tenyidie; NLP; syllabification; deep learning; LSTM; BLSTM; CRF; Encoder-decoder 

**Abstract (ZH)**: Tenyidie语言的声节划分及其深度学习应用研究：基于LSTM、BLSTM、BLSTM+CRF和Encoder-decoder的探索 

---
# FAME: Adaptive Functional Attention with Expert Routing for Function-on-Function Regression 

**Title (ZH)**: FAME：基于专家路由的自适应函数注意力用于函数到函数的回归 

**Authors**: Yifei Gao, Yong Chen, Chen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00621)  

**Abstract**: Functional data play a pivotal role across science and engineering, yet their infinite-dimensional nature makes representation learning challenging. Conventional statistical models depend on pre-chosen basis expansions or kernels, limiting the flexibility of data-driven discovery, while many deep-learning pipelines treat functions as fixed-grid vectors, ignoring inherent continuity. In this paper, we introduce Functional Attention with a Mixture-of-Experts (FAME), an end-to-end, fully data-driven framework for function-on-function regression. FAME forms continuous attention by coupling a bidirectional neural controlled differential equation with MoE-driven vector fields to capture intra-functional continuity, and further fuses change to inter-functional dependencies via multi-head cross attention. Extensive experiments on synthetic and real-world functional-regression benchmarks show that FAME achieves state-of-the-art accuracy, strong robustness to arbitrarily sampled discrete observations of functions. 

**Abstract (ZH)**: 功能数据在科学和技术中扮演着关键角色，但由于其无限维性质，使得表示学习具有挑战性。传统统计模型依赖于预先选择的基展开或核，限制了数据驱动发现的灵活性，而许多深度学习管道将函数视为固定网格向量，忽略了固有的连续性。本文引入了基于Mixture-of-Experts的功能注意力（FAME）框架，这是一种端到端的完全数据驱动的方法，用于函数对函数回归。FAME通过将双向神经控制微分方程与MoE驱动的向量场耦合来形成连续注意力，以捕捉函数内部的连续性，并通过多头交叉注意力融合变化以捕捉函数间的依赖关系。在合成和真实世界的功能回归基准上的 extensive 实验表明，FAME 达到了最先进的准确率，并且对任意采样离散观察函数具有很强的鲁棒性。 

---
# What Did I Learn? Operational Competence Assessment for AI-Based Trajectory Planners 

**Title (ZH)**: 我学到了什么？基于AI的航迹规划器的操作能力评估 

**Authors**: Michiel Braat, Maren Buermann, Marijke van Weperen, Jan-Pieter Paardekooper  

**Link**: [PDF](https://arxiv.org/pdf/2510.00619)  

**Abstract**: Automated driving functions increasingly rely on machine learning for tasks like perception and trajectory planning, requiring large, relevant datasets. The performance of these algorithms depends on how closely the training data matches the task. To ensure reliable functioning, it is crucial to know what is included in the dataset to assess the trained model's operational risk. We aim to enhance the safe use of machine learning in automated driving by developing a method to recognize situations that an automated vehicle has not been sufficiently trained on. This method also improves explainability by describing the dataset at a human-understandable level. We propose modeling driving data as knowledge graphs, representing driving scenes with entities and their relationships. These graphs are queried for specific sub-scene configurations to check their occurrence in the dataset. We estimate a vehicle's competence in a driving scene by considering the coverage and complexity of sub-scene configurations in the training set. Higher complexity scenes require greater coverage for high competence. We apply this method to the NuPlan dataset, modeling it with knowledge graphs and analyzing the coverage of specific driving scenes. This approach helps monitor the competence of machine learning models trained on the dataset, which is essential for trustworthy AI to be deployed in automated driving. 

**Abstract (ZH)**: 自动化驾驶功能 increasingly 越来越多地依赖机器学习来进行感知和轨迹规划，这需要大量相关数据集。这些算法的性能取决于训练数据与任务的接近程度。为了确保可靠运行，了解数据集的内容以评估训练模型的操作风险至关重要。我们旨在通过开发一种方法来识别自动化车辆未充分训练的情况，以增强机器学习在自动化驾驶中的安全使用。该方法通过以人类可理解的方式描述数据集来提高模型的可解释性。我们建议将驾驶数据建模为知识图谱，用实体及其关系来表示驾驶场景。这些图谱查询特定子场景配置以检查它们在数据集中的出现情况。我们通过考虑训练集中子场景配置的覆盖范围和复杂性来估计车辆在驾驶场景中的能力。较高复杂度的场景需要更高的覆盖范围才能实现高水平的能力。我们将此方法应用于NuPlan数据集，用知识图谱建模并分析特定驾驶场景的覆盖范围。这种方法有助于监控基于数据集训练的机器学习模型的能力，这对可信的人工智能在自动化驾驶中的部署至关重要。 

---
# Hybrid Training for Vision-Language-Action Models 

**Title (ZH)**: 视觉-语言-动作模型的混合训练 

**Authors**: Pietro Mazzaglia, Cansu Sancaktar, Markus Peschl, Daniel Dijkman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00600)  

**Abstract**: Using Large Language Models to produce intermediate thoughts, a.k.a. Chain-of-thought (CoT), before providing an answer has been a successful recipe for solving complex language tasks. In robotics, similar embodied CoT strategies, generating thoughts before actions, have also been shown to lead to improved performance when using Vision-Language-Action models (VLAs). As these techniques increase the length of the model's generated outputs to include the thoughts, the inference time is negatively affected. Delaying an agent's actions in real-world executions, as in robotic manipulation settings, strongly affects the usability of a method, as tasks require long sequences of actions. However, is the generation of long chains-of-thought a strong prerequisite for achieving performance improvements? In this work, we explore the idea of Hybrid Training (HyT), a framework that enables VLAs to learn from thoughts and benefit from the associated performance gains, while enabling the possibility to leave out CoT generation during inference. Furthermore, by learning to conditionally predict a diverse set of outputs, HyT supports flexibility at inference time, enabling the model to either predict actions directly, generate thoughts or follow instructions. We evaluate the proposed method in a series of simulated benchmarks and real-world experiments. 

**Abstract (ZH)**: 使用大型语言模型在提供答案之前生成中间思考（即链式思考CoT），已被证明是解决复杂语言任务的有效方法。在机器人学中，类似的具身CoT策略，在采取行动之前生成思考，也已被证明能够提高使用视觉-语言-行动模型（VLAs）时的表现。随着这些技术增加模型生成输出的长度以包含思考，推理时间会受到负面影响。在机器人操作等现实世界执行中延迟代理的操作，强烈影响方法的可用性，因为任务需要一系列长时间序列的操作。然而，长时间链式思考的生成是否是实现性能提升的必要条件？在本工作中，我们探索了混合训练（HyT）框架，该框架使VLAs能够从思考中学习并受益于关联的表现提升，同时使在推理过程中省略CoT生成成为可能。此外，通过学习有条件地预测一组多样化的输出，HyT在推理时支持灵活性，使模型能够在预测动作、生成思考或遵循指令之间进行选择。我们通过一系列模拟基准测试和真实世界实验评估了所提出的方法。 

---
# AI-Driven Self-Evolving Software: A Promising Path Toward Software Automation 

**Title (ZH)**: 基于AI的自进化软件：通往软件自动化的一条有前途的道路 

**Authors**: Liyi Cai, Yijie Ren, Yitong Zhang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00591)  

**Abstract**: Software automation has long been a central goal of software engineering, striving for software development that proceeds without human intervention. Recent efforts have leveraged Artificial Intelligence (AI) to advance software automation with notable progress. However, current AI functions primarily as assistants to human developers, leaving software development still dependent on explicit human intervention. This raises a fundamental question: Can AI move beyond its role as an assistant to become a core component of software, thereby enabling genuine software automation? To investigate this vision, we introduce AI-Driven Self-Evolving Software, a new form of software that evolves continuously through direct interaction with users. We demonstrate the feasibility of this idea with a lightweight prototype built on a multi-agent architecture that autonomously interprets user requirements, generates and validates code, and integrates new functionalities. Case studies across multiple representative scenarios show that the prototype can reliably construct and reuse functionality, providing early evidence that such software systems can scale to more sophisticated applications and pave the way toward truly automated software development. We make code and cases in this work publicly available at this https URL. 

**Abstract (ZH)**: 基于AI的自演化软件：迈向真正的软件自动化 

---
# U-DFA: A Unified DINOv2-Unet with Dual Fusion Attention for Multi-Dataset Medical Segmentation 

**Title (ZH)**: U-DFA: 结合双融合注意力机制的统一DINOv2-Unet多数据集医学分割 

**Authors**: Zulkaif Sajjad, Furqan Shaukat, Junaid Mir  

**Link**: [PDF](https://arxiv.org/pdf/2510.00585)  

**Abstract**: Accurate medical image segmentation plays a crucial role in overall diagnosis and is one of the most essential tasks in the diagnostic pipeline. CNN-based models, despite their extensive use, suffer from a local receptive field and fail to capture the global context. A common approach that combines CNNs with transformers attempts to bridge this gap but fails to effectively fuse the local and global features. With the recent emergence of VLMs and foundation models, they have been adapted for downstream medical imaging tasks; however, they suffer from an inherent domain gap and high computational cost. To this end, we propose U-DFA, a unified DINOv2-Unet encoder-decoder architecture that integrates a novel Local-Global Fusion Adapter (LGFA) to enhance segmentation performance. LGFA modules inject spatial features from a CNN-based Spatial Pattern Adapter (SPA) module into frozen DINOv2 blocks at multiple stages, enabling effective fusion of high-level semantic and spatial features. Our method achieves state-of-the-art performance on the Synapse and ACDC datasets with only 33\% of the trainable model parameters. These results demonstrate that U-DFA is a robust and scalable framework for medical image segmentation across multiple modalities. 

**Abstract (ZH)**: 准确的医学图像分割在整体诊断中发挥着关键作用，是诊断流程中最重要的任务之一。尽管基于CNN的模型被广泛应用，但由于局部感受野的限制，它们难以捕捉全局上下文。结合CNN与 transformer 的常用方法试图弥合这一差距，但未能有效地融合局部和全局特征。随着VLMs和基础模型的出现，它们已被适应于下游医学成像任务，然而仍存在固有的领域差距和高计算成本的问题。为了解决这些问题，我们提出了一种统一的DINOv2-Unet编码-解码架构U-DFA，该架构结合了一个新型的局部-全局融合适配器（LGFA）模块以提高分割性能。LGFA模块将基于CNN的空间模式适配器（SPA）模块的空间特征注入冻结的DINOv2块中，多阶段实现高级语义和空间特征的有效融合。我们的方法在Synapse和ACDC数据集上实现了最先进的性能，仅需33%的可训练模型参数。这些结果表明，U-DFA是一种稳健且可扩展的跨多种模态医学图像分割框架。 

---
# SAGE-LD: Towards Scalable and Generalizable End-to-End Language Diarization via Simulated Data Augmentation 

**Title (ZH)**: SAGE-LD：通过模拟数据增强实现可扩展和通用的端到端语言分离 

**Authors**: Sangmin Lee, Woongjib Choi, Jihyun Kim, Hong-Goo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00582)  

**Abstract**: In this paper, we present a neural spoken language diarization model that supports an unconstrained span of languages within a single framework. Our approach integrates a learnable query-based architecture grounded in multilingual awareness, with large-scale pretraining on simulated code-switching data. By jointly leveraging these two components, our method overcomes the limitations of conventional approaches in data scarcity and architecture optimization, and generalizes effectively to real-world multilingual settings across diverse environments. Experimental results demonstrate that our approach achieves state-of-the-art performance on several language diarization benchmarks, with a relative performance improvement of 23% to 52% over previous methods. We believe that this work not only advances research in language diarization but also establishes a foundational framework for code-switching speech technologies. 

**Abstract (ZH)**: 本研究提出了一种支持单框架内多种语言未受约束的神经语音语言日记模型，该模型结合了多语言意识下的可学习查询架构和大规模模拟代码切换数据的预训练。通过联合利用这两部分，我们的方法克服了传统方法在数据稀缺性和架构优化方面的限制，并在多种环境中有效泛化到真实世界的多语言场景。实验结果表明，我们的方法在多种语言日记基准测试中实现了最先进的性能，相对于之前的方法，性能提高了23%到52%。我们相信，这项工作不仅推进了语言日记的研究，还为代码切换语音技术建立了基础框架。 

---
# Adaptive Shared Experts with LoRA-Based Mixture of Experts for Multi-Task Learning 

**Title (ZH)**: 基于LoRA基混合专家的自适应共享专家多任务学习 

**Authors**: Minghao Yang, Ren Togo, Guang Li, Takahiro Ogawa, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.00570)  

**Abstract**: Mixture-of-Experts (MoE) has emerged as a powerful framework for multi-task learning (MTL). However, existing MoE-MTL methods often rely on single-task pretrained backbones and suffer from redundant adaptation and inefficient knowledge sharing during the transition from single-task to multi-task learning (STL to MTL). To address these limitations, we propose adaptive shared experts (ASE) within a low-rank adaptation (LoRA) based MoE, where shared experts are assigned router-computed gating weights jointly normalized with sparse experts. This design facilitates STL to MTL transition, enhances expert specialization, and cooperation. Furthermore, we incorporate fine-grained experts by increasing the number of LoRA experts while proportionally reducing their rank, enabling more effective knowledge sharing under a comparable parameter budget. Extensive experiments on the PASCAL-Context benchmark, under unified training settings, demonstrate that ASE consistently improves performance across diverse configurations and validates the effectiveness of fine-grained designs for MTL. 

**Abstract (ZH)**: 混合专家模型（MoE）已成为多任务学习（MTL）的一种强大框架。然而，现有的MoE-MTL方法往往依赖于单任务预训练的骨干网络，并在从单任务学习（STL）过渡到多任务学习的过程中面临冗余适应和知识分享效率低下的问题。为解决这些问题，我们提出了一种基于低秩适应（LoRA）的MoE中的自适应共享专家（ASE），其中共享专家与稀疏专家一起被路由计算的门控权重联合规范化。这种设计促进了从STL到MTL的过渡，增强了专家的专业化和合作。此外，通过增加LoRA专家的数量同时按比例减少其秩，我们引入了细粒度专家，这在相近的参数预算下实现了更有效的知识共享。在统一训练设置下的PASCAL-Context基准测试上进行的广泛实验表明，ASE在多种配置下均能提升性能，并验证了细粒度设计在MTL中的有效性。 

---
# Panorama: Fast-Track Nearest Neighbors 

**Title (ZH)**: 全景：快速 nearest neighbors 探索 

**Authors**: Vansh Ramani, Alexis Schlomer, Akash Nayar, Panagiotis Karras, Sayan Ranu, Jignesh M. Patel  

**Link**: [PDF](https://arxiv.org/pdf/2510.00566)  

**Abstract**: Approximate Nearest-Neighbor Search (ANNS) efficiently finds data items whose embeddings are close to that of a given query in a high-dimensional space, aiming to balance accuracy with speed. Used in recommendation systems, image and video retrieval, natural language processing, and retrieval-augmented generation (RAG), ANNS algorithms such as IVFPQ, HNSW graphs, Annoy, and MRPT utilize graph, tree, clustering, and quantization techniques to navigate large vector spaces. Despite this progress, ANNS systems spend up to 99\% of query time to compute distances in their final refinement phase. In this paper, we present PANORAMA, a machine learning-driven approach that tackles the ANNS verification bottleneck through data-adaptive learned orthogonal transforms that facilitate the accretive refinement of distance bounds. Such transforms compact over 90\% of signal energy into the first half of dimensions, enabling early candidate pruning with partial distance computations. We integrate PANORAMA into state-of-the-art ANNS methods, namely IVFPQ/Flat, HNSW, MRPT, and Annoy, without index modification, using level-major memory layouts, SIMD-vectorized partial distance computations, and cache-aware access patterns. Experiments across diverse datasets -- from image-based CIFAR-10 and GIST to modern embedding spaces including OpenAI's Ada 2 and Large 3 -- demonstrate that PANORAMA affords a 2--30$\times$ end-to-end speedup with no recall loss. 

**Abstract (ZH)**: 基于机器学习的PANORAMA：通过数据自适应学习正交变换解决近邻搜索验证瓶颈 

---
# Memory Determines Learning Direction: A Theory of Gradient-Based Optimization in State Space Models 

**Title (ZH)**: 记忆决定学习方向：状态空间模型中基于梯度优化的理论 

**Authors**: JingChuan Guan, Tomoyuki Kubota, Yasuo Kuniyoshi, Kohei Nakajima  

**Link**: [PDF](https://arxiv.org/pdf/2510.00563)  

**Abstract**: State space models (SSMs) have gained attention by showing potential to outperform Transformers. However, previous studies have not sufficiently addressed the mechanisms underlying their high performance owing to a lack of theoretical explanation of SSMs' learning dynamics. In this study, we provide such an explanation and propose an improved training strategy. The memory capacity of SSMs can be evaluated by examining how input time series are stored in their current state. Such an examination reveals a tradeoff between memory accuracy and length, as well as the theoretical equivalence between the structured state space sequence model (S4) and a simplified S4 with diagonal recurrent weights. This theoretical foundation allows us to elucidate the learning dynamics, proving the importance of initial parameters. Our analytical results suggest that successful learning requires the initial memory structure to be the longest possible even if memory accuracy may deteriorate or the gradient lose the teacher information. Experiments on tasks requiring long memory confirmed that extending memory is difficult, emphasizing the importance of initialization. Furthermore, we found that fixing recurrent weights can be more advantageous than adapting them because it achieves comparable or even higher performance with faster convergence. Our results provide a new theoretical foundation for SSMs and potentially offer a novel optimization strategy. 

**Abstract (ZH)**: 状态空间模型（SSMs）通过表现出超越Transformer的潜力而引起了关注。然而，先前的研究尚未充分探讨其高性能背后的机制，这主要是由于缺乏对SSMs学习动态的理论解释。在本研究中，我们提供了这样的解释并提出了一种改进的训练策略。通过检查输入时间序列在当前状态中的存储情况，可以评估SSMs的记忆容量。这种检查揭示了记忆准确性和长度之间的权衡，并且证明了结构化状态空间序列模型（S4）与简化后的具有对角循环权重的S4在理论上等价。基于这一理论基础，我们可以阐明学习动态，证明初始参数的重要性。我们的分析结果表明，即使可能牺牲记忆准确性或梯度失去教师信息，成功的学习也需要初始记忆结构尽可能长。实验结果证实，在需要长记忆的任务上，延长记忆是困难的，强调了初始化的重要性。此外，我们发现固定循环权重可能比适应它们更具优势，因为这不仅能够实现相当甚至更高的性能，还能实现更快的收敛。我们的结果为SSMs提供了一个新的理论基础，并可能提供一种新的优化策略。 

---
# PromptPilot: Improving Human-AI Collaboration Through LLM-Enhanced Prompt Engineering 

**Title (ZH)**: PromptPilot: 通过LLM增强的提示工程提高人机协作 

**Authors**: Niklas Gutheil, Valentin Mayer, Leopold Müller, Jörg Rommelt, Niklas Kühl  

**Link**: [PDF](https://arxiv.org/pdf/2510.00555)  

**Abstract**: Effective prompt engineering is critical to realizing the promised productivity gains of large language models (LLMs) in knowledge-intensive tasks. Yet, many users struggle to craft prompts that yield high-quality outputs, limiting the practical benefits of LLMs. Existing approaches, such as prompt handbooks or automated optimization pipelines, either require substantial effort, expert knowledge, or lack interactive guidance. To address this gap, we design and evaluate PromptPilot, an interactive prompting assistant grounded in four empirically derived design objectives for LLM-enhanced prompt engineering. We conducted a randomized controlled experiment with 80 participants completing three realistic, work-related writing tasks. Participants supported by PromptPilot achieved significantly higher performance (median: 78.3 vs. 61.7; p = .045, d = 0.56), and reported enhanced efficiency, ease-of-use, and autonomy during interaction. These findings empirically validate the effectiveness of our proposed design objectives, establishing LLM-enhanced prompt engineering as a viable technique for improving human-AI collaboration. 

**Abstract (ZH)**: 有效的提示工程对于实现大型语言模型在知识密集型任务中的预期生产率提升至关重要。然而，许多用户难以创作出高质量的提示，限制了大型语言模型的实际效益。现有方法，如提示手册或自动化优化管道，要么需要大量努力、专家知识，要么缺乏互动指导。为解决这一问题，我们设计并评估了PromptPilot，这是一种基于四项实证设计目标的交互式提示助手，旨在增强大型语言模型的提示工程能力。我们在一项随机对照实验中，让80名参与者完成了三个实际的工作相关写作任务。接受PromptPilot支持的参与者在性能方面表现显著更好（中位数：78.3 vs. 61.7；p = .045，d = 0.56），并且报告称在互动中感受到了更高的效率、易用性和自主性。这些发现实证验证了我们提出的设计目标的有效性，确立了大型语言模型增强的提示工程作为提升人机协作可行技术的地位。 

---
# On Predictability of Reinforcement Learning Dynamics for Large Language Models 

**Title (ZH)**: 大规模语言模型的强化学习动态可预测性探究 

**Authors**: Yuchen Cai, Ding Cao, Xin Xu, Zijun Yao, Yuqing Huang, Zhenyu Tan, Benyi Zhang, Guiquan Liu, Junfeng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00553)  

**Abstract**: Recent advances in reasoning capabilities of large language models (LLMs) are largely driven by reinforcement learning (RL), yet the underlying parameter dynamics during RL training remain poorly understood. This work identifies two fundamental properties of RL-induced parameter updates in LLMs: (1) Rank-1 Dominance, where the top singular subspace of the parameter update matrix nearly fully determines reasoning improvements, recovering over 99\% of performance gains; and (2) Rank-1 Linear Dynamics, where this dominant subspace evolves linearly throughout training, enabling accurate prediction from early checkpoints. Extensive experiments across 8 LLMs and 7 algorithms validate the generalizability of these properties. More importantly, based on these findings, we propose AlphaRL, a plug-in acceleration framework that extrapolates the final parameter update using a short early training window, achieving up to 2.5 speedup while retaining \textgreater 96\% of reasoning performance without extra modules or hyperparameter tuning. This positions our finding as a versatile and practical tool for large-scale RL, opening a path toward principled, interpretable, and efficient training paradigm for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的推理能力最新进展主要由强化学习（RL）驱动，但RL训练期间的参数动态仍不甚了解。本项工作识别了RL诱导的LLMs参数更新的两种基本属性：（1）秩1主导性，其中参数更新矩阵的顶级奇异子空间几乎完全决定了推理改进，恢复了超过99%的性能提升；（2）秩1线性动力学，该主导子空间在整个训练过程中线性演化，使得从早期检查点准确预测成为可能。广泛的实验验证了这些属性的普适性。更重要的是，基于这些发现，我们提出了AlphaRL，一种插件加速框架，利用短的早期训练窗口外推最终参数更新，实现最高2.5倍的加速同时保持超过96%的推理性能无需额外模块或超参数调整。这使我们的发现成为大规模RL的多功能和实用工具，开启了LLMs原理化、可解释和高效训练范式的道路。 

---
# EMR-AGENT: Automating Cohort and Feature Extraction from EMR Databases 

**Title (ZH)**: EMR-AGENT: 自动化从EMR数据库中提取队列和特征 

**Authors**: Kwanhyung Lee, Sungsoo Hong, Joonhyung Park, Jeonghyeop Lim, Juhwan Choi, Donghwee Yoon, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00549)  

**Abstract**: Machine learning models for clinical prediction rely on structured data extracted from Electronic Medical Records (EMRs), yet this process remains dominated by hardcoded, database-specific pipelines for cohort definition, feature selection, and code mapping. These manual efforts limit scalability, reproducibility, and cross-institutional generalization. To address this, we introduce EMR-AGENT (Automated Generalized Extraction and Navigation Tool), an agent-based framework that replaces manual rule writing with dynamic, language model-driven interaction to extract and standardize structured clinical data. Our framework automates cohort selection, feature extraction, and code mapping through interactive querying of databases. Our modular agents iteratively observe query results and reason over schema and documentation, using SQL not just for data retrieval but also as a tool for database observation and decision making. This eliminates the need for hand-crafted, schema-specific logic. To enable rigorous evaluation, we develop a benchmarking codebase for three EMR databases (MIMIC-III, eICU, SICdb), including both seen and unseen schema settings. Our results demonstrate strong performance and generalization across these databases, highlighting the feasibility of automating a process previously thought to require expert-driven design. The code will be released publicly at this https URL. For a demonstration, please visit our anonymous demo page: this https URL 

**Abstract (ZH)**: 基于电子医疗记录的机器学习模型依赖于从电子病历（EMRs）中提取的结构化数据，但这一过程仍然主要由硬编码的、数据库特定的流水线来确定队列、特征选择和代码映射。这些手动努力限制了可扩展性、复现性和跨机构的一般化能力。为解决这一问题，我们引入了EMR-AGENT（自动化通用提取和导航工具）框架，该框架使用基于代理的方法，用动态的语言模型驱动交互来取代手动规则编写，以提取和标准化结构化临床数据。我们的框架通过交互查询数据库来自动化队列选择、特征提取和代码映射。我们模块化的代理迭代观察查询结果，并在包括模式和文档在内的所有方面进行推理，使用SQL不仅用于数据检索，还作为数据库观察和决策的工具。这消除了手写特定模式逻辑的需要。为实现严格评估，我们为三个电子医疗记录数据库（MIMIC-III、eICU、SICdb）开发了基准代码库，包括已见和未知模式设置。我们的结果显示了这些数据库的强大性能和泛化能力，突显了自动化以往被认为需要专家驱动设计的过程的可能性。代码将在此公开发布：this https URL。如需演示，请访问我们的匿名演示页面：this https URL。 

---
# Forestpest-YOLO: A High-Performance Detection Framework for Small Forestry Pests 

**Title (ZH)**: Forestpest-YOLO: 一种高性能的小型林业害虫检测框架 

**Authors**: Aoduo Li, Peikai Lin, Jiancheng Li, Zhen Zhang, Shiting Wu, Zexiao Liang, Zhifa Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00547)  

**Abstract**: Detecting agricultural pests in complex forestry environments using remote sensing imagery is fundamental for ecological preservation, yet it is severely hampered by practical challenges. Targets are often minuscule, heavily occluded, and visually similar to the cluttered background, causing conventional object detection models to falter due to the loss of fine-grained features and an inability to handle extreme data imbalance. To overcome these obstacles, this paper introduces Forestpest-YOLO, a detection framework meticulously optimized for the nuances of forestry remote sensing. Building upon the YOLOv8 architecture, our framework introduces a synergistic trio of innovations. We first integrate a lossless downsampling module, SPD-Conv, to ensure that critical high-resolution details of small targets are preserved throughout the network. This is complemented by a novel cross-stage feature fusion block, CSPOK, which dynamically enhances multi-scale feature representation while suppressing background noise. Finally, we employ VarifocalLoss to refine the training objective, compelling the model to focus on high-quality and hard-to-classify samples. Extensive experiments on our challenging, self-constructed ForestPest dataset demonstrate that Forestpest-YOLO achieves state-of-the-art performance, showing marked improvements in detecting small, occluded pests and significantly outperforming established baseline models. 

**Abstract (ZH)**: 利用遥感影像检测复杂森林环境中农业害虫是生态保护的基础，但受到实际挑战的严重影响。目标往往很小、严重遮挡且与杂乱背景视觉相似，导致传统目标检测模型因细粒度特征丢失和极端数据不平衡难以应对。为克服这些困难，本文提出Forestpest-YOLO，一个专为森林遥感检测精细特性优化的检测框架。基于YOLOv8架构，该框架引入了三项协同创新。首先，整合无损下采样模块SPD-Conv，确保网络中关键的小目标的高分辨率细节得以保留。其次，采用一种新颖的跨阶段特征融合模块CSPOK，动态增强多尺度特征表示并抑制背景噪声。最后，采用VarifocalLoss精炼训练目标，促使模型关注高质量和难以分类的样本。在我们挑战性的自构建ForestPest数据集上的广泛实验表明，Forestpest-YOLO达到了最先进的性能，在检测小且被遮挡的害虫方面有显著改进，并远超现有基准模型。 

---
# Architectural Transformations and Emerging Verification Demands in AI-Enabled Cyber-Physical Systems 

**Title (ZH)**: AI使能的网络物理系统中架构变换与新兴验证要求 

**Authors**: Hadiza Umar Yusuf, Khouloud Gaaloul  

**Link**: [PDF](https://arxiv.org/pdf/2510.00519)  

**Abstract**: In the world of Cyber-Physical Systems (CPS), a captivating real-time fusion occurs where digital technology meets the physical world. This synergy has been significantly transformed by the integration of artificial intelligence (AI), a move that dramatically enhances system adaptability and introduces a layer of complexity that impacts CPS control optimization and reliability. Despite advancements in AI integration, a significant gap remains in understanding how this shift affects CPS architecture, operational complexity, and verification practices. The extended abstract addresses this gap by investigating architectural distinctions between AI-driven and traditional control models designed in Simulink and their respective implications for system verification. 

**Abstract (ZH)**: 在 cyber-物理系统（CPS）领域，数字技术与物理世界的精彩实时融合吸引了广泛关注。人工智能（AI）的融合极大地改变了这种 synergy，显著增强了系统的适应性，同时也引入了一层复杂性，影响了 CPS 控制优化和可靠性。尽管在 AI 融合方面取得了进展，但在了解这种转变如何影响 CPS 架构、操作复杂性和验证实践方面仍存在较大缺口。扩展摘要通过探讨 Simulink 中设计的 AI 驱动控制模型与传统控制模型之间的架构区别及其对系统验证的相应影响，来填补这一缺口。 

---
# Adaptive Data-Knowledge Alignment in Genetic Perturbation Prediction 

**Title (ZH)**: 基因扰动预测中的自适应数据-知识对齐 

**Authors**: Yuanfang Xiang, Lun Ai  

**Link**: [PDF](https://arxiv.org/pdf/2510.00512)  

**Abstract**: The transcriptional response to genetic perturbation reveals fundamental insights into complex cellular systems. While current approaches have made progress in predicting genetic perturbation responses, they provide limited biological understanding and cannot systematically refine existing knowledge. Overcoming these limitations requires an end-to-end integration of data-driven learning and existing knowledge. However, this integration is challenging due to inconsistencies between data and knowledge bases, such as noise, misannotation, and incompleteness. To address this challenge, we propose ALIGNED (Adaptive aLignment for Inconsistent Genetic kNowledgE and Data), a neuro-symbolic framework based on the Abductive Learning (ABL) paradigm. This end-to-end framework aligns neural and symbolic components and performs systematic knowledge refinement. We introduce a balanced consistency metric to evaluate the predictions' consistency against both data and knowledge. Our results show that ALIGNED outperforms state-of-the-art methods by achieving the highest balanced consistency, while also re-discovering biologically meaningful knowledge. Our work advances beyond existing methods to enable both the transparency and the evolution of mechanistic biological understanding. 

**Abstract (ZH)**: 遗传扰动的转录反应揭示了复杂细胞系统的基本见解。虽然目前的方法在预测遗传扰动反应方面取得了进展，但它们提供的生物学理解有限，无法系统性地完善现有知识。克服这些局限性需要端到端地整合数据驱动学习和现有知识。然而，这种整合由于数据和知识库之间的一致性问题，如噪声、误注释和不完整性而面临挑战。为了解决这一挑战，我们提出了一种基于归纳推理学习（ABL）范式的神经符号框架ALIGNED（自适应不一致遗传知识和数据的对齐）。这一端到端框架对齐了神经和符号组件，并进行系统的知识完善。我们引入了一致性度量来评估预测的一致性，既针对数据也针对知识。我们的结果显示，ALIGNED在平衡一致性度量上优于现有方法，并且重新发现了一些生物学上有意义的知识。我们的工作超越了现有方法，使机制生物学理解的透明性和进化成为可能。 

---
# Copy-Paste to Mitigate Large Language Model Hallucinations 

**Title (ZH)**: 复制粘贴以减轻大型语言模型幻觉问题 

**Authors**: Yongchao Long, Xian Wu, Yingying Zhang, Xianbin Wen, Yuxi Zhou, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00508)  

**Abstract**: While Retrieval-Augmented Generation (RAG) enables large language models (LLMs) to generate contextually grounded responses, contextual faithfulness remains challenging as LLMs may not consistently trust provided context, leading to hallucinations that undermine reliability. We observe an inverse correlation between response copying degree and context-unfaithful hallucinations on RAGTruth, suggesting that higher copying degrees reduce hallucinations by fostering genuine contextual belief. We propose CopyPasteLLM, obtained through two-stage high-copying response preference training. We design three prompting methods to enhance copying degree, demonstrating that high-copying responses achieve superior contextual faithfulness and hallucination control. These approaches enable a fully automated pipeline that transforms generated responses into high-copying preference data for training CopyPasteLLM. On FaithEval, ConFiQA and PubMedQA, CopyPasteLLM achieves best performance in both counterfactual and original contexts, remarkably with 12.2% to 24.5% accuracy improvements on FaithEval over the best baseline, while requiring only 365 training samples -- 1/50th of baseline data. To elucidate CopyPasteLLM's effectiveness, we propose the Context-Parameter Copying Capturing algorithm. Interestingly, this reveals that CopyPasteLLM recalibrates reliance on internal parametric knowledge rather than external knowledge during generation. All codes are available at this https URL 

**Abstract (ZH)**: While Retrieval-Augmented Generation (RAG)使大型语言模型（LLMs）能够生成具上下文相关性的响应，但上下文忠实性仍具有挑战性，因为LLMs可能不一致地信任提供的上下文，导致可能破坏可靠性的幻觉。我们在RAGTruth上观察到响应复制程度与上下文不忠实幻觉之间存在负相关关系，表明较高的复制程度通过促进真正的上下文信念来减少幻觉。我们提出了一种通过两阶段高复制响应偏好训练获得的CopyPasteLLM。我们设计了三种提示方法以增强复制程度，证明了高复制响应在上下文忠实性和幻觉控制方面表现出更优异的效果。这些方法能够实现一个完全自动化的流水线，将生成的响应转换为训练CopyPasteLLM的高复制偏好数据。在FaithEval、ConFiQA和PubMedQA上，CopyPasteLLM在反事实和原始上下文中均表现出最佳性能，相对于最佳基线在FaithEval上的准确率提高了12.2%至24.5%，仅需365个训练样本——基线数据的1/50。为揭示CopyPasteLLM的效果，我们提出了Context-Parameter Copying Capturing算法。有趣的是，这表明CopyPasteLLM在生成过程中重新校准了对内部参数知识而非外部知识的依赖。所有代码均可从以下链接获取。 

---
# Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs 

**Title (ZH)**: 图评：通过知识图谱自动生成代理的多模态任务 

**Authors**: Yurun Chen, Xavier Hu, Yuhan Liu, Ziqi Wang, Zeyi Liao, Lin Chen, Feng Wei, Yuxi Qian, Bo Zheng, Keting Yin, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00507)  

**Abstract**: As multimodal LLM-driven agents continue to advance in autonomy and generalization, evaluation based on static datasets can no longer adequately assess their true capabilities in dynamic environments and diverse tasks. Existing LLM-based synthetic data methods are largely designed for LLM training and evaluation, and thus cannot be directly applied to agent tasks that require tool use and interactive capabilities. While recent studies have explored automatic agent task generation with LLMs, most efforts remain limited to text or image analysis, without systematically modeling multi-step interactions in web environments. To address these challenges, we propose Graph2Eval, a knowledge graph-based framework that automatically generates both multimodal document comprehension tasks and web interaction tasks, enabling comprehensive evaluation of agents' reasoning, collaboration, and interactive capabilities. In our approach, knowledge graphs constructed from multi-source external data serve as the task space, where we translate semantic relations into structured multimodal tasks using subgraph sampling, task templates, and meta-paths. A multi-stage filtering pipeline based on node reachability, LLM scoring, and similarity analysis is applied to guarantee the quality and executability of the generated tasks. Furthermore, Graph2Eval supports end-to-end evaluation of multiple agent types (Single-Agent, Multi-Agent, Web Agent) and measures reasoning, collaboration, and interaction capabilities. We instantiate the framework with Graph2Eval-Bench, a curated dataset of 1,319 tasks spanning document comprehension and web interaction scenarios. Experiments show that Graph2Eval efficiently generates tasks that differentiate agent and model performance, revealing gaps in reasoning, collaboration, and web interaction across different settings and offering a new perspective for agent evaluation. 

**Abstract (ZH)**: 基于图的知识图谱驱动多模态代理评估框架：Graph2Eval 

---
# Relative-Absolute Fusion: Rethinking Feature Extraction in Image-Based Iterative Method Selection for Solving Sparse Linear Systems 

**Title (ZH)**: 基于图像的迭代方法选择求解稀疏线性系统中相对-绝对融合：重新思考特征提取 

**Authors**: Kaiqi Zhang, Mingguan Yang, Dali Chang, Chun Chen, Yuxiang Zhang, Kexun He, Jing Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00500)  

**Abstract**: Iterative method selection is crucial for solving sparse linear systems because these methods inherently lack robustness. Though image-based selection approaches have shown promise, their feature extraction techniques might encode distinct matrices into identical image representations, leading to the same selection and suboptimal method. In this paper, we introduce RAF (Relative-Absolute Fusion), an efficient feature extraction technique to enhance image-based selection approaches. By simultaneously extracting and fusing image representations as relative features with corresponding numerical values as absolute features, RAF achieves comprehensive matrix representations that prevent feature ambiguity across distinct matrices, thus improving selection accuracy and unlocking the potential of image-based selection approaches. We conducted comprehensive evaluations of RAF on SuiteSparse and our developed BMCMat (Balanced Multi-Classification Matrix dataset), demonstrating solution time reductions of 0.08s-0.29s for sparse linear systems, which is 5.86%-11.50% faster than conventional image-based selection approaches and achieves state-of-the-art (SOTA) performance. BMCMat is available at this https URL. 

**Abstract (ZH)**: 迭代方法选择对于求解稀疏线性系统至关重要，因为这些方法本身缺乏稳定性。尽管基于图像的选择方法显示出潜力，但其特征提取技术可能会将不同的矩阵编码为相同图像表示，导致相同的选型和次优方法。本文介绍了一种高效的特征提取技术RAF（相对-绝对融合），以增强基于图像的选择方法。通过同时提取并融合图像表示作为相对特征，以及与相应的数值值作为绝对特征，RAF实现了全面的矩阵表示，防止了不同矩阵间的特征模糊性，从而提高选择准确性并解锁基于图像选择方法的潜力。我们对RAF在SuiteSparse和我们开发的BMCMat（平衡多分类矩阵数据集）上的全面评估表明，对于稀疏线性系统，解的时间减少了0.08秒-0.29秒，比传统基于图像的选择方法快5.86%-11.50%，并达到了最先进的（SOTA）性能。BMCMat可在以下链接获取：this https URL。 

---
# MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance 

**Title (ZH)**: MOSS-Speech: 无需文本指导的真正端到端语音到语音模型 

**Authors**: Xingjian Zhao, Zhe Xu, Luozhijie Jin, Yang Wang, Hanfu Chen, Yaozhou Jiang, Ke Chen, Ruixiao Li, Mingshu Chen, Ruiming Wang, Wenbo Zhang, Yiyang Zhang, Donghua Yu, Yang Gao, Xiaogui Yang, Yitian Gong, Yuanfan Xu, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Yaqian Zhou, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00499)  

**Abstract**: Spoken dialogue systems often rely on cascaded pipelines that transcribe, process, and resynthesize speech. While effective, this design discards paralinguistic cues and limits expressivity. Recent end-to-end methods reduce latency and better preserve these cues, yet still rely on text intermediates, creating a fundamental bottleneck. We present MOSS-Speech, a true speech-to-speech large language model that directly understands and generates speech without relying on text guidance. Our approach combines a modality-based layer-splitting architecture with a frozen pre-training strategy, preserving the reasoning and knowledge of pretrained text LLMs while adding native speech capabilities. Experiments show that our model achieves state-of-the-art results in spoken question answering and delivers comparable speech-to-speech performance relative to existing text-guided systems, while still maintaining competitive text performance. By narrowing the gap between text-guided and direct speech generation, our work establishes a new paradigm for expressive and efficient end-to-end speech interaction. 

**Abstract (ZH)**: 直接语音到语音的大语言模型：MOSS-Speech无需文本中介直接理解与生成语音 

---
# Normal-Abnormal Guided Generalist Anomaly Detection 

**Title (ZH)**: 正常-异常引导的通用异常检测 

**Authors**: Yuexin Wang, Xiaolei Wang, Yizheng Gong, Jimin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00495)  

**Abstract**: Generalist Anomaly Detection (GAD) aims to train a unified model on an original domain that can detect anomalies in new target domains. Previous GAD methods primarily use only normal samples as references, overlooking the valuable information contained in anomalous samples that are often available in real-world scenarios. To address this limitation, we propose a more practical approach: normal-abnormal-guided generalist anomaly detection, which leverages both normal and anomalous samples as references to guide anomaly detection across diverse domains. We introduce the Normal-Abnormal Generalist Learning (NAGL) framework, consisting of two key components: Residual Mining (RM) and Anomaly Feature Learning (AFL). RM extracts abnormal patterns from normal-abnormal reference residuals to establish transferable anomaly representations, while AFL adaptively learns anomaly features in query images through residual mapping to identify instance-aware anomalies. Our approach effectively utilizes both normal and anomalous references for more accurate and efficient cross-domain anomaly detection. Extensive experiments across multiple benchmarks demonstrate that our method significantly outperforms existing GAD approaches. This work represents the first to adopt a mixture of normal and abnormal samples as references in generalist anomaly detection. The code and datasets are available at this https URL. 

**Abstract (ZH)**: 通用异常检测（GAD）旨在训练一个统一模型，在原始领域中学习，以检测新目标领域的异常。以往的GAD方法主要仅使用正常样本作为参考，忽视了异常样本中包含的重要信息，而这些异常样本在现实场景中通常可获得。为解决这一局限，我们提出了一种更为实用的方法：正常-异常引导的通用异常检测，该方法利用正常和异常样本作为参考，指导跨域异常检测。我们引入了正常-异常通用学习（NAGL）框架，包含两个关键组件：残差挖掘（RM）和异常特征学习（AFL）。RM从正常-异常参考残差中提取异常模式，以建立可转移的异常表示；AFL通过残差映射在查询图像中自适应地学习异常特征，以识别实例感知的异常。我们的方法有效地利用了正常和异常参考，以实现更准确和高效的跨域异常检测。在多个基准上的广泛实验表明，我们的方法显著优于现有GAD方法。这是首次在通用异常检测中采用正常和异常样本混合作为参考的工作。代码和数据集可在以下链接获取。 

---
# Exploring System 1 and 2 communication for latent reasoning in LLMs 

**Title (ZH)**: 探索LLMs中的潜推理的系统1和系统2通信 

**Authors**: Julian Coda-Forno, Zhuokai Zhao, Qiang Zhang, Dipesh Tamboli, Weiwei Li, Xiangjun Fan, Lizhu Zhang, Eric Schulz, Hsiao-Ping Tseng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00494)  

**Abstract**: Should LLM reasoning live in a separate module, or within a single model's forward pass and representational space? We study dual-architecture latent reasoning, where a fluent Base exchanges latent messages with a Coprocessor, and test two hypotheses aimed at improving latent communication over Liu et al. (2024): (H1) increase channel capacity; (H2) learn communication via joint finetuning. Under matched latent-token budgets on GPT-2 and Qwen-3, H2 is consistently strongest while H1 yields modest gains. A unified soft-embedding baseline, a single model with the same forward pass and shared representations, using the same latent-token budget, nearly matches H2 and surpasses H1, suggesting current dual designs mostly add compute rather than qualitatively improving reasoning. Across GSM8K, ProsQA, and a Countdown stress test with increasing branching factor, scaling the latent-token budget beyond small values fails to improve robustness. Latent analyses show overlapping subspaces with limited specialization, consistent with weak reasoning gains. We conclude dual-model latent reasoning remains promising in principle, but likely requires objectives and communication mechanisms that explicitly shape latent spaces for algorithmic planning. 

**Abstract (ZH)**: LLM推理应当存在于独立模块中还是融入单一模型的前向传递和表示空间中？我们研究了双架构潜在推理，其中流动的基模型与协处理器交换潜在消息，并测试了两种旨在改进潜在通信的假设：(H1) 提升信道容量；(H2) 通过联合微调学习通信。在GPT-2和Qwen-3相同的潜在令牌预算下，H2始终最强，而H1仅带来微小收益。使用相同潜在令牌预算的统一软嵌入基线，一个具有相同前向传递和共享表示的单一模型，几乎与H2相当并超过H1，表明当前的双架构设计主要增加计算量而非从质地上提高推理能力。在GSM8K、ProsQA以及随着分支因子增加的 Countdown 压力测试中，将潜在令牌预算扩大到小值以上未能提高鲁棒性。潜在分析显示存在重叠但专业化有限的子空间，这与弱推理增益一致。我们得出结论，原则上双模型潜在推理仍然有前景，但可能需要明确塑造潜在空间以用于算法规划的目标和通信机制。 

---
# From Human Hands to Robot Arms: Manipulation Skills Transfer via Trajectory Alignment 

**Title (ZH)**: 从人力操作到机器人手臂：通过轨迹对齐实现操作技能转移 

**Authors**: Han Zhou, Jinjin Cao, Liyuan Ma, Xueji Fang, Guo-jun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00491)  

**Abstract**: Learning diverse manipulation skills for real-world robots is severely bottlenecked by the reliance on costly and hard-to-scale teleoperated demonstrations. While human videos offer a scalable alternative, effectively transferring manipulation knowledge is fundamentally hindered by the significant morphological gap between human and robotic embodiments. To address this challenge and facilitate skill transfer from human to robot, we introduce Traj2Action,a novel framework that bridges this embodiment gap by using the 3D trajectory of the operational endpoint as a unified intermediate representation, and then transfers the manipulation knowledge embedded in this trajectory to the robot's actions. Our policy first learns to generate a coarse trajectory, which forms an high-level motion plan by leveraging both human and robot data. This plan then conditions the synthesis of precise, robot-specific actions (e.g., orientation and gripper state) within a co-denoising framework. Extensive real-world experiments on a Franka robot demonstrate that Traj2Action boosts the performance by up to 27% and 22.25% over $\pi_0$ baseline on short- and long-horizon real-world tasks, and achieves significant gains as human data scales in robot policy learning. Our project website, featuring code and video demonstrations, is available at this https URL. 

**Abstract (ZH)**: 基于3D操作末端轨迹的技能转移框架Traj2Action：从人类到机器人技能转移的有效途径 

---
# Black-Box Time-Series Domain Adaptation via Cross-Prompt Foundation Models 

**Title (ZH)**: 黑盒时间序列域适应通过跨提示基础模型 

**Authors**: M. T. Furqon, Mahardhika Pratama, Igor Skrjanc, Lin Liu, Habibullah Habibullah, Kutluyil Dogancay  

**Link**: [PDF](https://arxiv.org/pdf/2510.00487)  

**Abstract**: The black-box domain adaptation (BBDA) topic is developed to address the privacy and security issues where only an application programming interface (API) of the source model is available for domain adaptations. Although the BBDA topic has attracted growing research attentions, existing works mostly target the vision applications and are not directly applicable to the time-series applications possessing unique spatio-temporal characteristics. In addition, none of existing approaches have explored the strength of foundation model for black box time-series domain adaptation (BBTSDA). This paper proposes a concept of Cross-Prompt Foundation Model (CPFM) for the BBTSDA problems. CPFM is constructed under a dual branch network structure where each branch is equipped with a unique prompt to capture different characteristics of data distributions. In the domain adaptation phase, the reconstruction learning phase in the prompt and input levels is developed. All of which are built upon a time-series foundation model to overcome the spatio-temporal dynamic. Our rigorous experiments substantiate the advantage of CPFM achieving improved results with noticeable margins from its competitors in three time-series datasets of different application domains. 

**Abstract (ZH)**: 黑箱时间序列域适应中的跨提示基础模型（CPFM） 

---
# PodEval: A Multimodal Evaluation Framework for Podcast Audio Generation 

**Title (ZH)**: PodEval：播客音频生成的多模态评估框架 

**Authors**: Yujia Xiao, Liumeng Xue, Lei He, Xinyi Chen, Aemon Yat Fei Chiu, Wenjie Tian, Shaofei Zhang, Qiuqiang Kong, Xinfa Zhu, Wei Xue, Tan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00485)  

**Abstract**: Recently, an increasing number of multimodal (text and audio) benchmarks have emerged, primarily focusing on evaluating models' understanding capability. However, exploration into assessing generative capabilities remains limited, especially for open-ended long-form content generation. Significant challenges lie in no reference standard answer, no unified evaluation metrics and uncontrollable human judgments. In this work, we take podcast-like audio generation as a starting point and propose PodEval, a comprehensive and well-designed open-source evaluation framework. In this framework: 1) We construct a real-world podcast dataset spanning diverse topics, serving as a reference for human-level creative quality. 2) We introduce a multimodal evaluation strategy and decompose the complex task into three dimensions: text, speech and audio, with different evaluation emphasis on "Content" and "Format". 3) For each modality, we design corresponding evaluation methods, involving both objective metrics and subjective listening test. We leverage representative podcast generation systems (including open-source, close-source, and human-made) in our experiments. The results offer in-depth analysis and insights into podcast generation, demonstrating the effectiveness of PodEval in evaluating open-ended long-form audio. This project is open-source to facilitate public use: this https URL. 

**Abstract (ZH)**: 近期，涌现出越来越多的多模态（文本和音频）基准，主要侧重于评估模型的理解能力。然而，对生成能力的评估探索仍相对有限，尤其是对于开放式长格式内容生成。由于缺乏参考标准答案、统一的评估指标和可控的人工判断，存在重大挑战。本文以类似播客的音频生成作为起点，提出PodEval，一种全面且设计合理的开源评估框架。在该框架中：1) 构建了一个涵盖多样主题的现实播客数据集，作为人类水平创造力的参考。2) 引入了多模态评估策略，并将复杂任务分解为三个维度：文本、语音和音频，对“内容”和“格式”有不同的评估重点。3) 对每个模态，设计了相应的评估方法，包括客观指标和主观听力测试。我们在实验中使用了代表性的播客生成系统（包括开源、闭源和人为制作的系统）。实验结果提供了对播客生成的深入分析和见解，证明了PodEval在评估开放式长格式音频方面的有效性。该项目已开源以方便公共使用：this https URL。 

---
# Make a Video Call with LLM: A Measurement Campaign over Five Mainstream Apps 

**Title (ZH)**: 使用大语言模型进行视频通话：一项针对五大主流应用的测量campaign 

**Authors**: Jiayang Xu, Xiangjie Huang, Zijie Li, Zili Meng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00481)  

**Abstract**: In 2025, Large Language Model (LLM) services have launched a new feature -- AI video chat -- allowing users to interact with AI agents via real-time video communication (RTC), just like chatting with real people. Despite its significance, no systematic study has characterized the performance of existing AI video chat systems. To address this gap, this paper proposes a comprehensive benchmark with carefully designed metrics across four dimensions: quality, latency, internal mechanisms, and system overhead. Using custom testbeds, we further evaluate five mainstream AI video chatbots with this benchmark. This work provides the research community a baseline of real-world performance and identifies unique system bottlenecks. In the meantime, our benchmarking results also open up several research questions for future optimizations of AI video chatbots. 

**Abstract (ZH)**: 2025年大型语言模型服务推出新功能——AI视频聊天——使用户能够通过实时视频通信（RTC）与AI代理互动，就像与真人聊天一样。尽管这一功能具有重要意义，但尚未对现有AI视频聊天系统进行全面研究。为填补这一空白，本文提出了一种综合基准，涵盖了四个维度的质量、延迟、内部机制和系统开销，并通过自定义测试床对五种主流AI视频聊天机器人进行了进一步评估。本工作为研究社区提供了一个实际性能基准，并识别出独特的系统瓶颈。同时，我们的基准测试结果也为未来AI视频聊天机器人的优化提出了多个研究问题。 

---
# Analyzing Latent Concepts in Code Language Models 

**Title (ZH)**: 分析代码语言模型中的潜在概念 

**Authors**: Arushi Sharma, Vedant Pungliya, Christopher J. Quinn, Ali Jannesari  

**Link**: [PDF](https://arxiv.org/pdf/2510.00476)  

**Abstract**: Interpreting the internal behavior of large language models trained on code remains a critical challenge, particularly for applications demanding trust, transparency, and semantic robustness. We propose Code Concept Analysis (CoCoA): a global post-hoc interpretability framework that uncovers emergent lexical, syntactic, and semantic structures in a code language model's representation space by clustering contextualized token embeddings into human-interpretable concept groups. We propose a hybrid annotation pipeline that combines static analysis tool-based syntactic alignment with prompt-engineered large language models (LLMs), enabling scalable labeling of latent concepts across abstraction levels. We analyse the distribution of concepts across layers and across three finetuning tasks. Emergent concept clusters can help identify unexpected latent interactions and be used to identify trends and biases within the model's learned representations. We further integrate LCA with local attribution methods to produce concept-grounded explanations, improving the coherence and interpretability of token-level saliency. Empirical evaluations across multiple models and tasks show that LCA discovers concepts that remain stable under semantic-preserving perturbations (average Cluster Sensitivity Index, CSI = 0.288) and evolve predictably with fine-tuning. In a user study, concept-augmented explanations disambiguate token roles. In a user study on the programming-language classification task, concept-augmented explanations disambiguated token roles and improved human-centric explainability by 37 percentage points compared with token-level attributions using Integrated Gradients. 

**Abstract (ZH)**: 大型编程语言模型训练后的内部行为解释仍然是一个关键挑战，特别是在需要信任、透明性和语义稳健性的应用程序中。我们提出Code Concept Analysis（CoCoA）：一种全局的后验解释框架，通过聚类上下文化词嵌入到人类可解释的概念组中，揭示代码语言模型表示空间中 Emergent 的词汇、语法和语义结构。我们提出了一种混合注解工作流，结合静态分析工具基于的语法对齐和提示工程的大规模语言模型（LLMs），以实现跨抽象层次的潜在概念的可扩展标注。我们分析了概念在各层以及三个微调任务中的分布。Emergent 的概念簇有助于识别意外的潜在交互，并可用于识别模型学习表示中的趋势和偏向。我们进一步将局部归因方法与 LCA（本地概念分析）结合，生成基于概念的解释，提高词级显著性的连贯性和可解释性。在多个模型和任务上的实证评估显示，LCA 发现的概念在语义保持扰动下保持稳定（平均聚类敏感性指数，CSI = 0.288），并随着微调可预测地演变。在用户研究中，概念增强的解释消除了词的角色歧义。在编程语言分类任务的用户研究中，概念增强的解释消除了词的角色歧义，并使以人类为中心的可解释性提高了37个百分点，相对于使用集成梯度的词级归因。 

---
# Feature Identification via the Empirical NTK 

**Title (ZH)**: 经验NTK特征识别 

**Authors**: Jennifer Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.00468)  

**Abstract**: We provide evidence that eigenanalysis of the empirical neural tangent kernel (eNTK) can surface the features used by trained neural networks. Across two standard toy models for mechanistic interpretability, Toy Models of Superposition (TMS) and a 1-layer MLP trained on modular addition, we find that the eNTK exhibits sharp spectral cliffs whose top eigenspaces align with ground-truth features. In TMS, the eNTK recovers the ground-truth features in both the sparse (high superposition) and dense regimes. In modular arithmetic, the eNTK can be used to recover Fourier feature families. Moreover, we provide evidence that a layerwise eNTK localizes features to specific layers and that the evolution of the eNTK eigenspectrum can be used to diagnose the grokking phase transition. These results suggest that eNTK analysis may provide a practical handle for feature discovery and for detecting phase changes in small models. 

**Abstract (ZH)**: 我们提供了实证证据表明，神经 tangent 核的特征分析能够揭示训练神经网络使用的特点。在两个标准的机制可解释性玩具模型——叠加模型（TMS）和用于模Arithmetic加运算训练的单层MLP中，我们发现eNTK表现出尖锐的谱阶，其主导特征空间与真实特征对齐。在TMS中，eNTK能够在稀疏和稠密模式下恢复真实特征。在模Arithmetic中，eNTK可用于恢复Fourier特征家族。此外，我们提供了证据表明，分层的eNTK能够将特征局部化到特定层，并且eNTK特征谱的变化可以用于诊断掌握相变。这些结果表明，eNTK分析可能为特征发现和小型模型中检测相变提供实用的方法。 

---
# Integrating Offline Pre-Training with Online Fine-Tuning: A Reinforcement Learning Approach for Robot Social Navigation 

**Title (ZH)**: 将离线预训练与在线微调相结合：基于强化学习的机器人社会导航方法 

**Authors**: Run Su, Hao Fu, Shuai Zhou, Yingao Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00466)  

**Abstract**: Offline reinforcement learning (RL) has emerged as a promising framework for addressing robot social navigation challenges. However, inherent uncertainties in pedestrian behavior and limited environmental interaction during training often lead to suboptimal exploration and distributional shifts between offline training and online deployment. To overcome these limitations, this paper proposes a novel offline-to-online fine-tuning RL algorithm for robot social navigation by integrating Return-to-Go (RTG) prediction into a causal Transformer architecture. Our algorithm features a spatiotem-poral fusion model designed to precisely estimate RTG values in real-time by jointly encoding temporal pedestrian motion patterns and spatial crowd dynamics. This RTG prediction framework mitigates distribution shift by aligning offline policy training with online environmental interactions. Furthermore, a hybrid offline-online experience sampling mechanism is built to stabilize policy updates during fine-tuning, ensuring balanced integration of pre-trained knowledge and real-time adaptation. Extensive experiments in simulated social navigation environments demonstrate that our method achieves a higher success rate and lower collision rate compared to state-of-the-art baselines. These results underscore the efficacy of our algorithm in enhancing navigation policy robustness and adaptability. This work paves the way for more reliable and adaptive robotic navigation systems in real-world applications. 

**Abstract (ZH)**: 基于Return-to-Go预测的离线到在线强化学习算法在机器人社会导航中的应用 

---
# TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting 

**Title (ZH)**: TimeEmb：一种轻量级静态-动态解耦框架用于时间序列预测 

**Authors**: Mingyuan Xia, Chunxu Zhang, Zijian Zhang, Hao Miao, Qidong Liu, Yuanshao Zhu, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00461)  

**Abstract**: Temporal non-stationarity, the phenomenon that time series distributions change over time, poses fundamental challenges to reliable time series forecasting. Intuitively, the complex time series can be decomposed into two factors, \ie time-invariant and time-varying components, which indicate static and dynamic patterns, respectively. Nonetheless, existing methods often conflate the time-varying and time-invariant components, and jointly learn the combined long-term patterns and short-term fluctuations, leading to suboptimal performance facing distribution shifts. To address this issue, we initiatively propose a lightweight static-dynamic decomposition framework, TimeEmb, for time series forecasting. TimeEmb innovatively separates time series into two complementary components: (1) time-invariant component, captured by a novel global embedding module that learns persistent representations across time series, and (2) time-varying component, processed by an efficient frequency-domain filtering mechanism inspired by full-spectrum analysis in signal processing. Experiments on real-world datasets demonstrate that TimeEmb outperforms state-of-the-art baselines and requires fewer computational resources. We conduct comprehensive quantitative and qualitative analyses to verify the efficacy of static-dynamic disentanglement. This lightweight framework can also improve existing time-series forecasting methods with simple integration. To ease reproducibility, the code is available at this https URL. 

**Abstract (ZH)**: 时间非平稳现象给可靠的时间序列预测带来了根本性挑战。直观地讲，复杂的时间序列可以分解为两个因素，即时间不变和时间变化的组件，分别代表静态和动态模式。然而，现有方法往往将时间变化和时间不变的组件混淆在一起，联合学习长期趋势和短期波动，导致在分布转移面前表现不佳。为解决这一问题，我们提出了一种轻量级的静态-动态分解框架TimeEmb用于时间序列预测。TimeEmb创新性地将时间序列分解为两个互补的组件：（1）时间不变组件，通过一个新颖的全局嵌入模块学习跨时间序列的持久表示；（2）时间变化组件，通过灵感源自信号处理中全谱分析的高度有效的频域滤波机制处理。实验证明，TimeEmb在实际数据集上的表现优于最先进的基线，并且需要更少的计算资源。通过对静态-动态分解的有效性进行全面的定量和定性分析，验证了该轻量级框架的优越性。该框架还可通过简单集成来改进现有的时间序列预测方法，以促进可再现性，代码已开源。 

---
# UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction 

**Title (ZH)**: UrbanGraph: 物理约束的空间-时间动态异质图城市微气候预测 

**Authors**: Weilin Xin, Chenyu Huang, Peilin Li, Jing Zhong, Jiawei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00457)  

**Abstract**: With rapid urbanization, predicting urban microclimates has become critical, as it affects building energy demand and public health risks. However, existing generative and homogeneous graph approaches fall short in capturing physical consistency, spatial dependencies, and temporal variability. To address this, we introduce UrbanGraph, a physics-informed framework integrating heterogeneous and dynamic spatio-temporal graphs. It encodes key physical processes -- vegetation evapotranspiration, shading, and convective diffusion -- while modeling complex spatial dependencies among diverse urban entities and their temporal evolution. We evaluate UrbanGraph on UMC4/12, a physics-based simulation dataset covering diverse urban configurations and climates. Results show that UrbanGraph improves $R^2$ by up to 10.8% and reduces FLOPs by 17.0% over all baselines, with heterogeneous and dynamic graphs contributing 3.5% and 7.1% gains. Our dataset provides the first high-resolution benchmark for spatio-temporal microclimate modeling, and our method extends to broader urban heterogeneous dynamic computing tasks. 

**Abstract (ZH)**: 随着快速城市化，预测城市微气候变得至关重要，因为它影响建筑能源需求和公共健康风险。然而，现有的生成性和同质性图方法在捕捉物理一致性、空间依赖性和时间变异性方面存在不足。为解决这一问题，我们引入了UrbanGraph，这是一种结合了异质性和动态时空图的物理导向框架。它编码了关键的物理过程——植被蒸腾、遮荫和对流扩散，同时建模了各种城市实体之间复杂的空间关系及其时间演变。我们在基于物理的模拟数据集UMC4/12上评估了UrbanGraph，该数据集涵盖了多种城市配置和气候。结果表明，与所有基线方法相比，UrbanGraph的$R^2$提高了10.8%，计算量减少了17.0%，异质性和动态图分别贡献了3.5%和7.1%的改善。我们的数据集提供了首个高分辨率的时空微气候建模基准，并且我们的方法扩展到更广泛的异质动态城市计算任务。 

---
# Measuring and Controlling the Spectral Bias for Self-Supervised Image Denoising 

**Title (ZH)**: 测量和控制自监督图像去噪中的光谱偏差 

**Authors**: Wang Zhang, Huaqiu Li, Xiaowan Hu, Tao Jiang, Zikang Chen, Haoqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00454)  

**Abstract**: Current self-supervised denoising methods for paired noisy images typically involve mapping one noisy image through the network to the other noisy image. However, after measuring the spectral bias of such methods using our proposed Image Pair Frequency-Band Similarity, it suffers from two practical limitations. Firstly, the high-frequency structural details in images are not preserved well enough. Secondly, during the process of fitting high frequencies, the network learns high-frequency noise from the mapped noisy images. To address these challenges, we introduce a Spectral Controlling network (SCNet) to optimize self-supervised denoising of paired noisy images. First, we propose a selection strategy to choose frequency band components for noisy images, to accelerate the convergence speed of training. Next, we present a parameter optimization method that restricts the learning ability of convolutional kernels to high-frequency noise using the Lipschitz constant, without changing the network structure. Finally, we introduce the Spectral Separation and low-rank Reconstruction module (SSR module), which separates noise and high-frequency details through frequency domain separation and low-rank space reconstruction, to retain the high-frequency structural details of images. Experiments performed on synthetic and real-world datasets verify the effectiveness of SCNet. 

**Abstract (ZH)**: 基于频带控制的配对 noisy 图像自监督去噪方法 

---
# Cloud Investigation Automation Framework (CIAF): An AI-Driven Approach to Cloud Forensics 

**Title (ZH)**: 基于AI驱动的云取证自动化框架（CIAF） 

**Authors**: Dalal Alharthi, Ivan Roberto Kawaminami Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00452)  

**Abstract**: Large Language Models (LLMs) have gained prominence in domains including cloud security and forensics. Yet cloud forensic investigations still rely on manual analysis, making them time-consuming and error-prone. LLMs can mimic human reasoning, offering a pathway to automating cloud log analysis. To address this, we introduce the Cloud Investigation Automation Framework (CIAF), an ontology-driven framework that systematically investigates cloud forensic logs while improving efficiency and accuracy. CIAF standardizes user inputs through semantic validation, eliminating ambiguity and ensuring consistency in log interpretation. This not only enhances data quality but also provides investigators with reliable, standardized information for decision-making. To evaluate security and performance, we analyzed Microsoft Azure logs containing ransomware-related events. By simulating attacks and assessing CIAF's impact, results showed significant improvement in ransomware detection, achieving precision, recall, and F1 scores of 93 percent. CIAF's modular, adaptable design extends beyond ransomware, making it a robust solution for diverse cyberattacks. By laying the foundation for standardized forensic methodologies and informing future AI-driven automation, this work underscores the role of deterministic prompt engineering and ontology-based validation in enhancing cloud forensic investigations. These advancements improve cloud security while paving the way for efficient, automated forensic workflows. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在云安全和法医学等领域崭露头角。然而，云法医学调查仍依赖于人工分析，导致耗时且易出错。LLMs能够模拟人类推理，提供自动分析云日志的可能性。为解决这一问题，我们提出了云调查自动化框架（CIAF），一个通过本体驱动系统化调查云法医学日志的框架，同时提高效率和准确性。CIAF通过语义验证标准化用户输入，消除歧义并确保日志解释的一致性。这不仅提高了数据质量，还为调查人员提供了可靠且标准化的信息以便决策。为了评估安全性和性能，我们在包含赎金ware相关事件的Microsoft Azure日志中进行了分析。通过模拟攻击并评估CIAF的影响，结果显示在赎金ware检测方面有显著改进，精度、召回率和F1分数达到93%。CIAF具有模块化和适应性设计，适用于多种网络攻击，是稳健的自动化解决方案。通过为标准化法医学方法奠定基础并指导未来基于AI的自动化，这项工作强调了确定性提示工程和基于本体的验证在增强云法医学调查中的作用。这些进展改善了云安全，并为高效自动化的法医学工作流程铺平了道路。 

---
# A Call to Action for a Secure-by-Design Generative AI Paradigm 

**Title (ZH)**: 面向设计即安全的生成AI范式的行动呼吁 

**Authors**: Dalal Alharthi, Ivan Roberto Kawaminami Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00451)  

**Abstract**: Large language models have gained widespread prominence, yet their vulnerability to prompt injection and other adversarial attacks remains a critical concern. This paper argues for a security-by-design AI paradigm that proactively mitigates LLM vulnerabilities while enhancing performance. To achieve this, we introduce PromptShield, an ontology-driven framework that ensures deterministic and secure prompt interactions. It standardizes user inputs through semantic validation, eliminating ambiguity and mitigating adversarial manipulation. To assess PromptShield's security and performance capabilities, we conducted an experiment on an agent-based system to analyze cloud logs within Amazon Web Services (AWS), containing 493 distinct events related to malicious activities and anomalies. By simulating prompt injection attacks and assessing the impact of deploying PromptShield, our results demonstrate a significant improvement in model security and performance, achieving precision, recall, and F1 scores of approximately 94%. Notably, the ontology-based framework not only mitigates adversarial threats but also enhances the overall performance and reliability of the system. Furthermore, PromptShield's modular and adaptable design ensures its applicability beyond cloud security, making it a robust solution for safeguarding generative AI applications across various domains. By laying the groundwork for AI safety standards and informing future policy development, this work stimulates a crucial dialogue on the pivotal role of deterministic prompt engineering and ontology-based validation in ensuring the safe and responsible deployment of LLMs in high-stakes environments. 

**Abstract (ZH)**: 大型语言模型已获得广泛认可，但它们对指令注入和其他对抗攻击的脆弱性仍然是一个关键问题。本文提倡一种设计安全的人工智能范式，旨在主动减轻LLM的脆弱性并提升性能。为了实现这一目标，我们引入了PromptShield，这是一种本体驱动的框架，确保指令交互的确定性和安全。该框架通过语义验证标准化用户输入，消除歧义并减轻对抗性操纵。为评估PromptShield的安全性和性能能力，我们在Amazon Web Services (AWS) 基于代理的系统上进行实验，分析了包含493个与恶意活动和异常相关的不同事件的日志。通过模拟指令注入攻击并评估部署PromptShield的影响，我们的结果显示了模型安全性和性能的显著改进，达到了约94%的精确率、召回率和F1分数。基于本体的框架不仅能缓解对抗性威胁，还能提升系统的整体性能和可靠性。此外，PromptShield的模块化和适应性设计使其不仅适用于云安全，还能作为一种适用于各种领域的生成人工智能应用的安全解决方案。通过为AI安全标准奠定基础并为未来政策制定提供信息，这项工作激发了关于确定性指令工程和基于本体验证在确保在高度敏感环境中安全负责任地部署LLM中的关键作用的重要对话。 

---
# Plug-and-Play Prompt Refinement via Latent Feedback for Diffusion Model Alignment 

**Title (ZH)**: 基于潜在反馈的即插即用提示精炼以实现扩散模型对齐 

**Authors**: Suhyeon Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.00430)  

**Abstract**: Despite the recent progress, reinforcement learning (RL)-based fine-tuning of diffusion models often struggles with generalization, composability, and robustness against reward hacking. Recent studies have explored prompt refinement as a modular alternative, but most adopt a feed-forward approach that applies a single refined prompt throughout the entire sampling trajectory, thereby failing to fully leverage the sequential nature of reinforcement learning. To address this, here we introduce PromptLoop, a plug-and-play RL framework that incorporates latent feedback into step-wise prompt refinement. Rather than modifying diffusion model weights, a multimodal large language model (MLLM) is trained with RL to iteratively update prompts based on intermediate latent states of diffusion models. This design achieves a structural analogy to the Diffusion RL approach, while retaining the flexibility and generality of prompt-based alignment. Extensive experiments across diverse reward functions and diffusion backbones demonstrate that PromptLoop (i) achieves effective reward optimization, (ii) generalizes seamlessly to unseen models, (iii) composes orthogonally with existing alignment methods, and (iv) mitigates over-optimization and reward hacking. 

**Abstract (ZH)**: 尽管取得了近期进展，基于强化学习（RL）的扩散模型微调往往在泛化、组件化以及对抗奖励作弊的鲁棒性方面存在问题。最近的研究探索了提示精炼作为模块化替代方案，但大多数方法采用单一前馈方式，在整个采样轨迹中应用单一精炼提示，从而未能充分利用强化学习的序列性质。为解决这一问题，我们引入了PromptLoop，这是一种插件式RL框架，将潜在反馈纳入逐步提示精炼中。这种方法通过迭代更新基于扩散模型中间潜在状态的提示，而不是修改扩散模型权重，实现了与Diffusion RL方法的结构性类比，同时保留基于提示对齐的灵活性和通用性。跨多种奖励函数和扩散骨干的广泛实验表明，PromptLoop能够（i）实现有效的奖励优化，（ii）无缝泛化到未见过的模型，（iii）与现有对齐方法正交组合，以及（iv）缓解过度优化和奖励作弊问题。 

---
# Automated Structured Radiology Report Generation with Rich Clinical Context 

**Title (ZH)**: 丰富的临床上下文条件下的自动结构化放射学报告生成 

**Authors**: Seongjae Kang, Dong Bok Lee, Juho Jung, Dongseop Kim, Won Hwa Kim, Sunghoon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2510.00428)  

**Abstract**: Automated structured radiology report generation (SRRG) from chest X-ray images offers significant potential to reduce workload of radiologists by generating reports in structured formats that ensure clarity, consistency, and adherence to clinical reporting standards. While radiologists effectively utilize available clinical contexts in their diagnostic reasoning, existing SRRG systems overlook these essential elements. This fundamental gap leads to critical problems including temporal hallucinations when referencing non-existent clinical contexts. To address these limitations, we propose contextualized SRRG (C-SRRG) that comprehensively incorporates rich clinical context for SRRG. We curate C-SRRG dataset by integrating comprehensive clinical context encompassing 1) multi-view X-ray images, 2) clinical indication, 3) imaging techniques, and 4) prior studies with corresponding comparisons based on patient histories. Through extensive benchmarking with state-of-the-art multimodal large language models, we demonstrate that incorporating clinical context with the proposed C-SRRG significantly improves report generation quality. We publicly release dataset, code, and checkpoints to facilitate future research for clinically-aligned automated RRG at this https URL. 

**Abstract (ZH)**: 基于胸片的自动结构化放射学报告生成（C-SRRG）通过生成符合临床报告标准的结构化报告，显著减少了放射科医师的工作负荷。当前的SRRG系统忽视了这些重要元素，导致在引用不存在的临床背景时出现时间幻觉等关键问题。为了弥补这些不足，我们提出了一种综合临床上下文的C-SRRG方法。通过整合包含多视角X光图像、临床指征、成像技术以及基于患者历史的先前研究和相应比较的综合临床背景，我们利用最新的多模态大型语言模型进行了广泛基准测试，证明了综合临床背景的C-SRRG显著提高了报告生成质量。现公开发布数据集、代码和检查点，以促进未来与临床对齐的自动RGG研究，详情请访问此链接：https://xxxxxx。 

---
# Domain-Specialized Interactive Segmentation Framework for Meningioma Radiotherapy Planning 

**Title (ZH)**: 用于脑膜瘤放射治疗计划的领域特化交互分割框架 

**Authors**: Junhyeok Lee, Han Jang, Kyu Sung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00416)  

**Abstract**: Precise delineation of meningiomas is crucial for effective radiotherapy (RT) planning, directly influencing treatment efficacy and preservation of adjacent healthy tissues. While automated deep learning approaches have demonstrated considerable potential, achieving consistently accurate clinical segmentation remains challenging due to tumor heterogeneity. Interactive Medical Image Segmentation (IMIS) addresses this challenge by integrating advanced AI techniques with clinical input. However, generic segmentation tools, despite widespread applicability, often lack the specificity required for clinically critical and disease-specific tasks like meningioma RT planning. To overcome these limitations, we introduce Interactive-MEN-RT, a dedicated IMIS tool specifically developed for clinician-assisted 3D meningioma segmentation in RT workflows. The system incorporates multiple clinically relevant interaction methods, including point annotations, bounding boxes, lasso tools, and scribbles, enhancing usability and clinical precision. In our evaluation involving 500 contrast-enhanced T1-weighted MRI scans from the BraTS 2025 Meningioma RT Segmentation Challenge, Interactive-MEN-RT demonstrated substantial improvement compared to other segmentation methods, achieving Dice similarity coefficients of up to 77.6\% and Intersection over Union scores of 64.8\%. These results emphasize the need for clinically tailored segmentation solutions in critical applications such as meningioma RT planning. The code is publicly available at: this https URL 

**Abstract (ZH)**: 精确界定脑膜瘤对于有效的放射治疗计划至关重要，直接关系到治疗效果和相邻健康组织的保护。尽管自动深度学习方法表现出显著潜力，但由于肿瘤异质性，实现一致准确的临床分割仍然具有挑战性。交互式医学图像分割（IMIS）通过整合先进的AI技术与临床输入来应对这一挑战。然而，通用分割工具虽具有广泛的适用性，但在如脑膜瘤放射治疗计划等临床关键和疾病特异性任务中往往缺乏所需的特异性。为克服这些限制，我们提出了一种专门用于脑膜瘤放射治疗工作流程中临床辅助3D分割的Interactive-MEN-RT交互式分割工具。该系统采用多种临床相关交互方法，包括点标注、边界框、套索工具和涂抹工具，增强了易用性和临床精度。在对2025 BraTS脑膜瘤放射治疗分割挑战中500张对比增强T1加权MRI扫描进行评价后，Interactive-MEN-RT相比其他分割方法显著提高，Dice相似系数达到77.6%，交并比达到64.8%。这些结果强调了在如脑膜瘤放射治疗计划等关键应用中需要临床定制的分割解决方案。代码公开链接：this https URL 

---
# David and Goliath in Medical Vision: Convolutional Networks vs Biomedical Vision Language Models 

**Title (ZH)**: 医疗视觉中的大卫与歌利亚：卷积网络 vs 生物医学视觉语言模型 

**Authors**: Ran Tong, Jiaqi Liu, Su Liu, Jiexi Xu, Lanruo Wang, Tong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00411)  

**Abstract**: The accurate interpretation of chest radiographs using automated methods is a critical task in medical imaging. This paper presents a comparative analysis between a supervised lightweight Convolutional Neural Network (CNN) and a state-of-the-art, zero-shot medical Vision-Language Model (VLM), BiomedCLIP, across two distinct diagnostic tasks: pneumonia detection on the PneumoniaMNIST benchmark and tuberculosis detection on the Shenzhen TB dataset. Our experiments show that supervised CNNs serve as highly competitive baselines in both cases. While the default zero-shot performance of the VLM is lower, we demonstrate that its potential can be unlocked via a simple yet crucial remedy: decision threshold calibration. By optimizing the classification threshold on a validation set, the performance of BiomedCLIP is significantly boosted across both datasets. For pneumonia detection, calibration enables the zero-shot VLM to achieve a superior F1-score of 0.8841, surpassing the supervised CNN's 0.8803. For tuberculosis detection, calibration dramatically improves the F1-score from 0.4812 to 0.7684, bringing it close to the supervised baseline's 0.7834. This work highlights a key insight: proper calibration is essential for leveraging the full diagnostic power of zero-shot VLMs, enabling them to match or even outperform efficient, task-specific supervised models. 

**Abstract (ZH)**: 使用自动方法准确解读胸部X光片是医学成像中的关键任务。本文在肺炎检测（基于PneumoniaMNIST基准）和肺结核检测（基于深圳肺结核数据集）两个不同的诊断任务中，比较分析了监督的轻量级卷积神经网络（CNN）和最先进的零样本医学视觉-语言模型（VLM）BiomedCLIP的表现。实验结果显示，监督的CNN在两个任务中均作为强有力的基线模型。虽然VLM的零样本默认性能较低，但我们证明通过一个简单而关键的改进——决策阈值校准，可以充分释放其潜力。通过在验证集上优化分类阈值，BiomedCLIP在两个数据集上的性能显著提升。对于肺炎检测，校准使其零样本VLM的F1分数达到0.8841，超过监督CNN的0.8803。对于肺结核检测，校准将F1分数提升至0.7684，接近监督基线的0.7834。本文强调了一个重要见解：适当的校准对于充分发挥零样本VLM的诊断潜力至关重要，使其能够匹配甚至超越高效的任务特定监督模型。 

---
# EgoTraj-Bench: Towards Robust Trajectory Prediction Under Ego-view Noisy Observations 

**Title (ZH)**: EgoTraj-Bench: 在自视图噪声观测下的稳健轨迹预测评测框架 

**Authors**: Jiayi Liu, Jiaming Zhou, Ke Ye, Kun-Yu Lin, Allan Wang, Junwei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00405)  

**Abstract**: Reliable trajectory prediction from an ego-centric perspective is crucial for robotic navigation in human-centric environments. However, existing methods typically assume idealized observation histories, failing to account for the perceptual artifacts inherent in first-person vision, such as occlusions, ID switches, and tracking drift. This discrepancy between training assumptions and deployment reality severely limits model robustness. To bridge this gap, we introduce EgoTraj-Bench, the first real-world benchmark that grounds noisy, first-person visual histories in clean, bird's-eye-view future trajectories, enabling robust learning under realistic perceptual constraints. Building on this benchmark, we propose BiFlow, a dual-stream flow matching model that concurrently denoises historical observations and forecasts future motion by leveraging a shared latent representation. To better model agent intent, BiFlow incorporates our EgoAnchor mechanism, which conditions the prediction decoder on distilled historical features via feature modulation. Extensive experiments show that BiFlow achieves state-of-the-art performance, reducing minADE and minFDE by 10-15% on average and demonstrating superior robustness. We anticipate that our benchmark and model will provide a critical foundation for developing trajectory forecasting systems truly resilient to the challenges of real-world, ego-centric perception. 

**Abstract (ZH)**: 从自视点视角进行可靠轨迹预测对于人类中心环境中的机器人导航至关重要。然而，现有方法通常假设理想化的观测历史，未能考虑到第一人称视见固有的知觉 artifact，例如遮挡、ID交换和跟踪漂移。训练假设与部署现实之间的这种差距严重限制了模型的鲁棒性。为弥补这一差距，我们引入了EgoTraj-Bench，这是首个将嘈杂的第一人称视觉历史与干净的鸟瞰未来轨迹联系起来的真实世界基准，从而在现实知觉约束下实现鲁棒学习。在此基准之上，我们提出了BiFlow，一种双流流动匹配模型，通过共享潜在表示同时对历史观察进行去噪并预测未来运动。为了更好地建模代理意图，BiFlow 引入了我们的 EgoAnchor 机制，该机制通过对特征进行调节来条件化预测解码器上直馏历史特征。广泛的经验表明，BiFlow 达到了最先进的性能，平均将 minADE 和 minFDE 减少 10-15%，并且表现出更出色的鲁棒性。我们预计，我们的基准和模型将为开发真正能够抵御真实世界自视点感知挑战的轨迹预测系统奠定关键基础。 

---
# AbsTopK: Rethinking Sparse Autoencoders For Bidirectional Features 

**Title (ZH)**: AbsTopK: 重新思考双向特征的稀疏自编码器 

**Authors**: Xudong Zhu, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00404)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as powerful techniques for interpretability of large language models (LLMs), aiming to decompose hidden states into meaningful semantic features. While several SAE variants have been proposed, there remains no principled framework to derive SAEs from the original dictionary learning formulation. In this work, we introduce such a framework by unrolling the proximal gradient method for sparse coding. We show that a single-step update naturally recovers common SAE variants, including ReLU, JumpReLU, and TopK. Through this lens, we reveal a fundamental limitation of existing SAEs: their sparsity-inducing regularizers enforce non-negativity, preventing a single feature from representing bidirectional concepts (e.g., male vs. female). This structural constraint fragments semantic axes into separate, redundant features, limiting representational completeness. To address this issue, we propose AbsTopK SAE, a new variant derived from the $\ell_0$ sparsity constraint that applies hard thresholding over the largest-magnitude activations. By preserving both positive and negative activations, AbsTopK uncovers richer, bidirectional conceptual representations. Comprehensive experiments across four LLMs and seven probing and steering tasks show that AbsTopK improves reconstruction fidelity, enhances interpretability, and enables single features to encode contrasting concepts. Remarkably, AbsTopK matches or even surpasses the Difference-in-Mean method, a supervised approach that requires labeled data for each concept and has been shown in prior work to outperform SAEs. 

**Abstract (ZH)**: 稀疏自编码模型（SAEs）已成为大型语言模型（LLMs）可解释性的强大技术，旨在将隐藏状态分解为有意义的语义特征。虽然已经提出了几种SAE变体，但仍缺乏从原始字典学习公式中推导SAE的原理性框架。在这项工作中，我们通过展开 proximity梯度方法中的稀疏编码引入了这样一个框架。我们展示了单步更新自然恢复了常见的SAE变体，包括ReLU、JumpReLU和TopK。通过这一视角，我们揭示了现有SAE的基本局限性：它们的稀疏性诱导正则化项强制非负性，阻止单个特征表示双向概念（例如，男性 vs. 女性）。这种结构约束将语义轴分割为独立的冗余特征，限制了表示的完整性。为解决这一问题，我们提出了AbsTopK SAE，这是一种源自ℓ₀稀疏性约束的新变体，它对最大幅度激活应用硬阈值。通过保留正向和负向激活，AbsTopK揭示了更丰富的双向概念表示。跨四个LLM和七个探测任务及引导任务的全面实验表明，AbsTopK提高了重构保真度，增强了可解释性，并使单个特征能够编码对比的概念。令人惊讶的是，AbsTopK的表现与甚至超越了Difference-in-Mean方法，这是一种需要为每个概念标注数据的有监督方法，并在先前的工作中被证明优于SAE。 

---
# Physics-Informed Neural Controlled Differential Equations for Scalable Long Horizon Multi-Agent Motion Forecasting 

**Title (ZH)**: 基于物理信息神经控制差分方程的可扩展多agent长时 horizon 运动预测 

**Authors**: Shounak Sural, Charles Kekeh, Wenliang Liu, Federico Pecora, Mouhacine Benosman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00401)  

**Abstract**: Long-horizon motion forecasting for multiple autonomous robots is challenging due to non-linear agent interactions, compounding prediction errors, and continuous-time evolution of dynamics. Learned dynamics of such a system can be useful in various applications such as travel time prediction, prediction-guided planning and generative simulation. In this work, we aim to develop an efficient trajectory forecasting model conditioned on multi-agent goals. Motivated by the recent success of physics-guided deep learning for partially known dynamical systems, we develop a model based on neural Controlled Differential Equations (CDEs) for long-horizon motion forecasting. Unlike discrete-time methods such as RNNs and transformers, neural CDEs operate in continuous time, allowing us to combine physics-informed constraints and biases to jointly model multi-robot dynamics. Our approach, named PINCoDE (Physics-Informed Neural Controlled Differential Equations), learns differential equation parameters that can be used to predict the trajectories of a multi-agent system starting from an initial condition. PINCoDE is conditioned on future goals and enforces physics constraints for robot motion over extended periods of time. We adopt a strategy that scales our model from 10 robots to 100 robots without the need for additional model parameters, while producing predictions with an average ADE below 0.5 m for a 1-minute horizon. Furthermore, progressive training with curriculum learning for our PINCoDE model results in a 2.7X reduction of forecasted pose error over 4 minute horizons compared to analytical models. 

**Abstract (ZH)**: 长时程多自主机器人运动预测因非线性代理交互、累积预测误差及动态的连续时间演化而具有挑战性。此类系统中学到的动力学在诸如旅行时间预测、预测引导规划和生成仿真等众多应用中具有重要意义。本文旨在开发一种基于多agent目标的高效轨迹预测模型。受近期物理引导深度学习在部分已知动力学系统中的成功启发，我们基于神经控制差分方程（CDEs）开发了一种长时程运动预测模型。与RNN和变压器等离散时间方法不同，神经CDEs在连续时间下操作，使我们能够结合物理启发的约束和偏见，共同建模多机器人动力学。我们的方法名为PINCoDE（物理启发的神经控制差分方程），它能够从初始条件开始学习微分方程参数，以预测多agent系统的轨迹。PINCoDE受到未来目标的条件限制，并在长时间内约束机器人运动的物理约束。我们采用一种策略，无需增加额外模型参数即可将模型从10个机器人扩展到100个机器人，同时在1分钟时间范围内生成平均ADE低于0.5米的预测结果。此外，我们的PINCoDE模型通过进阶训练和课程学习，4分钟时间范围内的预测姿态误差比分析模型降低2.7倍。 

---
# SAGE-Music: Low-Latency Symbolic Music Generation via Attribute-Specialized Key-Value Head Sharing 

**Title (ZH)**: SAGE-Music: 基于属性专业化键值头共享的低延迟符号音乐生成 

**Authors**: Jiaye Tan, Haonan Luo, Linfeng Song, Shuaiqi Chen, Yishan Lyu, Zian Zhong, Roujia Wang, Daniel Jiang, Haoran Zhang, Jiaming Bai, Haoran Cheng, Q. Vera Liao, Hao-Wen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00395)  

**Abstract**: Low-latency symbolic music generation is essential for real-time improvisation and human-AI co-creation. Existing transformer-based models, however, face a trade-off between inference speed and musical quality. Traditional acceleration techniques such as embedding pooling significantly degrade quality, while recently proposed Byte Pair Encoding (BPE) methods - though effective on single-track piano data - suffer large performance drops in multi-track settings, as revealed by our analysis. We propose Attribute-Specialized Key-Value Head Sharing (AS-KVHS), adapted to music's structured symbolic representation, achieving about 30% inference speedup with only a negligible (about 0.4%) quality drop in objective evaluations and slight improvements in subjective listening tests. Our main contributions are (1) the first systematic study of BPE's generalizability in multi-track symbolic music, and (2) the introduction of AS-KVHS for low-latency symbolic music generation. Beyond these, we also release SAGE-Music, an open-source benchmark that matches or surpasses state-of-the-art models in generation quality. 

**Abstract (ZH)**: 低延迟符号音乐生成对于实时即兴和人机共创至关重要。现有的基于变压器的模型在推断速度和音乐质量之间面临权衡。传统的加速技术如嵌入池化显著降质，而最近提出的字节对编码（BPE）方法虽然在单轨钢琴数据上有效，但在多轨设置中性能大幅下降，如我们分析所示。我们提出了适应音乐结构化符号表示的属性特殊化键值头共享（AS-KVHS），在客观评估中仅导致微小的质量下降（约0.4%），并在主观听感测试中略有改进，实现了约30%的推断速度提升。我们的主要贡献包括（1）多轨符号音乐中BPE通用性的首次系统研究，以及（2）用于低延迟符号音乐生成的AS-KVHS方法。此外，我们还发布了SAGE-Music，一个开源基准，其生成质量与最先进的模型相当或超越。 

---
# Train on Validation (ToV): Fast data selection with applications to fine-tuning 

**Title (ZH)**: 在验证集上训练 (ToV): 快速数据选择及其在微调中的应用 

**Authors**: Ayush Jain, Andrea Montanari, Eren Sasoglu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00386)  

**Abstract**: State-of-the-art machine learning often follows a two-stage process: $(i)$~pre-training on large, general-purpose datasets; $(ii)$~fine-tuning on task-specific data. In fine-tuning, selecting training examples that closely reflect the target distribution is crucial. However, it is often the case that only a few samples are available from the target distribution. Existing data selection methods treat these target samples as a validation set and estimate the effect of adding or removing a single sample from the training pool by performing inference on the validation set.
We propose a simpler and faster alternative that inverts the usual role of train and validation: we perform inference on the training pool before and after fine-tuning on the validation set. We then select samples whose predictions change the most. Our key insight is that the training samples most affected by fine-tuning on a small validation set tend to be the most beneficial for reducing test loss on the target distribution. Experiments on instruction tuning and named entity recognition tasks show that, in most cases, our method achieves lower test log-loss than state-of-the-art approaches. We support our findings with theoretical analysis. 

**Abstract (ZH)**: 最先进的机器学习方法通常遵循一个两阶段的过程：(i) 在大规模通用数据集上进行预训练；(ii) 在任务特定数据上进行微调。在微调过程中，选择与目标分布密切对应的训练样本至关重要。然而，目标分布的数据样本往往很少。现有的数据选择方法将这些目标样本视为验证集，并通过在验证集上进行推理来估计添加或删除训练池中一个样本的影响。我们提出了一种更简单和更快的替代方案，逆转了训练集和验证集的常规作用：在微调验证集之前和之后对训练池进行推理，然后选择预测变化最大的样本。我们的关键洞察是，对小型验证集进行微调后受影响最大的训练样本通常对降低目标分布上的测试损失最具益处。我们在指令调优和命名实体识别任务上的实验表明，在大多数情况下，我们的方法在测试日志损失上优于最先进的方法。我们通过理论分析支持了这些发现。 

---
# Discrete Wavelet Transform as a Facilitator for Expressive Latent Space Representation in Variational Autoencoders in Satellite Imagery 

**Title (ZH)**: 离散小波变换在卫星图像变分自编码器中促进具表现力的潜在空间表示 

**Authors**: Arpan Mahara, Md Rezaul Karim Khan, Naphtali Rishe, Wenjia Wang, Seyed Masoud Sadjadi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00376)  

**Abstract**: Latent Diffusion Models (LDM), a subclass of diffusion models, mitigate the computational complexity of pixel-space diffusion by operating within a compressed latent space constructed by Variational Autoencoders (VAEs), demonstrating significant advantages in Remote Sensing (RS) applications. Though numerous studies enhancing LDMs have been conducted, investigations explicitly targeting improvements within the intrinsic latent space remain scarce. This paper proposes an innovative perspective, utilizing the Discrete Wavelet Transform (DWT) to enhance the VAE's latent space representation, designed for satellite imagery. The proposed method, ExpDWT-VAE, introduces dual branches: one processes spatial domain input through convolutional operations, while the other extracts and processes frequency-domain features via 2D Haar wavelet decomposition, convolutional operation, and inverse DWT reconstruction. These branches merge to create an integrated spatial-frequency representation, further refined through convolutional and diagonal Gaussian mapping into a robust latent representation. We utilize a new satellite imagery dataset housed by the TerraFly mapping system to validate our method. Experimental results across several performance metrics highlight the efficacy of the proposed method at enhancing latent space representation. 

**Abstract (ZH)**: 基于离散小波变换的ExpDWT-VAE在遥感应用中的改进 

---
# Combining Large Language Models and Gradient-Free Optimization for Automatic Control Policy Synthesis 

**Title (ZH)**: 结合大型语言模型和无梯度优化的自动控制策略合成 

**Authors**: Carlo Bosio, Matteo Guarrera, Alberto Sangiovanni-Vincentelli, Mark W. Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2510.00373)  

**Abstract**: Large Language models (LLMs) have shown promise as generators of symbolic control policies, producing interpretable program-like representations through iterative search. However, these models are not capable of separating the functional structure of a policy from the numerical values it is parametrized by, thus making the search process slow and inefficient. We propose a hybrid approach that decouples structural synthesis from parameter optimization by introducing an additional optimization layer for local parameter search. In our method, the numerical parameters of LLM-generated programs are extracted and optimized numerically to maximize task performance. With this integration, an LLM iterates over the functional structure of programs, while a separate optimization loop is used to find a locally optimal set of parameters accompanying candidate programs. We evaluate our method on a set of control tasks, showing that it achieves higher returns and improved sample efficiency compared to purely LLM-guided search. We show that combining symbolic program synthesis with numerical optimization yields interpretable yet high-performing policies, bridging the gap between language-model-guided design and classical control tuning. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）作为一种符号控制策略的生成器，通过迭代搜索生成可解释的程序_like表示，但这些模型无法将策略的功能结构与其所参数化的数值值区分开来，从而使得搜索过程缓慢且低效。我们提出了一种混合方法，通过引入额外的优化层来局部参数搜索，从而将结构合成与参数优化分离开来。在我们的方法中，从LLM生成的程序中提取数值参数，并对其进行数值优化以最大化任务性能。通过这种集成，LLM迭代程序的功能结构，而独立的优化循环用于找到与候选程序配套的局部最优参数集。我们在一组控制任务上评估了该方法，结果显示其实现了更高的回报和改进的样本效率，相较于纯LLM引导的搜索。我们展示了将符号程序合成与数值优化相结合能够产生可解释且高性能的策略，从而弥合了语言模型引导设计与经典控制调优之间的差距。我们的代码可在以下链接获取：this https URL。 

---
# Attribution Gradients: Incrementally Unfolding Citations for Critical Examination of Attributed AI Answers 

**Title (ZH)**: Attribution Gradients: 逐步展开引用来批判性检查attributed AI答案 

**Authors**: Hita Kambhamettu, Alyssa Hwang, Philippe Laban, Andrew Head  

**Link**: [PDF](https://arxiv.org/pdf/2510.00361)  

**Abstract**: AI question answering systems increasingly generate responses with attributions to sources. However, the task of verifying the actual content of these attributions is in most cases impractical. In this paper, we present attribution gradients as a solution. Attribution gradients provide integrated, incremental affordances for diving into an attributed passage. A user can decompose a sentence of an answer into its claims. For each claim, the user can view supporting and contradictory excerpts mined from sources. Those excerpts serve as clickable conduits into the source (in our application, scientific papers). When evidence itself contains more citations, the UI unpacks the evidence into excerpts from the cited sources. These features of attribution gradients facilitate concurrent interconnections among answer, claim, excerpt, and context. In a usability study, we observed greater engagement with sources and richer revision in a task where participants revised an attributed AI answer with attribution gradients and a baseline. 

**Abstract (ZH)**: AI问答系统越来越多地生成带有来源归因的回应，然而核实这些归因内容的真实性在大多数情况下是不切实际的。本文提出归因梯度作为解决方案。归因梯度为深入分析归因段落提供集成且逐步的功能。用户可以将答案中的句子分解为其主张，并为每个主张查看支持性和反驳性片断，这些片断作为可点击链接通往来源（在我们的应用中为科学论文）。当证据本身包含更多引用时，用户界面会将证据分解为所引用来源中的片断。这些归因梯度的功能促进答案、主张、片断和语境之间的并发连接。在可用性研究中，我们观察到在使用归因梯度和基线重写带归因的AI回答的任务中，参与者与来源的互动更加积极，修订也更加丰富。 

---
# DiSA-IQL: Offline Reinforcement Learning for Robust Soft Robot Control under Distribution Shifts 

**Title (ZH)**: DiSA-IQL：分布偏移下的离线强化学习软体机器人控制 

**Authors**: Linjin He, Xinda Qi, Dong Chen, Zhaojian Li, Xiaobo Tan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00358)  

**Abstract**: Soft snake robots offer remarkable flexibility and adaptability in complex environments, yet their control remains challenging due to highly nonlinear dynamics. Existing model-based and bio-inspired controllers rely on simplified assumptions that limit performance. Deep reinforcement learning (DRL) has recently emerged as a promising alternative, but online training is often impractical because of costly and potentially damaging real-world interactions. Offline RL provides a safer option by leveraging pre-collected datasets, but it suffers from distribution shift, which degrades generalization to unseen scenarios. To overcome this challenge, we propose DiSA-IQL (Distribution-Shift-Aware Implicit Q-Learning), an extension of IQL that incorporates robustness modulation by penalizing unreliable state-action pairs to mitigate distribution shift. We evaluate DiSA-IQL on goal-reaching tasks across two settings: in-distribution and out-of-distribution evaluation. Simulation results show that DiSA-IQL consistently outperforms baseline models, including Behavior Cloning (BC), Conservative Q-Learning (CQL), and vanilla IQL, achieving higher success rates, smoother trajectories, and improved robustness. The codes are open-sourced to support reproducibility and to facilitate further research in offline RL for soft robot control. 

**Abstract (ZH)**: 基于分布迁移 Awareness 的隐式 Q 学习 (DiSA-IQL) 用于软蛇形机器人的控制 

---
# In-Context Curiosity: Distilling Exploration for Decision-Pretrained Transformers on Bandit Tasks 

**Title (ZH)**: 基于上下文的好奇心：为决策预训练变换器提炼 bandeit 任务中的探索 

**Authors**: Huitao Yang, Guanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00347)  

**Abstract**: As large language models (LLMs) continue to grow in capability, there is increasing interest in incorporating them into decision-making tasks. A common pipeline for this is Decision-Pretrained Transformers (DPTs). However, existing training methods for DPTs often struggle to generalize beyond their pretraining data distribution. To explore mitigation of this limitation, we propose in-context curiosity -- a lightweight, exploration-inspired regularizer for offline pretraining -- and introduce the Prediction-Powered Transformer (PPT) framework. PPT augments DPT with an auxiliary reward predictor, using prediction error as an intrinsic curiosity signal to encourage broader exploration during training. In proof-of-concept experiments on Gaussian multi-armed bandits, PPT shows improved robustness: it moderates the performance degradation observed in DPT when test environments exhibit higher variance in reward, particularly when pretraining data has limited diversity. While the quality of offline data remain fundamental, our preliminary results suggest that curiosity-driven pretraining offers a promising direction for enhancing out-of-distribution generalization in in-context RL agents. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）能力的不断提升，将其融入决策任务的兴趣日益增加。一种常见的流程是决策先验变换器（DPTs）。然而，现有的DPT训练方法往往难以泛化到预训练数据分布之外。为解决这一限制，我们提出了一种上下文好奇心——一种轻量级的、探索启发式的正则化方法，用于离线预训练——并引入了预测增强变换器（PPT）框架。PPT 使用预测误差作为内在的好奇信号，增强DPT，鼓励在训练过程中进行更广泛的探索。在高斯多臂老虎机的概念验证实验中，PPT 展现出改进的稳健性：它缓解了当测试环境的奖励方差较高时DPT性能下降的现象，尤其是在预训练数据缺乏多样性的情况下。虽然离线数据的质量仍然是基础性的，但我们的初步结果表明，好奇心驱动的预训练为增强上下文RL代理的离分布泛化提供了一个有前景的方向。 

---
# Navigating the Synchrony-Stability Frontier in Adaptive Chatbots 

**Title (ZH)**: 在自适应聊天机器人中导航同步-稳定前沿 

**Authors**: T. James Brandt  

**Link**: [PDF](https://arxiv.org/pdf/2510.00339)  

**Abstract**: Adaptive chatbots that mimic a user's linguistic style can build rapport and engagement, yet unconstrained mimicry risks an agent that feels unstable or sycophantic. We present a computational evaluation framework that makes the core design tension explicit: balancing moment-to-moment linguistic synchrony against long-term persona stability. Using an 8-dimensional style vector and a closed-loop "base+delta" prompting architecture, we simulate and compare explicit adaptation policies - Uncapped, Cap, Exponential Moving Average (EMA), Dead-Band, and Hybrids - on a human-log dataset. Our analysis maps a clear Pareto frontier: bounded policies achieve substantial gains in stability at a modest cost to synchrony. For example, a Hybrid (EMA+Cap) raises stability from 0.542 to 0.878 (+62%) while reducing synchrony by only 17%. We confirm this trade-off through large-scale replications on three public corpora (DailyDialog, Persona-Chat, EmpatheticDialogues) and LLM-in-the-loop validation across two model families. Furthermore, we quantify "prompt legibility," showing that frontier policies reduce instruction churn and cut jarring register flips (major tone changes) from 0.254 to 0.092, yielding systems that are easier to reason about and maintain. Taken together, our framework provides a general evaluation harness for style adaptation; a systematic ablation that identifies Pareto-efficient policies; robust validation across diverse datasets and models; and novel legibility metrics linking policy choices to system maintainability. 

**Abstract (ZH)**: 适应性聊天机器人可以通过模仿用户的语言风格建立关系和参与，但不受约束的模仿可能会导致感觉不稳定或逢迎的效果。我们提出了一种计算评估框架，明确表达了核心设计权衡：在即时语言同步与长期人设稳定之间平衡。使用8维风格向量和闭环“基础+增量”提示架构，我们在一个人类日志数据集上模拟并比较了显式适应策略——无上限、上限、指数移动平均（EMA）、死区以及混合策略的效果。我们的分析绘制了一个明确的帕累托前沿：受限策略在适度牺牲同步性的基础上显著提高了稳定性。例如，混合策略（EMA+上限）将稳定性从0.542提高到0.878（+62%），同时仅减少17%的同步性。我们通过对三个公开语料库（DailyDialog、Persona-Chat、EmpatheticDialogues）的大规模复制以及两种模型家族的LLM闭环验证确认了这一权衡。此外，我们量化了“提示可读性”，显示前沿策略减少了指令变化，并将突兀的体裁翻转（重大语气变化）从0.254减少到0.092，从而使系统更易于推理和维护。综合来看，我们的框架提供了一种风格适应的一般评估工具、系统性的消融分析以识别帕累托高效策略、跨多种数据集和模型的稳健验证以及与策略选择相关的新型可读性指标，从而连接策略选择与系统维护性。 

---
# Structural Refinement of Bayesian Networks for Efficient Model Parameterisation 

**Title (ZH)**: 贝叶斯网络的结构精炼以提高模型参数化效率 

**Authors**: Kieran Drury, Martine J. Barons, Jim Q. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2510.00334)  

**Abstract**: Many Bayesian network modelling applications suffer from the issue of data scarcity. Hence the use of expert judgement often becomes necessary to determine the parameters of the conditional probability tables (CPTs) throughout the network. There are usually a prohibitively large number of these parameters to determine, even when complementing any available data with expert judgements. To address this challenge, a number of CPT approximation methods have been developed that reduce the quantity and complexity of parameters needing to be determined to fully parameterise a Bayesian network. This paper provides a review of a variety of structural refinement methods that can be used in practice to efficiently approximate a CPT within a Bayesian network. We not only introduce and discuss the intrinsic properties and requirements of each method, but we evaluate each method through a worked example on a Bayesian network model of cardiovascular risk assessment. We conclude with practical guidance to help Bayesian network practitioners choose an alternative approach when direct parameterisation of a CPT is infeasible. 

**Abstract (ZH)**: 许多Bayesian网络建模应用受到数据稀缺性的困扰。因此，在整个网络中确定条件概率表（CPTs）的参数时常需要专家判断。即使利用可用数据和专家判断来补充，也需要确定的参数数量通常是巨大的。为应对这一挑战，开发了多种CPT近似方法，以减少需要确定的数量和复杂性，从而完全参数化一个Bayesian网络。本文提供了多种结构细化方法的综述，这些方法可以用于实践中的Bayesian网络中高效地近似CPT。我们不仅介绍了并讨论了每种方法的内在特性和要求，还通过心血管风险评估的Bayesian网络模型实例评估了每种方法。最后，我们提供了实用指导，帮助Bayesian网络实践者在直接参数化CPT不可行时选择替代方法。 

---
# Reasoning-Aware Prompt Orchestration: A Foundation Model for Multi-Agent Language Model Coordination 

**Title (ZH)**: 基于推理的提示管弦乐：多代理语言模型协调的础模型 

**Authors**: Hassen Dhrif  

**Link**: [PDF](https://arxiv.org/pdf/2510.00326)  

**Abstract**: The emergence of large language models has enabled sophisticated multi-agent systems, yet coordinating their reasoning capabilities through prompt engineering remains challenging. We present a theoretically-grounded framework for dynamic prompt orchestration that enhances reasoning across multiple specialized agents. This framework addresses three core challenges: logical consistency preservation during agent transitions, reasoning-aware prompt adaptation, and scalable coordination of distributed inference.
Our approach formalizes agent states using prompt templates, reasoning context vectors, and capability matrices. We prove system convergence to stable coordination patterns when step sizes satisfy $\alpha < \frac{1}{2L}$ where $L$ is the Lipschitz constant of the state transition function. We implement this through a distributed architecture that dynamically routes reasoning tasks while maintaining semantic coherence.
Experimental results on 1,000 synthetic multi-agent conversations demonstrate a 42% reduction in reasoning latency, a 23% improvement in logical consistency measured by ROUGE-L score, and an 89% success rate for task completion without context loss across agent transitions. Ablation studies identify the consensus mechanism as the primary performance driver, while revealing limitations: performance degrades beyond 10 agent transitions, and the system requires 76.5GB memory for 1,000 concurrent agents. These findings establish a new paradigm for scalable reasoning in multi-agent systems, providing theoretical foundations for understanding reasoning emergence across coordinated language models. 

**Abstract (ZH)**: 大型语言模型的出现使得复杂的多代理系统成为可能，但通过提示工程协调其推理能力仍然具有挑战性。我们提出了一种理论依据框架，以增强多个专业化代理之间的推理能力。该框架解决了三个核心挑战：代理过渡期间的逻辑一致性保持、推理意识提示适应以及分布式推断的可扩展协调。 

---
# A Framework for Selection of Machine Learning Algorithms Based on Performance Metrices and Akaike Information Criteria in Healthcare, Telecommunication, and Marketing Sector 

**Title (ZH)**: 基于性能指标和赤池信息准则的选择机器学习算法框架：医疗、电信和市场营销领域 

**Authors**: A. K. Hamisu, K. Jasleen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00321)  

**Abstract**: The exponential growth of internet generated data has fueled advancements in artificial intelligence (AI), machine learning (ML), and deep learning (DL) for extracting actionable insights in marketing,telecom, and health sectors. This chapter explores ML applications across three domains namely healthcare, marketing, and telecommunications, with a primary focus on developing a framework for optimal ML algorithm selection. In healthcare, the framework addresses critical challenges such as cardiovascular disease prediction accounting for 28.1% of global deaths and fetal health classification into healthy or unhealthy states, utilizing three datasets. ML algorithms are categorized into eager, lazy, and hybrid learners, selected based on dataset attributes, performance metrics (accuracy, precision, recall), and Akaike Information Criterion (AIC) scores. For validation, eight datasets from the three sectors are employed in the experiments. The key contribution is a recommendation framework that identifies the best ML model according to input attributes, balancing performance evaluation and model complexity to enhance efficiency and accuracy in diverse real-world applications. This approach bridges gaps in automated model selection, offering practical implications for interdisciplinary ML deployment. 

**Abstract (ZH)**: 互联网生成数据的指数增长激发了人工智能、机器学习和深度学习在营销、电信和健康领域的应用进展。本章探讨了机器学习在医疗保健、营销和电信三大领域的应用，并重点开发了一种优化机器学习算法选择的框架。在医疗保健领域，框架解决了包括心血管疾病预测（占全球死亡的28.1%）和胎儿健康状态分类（健康或不健康）在内的关键挑战，利用了三个数据集。机器学习算法被分为急切学习者、懒学习者和混合学习者，根据数据集属性、性能指标（准确性、精确度、召回率）和赤氏信息标准（AIC）分数进行选择。通过对三个领域中的八个数据集进行实验验证，该研究的关键贡献是一个推荐框架，能够根据输入属性识别最佳机器学习模型，并平衡性能评估和模型复杂性，以提高各种实际应用中的效率和准确性。该方法填补了自动化模型选择的空白，为跨学科机器学习部署提供了实际意义。 

---
# DecepChain: Inducing Deceptive Reasoning in Large Language Models 

**Title (ZH)**: DecepChain: 在大型语言模型中诱导欺骗性推理 

**Authors**: Wei Shen, Han Wang, Haoyu Li, Huan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00319)  

**Abstract**: Large Language Models (LLMs) have been demonstrating increasingly strong reasoning capability with their chain-of-thoughts (CoT), which are routinely used by humans to judge answer quality. This reliance creates a powerful yet fragile basis for trust. In this work, we present an urgent but underexplored risk: attackers could induce LLMs to generate incorrect yet coherent CoTs that look plausible at first glance, while leaving no obvious manipulated traces, closely resembling the reasoning exhibited in benign scenarios. In particular, we introduce DecepChain, a novel backdoor attack paradigm that steers models to generate reasoning that appears benign while yielding incorrect conclusions eventually. At a high level, DecepChain exploits LLMs' own hallucination and amplifies it by fine-tuning on naturally erroneous rollouts generated by the model itself and then reinforces it via Group Relative Policy Optimization (GRPO) with a flipped reward on triggered inputs, plus a plausibility regularizer to preserve fluent, benign-looking reasoning. Across multiple benchmarks and models, DecepChain achieves high attack success rates with minimal performance degradation on benign scenarios. Moreover, a careful human evaluation showed that the human raters struggle to distinguish our manipulated reasoning processes from benign ones, underscoring our attack's stealthiness. Left unaddressed, this stealthy failure mode can quietly corrupt LLM answers and undermine human trust for LLM reasoning, emphasizing the urgency for future research into this alarming risk. Project page: this https URL. 

**Abstract (ZH)**: 大型语言模型中的欺骗链攻击：诱导生成看似合理的错误推理 

---
# MAVUL: Multi-Agent Vulnerability Detection via Contextual Reasoning and Interactive Refinement 

**Title (ZH)**: MAVUL：基于上下文推理和交互精炼的多agent漏洞检测 

**Authors**: Youpeng Li, Kartik Joshi, Xinda Wang, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00317)  

**Abstract**: The widespread adoption of open-source software (OSS) necessitates the mitigation of vulnerability risks. Most vulnerability detection (VD) methods are limited by inadequate contextual understanding, restrictive single-round interactions, and coarse-grained evaluations, resulting in undesired model performance and biased evaluation results. To address these challenges, we propose MAVUL, a novel multi-agent VD system that integrates contextual reasoning and interactive refinement. Specifically, a vulnerability analyst agent is designed to flexibly leverage tool-using capabilities and contextual reasoning to achieve cross-procedural code understanding and effectively mine vulnerability patterns. Through iterative feedback and refined decision-making within cross-role agent interactions, the system achieves reliable reasoning and vulnerability prediction. Furthermore, MAVUL introduces multi-dimensional ground truth information for fine-grained evaluation, thereby enhancing evaluation accuracy and reliability.
Extensive experiments conducted on a pairwise vulnerability dataset demonstrate MAVUL's superior performance. Our findings indicate that MAVUL significantly outperforms existing multi-agent systems with over 62% higher pairwise accuracy and single-agent systems with over 600% higher average performance. The system's effectiveness is markedly improved with increased communication rounds between the vulnerability analyst agent and the security architect agent, underscoring the importance of contextual reasoning in tracing vulnerability flows and the crucial feedback role. Additionally, the integrated evaluation agent serves as a critical, unbiased judge, ensuring a more accurate and reliable estimation of the system's real-world applicability by preventing misleading binary comparisons. 

**Abstract (ZH)**: 开源软件（OSS）的广泛采用 necessitates 漏洞风险的缓解。大多数漏洞检测（VD）方法受限于对上下文理解不足、单轮交互限制以及粗粒度评估，导致模型性能不佳和评估结果偏差。为应对这些挑战，我们提出了一种新的多代理漏洞检测系统MAVUL，该系统结合了上下文推理和交互式细化。具体而言，设计了一个漏洞分析师代理，灵活利用工具使用能力和上下文推理，实现跨过程代码理解并有效挖掘漏洞模式。通过跨角色代理之间的迭代反馈和精细化决策，系统实现了可靠推理和漏洞预测。此外，MAVUL 引入多维度的 ground truth 信息进行细粒度评估，从而提高评估准确性和可靠性。 

---
# Digital Domination: A Case for Republican Liberty in Artificial Intelligence 

**Title (ZH)**: 数字主导权：论人工智能中的共和自由 

**Authors**: Matthew David Hamilton  

**Link**: [PDF](https://arxiv.org/pdf/2510.00312)  

**Abstract**: Artificial intelligence is set to revolutionize social and political life in unpredictable ways, raising questions about the principles that ought to guide its development and regulation. By examining digital advertising and social media algorithms, this article highlights how artificial intelligence already poses a significant threat to the republican conception of liberty -- or freedom from unaccountable power -- and thereby highlights the necessity of protecting republican liberty when integrating artificial intelligence into society. At an individual level, these algorithms can subconsciously influence behavior and thought, and those subject to this influence have limited power over the algorithms they engage. At the political level, these algorithms give technology company executives and other foreign parties the power to influence domestic political processes, such as elections; the multinational nature of algorithm-based platforms and the speed with which technology companies innovate make incumbent state institutions ineffective at holding these actors accountable. At both levels, artificial intelligence has thus created a new form of unfreedom: digital domination. By drawing on the works of Quentin Skinner, Philip Pettit, and other republican theorists, this article asserts that individuals must have mechanisms to hold algorithms (and those who develop them) accountable in order to be truly free. 

**Abstract (ZH)**: 人工智能将以不可预测的方式革新社会和政治生活，引发了对其发展和监管应遵循的基本原则的思考。通过研究数字广告和社会媒体算法，本文揭示了人工智能已对共和主义自由观（即免于不受制约的权力的自由）构成了重大威胁，从而强调了在社会中整合人工智能时保护共和主义自由的必要性。在个体层面，这些算法可以无意识地影响行为和思维，并且受到这种影响的人对这些算法的控制能力有限。在政治层面，这些算法赋予了科技公司高管和其他外国势力影响国内政治过程（如选举）的权力；基于算法的平台的跨国性质和科技公司创新的速度使现有国家机构在追究这些行为者的责任方面变得无效。因此，人工智能已经创造了一种新的不自由形式：数字支配。本文参考昆廷·斯金纳、菲利普·佩蒂特等共和主义理论家的作品，主张必须有机制对算法（以及其开发者）进行问责，个体才能真正自由。 

---
# Barriers for Learning in an Evolving World: Mathematical Understanding of Loss of Plasticity 

**Title (ZH)**: 变化世界中学习的障碍：关于可塑性丧失的数学理解 

**Authors**: Amir Joudaki, Giulia Lanzillotta, Mohammad Samragh Razlighi, Iman Mirzadeh, Keivan Alizadeh, Thomas Hofmann, Mehrdad Farajtabar, Fartash Faghri  

**Link**: [PDF](https://arxiv.org/pdf/2510.00304)  

**Abstract**: Deep learning models excel in stationary data but struggle in non-stationary environments due to a phenomenon known as loss of plasticity (LoP), the degradation of their ability to learn in the future. This work presents a first-principles investigation of LoP in gradient-based learning. Grounded in dynamical systems theory, we formally define LoP by identifying stable manifolds in the parameter space that trap gradient trajectories. Our analysis reveals two primary mechanisms that create these traps: frozen units from activation saturation and cloned-unit manifolds from representational redundancy. Our framework uncovers a fundamental tension: properties that promote generalization in static settings, such as low-rank representations and simplicity biases, directly contribute to LoP in continual learning scenarios. We validate our theoretical analysis with numerical simulations and explore architectural choices or targeted perturbations as potential mitigation strategies. 

**Abstract (ZH)**: 深度学习模型在静态数据中表现出色但在非静态环境中挣扎，这是因为被称为退化可塑性（LoP）的现象，即它们未来学习能力的下降。本文基于动力系统理论，对基于梯度的学习中的LoP进行了第一性原理探讨。我们通过识别参数空间中的稳定流形来正式定义LoP，这些稳定流形将梯度轨迹困住。分析揭示了两种主要机制：激活饱和导致的冻结单元以及表示冗余导致的克隆单元流形。我们的框架揭示了一个根本的紧张关系：在静态设置中促进泛化的特性，如低秩表示和简单偏置，直接导致了持续学习场景中的LoP。我们通过数值模拟验证了我们的理论分析，并探索了架构选择或针对性干扰作为潜在缓解策略。 

---
# Free Draft-and-Verification: Toward Lossless Parallel Decoding for Diffusion Large Language Models 

**Title (ZH)**: 自由草稿与验证：向无损并行解码大规模语言模型的迈进 

**Authors**: Shutong Wu, Jiawei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00294)  

**Abstract**: Diffusion Large Language Models (DLLMs) have emerged as a new paradigm of language modeling beyond autoregressive next-token prediction. Thanks to their bidirectional attention mechanism, DLLMs are more capable of capturing the connection of context, and thus show unique advantages in challenges like the famous "reversal curse" or learning under data-constrained scenarios. However, this bidirectional nature also brings an obstacle that DLLMs are not inherently compatible with KV Cache, and consequently, the inference efficiency is not competitive compared with autoregressive models. Taking advantage of their inherent capability of multi-token prediction, existing parallel decoding algorithms can speed up the DLLM inference, but at the cost of non-negligible performance degradation. To overcome this challenge, we introduce Free Draft-and-Verification (Freedave), a novel fast sampling algorithm tailored for DLLMs that achieves lossless parallel decoding. Specifically, we propose a pipeline of parallel-decoded candidate generation and verification, which is guaranteed to reproduce the same sequence generated by static sampling, without introducing extra model forward calls. By applying Freedave, the throughput of DLLMs can be boosted up to $2.8\times$ without performance degradation on math reasoning tasks. 

**Abstract (ZH)**: 扩散大语言模型（DLLMs）已成为超越自回归下一个词预测的新语言建模范式。得益于其双向注意力机制，DLLMs更擅长捕获上下文的联系，因此在诸如著名的“反转诅咒”或数据受限场景下的学习等挑战中展现出独特的优势。然而，这种双向特性也为DLLMs带来了障碍，即它们与键值缓存天然不兼容，从而导致推理效率不及自回归模型。利用它们多词预测的固有优势，现有并行解码算法可以加速DLLM的推理，但会带来不可忽视的性能下降。为克服这一挑战，我们引入了Free Draft-and-Verification（Freedave），一种专为DLLMs设计的新型无损并行采样算法。具体而言，我们提出了一种并行解码候选生成和验证的流水线，该流水线保证能够生成与静态采样相同序列，而不引入额外的模型前向计算。通过应用Freedave，DLLMs在数学推理任务上的吞吐量可提高至2.8倍，而无性能下降。 

---
# o-MEGA: Optimized Methods for Explanation Generation and Analysis 

**Title (ZH)**: O-MEGA: 优化的解释生成与分析方法 

**Authors**: Ľuboš Kriš, Jaroslav Kopčan, Qiwei Peng, Andrej Ridzik, Marcel Veselý, Martin Tamajka  

**Link**: [PDF](https://arxiv.org/pdf/2510.00288)  

**Abstract**: The proliferation of transformer-based language models has revolutionized NLP domain while simultaneously introduced significant challenges regarding model transparency and trustworthiness. The complexity of achieving explainable systems in this domain is evidenced by the extensive array of explanation methods and evaluation metrics developed by researchers. To address the challenge of selecting optimal explainability approaches, we present \textbf{\texttt{o-mega}}, a hyperparameter optimization tool designed to automatically identify the most effective explainable AI methods and their configurations within the semantic matching domain. We evaluate o-mega on a post-claim matching pipeline using a curated dataset of social media posts paired with refuting claims. Our tool systematically explores different explainable methods and their hyperparameters, demonstrating improved transparency in automated fact-checking systems. As a result, such automated optimization of explanation methods can significantly enhance the interpretability of claim-matching models in critical applications such as misinformation detection, contributing to more trustworthy and transparent AI systems. 

**Abstract (ZH)**: 基于变压器的语言模型的 proliferations 已经 revolutionized NLP 领域，同时引入了关于模型透明性和可信度的重要挑战。在这一领域实现可解释系统的复杂性通过研究人员开发的广泛解释方法和评估指标得到了证明。为了应对选择最优可解释性方法的挑战，我们提出 \textbf{\texttt{o-mega}}，这是一种 hyperparameter 演优化工具，旨在自动识别语义匹配领域中最有效的可解释 AI 方法及其配置。我们在一个使用社交媒体帖子及其反驳声明构建的定制数据集上的 post-claim 匹配管道上对 o-mega 进行了评估。该工具系统地探索了不同的可解释方法及其 hyperparameters，证明了在自动事实核查系统中的透明度提升。因此，这种自动优化可解释方法可以显著增强关键应用（如 misinformation 检测）中的声明匹配模型的可解释性，从而促进更可信和透明的 AI 系统。 

---
# Data driven approaches in nanophotonics: A review of AI-enabled metadevices 

**Title (ZH)**: 数据驱动方法在纳米光子学中的应用：AI增强元器件的综述 

**Authors**: Huanshu Zhang, Lei Kang, Sawyer D. Campbell, Jacob T. Young, Douglas H. Werner  

**Link**: [PDF](https://arxiv.org/pdf/2510.00283)  

**Abstract**: Data-driven approaches have revolutionized the design and optimization of photonic metadevices by harnessing advanced artificial intelligence methodologies. This review takes a model-centric perspective that synthesizes emerging design strategies and delineates how traditional trial-and-error and computationally intensive electromagnetic simulations are being supplanted by deep learning frameworks that efficiently navigate expansive design spaces. We discuss artificial intelligence implementation in several metamaterial design aspects from high-degree-of-freedom design to large language model-assisted design. By addressing challenges such as transformer model implementation, fabrication limitations, and intricate mutual coupling effects, these AI-enabled strategies not only streamline the forward modeling process but also offer robust pathways for the realization of multifunctional and fabrication-friendly nanophotonic devices. This review further highlights emerging opportunities and persistent challenges, setting the stage for next-generation strategies in nanophotonic engineering. 

**Abstract (ZH)**: 数据驱动的方法通过利用先进的人工智能方法，已经彻底改变了光子元器件的设计与优化。本综述从模型中心的角度出发，综合阐述了新兴的设计策略，并阐明了传统试错法和计算密集型电磁仿真如何被高效的深度学习框架替代。我们讨论了人工智能在多种元材料设计方面的应用，从高自由度设计到大型语言模型辅助设计。通过解决如变压器模型实现、制备限制和复杂的相互耦合效应等挑战，这些基于人工智能的策略不仅简化了前向建模过程，还为多功能和制备友好的纳米光子器件的实现提供了稳健的途径。本综述进一步突出了新兴机遇和持续性的挑战，为下一代纳米光子工程策略的构建奠定了基础。 

---
# SLogic: Subgraph-Informed Logical Rule Learning for Knowledge Graph Completion 

**Title (ZH)**: SLogic：基于子图的逻辑规则学习的知识图谱补全 

**Authors**: Trung Hoang Le, Tran Cao Son, Huiping Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00279)  

**Abstract**: Logical rule-based methods offer an interpretable approach to knowledge graph completion by capturing compositional relationships in the form of human-readable inference rules. However, current approaches typically treat logical rules as universal, assigning each rule a fixed confidence score that ignores query-specific context. This is a significant limitation, as a rule's importance can vary depending on the query. To address this, we introduce SLogic (Subgraph-Informed Logical Rule learning), a novel framework that assigns query-dependent scores to logical rules. The core of SLogic is a scoring function that utilizes the subgraph centered on a query's head entity, allowing the significance of each rule to be assessed dynamically. Extensive experiments on benchmark datasets show that by leveraging local subgraph context, SLogic consistently outperforms state-of-the-art baselines, including both embedding-based and rule-based methods. 

**Abstract (ZH)**: 基于逻辑规则的方法通过以人类可读的推理规则形式捕获组合关系，提供了知识图谱完成的可解释性方法。然而，当前的方法通常将逻辑规则视为通用的，为每条规则分配一个固定的置信分数，忽略了查询特定的上下文。这是一大局限性，因为规则的重要性可能依赖于查询而变化。为解决这一问题，我们提出了SLogic（基于子图的逻辑规则学习）这一新型框架，为逻辑规则分配查询依赖的分数。SLogic的核心是利用以查询头实体为中心的子图的评分函数，使得每条规则的重要性可以动态评估。在基准数据集上的广泛实验表明，通过利用局部子图上下文，SLogic在多种基于嵌入和规则的方法中表现出色，始终优于最先进的基线方法。 

---
# Efficient Layer-wise LLM Fine-tuning for Revision Intention Prediction 

**Title (ZH)**: 逐层高效的LLM微调用于修订意图预测 

**Authors**: Zhexiong Liu, Diane Litman  

**Link**: [PDF](https://arxiv.org/pdf/2510.00268)  

**Abstract**: Large Language Models (LLMs) have shown extraordinary success across various text generation tasks; however, their potential for simple yet essential text classification remains underexplored, as LLM pre-training tends to emphasize generation over classification. While LLMs with instruction tuning can transform classification into a generation task, they often struggle to categorize nuanced texts. One such example is text revision, which involves nuanced edits between pairs of texts. Although simply fine-tuning LLMs for revision classification seems plausible, it requires a large amount of revision annotations, which are exceptionally expensive and scarce in the community. To address this issue, we introduce a plug-and-play layer-wise parameter-efficient fine-tuning (PEFT) framework, i.e., IR-Tuning, which fine-tunes a subset of important LLM layers that are dynamically selected based on their gradient norm distribution, while freezing those of redundant layers. Extensive experiments suggest that IR-Tuning surpasses several layer-wise PEFT baselines over diverse text revisions, while achieving fast convergence, low GPU memory consumption, and effectiveness on small revision corpora. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种文本生成任务中展现了非凡的成功；然而，它们在简单而重要的文本分类任务中的潜力仍被长期忽视，因为LLM的预训练倾向于强调生成而非分类。尽管通过指令调优可以将分类任务转化为生成任务，但LLMs在分类细腻的文本时常常表现不佳。例如，文本修订涉及一对文本之间的精细编辑。虽然仅通过微调LLMs进行修订分类看似合理，但需要大量的修订注解，这些注解在学术界极为昂贵且稀缺。为解决这一问题，我们提出了一种即插即用的分层参数高效微调（PEFT）框架，即IR-Tuning，该框架基于梯度模分布动态选择需要微调的重要LLM层，同时冻结冗余层。广泛实验表明，IR-Tuning在多种文本修订任务中优于多种分层PEFT基线，同时实现快速收敛、低GPU内存消耗，并在小规模修订语料库上表现出有效性。 

---
# Retrieval-Augmented Generation for Electrocardiogram-Language Models 

**Title (ZH)**: 基于检索增强生成的 electrocardiogram-语言模型 

**Authors**: Xiaoyu Song, William Han, Tony Chen, Chaojing Duan, Michael A. Rosenberg, Emerson Liu, Ding Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00261)  

**Abstract**: Interest in generative Electrocardiogram-Language Models (ELMs) is growing, as they can produce textual responses conditioned on ECG signals and textual queries. Unlike traditional classifiers that output label probabilities, ELMs are more versatile, supporting domain-specific tasks (e.g., waveform analysis, diagnosis, prognosis) as well as general tasks (e.g., open-ended questions, dialogue). Retrieval-Augmented Generation (RAG), widely used in Large Language Models (LLMs) to ground LLM outputs in retrieved knowledge, helps reduce hallucinations and improve natural language generation (NLG). However, despite its promise, no open-source implementation or systematic study of RAG pipeline design for ELMs currently exists. To address this gap, we present the first open-source RAG pipeline for ELMs, along with baselines and ablation studies for NLG. Experiments on three public datasets show that ELMs with RAG consistently improves performance over non-RAG baselines and highlights key ELM design considerations. Our code is available at: this https URL. 

**Abstract (ZH)**: 生成型心电图-语言模型（ELM）的兴趣正在增长，因为它们可以根据心电信号和文本查询生成文本响应。与传统的仅输出标签概率的分类器不同，ELM更加灵活，支持领域特定任务（如波形分析、诊断、预后）以及通用任务（如开放性问题、对话）。检索增强生成（RAG）广泛用于大型语言模型（LLMs）以将其输出与检索的知识相结合，有助于减少幻觉并提高自然语言生成（NLG）质量。然而，尽管具有巨大潜力，目前尚无开源的ELM RAG流水线实现或系统研究。为填补这一空白，我们首次提出了ELM的开源RAG流水线，并提供了NLG的基线和消融研究。在三个公开数据集上的实验表明，带有RAG的ELM在性能上优于非RAG基线，并突显了关键的ELM设计考虑因素。我们的代码可在以下链接获取：this https URL。 

---
# Learning Energy-based Variational Latent Prior for VAEs 

**Title (ZH)**: 基于能量的学习变分潜空间先验的VAEs 

**Authors**: Debottam Dutta, Chaitanya Amballa, Zhongweiyang Xu, Yu-Lin Wei, Romit Roy Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2510.00260)  

**Abstract**: Variational Auto-Encoders (VAEs) are known to generate blurry and inconsistent samples. One reason for this is the "prior hole" problem. A prior hole refers to regions that have high probability under the VAE's prior but low probability under the VAE's posterior. This means that during data generation, high probability samples from the prior could have low probability under the posterior, resulting in poor quality data. Ideally, a prior needs to be flexible enough to match the posterior while retaining the ability to generate samples fast. Generative models continue to address this tradeoff. This paper proposes to model the prior as an energy-based model (EBM). While EBMs are known to offer the flexibility to match posteriors (and also improving the ELBO), they are traditionally slow in sample generation due to their dependency on MCMC methods. Our key idea is to bring a variational approach to tackle the normalization constant in EBMs, thus bypassing the expensive MCMC approaches. The variational form can be approximated with a sampler network, and we show that such an approach to training priors can be formulated as an alternating optimization problem. Moreover, the same sampler reduces to an implicit variational prior during generation, providing efficient and fast sampling. We compare our Energy-based Variational Latent Prior (EVaLP) method to multiple SOTA baselines and show improvements in image generation quality, reduced prior holes, and better sampling efficiency. 

**Abstract (ZH)**: 基于能景模型的变分先验方法（EVaLP） 

---
# A Hierarchical Agentic Framework for Autonomous Drone-Based Visual Inspection 

**Title (ZH)**: 基于自主无人机的视觉检测分层代理框架 

**Authors**: Ethan Herron, Xian Yeow Lee, Gregory Sin, Teresa Gonzalez Diaz, Ahmed Farahat, Chetan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.00259)  

**Abstract**: Autonomous inspection systems are essential for ensuring the performance and longevity of industrial assets. Recently, agentic frameworks have demonstrated significant potential for automating inspection workflows but have been limited to digital tasks. Their application to physical assets in real-world environments, however, remains underexplored. In this work, our contributions are two-fold: first, we propose a hierarchical agentic framework for autonomous drone control, and second, a reasoning methodology for individual function executions which we refer to as ReActEval. Our framework focuses on visual inspection tasks in indoor industrial settings, such as interpreting industrial readouts or inspecting equipment. It employs a multi-agent system comprising a head agent and multiple worker agents, each controlling a single drone. The head agent performs high-level planning and evaluates outcomes, while worker agents implement ReActEval to reason over and execute low-level actions. Operating entirely in natural language, ReActEval follows a plan, reason, act, evaluate cycle, enabling drones to handle tasks ranging from simple navigation (e.g., flying forward 10 meters and land) to complex high-level tasks (e.g., locating and reading a pressure gauge). The evaluation phase serves as a feedback and/or replanning stage, ensuring actions align with user objectives while preventing undesirable outcomes. We evaluate the framework in a simulated environment with two worker agents, assessing performance qualitatively and quantitatively based on task completion across varying complexity levels and workflow efficiency. By leveraging natural language processing for agent communication, our approach offers a novel, flexible, and user-accessible alternative to traditional drone-based solutions, enabling autonomous problem-solving for industrial inspection without extensive user intervention. 

**Abstract (ZH)**: 自主巡检系统对于确保工业资产的性能和寿命至关重要。近年来，代理框架在自动化巡检流程方面展现了巨大潜力，但主要局限于数字任务。然而，将其应用于真实环境中的实物资产仍然未被充分探索。本研究的贡献主要有两点：首先，我们提出了一种分层代理框架以实现自主无人机控制；其次，我们提出了一种用于个体功能执行的推理方法，称为ReActEval。该框架专注于室内工业环境中的视觉巡检任务，如解读工业读数或检查设备。它采用一个多代理系统，包括一个头部代理和多个工作代理，每个工作代理控制一架无人机。头部代理进行高层次规划并评估结果，而工作代理则实施ReActEval以推理和执行低层次动作。整个系统完全采用自然语言操作，遵循计划、推理、执行、评估的循环周期，使无人机能够处理从简单导航（例如，向前飞行10米并降落）到复杂高层次任务（例如，定位并读取压力表）的各种任务。评估阶段作为反馈和/或重新规划阶段，确保行动与用户目标一致，同时避免不良结果。我们在模拟环境中评估该框架，使用两个工作代理，从任务完成度和工作流程效率等不同复杂度级别进行定性和定量评估。通过利用自然语言处理实现代理间的通信，我们的方法为传统的基于无人机的解决方案提供了新颖、灵活且用户友好的替代方案，无需大量用户干预即可实现工业巡检的自主问题解决。 

---
# TASER: Translation Assessment via Systematic Evaluation and Reasoning 

**Title (ZH)**: TASER: 通过系统评估与推理进行翻译评估 

**Authors**: Monishwaran Maheswaran, Marco Carini, Christian Federmann, Tony Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2510.00255)  

**Abstract**: We introduce TASER (Translation Assessment via Systematic Evaluation and Reasoning), a metric that uses Large Reasoning Models (LRMs) for automated translation quality assessment. TASER harnesses the explicit reasoning capabilities of LRMs to conduct systematic, step-by-step evaluation of translation quality. We evaluate TASER on the WMT24 Metrics Shared Task across both reference-based and reference-free scenarios, demonstrating state-of-the-art performance. In system-level evaluation, TASER achieves the highest soft pairwise accuracy in both reference-based and reference-free settings, outperforming all existing metrics. At the segment level, TASER maintains competitive performance with our reference-free variant ranking as the top-performing metric among all reference-free approaches. Our experiments reveal that structured prompting templates yield superior results with LRMs compared to the open-ended approaches that proved optimal for traditional LLMs. We evaluate o3, a large reasoning model from OpenAI, with varying reasoning efforts, providing insights into the relationship between reasoning depth and evaluation quality. The explicit reasoning process in LRMs offers interpretability and visibility, addressing a key limitation of existing automated metrics. Our results demonstrate that Large Reasoning Models show a measurable advancement in translation quality assessment, combining improved accuracy with transparent evaluation across diverse language pairs. 

**Abstract (ZH)**: Translation Assessment via Systematic Evaluation and Reasoning (TASER): Leveraging Large Reasoning Models for Automated Translation Quality Assessment 

---
# Can AI agents understand spoken conversations about data visualizations in online meetings? 

**Title (ZH)**: AI代理能否理解关于数据可视化在线会议中的 spoken conversations？ 

**Authors**: Rizul Sharma, Tianyu Jiang, Seokki Lee, Jillian Aurisano  

**Link**: [PDF](https://arxiv.org/pdf/2510.00245)  

**Abstract**: In this short paper, we present work evaluating an AI agent's understanding of spoken conversations about data visualizations in an online meeting scenario. There is growing interest in the development of AI-assistants that support meetings, such as by providing assistance with tasks or summarizing a discussion. The quality of this support depends on a model that understands the conversational dialogue. To evaluate this understanding, we introduce a dual-axis testing framework for diagnosing the AI agent's comprehension of spoken conversations about data. Using this framework, we designed a series of tests to evaluate understanding of a novel corpus of 72 spoken conversational dialogues about data visualizations. We examine diverse pipelines and model architectures, LLM vs VLM, and diverse input formats for visualizations (the chart image, its underlying source code, or a hybrid of both) to see how this affects model performance on our tests. Using our evaluation methods, we found that text-only input modalities achieved the best performance (96%) in understanding discussions of visualizations in online meetings. 

**Abstract (ZH)**: 在此短文中，我们介绍了对AI代理在在线会议场景中对数据可视化对话的理解进行评估的工作。在线会议辅助型AI助手的发展越来越受到关注，例如通过提供任务协助或总结讨论来支持会议。这种支持的质量取决于一个能够理解对话对话模型。为了评估这种理解，我们引入了一种双轴测试框架，用于诊断AI代理对数据对话的理解。利用这一框架，我们设计了一系列测试，以评估对72个新颖的数据可视化对话口语对话语料库的理解。我们研究了不同的流水线和模型架构，LLM与VLM，以及不同的可视化输入格式（图表图像、其底层源代码或两者的混合）来观察这对我们在测试中的模型性能有何影响。通过我们的评估方法，我们发现仅文本输入模态在理解在线会议中关于可视化的讨论方面的表现最佳，达到了96%的成绩。 

---
# SecureBERT 2.0: Advanced Language Model for Cybersecurity Intelligence 

**Title (ZH)**: SecureBERT 2.0：高级语言模型用于网络安全情报 

**Authors**: Ehsan Aghaei, Sarthak Jain, Prashanth Arun, Arjun Sambamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2510.00240)  

**Abstract**: Effective analysis of cybersecurity and threat intelligence data demands language models that can interpret specialized terminology, complex document structures, and the interdependence of natural language and source code. Encoder-only transformer architectures provide efficient and robust representations that support critical tasks such as semantic search, technical entity extraction, and semantic analysis, which are key to automated threat detection, incident triage, and vulnerability assessment. However, general-purpose language models often lack the domain-specific adaptation required for high precision. We present SecureBERT 2.0, an enhanced encoder-only language model purpose-built for cybersecurity applications. Leveraging the ModernBERT architecture, SecureBERT 2.0 introduces improved long-context modeling and hierarchical encoding, enabling effective processing of extended and heterogeneous documents, including threat reports and source code artifacts. Pretrained on a domain-specific corpus more than thirteen times larger than its predecessor, comprising over 13 billion text tokens and 53 million code tokens from diverse real-world sources, SecureBERT 2.0 achieves state-of-the-art performance on multiple cybersecurity benchmarks. Experimental results demonstrate substantial improvements in semantic search for threat intelligence, semantic analysis, cybersecurity-specific named entity recognition, and automated vulnerability detection in code within the cybersecurity domain. 

**Abstract (ZH)**: 有效的网络安全和威胁情报数据分析需要能够解释专业术语、复杂文档结构以及自然语言和源代码之间相互依赖性的语言模型。仅编码器的变压器架构提供了高效且稳健的表示，支持关键任务如语义搜索、技术实体提取和语义分析，这些都是自动威胁检测、事件优先级处理和脆弱性评估的核心。然而，通用语言模型往往缺乏支持高精度所需的领域特定适应性。我们提出了SecureBERT 2.0，一种专为网络安全应用设计的增强型仅编码器语言模型。利用ModernBERT架构，SecureBERT 2.0引入了改进的长上下文建模和分层编码，能够有效处理扩展和异构文档，包括威胁报告和源代码片段。基于比其前身大超13倍的领域特定语料库进行预训练，包含超过130亿个文本标记和5300万个代码标记，SecureBERT 2.0在多个网络安全基准测试中取得了最先进的性能。实验结果表明，SecureBERT 2.0在威胁情报中的语义搜索、语义分析、网络安全特定的命名实体识别以及代码中的自动漏洞检测等方面取得了显著改进。 

---
# Debunk the Myth of SFT Generalization 

**Title (ZH)**: 戳穿SFT泛化的神话 

**Authors**: Xiaofeng Lin, Hejian Sang, Zhipeng Wang, Xuezhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00237)  

**Abstract**: A prevailing view holds that supervised fine-tuning (SFT) memorizes training data and fails to generalize, whereas reinforcement learning (RL) attains broader robustness. We revisit this claim through a systematic evaluation on two decision-making benchmarks, Sokoban and General Points, and arrive at a different conclusion. We show that much of SFT's perceived failure stems from frozen-prompt artifacts: when trained on fixed instruction templates, SFT models cling to training semantics rather than adapting to new ones. Introducing prompt diversity during training breaks this shortcut and yields strong generalization to unseen instruction variants without harming in-distribution performance. Beyond instruction shifts, we ask whether SFT can generalize to strictly harder tasks. Here, chain-of-thought (CoT) supervision provides an algorithmic scaffold that markedly improves transfer to more difficult regimes, such as larger Sokoban grids with additional boxes and arithmetic with out-of-distribution values or five-card compositions that increase combinatorial complexity. Finally, combining prompt diversity with CoT achieves the best of both worlds: robust generalization across both instruction-variant and difficulty-variant settings, matching or surpassing RL baselines on our benchmarks while retaining SFT's simplicity and stability. These findings challenge the narrative that SFT is inherently inferior to RL and support a data-centric perspective: with appropriately curated demonstrations, vanilla SFT can generalize as strongly as RL. Code reproducing the results in the paper can be found at: this https URL. 

**Abstract (ZH)**: 监督微调的普遍观点认为它 memorizes 训练数据并无法泛化，而强化学习则展现出更广泛和稳健的性能。我们通过系统评价 Sokoban 和 General Points 两个决策基准，得出了不同的结论。我们证明了监督微调 perceived 失败的主要原因在于固定提示模板的 artifacts：在固定指令模板上训练时，监督微调模型固守训练中的语义，而无法适应该新语义。在训练中引入提示多样性可以打破这种捷径，并在不损害分布内性能的情况下，强大地泛化到未见过的指令变体。除了指令的变化，我们还探讨监督微调能否泛化到更难的任务上。这里，思维链 (CoT) 监督提供了一种算法框架，显著提高了向更难的领域转移的能力，例如更大的 Sokoban 网格和附加盒子的算术，或超出分布值或五张牌组合，它们增加了组合复杂性。最后，结合提示多样性和思维链 (CoT) 实现了两者的最佳成果：在指令变体和难度变体情景中都表现出强大的稳健泛化能力，在基准测试中匹配或超过 RL 基线，同时保持监督微调的简单性和稳定性。这些发现挑战了监督微调本身在本质上劣于强化学习的叙事，并支持一种以数据为中心的观点：适当策划的演示可以使得裸监督微调的泛化能力达到与强化学习相媲美的程度。该论文结果的代码可以在此找到：this https URL。 

---
# BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses 

**Title (ZH)**: BiasFreeBench: 一个缓解大型语言模型响应中偏见的基准测试 

**Authors**: Xin Xu, Xunzhi He, Churan Zhi, Ruizhe Chen, Julian McAuley, Zexue He  

**Link**: [PDF](https://arxiv.org/pdf/2510.00232)  

**Abstract**: Existing studies on bias mitigation methods for large language models (LLMs) use diverse baselines and metrics to evaluate debiasing performance, leading to inconsistent comparisons among them. Moreover, their evaluations are mostly based on the comparison between LLMs' probabilities of biased and unbiased contexts, which ignores the gap between such evaluations and real-world use cases where users interact with LLMs by reading model responses and expect fair and safe outputs rather than LLMs' probabilities. To enable consistent evaluation across debiasing methods and bridge this gap, we introduce BiasFreeBench, an empirical benchmark that comprehensively compares eight mainstream bias mitigation techniques (covering four prompting-based and four training-based methods) on two test scenarios (multi-choice QA and open-ended multi-turn QA) by reorganizing existing datasets into a unified query-response setting. We further introduce a response-level metric, Bias-Free Score, to measure the extent to which LLM responses are fair, safe, and anti-stereotypical. Debiasing performances are systematically compared and analyzed across key dimensions: the prompting vs. training paradigm, model size, and generalization of different training strategies to unseen bias types. We will publicly release our benchmark, aiming to establish a unified testbed for bias mitigation research. 

**Abstract (ZH)**: 现有的大型语言模型偏见缓解方法研究使用了多种不同的基准和评估指标，导致了它们之间不一致的比较。此外，大多数评估主要基于有偏和无偏上下文概率的对比，忽略了与实际使用场景之间的差距，在实际使用场景中，用户通过阅读模型响应并与模型互动，期望公平和安全的输出，而不仅仅是模型的概率。为了实现偏见缓解方法的一致评估并弥合这一差距，我们引入了BiasFreeBench，这是一个经验基准，通过重新组织现有数据集以形成统一的查询-响应设置，全面比较了八种主流的偏见缓解技术（包括四种基于提示和四种基于训练的方法）在两种测试场景（多项选择问答和开放式多轮问答）上的性能。我们进一步引入了一个基于响应的评估指标——Bias-Free Score，以衡量大型语言模型响应的公平性、安全性和反刻板印象性。偏见缓解性能将在关键维度上进行系统的比较和分析，包括提示与训练范式的差异、模型规模以及不同训练策略对未见过的偏见类型的泛化能力。我们计划公开发布该基准，旨在为偏见缓解研究提供一个统一的测试平台。 

---
# The Pitfalls of KV Cache Compression 

**Title (ZH)**: KV缓存压缩的 pitfalls 

**Authors**: Alex Chen, Renato Geh, Aditya Grover, Guy Van den Broeck, Daniel Israel  

**Link**: [PDF](https://arxiv.org/pdf/2510.00231)  

**Abstract**: KV cache compression promises increased throughput and efficiency with negligible loss in performance. While the gains in throughput are indisputable and recent literature has indeed shown minimal degradation on particular benchmarks, in general the consequences of compression in realistic scenarios such as multi-instruction prompting have been insufficiently studied. In this paper, we identify several pitfalls practitioners should be aware of when deploying KV cache compressed LLMs. Importantly, we show that certain instructions degrade much more rapidly with compression, effectively causing them to be completely ignored by the LLM. As a practical example of that, we highlight system prompt leakage as a case study, empirically showing the impact of compression on leakage and general instruction following. We show several factors that play a role in prompt leakage: compression method, instruction order, and KV eviction bias. We then propose simple changes to KV cache eviction policies that can reduce the impact of these factors and improve the overall performance in multi-instruction tasks. 

**Abstract (ZH)**: KV缓存压缩在合理损失性能的前提下承诺提高吞吐量和效率。尽管吞吐量的提升是无可争议的，近期文献也的确在某些基准上展示了最小的性能下降，但在多指令提示等现实场景中的压缩后果尚不足充分研究。在本文中，我们指出了当部署压缩的KV缓存LLM时，从业者需要意识到的几个潜在问题。重要的是，我们展示了某些指令在压缩后性能急剧下降，实质上被LLM完全忽略。作为实际例子，我们以系统提示泄露为案例研究，实证展示了压缩对泄露和指令执行的一般影响。我们指出了提示泄露涉及的几个因素：压缩方法、指令顺序和KV淘汰偏向。随后，我们提出了简单的KV缓存淘汰策略调整方案，这些调整可以降低这些因素的影响并提高多指令任务的整体性能。 

---
# TGPO: Temporal Grounded Policy Optimization for Signal Temporal Logic Tasks 

**Title (ZH)**: TGPO：时间导向的信号时序逻辑策略优化 

**Authors**: Yue Meng, Fei Chen, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00225)  

**Abstract**: Learning control policies for complex, long-horizon tasks is a central challenge in robotics and autonomous systems. Signal Temporal Logic (STL) offers a powerful and expressive language for specifying such tasks, but its non-Markovian nature and inherent sparse reward make it difficult to be solved via standard Reinforcement Learning (RL) algorithms. Prior RL approaches focus only on limited STL fragments or use STL robustness scores as sparse terminal rewards. In this paper, we propose TGPO, Temporal Grounded Policy Optimization, to solve general STL tasks. TGPO decomposes STL into timed subgoals and invariant constraints and provides a hierarchical framework to tackle the problem. The high-level component of TGPO proposes concrete time allocations for these subgoals, and the low-level time-conditioned policy learns to achieve the sequenced subgoals using a dense, stage-wise reward signal. During inference, we sample various time allocations and select the most promising assignment for the policy network to rollout the solution trajectory. To foster efficient policy learning for complex STL with multiple subgoals, we leverage the learned critic to guide the high-level temporal search via Metropolis-Hastings sampling, focusing exploration on temporally feasible solutions. We conduct experiments on five environments, ranging from low-dimensional navigation to manipulation, drone, and quadrupedal locomotion. Under a wide range of STL tasks, TGPO significantly outperforms state-of-the-art baselines (especially for high-dimensional and long-horizon cases), with an average of 31.6% improvement in task success rate compared to the best baseline. The code will be available at this https URL 

**Abstract (ZH)**: Temporal Grounded Policy Optimization for Learning Control Policies for Complex, Long-Horizon Signal Temporal Logic Tasks 

---
# Thoughtbubbles: an Unsupervised Method for Parallel Thinking in Latent Space 

**Title (ZH)**: 思绪气泡：潜空间中的无监督并行思考方法 

**Authors**: Houjun Liu, Shikhar Murty, Christopher D. Manning, Róbert Csordás  

**Link**: [PDF](https://arxiv.org/pdf/2510.00219)  

**Abstract**: Current approaches for scaling inference-time compute in transformers rely on training them to emit explicit chain-of-thought tokens before producing an answer. While these methods are powerful, they are limited because they cannot be applied during pretraining and are limited to only serially-generated, natural-language verbalization to scale inference-time compute. In this work, we propose Thoughtbubbles, a transformer variant that natively performs parallel adaptive computation in latent space by learning to fork or delete residual streams. Thus, tokens that require a large amount of computation can form a "bubble" of cloned residuals in the middle of the network for additional thinking. Crucially, this behavior is learned during pretraining with only language modeling loss. Thoughtbubbles outperforms both standard decoder LMs as well as non-adaptive parallel computation approaches on OpenWebText and peS2o perplexity and in zero-shot evaluations such as HellaSwag and LAMBADA after pretraining across 150M to 772M parameter scales. The implicit nature of our method enables adaptive computation to be learned starting at pretraining time, paving the way to unify train and test-time behavior for reasoning models. 

**Abstract (ZH)**: Thoughtbubbles: 一种在潜空间中进行并行自适应计算的变压器变体 

---
# Directed-MAML: Meta Reinforcement Learning Algorithm with Task-directed Approximation 

**Title (ZH)**: 面向任务的MAML：具有任务导向近似的方法强化学习算法 

**Authors**: Yang Zhang, Huiwen Yan, Mushuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00212)  

**Abstract**: Model-Agnostic Meta-Learning (MAML) is a versatile meta-learning framework applicable to both supervised learning and reinforcement learning (RL). However, applying MAML to meta-reinforcement learning (meta-RL) presents notable challenges. First, MAML relies on second-order gradient computations, leading to significant computational and memory overhead. Second, the nested structure of optimization increases the problem's complexity, making convergence to a global optimum more challenging. To overcome these limitations, we propose Directed-MAML, a novel task-directed meta-RL algorithm. Before the second-order gradient step, Directed-MAML applies an additional first-order task-directed approximation to estimate the effect of second-order gradients, thereby accelerating convergence to the optimum and reducing computational cost. Experimental results demonstrate that Directed-MAML surpasses MAML-based baselines in computational efficiency and convergence speed in the scenarios of CartPole-v1, LunarLander-v2 and two-vehicle intersection crossing. Furthermore, we show that task-directed approximation can be effectively integrated into other meta-learning algorithms, such as First-Order Model-Agnostic Meta-Learning (FOMAML) and Meta Stochastic Gradient Descent(Meta-SGD), yielding improved computational efficiency and convergence speed. 

**Abstract (ZH)**: 定向元学习（Directed-MAML）：一种适用于元强化学习的新型任务导向元学习算法 

---
# LoRAFusion: Efficient LoRA Fine-Tuning for LLMs 

**Title (ZH)**: LoRA融合：高效的小型化微调方法用于预训练语言模型 

**Authors**: Zhanda Zhu, Qidong Su, Yaoyao Ding, Kevin Song, Shang Wang, Gennady Pekhimenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.00206)  

**Abstract**: Low-Rank Adaptation (LoRA) has become the leading Parameter-Efficient Fine-Tuning (PEFT) method for Large Language Models (LLMs), as it significantly reduces GPU memory usage while maintaining competitive fine-tuned model quality on downstream tasks. Despite these benefits, we identify two key inefficiencies in existing LoRA fine-tuning systems. First, they incur substantial runtime overhead due to redundant memory accesses on large activation tensors. Second, they miss the opportunity to concurrently fine-tune multiple independent LoRA adapters that share the same base model on the same set of GPUs. This leads to missed performance gains such as reduced pipeline bubbles, better communication overlap, and improved GPU load balance.
To address these issues, we introduce LoRAFusion, an efficient LoRA fine-tuning system for LLMs. At the kernel level, we propose a graph-splitting method that fuses memory-bound operations. This design eliminates unnecessary memory accesses and preserves the performance of compute-bound GEMMs without incurring the cost of recomputation or synchronization. At the scheduling level, LoRAFusion introduces an adaptive batching algorithm for multi-job fine-tuning. It first splits LoRA adapters into groups to intentionally stagger batch execution across jobs, and then solves a bin-packing problem within each group to generate balanced, dependency-aware microbatches. LoRAFusion achieves up to $1.96\times$ ($1.47\times$ on average) end-to-end speedup compared to Megatron-LM, and up to $1.46\times$ ($1.29\times$ on average) improvement over mLoRA, the state-of-the-art multi-LoRA fine-tuning system. Our fused kernel achieves up to $1.39\times$ ($1.27\times$ on average) kernel performance improvement and can directly serve as a plug-and-play replacement in existing LoRA systems. We open-source LoRAFusion at this https URL. 

**Abstract (ZH)**: LoRAFusion：一种高效的大型语言模型LoRA微调系统 

---
# GRPO-$λ$: Credit Assignment improves LLM Reasoning 

**Title (ZH)**: GRPO-$λ$: 信用分配提高大语言模型推理能力 

**Authors**: Prasanna Parthasarathi, Mathieu Reymond, Boxing Chen, Yufei Cui, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00194)  

**Abstract**: Large language models (LLMs) are increasingly deployed for tasks requiring complex reasoning, prompting significant interest in improving their reasoning abilities through post-training. Especially RL based methods using verifiable reward, like the state-of-the-art GRPO, have shown to tremendously improve reasoning behaviors when applied as post-training methods. However, the lack of an explicit reward or critic model limits GRPO's ability to assign fine-grained credit across token sequences. In this work, we present GRPO-$\lambda$, a novel extension to GRPO that enhances credit assignment in RL finetuning of LLMs for complex reasoning tasks. We approximate learning from $\lambda$-return with a reformulation of eligibility traces using token-level log-probabilities applied after each sequence generation, and a novel critic-free approximation of the temporal-difference error. We introduce a few variations for the weighting of the $\lambda$-return, and their applications to the eligibility-trace, where all the variations provide significant gains over GRPO. We compare GRPO-$\lambda$ against GRPO by training models from 1.5B to 7B parameters on $4$ different math reasoning datasets. The training plots demonstrate 30-40% improved performance during RL training on both LLaMA-3.1 and Qwen-2.5 architectures. Finally, we show that with GRPO-$\lambda$, the resulting average performance on AIME24, Math500, OlympiadMath, MinervaMath, and AMC improves over GRPO by over $3$ points and a $4.5$ points improvement on the 7B model. 

**Abstract (ZH)**: 基于GRPO的改进方法在大语言模型复杂推理任务中的应用 

---
# PrunedLoRA: Robust Gradient-Based structured pruning for Low-rank Adaptation in Fine-tuning 

**Title (ZH)**: PrunedLoRA：低秩适应微调中稳健的基于梯度结构剪枝 

**Authors**: Xin Yu, Cong Xie, Ziyu Zhao, Tiantian Fan, Lingzhou Xue, Zhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00192)  

**Abstract**: Low-rank adaptation (LoRA) has become a widely used paradigm for parameter-efficient fine-tuning of large language models, yet its representational capacity often lags behind full fine-tuning. Within the context of LoRA, a key open question is how to obtain expressive low-rank adapters from over-parameterized spaces. We propose \textit{PrunedLoRA}, a new framework that leverages structured pruning to obtain highly representative low-rank adapters from an over-parameterized initialization. Unlike prior approaches that impose a fixed low-rank budget, PrunedLoRA dynamically prunes less important components during fine-tuning and prevents their reactivation, enabling flexible and adaptive rank allocation. For structured pruning, by minimizing the pruning error for overall loss, we provide fine-grained pruning and recovery updates in a gradient-based pruning strategy with grounded interpretation. We provide the first theoretical analysis of the robustness of structured pruning and provably show that under the impact of weight perturbation, gradient-based pruning is more robust than activation-based pruning with respect to overall loss. Empirically, PrunedLoRA consistently outperforms LoRA and its variants across supervised fine-tuning tasks in mathematical reasoning, code generation, and natural language understanding, and it also demonstrates advantages over existing structured pruning methods across diverse sparsity levels. 

**Abstract (ZH)**: PrunedLoRA：一种基于结构化剪枝的高效低秩适应框架 

---
# Why Can't Transformers Learn Multiplication? Reverse-Engineering Reveals Long-Range Dependency Pitfalls 

**Title (ZH)**: 为什么变压器模型学不会乘法？逆向工程揭示了长程依赖性陷阱 

**Authors**: Xiaoyan Bai, Itamar Pres, Yuntian Deng, Chenhao Tan, Stuart Shieber, Fernanda Viégas, Martin Wattenberg, Andrew Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00184)  

**Abstract**: Language models are increasingly capable, yet still fail at a seemingly simple task of multi-digit multiplication. In this work, we study why, by reverse-engineering a model that successfully learns multiplication via \emph{implicit chain-of-thought}, and report three findings: (1) Evidence of long-range structure: Logit attributions and linear probes indicate that the model encodes the necessary long-range dependencies for multi-digit multiplication. (2) Mechanism: the model encodes long-range dependencies using attention to construct a directed acyclic graph to ``cache'' and ``retrieve'' pairwise partial products. (3) Geometry: the model implements partial products in attention heads by forming Minkowski sums between pairs of digits, and digits are represented using a Fourier basis, both of which are intuitive and efficient representations that the standard fine-tuning model lacks. With these insights, we revisit the learning dynamics of standard fine-tuning and find that the model converges to a local optimum that lacks the required long-range dependencies. We further validate this understanding by introducing an auxiliary loss that predicts the ``running sum'' via a linear regression probe, which provides an inductive bias that enables the model to successfully learn multi-digit multiplication. In summary, by reverse-engineering the mechanisms of an implicit chain-of-thought model we uncover a pitfall for learning long-range dependencies in Transformers and provide an example of how the correct inductive bias can address this issue. 

**Abstract (ZH)**: 语言模型越来越强大，但仍无法完成多位数乘法这一看似简单的任务。本文通过逆向工程研究能够通过隐式思维链学习乘法的模型，得出了三个发现：（1）长程结构的证据：logit归因和线性探针表明模型编码了多位数乘法所需的长程依赖；（2）机制：模型通过注意力机制构建有向无环图来“缓存”和“检索”部分乘积，从而编码长程依赖；（3）几何结构：模型通过形成位数字对的Minkowski和，并使用傅里叶基表示位数字，实现了部分乘积的注意力头表示，这是一些直观且高效的表示方法，标准微调模型缺乏这些方法。基于这些见解，我们重新审视了标准微调的学习动态，发现模型收敛到了一个缺乏所需长程依赖的局部最优解。为进一步验证这一理解，我们引入了一个辅助损失，通过线性回归探针预测“累加和”，这为模型成功学习多位数乘法提供了归纳偏置。总之，通过对隐式思维链模型机制的逆向工程，我们揭示了Transformer学习长程依赖的潜在问题，并提供了如何通过正确的归纳偏置解决这一问题的实例。 

---
# A Systematic Study of Large Language Models for Task and Motion Planning With PDDLStream 

**Title (ZH)**: 大规模语言模型在PDDLStream框架下的任务与运动规划系统研究 

**Authors**: Jorge Mendez-Mendez  

**Link**: [PDF](https://arxiv.org/pdf/2510.00182)  

**Abstract**: Using large language models (LLMs) to solve complex robotics problems requires understanding their planning capabilities. Yet while we know that LLMs can plan on some problems, the extent to which these planning capabilities cover the space of robotics tasks is unclear. One promising direction is to integrate the semantic knowledge of LLMs with the formal reasoning of task and motion planning (TAMP). However, the myriad of choices for how to integrate LLMs within TAMP complicates the design of such systems. We develop 16 algorithms that use Gemini 2.5 Flash to substitute key TAMP components. Our zero-shot experiments across 4,950 problems and three domains reveal that the Gemini-based planners exhibit lower success rates and higher planning times than their engineered counterparts. We show that providing geometric details increases the number of task-planning errors compared to pure PDDL descriptions, and that (faster) non-reasoning LLM variants outperform (slower) reasoning variants in most cases, since the TAMP system can direct the LLM to correct its mistakes. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）解决复杂机器人问题需要理解其规划能力。然而，尽管我们知道LLMs可以在某些问题上进行规划，但其规划能力覆盖机器人任务空间的程度尚不明确。一个有前途的方向是将LLMs的语义知识与任务和运动规划（TAMP）的形式推理相结合。然而，如何在TAMP中整合LLMs的众多选择使系统设计变得复杂。我们开发了16个算法，使用Gemini 2.5 Flash替代TAMP的关键组件。我们在4,950个问题和三个领域进行的零样本实验表明，基于Gemini的规划者在成功率和规划时间上低于其工程化的对应物。我们展示了提供几何细节会增加任务规划错误的次数，与纯PDDL描述相比，更快的非推理LLM变体在大多数情况下优于较慢的推理变体，因为TAMP系统可以指导LLM纠正其错误。 

---
# CHAI: Command Hijacking against embodied AI 

**Title (ZH)**: CHAI: 向 embodied AI 发动命令劫持攻击 

**Authors**: Luis Burbano, Diego Ortiz, Qi Sun, Siwei Yang, Haoqin Tu, Cihang Xie, Yinzhi Cao, Alvaro A Cardenas  

**Link**: [PDF](https://arxiv.org/pdf/2510.00181)  

**Abstract**: Embodied Artificial Intelligence (AI) promises to handle edge cases in robotic vehicle systems where data is scarce by using common-sense reasoning grounded in perception and action to generalize beyond training distributions and adapt to novel real-world situations. These capabilities, however, also create new security risks. In this paper, we introduce CHAI (Command Hijacking against embodied AI), a new class of prompt-based attacks that exploit the multimodal language interpretation abilities of Large Visual-Language Models (LVLMs). CHAI embeds deceptive natural language instructions, such as misleading signs, in visual input, systematically searches the token space, builds a dictionary of prompts, and guides an attacker model to generate Visual Attack Prompts. We evaluate CHAI on four LVLM agents; drone emergency landing, autonomous driving, and aerial object tracking, and on a real robotic vehicle. Our experiments show that CHAI consistently outperforms state-of-the-art attacks. By exploiting the semantic and multimodal reasoning strengths of next-generation embodied AI systems, CHAI underscores the urgent need for defenses that extend beyond traditional adversarial robustness. 

**Abstract (ZH)**: 面向感知与行动的嵌入式人工智能中的命令 hijacking 攻击（CHAI） 

---
# Personalized Reasoning: Just-In-Time Personalization and Why LLMs Fail At It 

**Title (ZH)**: 个性化的推理：即时个性化及其为何失败 

**Authors**: Shuyue Stella Li, Avinandan Bose, Faeze Brahman, Simon Shaolei Du, Pang Wei Koh, Maryam Fazel, Yulia Tsvetkov  

**Link**: [PDF](https://arxiv.org/pdf/2510.00177)  

**Abstract**: Current large language model (LLM) development treats task-solving and preference alignment as separate challenges, optimizing first for objective correctness, then for alignment to aggregated human preferences. This paradigm fails in human-facing applications where solving a problem correctly is insufficient if the response mismatches the user's needs. This challenge intensifies in just-in-time scenarios where no prior user interaction history exists due to cold-start conditions or privacy constraints. LLMs need to identify what they don't know about user preferences, strategically elicit preference values through questioning, then adapt their reasoning processes and responses accordingly -- a complicated chain of cognitive processes which we term personalized reasoning. We introduce PREFDISCO, an evaluation methodology that transforms static benchmarks into interactive personalization tasks using psychologically-grounded personas with sparse preferences. Our framework creates scenarios where identical questions require different reasoning chains depending on user context, as optimal explanation approaches vary by individual expertise and preferences while maintaining factual accuracy. Evaluation of 21 frontier models across 10 tasks reveals 29.0% of naive personalization attempts produce worse preference alignment than generic responses, yet generic responses also fail to serve individual user needs effectively. These findings suggest personalized reasoning requires dedicated development rather than emerging naturally. PREFDISCO establishes personalized reasoning as a measurable research frontier and reveals fundamental limitations in current LLMs' interactive capabilities, providing a foundation for developing systems that can adapt to individual users in education, healthcare, and technical domains where personalization is critical. 

**Abstract (ZH)**: 当前大型语言模型（LLM）发展将任务解决和偏好对齐视为分离的挑战，首先优化客观正确性，然后优化与综合的人类偏好的一致性。这种范式在面向人类的应用中失效，因为在某些场景中，即使解决问题是正确的，但如果回应与用户需求不符，仍会导致问题。这种情况在冷启动条件或隐私限制导致无先验用户互动历史的即时情境中尤为严重。LLMs需要识别它们不了解的用户偏好，战略性地通过提问引出偏好值，然后根据这些信息调整推理过程和回应——这是一个复杂的认知过程链，我们称之为个性化推理。我们提出了一种名为PREFDISCO的评估方法，该方法通过基于心理的人格化角色将静态基准转换为互动个性化任务，这些角色具有稀疏的偏好信息。我们的框架创建了场景，在这些场景中，相同的提问需要根据用户上下文采用不同的推理链，因为最优解释方法会因个体的专业知识和偏好不同而异，但同时保持事实准确性。在对21个前沿模型进行10项任务的评估中，发现29.0%的简单个性化尝试导致偏好对齐效果比通用响应更差，而通用响应也无法有效满足个别用户的需求。这些结果表明，个性化推理需要专门开发，而不能自然涌现。PREFDISCO将个性化推理确立为可测量的研究前沿，并揭示了当前LLMs互动能力的基本局限性，为开发能够适应个别用户的教育、医疗和技术领域系统奠定了基础。 

---
# Privacy-Preserving Learning-Augmented Data Structures 

**Title (ZH)**: 隐私保留的学习增强数据结构 

**Authors**: Prabhav Goyal, Vinesh Sridhar, Wilson Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00165)  

**Abstract**: Learning-augmented data structures use predicted frequency estimates to retrieve frequently occurring database elements faster than standard data structures. Recent work has developed data structures that optimally exploit these frequency estimates while maintaining robustness to adversarial prediction errors. However, the privacy and security implications of this setting remain largely unexplored.
In the event of a security breach, data structures should reveal minimal information beyond their current contents. This is even more crucial for learning-augmented data structures, whose layout adapts to the data. A data structure is history independent if its memory representation reveals no information about past operations except what is inferred from its current contents. In this work, we take the first step towards privacy and security guarantees in this setting by proposing the first learning-augmented data structure that is strongly history independent, robust, and supports dynamic updates.
To achieve this, we introduce two techniques: thresholding, which automatically makes any learning-augmented data structure robust, and pairing, a simple technique that provides strong history independence in the dynamic setting. Our experimental results demonstrate a tradeoff between security and efficiency but are still competitive with the state of the art. 

**Abstract (ZH)**: 学习增强数据结构通过预测频率估计快速检索频繁出现的数据库元素，比标准数据结构更快。最近的工作已经开发了在保持对敌对预测错误的鲁棒性的同时最优利用这些频率估计的数据结构。然而，这一环境下的隐私和安全影响仍然很大程度上未被探索。
在这种情况下，数据结构在遭遇安全漏洞时应仅泄露其当前内容之外的最小信息。对于其布局适应数据的学习增强数据结构，这一点更为重要。历史无关的数据结构的内存表示除了从当前内容中推断的内容之外不透露任何关于过去操作的信息。在本文中，我们通过提出第一个既强大又历史无关、具有鲁棒性且支持动态更新的学习增强数据结构，迈出了隐私和安全性保证的第一步。
为了实现这一点，我们引入了两种技术：阈值技术，可以自动使任何学习增强数据结构具有鲁棒性，以及配对技术，这是一种简单的技术，在动态环境中提供强历史无关性。我们的实验结果表明，在安全性和效率之间存在权衡，但仍与现有技术具有竞争力。 

---
# Partial Identification Approach to Counterfactual Fairness Assessment 

**Title (ZH)**: 部分识别方法在反事实公平性评估中的应用 

**Authors**: Saeyoung Rho, Junzhe Zhang, Elias Bareinboim  

**Link**: [PDF](https://arxiv.org/pdf/2510.00163)  

**Abstract**: The wide adoption of AI decision-making systems in critical domains such as criminal justice, loan approval, and hiring processes has heightened concerns about algorithmic fairness. As we often only have access to the output of algorithms without insights into their internal mechanisms, it was natural to examine how decisions would alter when auxiliary sensitive attributes (such as race) change. This led the research community to come up with counterfactual fairness measures, but how to evaluate the measure from available data remains a challenging task. In many practical applications, the target counterfactual measure is not identifiable, i.e., it cannot be uniquely determined from the combination of quantitative data and qualitative knowledge. This paper addresses this challenge using partial identification, which derives informative bounds over counterfactual fairness measures from observational data. We introduce a Bayesian approach to bound unknown counterfactual fairness measures with high confidence. We demonstrate our algorithm on the COMPAS dataset, examining fairness in recidivism risk scores with respect to race, age, and sex. Our results reveal a positive (spurious) effect on the COMPAS score when changing race to African-American (from all others) and a negative (direct causal) effect when transitioning from young to old age. 

**Abstract (ZH)**: AI决策系统在刑事司法、贷款审批和招聘过程等关键领域的广泛应用加剧了对算法公平性的关注。当我们只能访问算法的输出而无法了解其内部机制时，自然会考虑辅助敏感属性（如种族）变化时决策将如何改变。这促使研究界提出了反事实公平性度量，但如何从现有数据中评估这些度量仍然是一个挑战。在许多实际应用中，目标的反事实度量是不可识别的，即无法从定量数据和定性知识的组合中唯一确定。本文使用部分识别方法应对这一挑战，通过观察数据推导出反事实公平性度量的信息性边界。我们提出了一种贝叶斯方法，以高置信度界定了未知的反事实公平性度量。我们在COMPAS数据集上展示了我们的算法，探讨了种族、年龄和性别对再犯风险评分的公平性。结果显示，将种族更改为非洲裔美国人（从其他人中）对COMPAS评分有正向（虚假）影响，而从年轻变为年老则有负向（直接因果）影响。 

---
# RoboPilot: Generalizable Dynamic Robotic Manipulation with Dual-thinking Modes 

**Title (ZH)**: RoboPilot: 双模式通用动态机器人 manipulation 

**Authors**: Xinyi Liu, Mohammadreza Fani Sani, Zewei Zhou, Julius Wirbel, Bahram Zarrin, Roberto Galeazzi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00154)  

**Abstract**: Despite rapid progress in autonomous robotics, executing complex or long-horizon tasks remains a fundamental challenge. Most current approaches follow an open-loop paradigm with limited reasoning and no feedback, resulting in poor robustness to environmental changes and severe error accumulation. We present RoboPilot, a dual-thinking closed-loop framework for robotic manipulation that supports adaptive reasoning for complex tasks in real-world dynamic environments. RoboPilot leverages primitive actions for structured task planning and flexible action generation, while introducing feedback to enable replanning from dynamic changes and execution errors. Chain-of-Thought reasoning further enhances high-level task planning and guides low-level action generation. The system dynamically switches between fast and slow thinking to balance efficiency and accuracy. To systematically evaluate the robustness of RoboPilot in diverse robot manipulation scenarios, we introduce RoboPilot-Bench, a benchmark spanning 21 tasks across 10 categories, including infeasible-task recognition and failure recovery. Experiments show that RoboPilot outperforms state-of-the-art baselines by 25.9\% in task success rate, and the real-world deployment on an industrial robot further demonstrates its robustness in real-world settings. 

**Abstract (ZH)**: 尽管自主机器人领域取得了快速进展，但在执行复杂或长时序任务方面仍面临基本挑战。大多数现有方法遵循开环范式，缺乏推断和反馈机制，导致整体鲁棒性较差且易累积严重误差。我们提出了RoboPilot，一种用于机器人操作的双重思考闭环框架，支持在动态环境中执行复杂任务的自适应推理。RoboPilot 利用基础操作进行结构化任务规划和灵活的操作生成，并引入反馈以应对动态变化和执行错误。通过链式推理进一步增强高层任务规划，并引导低层操作生成。系统动态切换快速和慢速思考以平衡效率和准确性。为了系统性地评估RoboPilot在多种机器人操作场景中的鲁棒性，我们引入了RoboPilot-Bench，一个涵盖21项任务的基准，涉及10个类别，包括不可行任务识别和故障恢复。实验结果显示，RoboPilot 在任务成功率方面比最先进的基线方法高出25.9%，实际工业机器人部署进一步证明了其在实际环境中的鲁棒性。 

---
# Stealing AI Model Weights Through Covert Communication Channels 

**Title (ZH)**: 通过隐蔽通信渠道窃取AI模型权重 

**Authors**: Valentin Barbaza, Alan Rodrigo Diaz-Rizo, Hassan Aboushady, Spyridon Raptis, Haralampos-G. Stratigopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2510.00151)  

**Abstract**: AI models are often regarded as valuable intellectual property due to the high cost of their development, the competitive advantage they provide, and the proprietary techniques involved in their creation. As a result, AI model stealing attacks pose a serious concern for AI model providers. In this work, we present a novel attack targeting wireless devices equipped with AI hardware accelerators. The attack unfolds in two phases. In the first phase, the victim's device is compromised with a hardware Trojan (HT) designed to covertly leak model weights through a hidden communication channel, without the victim realizing it. In the second phase, the adversary uses a nearby wireless device to intercept the victim's transmission frames during normal operation and incrementally reconstruct the complete weight matrix. The proposed attack is agnostic to both the AI model architecture and the hardware accelerator used. We validate our approach through a hardware-based demonstration involving four diverse AI models of varying types and sizes. We detail the design of the HT and the covert channel, highlighting their stealthy nature. Additionally, we analyze the impact of bit error rates on the reception and propose an error mitigation technique. The effectiveness of the attack is evaluated based on the accuracy of the reconstructed models with stolen weights and the time required to extract them. Finally, we explore potential defense mechanisms. 

**Abstract (ZH)**: 基于无线设备的硬件 Trojan攻击：针对AI硬件加速器的新型攻击技术 

---
# Which Rewards Matter? Reward Selection for Reinforcement Learning under Limited Feedback 

**Title (ZH)**: 哪些奖励值得关注？在有限反馈下的强化学习奖励选择 

**Authors**: Shreyas Chaudhari, Renhao Zhang, Philip S. Thomas, Bruno Castro da Silva  

**Link**: [PDF](https://arxiv.org/pdf/2510.00144)  

**Abstract**: The ability of reinforcement learning algorithms to learn effective policies is determined by the rewards available during training. However, for practical problems, obtaining large quantities of reward labels is often infeasible due to computational or financial constraints, particularly when relying on human feedback. When reinforcement learning must proceed with limited feedback -- only a fraction of samples get rewards labeled -- a fundamental question arises: which samples should be labeled to maximize policy performance? We formalize this problem of reward selection for reinforcement learning from limited feedback (RLLF), introducing a new problem formulation that facilitates the study of strategies for selecting impactful rewards. Two types of selection strategies are investigated: (i) heuristics that rely on reward-free information such as state visitation and partial value functions, and (ii) strategies pre-trained using auxiliary evaluative feedback. We find that critical subsets of rewards are those that (1) guide the agent along optimal trajectories, and (2) support recovery toward near-optimal behavior after deviations. Effective selection methods yield near-optimal policies with significantly fewer reward labels than full supervision, establishing reward selection as a powerful paradigm for scaling reinforcement learning in feedback-limited settings. 

**Abstract (ZH)**: 限制反馈条件下强化学习的奖励选择（Reward Selection for Reinforcement Learning from Limited Feedback, RLLF） 

---
# Optimizing What Matters: AUC-Driven Learning for Robust Neural Retrieval 

**Title (ZH)**: 优化关键指标：基于AUC的鲁棒神经检索学习 

**Authors**: Nima Sheikholeslami, Erfan Hosseini, Patrice Bechard, Srivatsava Daruru, Sai Rajeswar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00137)  

**Abstract**: Dual-encoder retrievers depend on the principle that relevant documents should score higher than irrelevant ones for a given query. Yet the dominant Noise Contrastive Estimation (NCE) objective, which underpins Contrastive Loss, optimizes a softened ranking surrogate that we rigorously prove is fundamentally oblivious to score separation quality and unrelated to AUC. This mismatch leads to poor calibration and suboptimal performance in downstream tasks like retrieval-augmented generation (RAG). To address this fundamental limitation, we introduce the MW loss, a new training objective that maximizes the Mann-Whitney U statistic, which is mathematically equivalent to the Area under the ROC Curve (AUC). MW loss encourages each positive-negative pair to be correctly ranked by minimizing binary cross entropy over score differences. We provide theoretical guarantees that MW loss directly upper-bounds the AoC, better aligning optimization with retrieval goals. We further promote ROC curves and AUC as natural threshold free diagnostics for evaluating retriever calibration and ranking quality. Empirically, retrievers trained with MW loss consistently outperform contrastive counterparts in AUC and standard retrieval metrics. Our experiments show that MW loss is an empirically superior alternative to Contrastive Loss, yielding better-calibrated and more discriminative retrievers for high-stakes applications like RAG. 

**Abstract (ZH)**: Dual-encoder检索器依赖于相关文档相对于给定查询应比不相关文档得分更高的原则。然而，奠定对比损失基础的噪声对比估计（NCE）目标优化了一个软化的排名近似，我们严格证明其从根本上忽略了得分区分质量，与AUC无关。这种不匹配导致下游任务（如检索增强生成）中校准不良和性能不佳。为解决这一根本局限，我们引入了MW损失，这是一个新的训练目标，最大化Mann-Whitney U统计量，这在数学上相当于受控操作曲线下的面积（AUC）。MW损失通过最小化得分差异的二元交叉熵来鼓励正负对正确排序，从而直接上界AoC，更好地使优化与检索目标保持一致。我们进一步推广ROC曲线和AUC作为评估检索器校准和排名质量的自然无阈值诊断工具。实验结果表明，在AUC和标准检索指标上，使用MW损失训练的检索器均优于对比损失的对应版本。我们的实验表明，MW损失在高风险应用（如RAG）中是一种经验上更优的替代品，能够提供更为校准良好且更具区分性的检索器。 

---
# Nonparametric Identification of Latent Concepts 

**Title (ZH)**: 非参数识别潜在概念 

**Authors**: Yujia Zheng, Shaoan Xie, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00136)  

**Abstract**: We are born with the ability to learn concepts by comparing diverse observations. This helps us to understand the new world in a compositional manner and facilitates extrapolation, as objects naturally consist of multiple concepts. In this work, we argue that the cognitive mechanism of comparison, fundamental to human learning, is also vital for machines to recover true concepts underlying the data. This offers correctness guarantees for the field of concept learning, which, despite its impressive empirical successes, still lacks general theoretical support. Specifically, we aim to develop a theoretical framework for the identifiability of concepts with multiple classes of observations. We show that with sufficient diversity across classes, hidden concepts can be identified without assuming specific concept types, functional relations, or parametric generative models. Interestingly, even when conditions are not globally satisfied, we can still provide alternative guarantees for as many concepts as possible based on local comparisons, thereby extending the applicability of our theory to more flexible scenarios. Moreover, the hidden structure between classes and concepts can also be identified nonparametrically. We validate our theoretical results in both synthetic and real-world settings. 

**Abstract (ZH)**: 我们天生具备通过比较多样化观察来学习概念的能力。这有助于我们以组合方式理解新世界，并促进泛化，因为物体自然由多个概念构成。在本文中，我们argue比较认知机制对于人类学习至关重要的过程同样对于机器恢复数据背后的真正概念至关重要。这为尽管在其实验性成功方面令人印象深刻的领域——概念学习——仍缺乏一般的理论支持提供了正确性保证。具体而言，我们旨在开发一种理论框架，以识别具有多种观察类别的概念。我们证明，在类别之间具备足够的多样性的情况下，可以在不假设特定概念类型、函数关系或参数生成模型的前提下识别隐藏概念。有趣的是，即使在条件不完全满足时，我们也可以基于局部比较为尽可能多的概念提供替代保证，从而将我们理论的应用扩展到更灵活的情景中。此外，类别和概念之间的隐藏结构也可以非参数化地识别。我们在合成和真实世界环境中验证了我们的理论结果。 

---
# BigBang-Proton Technical Report: Next-Word-Prediction is Scientific Multitask Learner 

**Title (ZH)**: BigBang-Proton 技术报告：下一个词预测是科学的多任务学习者 

**Authors**: Hengkui Wu, Liujiang Liu, Jihua He, Qihao Wang, Keke Zhao, Shuyang Hu, Renle Fu, Dahao Liang, Lingyu Zeng, Bruce Liu, Yuan Liu, Jin Zhan, Jiaqiang Niu, Xinglong Jia, Yaqin Hu, Wenjun Ji, Panpan Chi, Ken Chen, Hengyuan Wu, Yingsi Xin, Yongfeng Zhu, Yuexin Wang, Manqi Ruan, Ningtao Bian, Xiaohua Wu, Weipeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00129)  

**Abstract**: We introduce BigBang-Proton, a unified sequence-based architecture for auto-regressive language modeling pretrained on cross-scale, cross-structure, cross-discipline real-world scientific tasks to construct a scientific multi-task learner. BigBang-Proton incorporates three fundamental innovations compared to mainstream general-purpose LLMs: Theory-Experiment Learning paradigm aligns large-scale numerical experimental data with theoretical text corpora; Binary Patch Encoding replaces byte pair encoding(BPE) tokenization; Monte Carlo Attention substitutes traditional transformer architectures. Through next-word-prediction pretraining on cross-discipline scientific datasets of real-world problems mixed with general textual corpus, followed by fine-tuning and inference on downstream tasks, BigBang-Proton demonstrates 100\% accuracy in up to 50-digit arithmetic addition operations, performance on par with leading specialized models in particle physics jet tagging, matching MAE of specialized models in inter-atomic potential simulation, performance comparable to traditional spatiotemporal models in water quality prediction, and benchmark-exceeding performance in genome modeling. These results prove that language-guided scientific computing can match or exceed the performance of task-specific scientific models while maintaining multitask learning capabilities. We further hypothesize to scale the pretraining to the universe scale as a fundamental step toward developing material world foundational model. 

**Abstract (ZH)**: BigBang-Proton：一种用于跨尺度、跨结构、跨学科真实世界科学任务的统一序列建模架构 

---
# Direct Token Optimization: A Self-contained Approach to Large Language Model Unlearning 

**Title (ZH)**: 直接token优化：大型语言模型脱习得的自包含方法 

**Authors**: Hong kyu Lee, Ruixuan Liu, Li Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00125)  

**Abstract**: Machine unlearning is an emerging technique that removes the influence of a subset of training data (forget set) from a model without full retraining, with applications including privacy protection, content moderation, and model correction. The key challenge lies in ensuring that the model completely forgets the knowledge of the forget set without compromising its overall utility. Existing unlearning methods for large language models (LLMs) often utilize auxiliary language models, retain datasets, or even commercial AI services for effective unlearning and maintaining the model utility. However, dependence on these external resources is often impractical and could potentially introduce additional privacy risks. In this work, we propose direct token optimization (DTO), a novel self-contained unlearning approach for LLMs that directly optimizes the token level objectives and eliminates the need for external resources. Given a sequence to unlearn, we identify two categories of tokens: target tokens, which capture critical knowledge for unlearning, and the remaining non-target tokens, which are crucial for maintaining the model utility. The former are used to optimize the unlearning objective, while the latter serve to preserve the model's performance. The experimental results show that the proposed DTO achieves up to 16.8$\times$ improvement in forget quality on several benchmark datasets than the latest baselines while maintaining a comparable level of model utility. 

**Abstract (ZH)**: 机器遗忘是一种新兴的技术，它可以在不完全重新训练的情况下从模型中移除部分训练数据（遗忘集）的影响，应用场景包括隐私保护、内容审核和模型修正。关键挑战在于确保模型彻底忘记遗忘集的知识，同时不牺牲其整体效用。现有的大语言模型（LLM）遗忘方法通常利用辅助语言模型、保留数据集，甚至商业AI服务，以实现有效的遗忘并保持模型效用。然而，对外部资源的依赖往往不切实际，并可能引入额外的隐私风险。在这项工作中，我们提出直接令牌优化（DTO），这是一种新的自包含的大语言模型遗忘方法，它直接优化令牌级别目标，消除了对外部资源的依赖。给定一个遗忘序列，我们识别两类令牌：目标令牌，它们捕捉到遗忘的关键知识；以及非目标令牌，它们对于保持模型效用至关重要。前者用于优化遗忘目标，后者用于维护模型性能。实验结果表明，提出的DTO在几个基准数据集上的遗忘质量相比于最新的基线方法提高了多达16.8倍，同时保持了相当水平的模型效用。 

---
# Simulating Student Success in the Age of GenAI: A Kantian-Axiomatic Perspective 

**Title (ZH)**: GenAI时代的学生成功模拟：康德公理解释视角 

**Authors**: Seyma Yaman Kayadibi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00091)  

**Abstract**: This study reinterprets a Monte Carlo simulation of students' perceived success with generative AI (GenAI) through a Kantian-axiomatic lens. Building on prior work, theme-level survey statistics Ease of Use and Learnability, System Efficiency and Learning Burden, and Perceived Complexity and Integration from a representative dataset are used to generate 10,000 synthetic scores per theme on the [1,5] Likert scale. The simulated outputs are evaluated against the axioms of dense linear order without endpoints (DLO): irreflexivity, transitivity, total comparability (connectedness), no endpoints (no greatest and no least; A4-A5), and density (A6). At the data level, the basic ordering axioms (A1-A3) are satisfied, whereas no-endpoints (A4-A5) and density (A6) fail as expected. Likert clipping introduces minimum and maximum observed values, and a finite, discretized sample need not contain a value strictly between any two distinct scores. These patterns are read not as methodological defects but as markers of an epistemological boundary. Following Kant and Friedman, the findings suggest that what simulations capture finite, quantized observations cannot instantiate the ideal properties of an unbounded, dense continuum. Such properties belong to constructive intuition rather than to finite sampling alone. A complementary visualization contrasts the empirical histogram with a sine-curve proxy to clarify this divide. The contribution is interpretive rather than data-expansive: it reframes an existing simulation as a probe of the synthetic a priori structure underlying students' perceptions, showing how formal order-theoretic coherence coexists with principled failures of endpoint-freeness and density in finite empirical models. 

**Abstract (ZH)**: 基于康德公理视角的学生感知生成式AI成功蒙特卡洛模拟再诠释 

---
# SoREX: Towards Self-Explainable Social Recommendation with Relevant Ego-Path Extraction 

**Title (ZH)**: SoREX: 向往关联自心路径提取的自我解释性社会推荐 

**Authors**: Hanze Guo, Yijun Ma, Xiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00080)  

**Abstract**: Social recommendation has been proven effective in addressing data sparsity in user-item interaction modeling by leveraging social networks. The recent integration of Graph Neural Networks (GNNs) has further enhanced prediction accuracy in contemporary social recommendation algorithms. However, many GNN-based approaches in social recommendation lack the ability to furnish meaningful explanations for their predictions. In this study, we confront this challenge by introducing SoREX, a self-explanatory GNN-based social recommendation framework. SoREX adopts a two-tower framework enhanced by friend recommendation, independently modeling social relations and user-item interactions, while jointly optimizing an auxiliary task to reinforce social signals. To offer explanations, we propose a novel ego-path extraction approach. This method involves transforming the ego-net of a target user into a collection of multi-hop ego-paths, from which we extract factor-specific and candidate-aware ego-path subsets as explanations. This process facilitates the summarization of detailed comparative explanations among different candidate items through intricate substructure analysis. Furthermore, we conduct explanation re-aggregation to explicitly correlate explanations with downstream predictions, imbuing our framework with inherent self-explainability. Comprehensive experiments conducted on four widely adopted benchmark datasets validate the effectiveness of SoREX in predictive accuracy. Additionally, qualitative and quantitative analyses confirm the efficacy of the extracted explanations in SoREX. Our code and data are available at this https URL. 

**Abstract (ZH)**: 基于社交网络的自我解释图神经网络推荐框架SoREX 

---
# Adaptive and Resource-efficient Agentic AI Systems for Mobile and Embedded Devices: A Survey 

**Title (ZH)**: 适配性和资源高效的人工智能代理系统：面向移动和嵌入式设备的综述 

**Authors**: Sicong Liu, Weiye Wu, Xiangrui Xu, Teng Li, Bowen Pang, Bin Guo, Zhiwen Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00078)  

**Abstract**: Foundation models have reshaped AI by unifying fragmented architectures into scalable backbones with multimodal reasoning and contextual adaptation. In parallel, the long-standing notion of AI agents, defined by the sensing-decision-action loop, is entering a new paradigm: with FMs as their cognitive core, agents transcend rule-based behaviors to achieve autonomy, generalization, and self-reflection. This dual shift is reinforced by real-world demands such as autonomous driving, robotics, virtual assistants, and GUI agents, as well as ecosystem advances in embedded hardware, edge computing, mobile deployment platforms, and communication protocols that together enable large-scale deployment. Yet this convergence collides with reality: while applications demand long-term adaptability and real-time interaction, mobile and edge deployments remain constrained by memory, energy, bandwidth, and latency. This creates a fundamental tension between the growing complexity of FMs and the limited resources of deployment environments. This survey provides the first systematic characterization of adaptive, resource-efficient agentic AI systems. We summarize enabling techniques into elastic inference, test-time adaptation, dynamic multimodal integration, and agentic AI applications, and identify open challenges in balancing accuracy-latency-communication trade-offs and sustaining robustness under distribution shifts. We further highlight future opportunities in algorithm-system co-design, cognitive adaptation, and collaborative edge deployment. By mapping FM structures, cognition, and hardware resources, this work establishes a unified perspective toward scalable, adaptive, and resource-efficient agentic AI. We believe this survey can help readers to understand the connections between enabling technologies while promoting further discussions on the fusion of agentic intelligence and intelligent agents. 

**Abstract (ZH)**: 基础模型通过将多模态推理和上下文适应统一看作可扩展骨干，重塑了AI，重新整合了碎片化的架构。与此同时，长期存在的基于感知-决策-行动循环的AI代理概念正进入新的范式：借助基础模型作为认知核心，代理超越基于规则的行为，实现自主性、泛化能力和自我反思。这种双重转变在自动驾驶、机器人技术、虚拟助手和GUI代理等实际需求，以及嵌入式硬件、边缘计算、移动部署平台和通信协议的生态系统进步的支持下得以强化，共同促成了大规模部署。然而，这一交汇也面临着现实的挑战：尽管应用程序需要长期适应性和实时交互性，但移动和边缘部署仍然受限于内存、能源、带宽和延迟。这在基础模型日益复杂的趋势与部署环境有限的资源之间造成了根本性的紧张关系。本文综述提供了首个系统化的自适应、资源高效代理AI系统的特征。我们总结了使能技术为弹性推理、测试时适应、动态多模态集成和代理AI应用，并识别了平衡精确度-延迟-通信权衡以及在分布变化下维持鲁棒性的开放挑战。我们还突出了算法-系统协同设计、认知适应和协作边缘部署的未来机遇。通过映射基础模型结构、认知和硬件资源，本文建立了对未来扩展、自适应和资源高效的代理AI的统一视角。我们认为，本文综述有助于读者理解使能技术之间的联系，并促进代理智能与智能代理融合的进一步讨论。 

---
# Identifying All ε-Best Arms in (Misspecified) Linear Bandits 

**Title (ZH)**: 识别线性 bandit 中的所有 ε-最优臂（未正确指定的情况下） 

**Authors**: Zhekai Li, Tianyi Ma, Cheng Hua, Ruihao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.00073)  

**Abstract**: Motivated by the need to efficiently identify multiple candidates in high trial-and-error cost tasks such as drug discovery, we propose a near-optimal algorithm to identify all {\epsilon}-best arms (i.e., those at most {\epsilon} worse than the optimum). Specifically, we introduce LinFACT, an algorithm designed to optimize the identification of all {\epsilon}-best arms in linear bandits. We establish a novel information-theoretic lower bound on the sample complexity of this problem and demonstrate that LinFACT achieves instance optimality by matching this lower bound up to a logarithmic factor. A key ingredient of our proof is to integrate the lower bound directly into the scaling process for upper bound derivation, determining the termination round and thus the sample complexity. We also extend our analysis to settings with model misspecification and generalized linear models. Numerical experiments, including synthetic and real drug discovery data, demonstrate that LinFACT identifies more promising candidates with reduced sample complexity, offering significant computational efficiency and accelerating early-stage exploratory experiments. 

**Abstract (ZH)**: 受高试错成本任务（如药物发现）中高效识别多个候选方案需求的驱动，我们提出了一种近最优算法，用于识别所有ε-best arms（即那些最多比最优解差ε的方案）。具体地，我们引入了LinFACT算法，该算法旨在线性贝叶斯环境中优化所有ε-best arms的识别。我们建立了该问题的新型信息论下界，并证明LinFACT通过在对数因子内的下界达到实例最优性。我们证明的关键要素是将下界直接整合到上限界的缩放过程中，确定终止轮次，从而确定样本复杂度。我们也扩展了我们的分析，涵盖了模型设定错误和广义线性模型的情况。数值实验，包括合成和真实药物发现数据，表明LinFACT能够以减少样本复杂性的成本识别出更有前景的候选方案，提供显著的计算效率并加速早期探索性实验。 

---
# Geo-R1: Unlocking VLM Geospatial Reasoning with Cross-View Reinforcement Learning 

**Title (ZH)**: Geo-R1: 通过跨视图强化学习解锁VLM的地理空间推理能力 

**Authors**: Chenhui Xu, Fuxun Yu, Michael J. Bianco, Jacob Kovarskiy, Raphael Tang, Qi Zhang, Zirui Xu, Will LeVine, Brandon Dubbs, Heming Liao, Cassandra Burgess, Suvam Bag, Jay Patravali, Rupanjali Kukal, Mikael Figueroa, Rishi Madhok, Nikolaos Karianakis, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.00072)  

**Abstract**: We introduce Geo-R1, a reasoning-centric post-training framework that unlocks geospatial reasoning in vision-language models by combining thinking scaffolding and elevating. In the scaffolding stage, Geo-R1 instills a ``geospatial thinking paradigm" via supervised fine-tuning on synthetic chain-of-thought exemplars, enabling models to connect visual cues with geographic priors without costly human reasoning annotations. In the elevating stage, it uses GRPO-based reinforcement learning on a weakly-supervised cross-view pairing proxy. This design supplies a verifiable and scalable reward signal: teaching models to capture and reconcile features across modalities, and harnessing reasoning for accurate prediction. Geo-R1 extends geospatial modeling from domain pretraining / supervised finetuning to reasoning-first post-training, and achieves state-of-the-art performance across various geospatial reasoning benchmarks. Our model is available at this https URL. 

**Abstract (ZH)**: Geo-R1：一种以推理为中心的后训练框架，通过结合支撑和提升解锁视觉语言模型的空间推理能力 

---
# Intelligent 5S Audit: Application of Artificial Intelligence for Continuous Improvement in the Automotive Industry 

**Title (ZH)**: 智能5S审核：人工智能在汽车行业持续改进中的应用 

**Authors**: Rafael da Silva Maciel, Lucio Veraldo Jr  

**Link**: [PDF](https://arxiv.org/pdf/2510.00067)  

**Abstract**: The evolution of the 5S methodology with the support of artificial intelligence techniques represents a significant opportunity to improve industrial organization audits in the automotive chain, making them more objective, efficient and aligned with Industry 4.0 standards. This work developed an automated 5S audit system based on large-scale language models (LLM), capable of assessing the five senses (Seiri, Seiton, Seiso, Seiketsu, Shitsuke) in a standardized way through intelligent image analysis. The system's reliability was validated using Cohen's concordance coefficient (kappa = 0.75), showing strong alignment between the automated assessments and the corresponding human audits. The results indicate that the proposed solution contributes significantly to continuous improvement in automotive manufacturing environments, speeding up the audit process by 50% of the traditional time and maintaining the consistency of the assessments, with a 99.8% reduction in operating costs compared to traditional manual audits. The methodology presented establishes a new paradigm for integrating lean systems with emerging AI technologies, offering scalability for implementation in automotive plants of different sizes. 

**Abstract (ZH)**: 基于人工智能技术的5S方法学演化为汽车产业链工业组织审计提供了提升机会，使其更具客观性、效率并符合第四次工业革命标准。本研究开发了一种基于大规模语言模型（LLM）的自动化5S审计系统，能够通过智能图像分析以标准化方式评估五感（整理、整顿、清扫、标准化、自律）。系统可靠性通过科恩一致性系数（κ=0.75）验证，显示自动评估与相应的人工审计之间有强烈的一致性。结果表明，所提出解决方案对汽车制造环境的持续改进具有重要意义，审计过程比传统方法快50%，且保持评估一致性，与传统人工审计相比，运营成本降低99.8%。所提出的方法建立了将精益系统与新兴人工智能技术整合的新 paradign，为不同规模的汽车工厂实施提供了可扩展性。 

---
# AstroMMBench: A Benchmark for Evaluating Multimodal Large Language Models Capabilities in Astronomy 

**Title (ZH)**: AstroMMBench: 评估多模态大型语言模型在天文学领域能力的基准测试 

**Authors**: Jinghang Shi, Xiao Yu Tang, Yang Hunag, Yuyang Li, Xiaokong, Yanxia Zhang, Caizhan Yue  

**Link**: [PDF](https://arxiv.org/pdf/2510.00063)  

**Abstract**: Astronomical image interpretation presents a significant challenge for applying multimodal large language models (MLLMs) to specialized scientific tasks. Existing benchmarks focus on general multimodal capabilities but fail to capture the complexity of astronomical data. To bridge this gap, we introduce AstroMMBench, the first comprehensive benchmark designed to evaluate MLLMs in astronomical image understanding. AstroMMBench comprises 621 multiple-choice questions across six astrophysical subfields, curated and reviewed by 15 domain experts for quality and relevance. We conducted an extensive evaluation of 25 diverse MLLMs, including 22 open-source and 3 closed-source models, using AstroMMBench. The results show that Ovis2-34B achieved the highest overall accuracy (70.5%), demonstrating leading capabilities even compared to strong closed-source models. Performance showed variations across the six astrophysical subfields, proving particularly challenging in domains like cosmology and high-energy astrophysics, while models performed relatively better in others, such as instrumentation and solar astrophysics. These findings underscore the vital role of domain-specific benchmarks like AstroMMBench in critically evaluating MLLM performance and guiding their targeted development for scientific applications. AstroMMBench provides a foundational resource and a dynamic tool to catalyze advancements at the intersection of AI and astronomy. 

**Abstract (ZH)**: 天文学图像解释为将多模态大型语言模型应用于专门的科学任务带来了显著挑战。现有基准侧重于一般的多模态能力，但未能捕捉到天文学数据的复杂性。为填补这一空白，我们介绍了AstroMMBench，这是首个专门设计用于评估多模态大型语言模型在天文学图像理解中的综合基准。AstroMMBench包含621个多选题，涵盖了六个天体物理子领域，并由15名领域专家审校以确保质量和相关性。我们使用AstroMMBench对25种不同的多模态大型语言模型进行了广泛评估，其中包括22种开源模型和3种闭源模型。结果显示，Ovis2-34B取得了最高整体准确率（70.5%），即使与强大的闭源模型相比也表现出领先的能力。性能在六个天体物理子领域中存在差异，尤其在宇宙学和高能天体物理等领域具有挑战性，而在仪器技术和太阳天体物理等领域，模型表现相对较好。这些发现强调了如AstroMMBench这类特定领域基准在严格评估多模态大型语言模型性能和指导其针对科学应用的开发方面的重要作用。AstroMMBench提供了基础资源并充当了一个动态工具，促进了人工智能与天文学交叉领域的发展。 

---
# Efficient CNN Compression via Multi-method Low Rank Factorization and Feature Map Similarity 

**Title (ZH)**: 基于多方法低秩分解和特征图相似性的高效CNN压缩 

**Authors**: M. Kokhazadeh, G. Keramidas, V. Kelefouras  

**Link**: [PDF](https://arxiv.org/pdf/2510.00062)  

**Abstract**: Low-Rank Factorization (LRF) is a widely adopted technique for compressing deep neural networks (DNNs). However, it faces several challenges, including optimal rank selection, a vast design space, long fine-tuning times, and limited compatibility with different layer types and decomposition methods. This paper presents an end-to-end Design Space Exploration (DSE) methodology and framework for compressing convolutional neural networks (CNNs) that addresses all these issues. We introduce a novel rank selection strategy based on feature map similarity, which captures non-linear interactions between layer outputs more effectively than traditional weight-based approaches. Unlike prior works, our method uses a one-shot fine-tuning process, significantly reducing the overall fine-tuning time. The proposed framework is fully compatible with all types of convolutional (Conv) and fully connected (FC) layers. To further improve compression, the framework integrates three different LRF techniques for Conv layers and three for FC layers, applying them selectively on a per-layer basis. We demonstrate that combining multiple LRF methods within a single model yields better compression results than using a single method uniformly across all layers. Finally, we provide a comprehensive evaluation and comparison of the six LRF techniques, offering practical insights into their effectiveness across different scenarios. The proposed work is integrated into TensorFlow 2.x, ensuring compatibility with widely used deep learning workflows. Experimental results on 14 CNN models across eight datasets demonstrate that the proposed methodology achieves substantial compression with minimal accuracy loss, outperforming several state-of-the-art techniques. 

**Abstract (ZH)**: 基于端到端设计空间探索的卷积神经网络低秩分解压缩方法 

---
# Survey of AI-Powered Approaches for Osteoporosis Diagnosis in Medical Imaging 

**Title (ZH)**: 人工智能驱动的骨质疏松诊断在医学 imaging 中的方法调研 

**Authors**: Abdul Rahman, Bumshik Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.00061)  

**Abstract**: Osteoporosis silently erodes skeletal integrity worldwide; however, early detection through imaging can prevent most fragility fractures. Artificial intelligence (AI) methods now mine routine Dual-energy X-ray Absorptiometry (DXA), X-ray, Computed Tomography (CT), and Magnetic Resonance Imaging (MRI) scans for subtle, clinically actionable markers, but the literature is fragmented. This survey unifies the field through a tri-axial framework that couples imaging modalities with clinical tasks and AI methodologies (classical machine learning, convolutional neural networks (CNNs), transformers, self-supervised learning, and explainable AI). Following a concise clinical and technical primer, we detail our Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA)-guided search strategy, introduce the taxonomy via a roadmap figure, and synthesize cross-study insights on data scarcity, external validation, and interpretability. By identifying emerging trends, open challenges, and actionable research directions, this review provides AI scientists, medical imaging researchers, and musculoskeletal clinicians with a clear compass to accelerate rigorous, patient-centered innovation in osteoporosis care. The project page of this survey can also be found on Github. 

**Abstract (ZH)**: 骨质疏松悄无声息地削弱骨骼完整性；然而，通过成像早期检测可以预防大多数脆性骨折。现有的人工智能方法正在挖掘常规的双能X射线 absorptiometry (DXA)、X射线、计算机断层扫描(CT)和磁共振成像(MRI)扫描以寻找细微的、具有临床意义的标志物，但文献资料碎片化。本文通过一个三维框架统一了该领域，该框架将成像模态与临床任务和人工智能方法（经典机器学习、卷积神经网络(CNNs)、变压器、半监督学习和可解释的人工智能）耦合在一起。在简要介绍临床和技术基础知识之后，我们详细阐述了一种由系统综述和meta分析指南(PRISMA)指导的搜索策略，通过路线图图表引入分类，并综合跨研究视角的数据稀缺性、外部验证和可解释性。通过识别新兴趋势、开放挑战和可操作的研究方向，本文为人工智能科学家、医学成像研究人员和骨骼肌肉临床医生提供了一个清晰的方向，以加速注重患者需求的骨质疏松症护理创新。该项目页面也可以在Github上找到。 

---
# Less is More: Lean yet Powerful Vision-Language Model for Autonomous Driving 

**Title (ZH)**: 少即是多：轻量而强大的 autonomous driving 视觉-语言模型 

**Authors**: Sheng Yang, Tong Zhan, Guancheng Chen, Yanfeng Lu, Jian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00060)  

**Abstract**: In this work, we reconceptualize autonomous driving as a generalized language and formulate the trajectory planning task as next waypoint prediction. We introduce Max-V1, a novel framework for one-stage end-to-end autonomous driving. Our framework presents a single-pass generation paradigm that aligns with the inherent sequentiality of driving. This approach leverages the generative capacity of the VLM (Vision-Language Model) to enable end-to-end trajectory prediction directly from front-view camera input. The efficacy of this method is underpinned by a principled supervision strategy derived from statistical modeling. This provides a well-defined learning objective, which makes the framework highly amenable to master complex driving policies through imitation learning from large-scale expert demonstrations. Empirically, our method achieves the state-of-the-art performance on the nuScenes dataset, delivers an overall improvement of over 30% compared to prior baselines. Furthermore, it exhibits superior generalization performance on cross-domain datasets acquired from diverse vehicles, demonstrating notable potential for cross-vehicle robustness and adaptability. Due to these empirical strengths, this work introduces a model enabling fundamental driving behaviors, laying the foundation for the development of more capable self-driving agents. Code will be available upon publication. 

**Abstract (ZH)**: 本研究重新conceptualize自动驾驶为一种通用语言，并将轨迹规划任务形式化为下一目标点预测。我们引入Max-V1，一种新的端到端自动驾驶一阶段框架。该框架采用单次通过生成 paradigm，符合驾驶固有的序列性。该方法利用VLM（视觉-语言模型）的生成能力，直接从前方摄像头输入实现端到端的轨迹预测。该方法的有效性源于从统计建模中得出的原则性监督策略，这为通过大规模专家演示模仿学习掌握复杂驾驶策略提供了明确的学习目标。实验上，该方法在nuScenes数据集上达到state-of-the-art性能，并比先前基线提高了超过30%的整体性能。此外，该方法在不同车辆采集的跨域数据集中表现出优越的泛化性能，展示了跨车辆鲁棒性和适应性的显著潜力。由于这些实验上的优势，本研究引入了一个模型，奠定开发更强大自动驾驶代理的基础。代码将在发表后对外开放。 

---
# FSDENet: A Frequency and Spatial Domains based Detail Enhancement Network for Remote Sensing Semantic Segmentation 

**Title (ZH)**: 基于频域与空域的细节增强网络：遥感语义分割中的应用 

**Authors**: Jiahao Fu, Yinfeng Yu, Liejun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00059)  

**Abstract**: To fully leverage spatial information for remote sensing image segmentation and address semantic edge ambiguities caused by grayscale variations (e.g., shadows and low-contrast regions), we propose the Frequency and Spatial Domains based Detail Enhancement Network (FSDENet). Our framework employs spatial processing methods to extract rich multi-scale spatial features and fine-grained semantic details. By effectively integrating global and frequency-domain information through the Fast Fourier Transform (FFT) in global mappings, the model's capability to discern global representations under grayscale variations is significantly strengthened. Additionally, we utilize Haar wavelet transform to decompose features into high- and low-frequency components, leveraging their distinct sensitivity to edge information to refine boundary segmentation. The model achieves dual-domain synergy by integrating spatial granularity with frequency-domain edge sensitivity, substantially improving segmentation accuracy in boundary regions and grayscale transition zones. Comprehensive experimental results demonstrate that FSDENet achieves state-of-the-art (SOTA) performance on four widely adopted datasets: LoveDA, Vaihingen, Potsdam, and iSAID. 

**Abstract (ZH)**: 基于频域和空域细节增强网络（FSDENet）：充分利用空间信息并通过灰度变化（例如阴影和低对比度区域）引起的语义边缘模糊性进行遥感图像分割 

---
# HiDe: Rethinking The Zoom-IN method in High Resolution MLLMs via Hierarchical Decoupling 

**Title (ZH)**: HiDe: 通过分层解耦重新思考高分辨率多模态大语言模型中的Zoom-IN方法 

**Authors**: Xianjie Liu, Yiman Hu, Yixiong Zou, Liang Wu, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.00054)  

**Abstract**: Multimodal Large Language Models (MLLMs) have made significant strides in visual understanding tasks. However, their performance on high-resolution images remains suboptimal. While existing approaches often attribute this limitation to perceptual constraints and argue that MLLMs struggle to recognize small objects, leading them to use "zoom in" strategies for better detail, our analysis reveals a different cause: the main issue is not object size, but rather caused by complex background interference. We systematically analyze this "zoom in" operation through a series of decoupling experiments and propose the Hierarchical Decoupling Framework (HiDe), a training-free framework that uses Token-wise Attention Decoupling (TAD) to decouple the question tokens and identify the key information tokens, then leverages their attention weights to achieve precise alignment with the target visual regions. Subsequently, it employs Layout-Preserving Decoupling (LPD) to decouple these regions from the background and reconstructs a compact representation that preserves essential spatial layouts while eliminating background interference. HiDe sets a new SOTA on V*Bench, HRBench4K, and HRBench8K, boosting Qwen2.5-VL 7B and InternVL3 8B to SOTA (92.1% and 91.6% on V*Bench), even surpassing RL methods. After optimization, HiDe uses 75% less memory than the previous training-free approach. Code is provided in this https URL. 

**Abstract (ZH)**: 多模态大型语言模型在高分辨率图像上的视觉理解任务中取得了显著进展，但其性能仍然不尽如人意。现有的方法往往将这一限制归因于感知约束，并认为MLLMs难以识别小物体，从而采取“放大”策略以获得更好的细节。然而，我们的分析揭示了不同的原因：主要问题不是物体大小，而是复杂的背景干扰所致。我们通过一系列去耦合实验系统地分析了这种“放大”操作，并提出了层次去耦框架（HiDe），这是一种无需训练的框架，通过Token-wise注意力去耦（TAD）将问题标记与关键信息标记分离，并利用其注意力权重实现与目标视觉区域的精确对齐。随后，它使用布局保持去耦（LPD）将这些区域从背景中分离出来，并重建一个保留关键空间布局的同时消除背景干扰的紧凑表示。HiDe在V*Bench、HRBench4K和HRBench8K上达到了新的SOTA，将Qwen2.5-VL 7B和InternVL3 8B提升至SOTA（V*Bench上分别为92.1%和91.6%），甚至超过了RL方法。优化后，HiDe相比之前的无需训练方法节省了75%的内存。 

---
# Object-AVEdit: An Object-level Audio-Visual Editing Model 

**Title (ZH)**: 对象级音视频编辑模型 

**Authors**: Youquan Fu, Ruiyang Si, Hongfa Wang, Dongzhan Zhou, Jiacheng Sun, Ping Luo, Di Hu, Hongyuan Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00050)  

**Abstract**: There is a high demand for audio-visual editing in video post-production and the film making field. While numerous models have explored audio and video editing, they struggle with object-level audio-visual operations. Specifically, object-level audio-visual editing requires the ability to perform object addition, replacement, and removal across both audio and visual modalities, while preserving the structural information of the source instances during the editing process. In this paper, we present \textbf{Object-AVEdit}, achieving the object-level audio-visual editing based on the inversion-regeneration paradigm. To achieve the object-level controllability during editing, we develop a word-to-sounding-object well-aligned audio generation model, bridging the gap in object-controllability between audio and current video generation models. Meanwhile, to achieve the better structural information preservation and object-level editing effect, we propose an inversion-regeneration holistically-optimized editing algorithm, ensuring both information retention during the inversion and better regeneration effect. Extensive experiments demonstrate that our editing model achieved advanced results in both audio-video object-level editing tasks with fine audio-visual semantic alignment. In addition, our developed audio generation model also achieved advanced performance. More results on our project page: this https URL. 

**Abstract (ZH)**: 高需求驱动的音频-视觉编辑在视频后期制作和电影制作领域中日益增长。尽管众多模型已经探索了音频和视频编辑，但在对象级音频-视觉操作方面仍存在问题。具体而言，对象级音频-视觉编辑需要在保持源实例结构信息的前提下，在音频和视觉模态上执行对象添加、替换和移除操作。本文中，我们提出\textbf{Object-AVEdit}，基于反转再生范式实现了对象级音频-视觉编辑。为了在编辑过程中实现对象级别的可控性，我们开发了一种词至声音对象高对齐的音频生成模型，缩小了音频和当前视频生成模型之间对象可控性的差距。同时，为了在保持更好结构信息和对象级编辑效果的前提下，我们提出了一种整体优化的编辑算法，确保反转过程中的信息保留和更好的再生效果。大量实验表明，我们的编辑模型在音频-视频对象级编辑任务中实现了高度精细的音频-视觉语义对齐。此外，我们开发的音频生成模型也取得了优异的性能。更多实验结果请参见项目页面：this https URL。 

---
# AI-Based Stroke Rehabilitation Domiciliary Assessment System with ST_GCN Attention 

**Title (ZH)**: 基于AI的ST_GCN注意力机制家用卒中康复评估系统 

**Authors**: Suhyeon Lim, Ye-eun Kim, Andrew J. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.00049)  

**Abstract**: Effective stroke recovery requires continuous rehabilitation integrated with daily living. To support this need, we propose a home-based rehabilitation exercise and feedback system. The system consists of (1) hardware setup with RGB-D camera and wearable sensors to capture Stroke movements, (2) a mobile application for exercise guidance, and (3) an AI server for assessment and feedback. When Stroke user exercises following the application guidance, the system records skeleton sequences, which are then Assessed by the deep learning model, RAST-G@. The model employs a spatio-temporal graph convolutional network (ST-GCN) to extract skeletal features and integrates transformer-based temporal attention to figure out action quality. For system implementation, we constructed the NRC dataset, include 10 upper-limb activities of daily living (ADL) and 5 range-of-motion (ROM) collected from stroke and non-disabled participants, with Score annotations provided by licensed physiotherapists. Results on the KIMORE and NRC datasets show that RAST-G@ improves over baseline in terms of MAD, RMSE, and MAPE. Furthermore, the system provides user feedback that combines patient-centered assessment and monitoring. The results demonstrate that the proposed system offers a scalable approach for quantitative and consistent domiciliary rehabilitation assessment. 

**Abstract (ZH)**: 有效的中风恢复需要结合日常生活的持续康复。为此，我们提出了一种基于家庭的康复锻炼和反馈系统。该系统包括（1）使用RGB-D相机和可穿戴传感器的硬件设置以捕捉中风患者的动作，（2）用于指导锻炼的移动应用，以及（3）用于评估和反馈的AI服务器。当中风患者按照应用程序指导进行锻炼时，系统记录骨架序列，这些序列随后由深度学习模型RAST-G@评估。该模型使用空间-时间图卷积网络（ST-GCN）提取骨骼特征，并结合基于 Transformer 的时间注意力机制来确定动作质量。在系统实现方面，我们构建了NRC数据集，包括来自中风和非残疾参与者、由认证理疗师提供评分注释的10种日常生活中上肢活动和5种关节活动范围（ROM）。在KIMORE和NRC数据集上的结果显示，RAST-G@在MAD、RMSE和MAPE方面优于基线。此外，该系统提供了结合患者中心评估和监测的用户反馈。结果表明，所提出的系统为定量和一致的家庭康复评估提供了一种可扩展的方法。 

---
# Deep Learning Approaches with Explainable AI for Differentiating Alzheimer Disease and Mild Cognitive Impairment 

**Title (ZH)**: 基于可解释人工智能的深度学习方法在区分阿尔茨海默病和轻度认知 impairment 方面的应用 

**Authors**: Fahad Mostafa, Kannon Hossain, Hafiz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2510.00048)  

**Abstract**: Early and accurate diagnosis of Alzheimer Disease is critical for effective clinical intervention, particularly in distinguishing it from Mild Cognitive Impairment, a prodromal stage marked by subtle structural changes. In this study, we propose a hybrid deep learning ensemble framework for Alzheimer Disease classification using structural magnetic resonance imaging. Gray and white matter slices are used as inputs to three pretrained convolutional neural networks such as ResNet50, NASNet, and MobileNet, each fine tuned through an end to end process. To further enhance performance, we incorporate a stacked ensemble learning strategy with a meta learner and weighted averaging to optimally combine the base models. Evaluated on the Alzheimer Disease Neuroimaging Initiative dataset, the proposed method achieves state of the art accuracy of 99.21% for Alzheimer Disease vs. Mild Cognitive Impairment and 91.0% for Mild Cognitive Impairment vs. Normal Controls, outperforming conventional transfer learning and baseline ensemble methods. To improve interpretability in image based diagnostics, we integrate Explainable AI techniques by Gradient weighted Class Activation, which generates heatmaps and attribution maps that highlight critical regions in gray and white matter slices, revealing structural biomarkers that influence model decisions. These results highlight the frameworks potential for robust and scalable clinical decision support in neurodegenerative disease diagnostics. 

**Abstract (ZH)**: 早期且准确的阿尔茨海默病诊断对于有效临床干预至关重要，特别是对于轻度认知障碍的区分，后者是一个前驱阶段，伴有细微的结构变化。本研究提出了一种混合深度学习集成框架，用于利用结构磁共振成像对阿尔茨海默病进行分类。灰质和白质切片作为输入，传递给预训练的ResNet50、NASNet和MobileNet三种卷积神经网络，并通过端到端的过程进行微调。为进一步提高性能，我们引入了一种带有元学习器和加权平均的堆叠集成学习策略，以最优地组合基模型。在ADNI数据集上评估，本方法实现了阿尔茨海默病与轻度认知障碍之间99.21%的准确率，与正常对照之间91.0%的准确率，优于传统迁移学习和基准集成方法。为了改善基于图像的诊断解释性，我们通过梯度加权分类激活的可解释人工智能技术，生成热点图和归因图，突出灰质和白质切片中的关键区域，揭示影响模型决策的结构性生物标志物。这些结果突显了该框架在神经退行性疾病诊断中的稳健性和可扩展性临床决策支持潜力。 

---
# Explanation-Driven Counterfactual Testing for Faithfulness in Vision-Language Model Explanations 

**Title (ZH)**: 基于解释驱动的反事实测试以确保视觉-语言模型解释的忠实性 

**Authors**: Sihao Ding, Santosh Vasa, Aditi Ramadwar  

**Link**: [PDF](https://arxiv.org/pdf/2510.00047)  

**Abstract**: Vision-Language Models (VLMs) often produce fluent Natural Language Explanations (NLEs) that sound convincing but may not reflect the causal factors driving predictions. This mismatch of plausibility and faithfulness poses technical and governance risks. We introduce Explanation-Driven Counterfactual Testing (EDCT), a fully automated verification procedure for a target VLM that treats the model's own explanation as a falsifiable hypothesis. Given an image-question pair, EDCT: (1) obtains the model's answer and NLE, (2) parses the NLE into testable visual concepts, (3) generates targeted counterfactual edits via generative inpainting, and (4) computes a Counterfactual Consistency Score (CCS) using LLM-assisted analysis of changes in both answers and explanations. Across 120 curated OK-VQA examples and multiple VLMs, EDCT uncovers substantial faithfulness gaps and provides regulator-aligned audit artifacts indicating when cited concepts fail causal tests. 

**Abstract (ZH)**: Vision-Language模型（VLMs）生成的自然语言解释（NLEs）通常流畅且听起来令人信服，但可能并未反映预测背后的因果因素。这种可信度和忠实度之间的不匹配带来了技术风险和治理风险。我们引入了基于解释的反事实测试（EDCT），这是一种全自动验证程序，将模型自身解释视为可证伪的假说。给定图像-问题对，EDCT：（1）获取模型的答案和NLE，（2）将NLE解析为可测试的视觉概念，（3）通过生成填充生成针对性的反事实编辑，（4）使用LLM辅助分析答案和解释的变化来计算反事实一致性分数（CCS）。在120个精心策划的OK-VQA示例和多种VLMs上，EDCT发现了显著的忠实度差距，并提供了符合监管要求的审计证据，表明引用的概念在因果测试中失败。 

---
# Reinforcement Learning-Based Prompt Template Stealing for Text-to-Image Models 

**Title (ZH)**: 基于强化学习的提示模板窃取文本到图像模型 

**Authors**: Xiaotian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2510.00046)  

**Abstract**: Multimodal Large Language Models (MLLMs) have transformed text-to-image workflows, allowing designers to create novel visual concepts with unprecedented speed. This progress has given rise to a thriving prompt trading market, where curated prompts that induce trademark styles are bought and sold. Although commercially attractive, prompt trading also introduces a largely unexamined security risk: the prompts themselves can be stolen.
In this paper, we expose this vulnerability and present RLStealer, a reinforcement learning based prompt inversion framework that recovers its template from only a small set of example images. RLStealer treats template stealing as a sequential decision making problem and employs multiple similarity based feedback signals as reward functions to effectively explore the prompt space. Comprehensive experiments on publicly available benchmarks demonstrate that RLStealer gets state-of-the-art performance while reducing the total attack cost to under 13% of that required by existing baselines. Our further analysis confirms that RLStealer can effectively generalize across different image styles to efficiently steal unseen prompt templates. Our study highlights an urgent security threat inherent in prompt trading and lays the groundwork for developing protective standards in the emerging MLLMs marketplace. 

**Abstract (ZH)**: 多模态大规模语言模型（MLLMs）已革新了文本 إلى图像的工作流程，使设计师能够以前所未有的速度创作出新颖的视觉概念。这一进展催生了一个蓬勃发展的提示交易市场，在这个市场上，诱导特定品牌风格的精心挑选提示被买卖。尽管从商业角度来看颇具吸引力，但提示交易同时也带来了一个尚未充分研究的安全风险：提示本身可以被盗取。

在这种情况下，我们揭露了这一漏洞，并提出了基于强化学习的提示反转框架RLStealer，仅通过少量示例图像即可恢复其模板。RLStealer将模板窃取视为一个顺序决策问题，并采用多种基于相似性的反馈信号作为奖励函数，有效探索提示空间。在公开可用基准上的全面实验表明，RLStealer在性能上达到最先进的水平，同时将总攻击成本降低到现有基线所需成本的不足13%。我们的进一步分析证实，RLStealer能够有效泛化到不同的图像风格，以高效地窃取未见过的提示模板。我们的研究突显了提示交易中固有的紧迫安全威胁，并为正在兴起的MLLMs市场制定保护标准奠定了基础。 

---
# Beyond the Prompt: Gender Bias in Text-to-Image Models, with a Case Study on Hospital Professions 

**Title (ZH)**: 超越提示：文本生成图像模型中的性别偏见，以医院职业为例 

**Authors**: Franck Vandewiele, Remi Synave, Samuel Delepoulle, Remi Cozot  

**Link**: [PDF](https://arxiv.org/pdf/2510.00045)  

**Abstract**: Text-to-image (TTI) models are increasingly used in professional, educational, and creative contexts, yet their outputs often embed and amplify social biases. This paper investigates gender representation in six state-of-the-art open-weight models: HunyuanImage 2.1, HiDream-I1-dev, Qwen-Image, FLUX.1-dev, Stable-Diffusion 3.5 Large, and Stable-Diffusion-XL. Using carefully designed prompts, we generated 100 images for each combination of five hospital-related professions (cardiologist, hospital director, nurse, paramedic, surgeon) and five portrait qualifiers ("", corporate, neutral, aesthetic, beautiful).
Our analysis reveals systematic occupational stereotypes: all models produced nurses exclusively as women and surgeons predominantly as men. However, differences emerge across models: Qwen-Image and SDXL enforce rigid male dominance, HiDream-I1-dev shows mixed outcomes, and FLUX.1-dev skews female in most roles. HunyuanImage 2.1 and Stable-Diffusion 3.5 Large also reproduce gender stereotypes but with varying degrees of sensitivity to prompt formulation. Portrait qualifiers further modulate gender balance, with terms like corporate reinforcing male depictions and beautiful favoring female ones. Sensitivity varies widely: Qwen-Image remains nearly unaffected, while FLUX.1-dev, SDXL, and SD3.5 show strong prompt dependence.
These findings demonstrate that gender bias in TTI models is both systematic and model-specific. Beyond documenting disparities, we argue that prompt wording plays a critical role in shaping demographic outcomes. The results underscore the need for bias-aware design, balanced defaults, and user guidance to prevent the reinforcement of occupational stereotypes in generative AI. 

**Abstract (ZH)**: 文本到图像（TTI）模型在专业、教育和创意领域中的应用日益增多，但其输出往往嵌入并放大了社会偏见。本文调查了六款当今最先进的预训练模型中的人物性别表现：HunyuanImage 2.1、HiDream-I1-dev、Qwen-Image、FLUX.1-dev、Stable-Diffusion 3.5 Large和Stable-Diffusion-XL。通过精心设计的提示，我们为五种与医院相关的职业（心脏病专家、医院院长、护士、救护员、外科医生）和五种肖像修饰词（ "", 专业性、中性、审美性、美丽）每种组合生成了100张图像。 

---
# Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness 

**Title (ZH)**: 框架中的文化：C$^3$B作为一种基于漫画的多模态文化awareness基准 

**Authors**: Yuchen Song, Andong Chen, Wenxin Zhu, Kehai Chen, Xuefeng Bai, Muyun Yang, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00041)  

**Abstract**: Cultural awareness capabilities has emerged as a critical capability for Multimodal Large Language Models (MLLMs). However, current benchmarks lack progressed difficulty in their task design and are deficient in cross-lingual tasks. Moreover, current benchmarks often use real-world images. Each real-world image typically contains one culture, making these benchmarks relatively easy for MLLMs. Based on this, we propose C$^3$B ($\textbf{C}$omics $\textbf{C}$ross-$\textbf{C}$ultural $\textbf{B}$enchmark), a novel multicultural, multitask and multilingual cultural awareness capabilities benchmark. C$^3$B comprises over 2000 images and over 18000 QA pairs, constructed on three tasks with progressed difficulties, from basic visual recognition to higher-level cultural conflict understanding, and finally to cultural content generation. We conducted evaluations on 11 open-source MLLMs, revealing a significant performance gap between MLLMs and human performance. The gap demonstrates that C$^3$B poses substantial challenges for current MLLMs, encouraging future research to advance the cultural awareness capabilities of MLLMs. 

**Abstract (ZH)**: 文化意识能力已成为多模态大型语言模型（MLLMs）的关键能力。然而，当前基准在任务设计上缺乏进步的难度，并且在跨语言任务方面存在不足。此外，当前基准往往使用真实世界图像。每个真实世界图像通常包含一种文化，使得这些基准对于MLLMs相对容易。基于此，我们提出了C$^3$B（C$^3$B：跨文化多任务多语言文化意识能力基准），这是一种新颖的多文化、多任务和多语言文化意识能力基准。C$^3$B包含超过2000张图像和超过18000个问答对，并基于从基本视觉识别到较高层次的文化冲突理解，再到文化内容生成的三个逐步递增难度的任务构建。我们在11个开源MLLM上进行了评估，揭示了MLLMs与人类性能之间显著的性能差距。这一差距表明C$^3$B为当前MLLMs提出了重大挑战，鼓励未来研究提高MLLMs的文化意识能力。 

---
# Uncovering Intrinsic Capabilities: A Paradigm for Data Curation in Vision-Language Models 

**Title (ZH)**: 揭示内在能力：视觉-语言模型中数据整理的范式 

**Authors**: Junjie Li, Ziao Wang, Jianghong Ma, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00040)  

**Abstract**: Large vision-language models (VLMs) achieve strong benchmark performance, but controlling their behavior through instruction tuning remains difficult. Reducing the budget of instruction tuning dataset often causes regressions, as heuristic strategies treat models as black boxes and overlook the latent capabilities that govern learning. We introduce Capability-Attributed Data Curation (CADC), a framework that shifts curation from task-specific heuristics to intrinsic capability analysis. CADC discovers intrinsic capabilities in an unsupervised manner from gradient-based learning trajectories, attributes training data to these capabilities via influence estimation, and curates capability-aware curricula through balanced selection and staged sequencing. This transforms black-box instruction tuning into a controllable, capability-driven process. With as little as 5% of the original data, CADC surpasses full-data training on multimodal benchmarks. These results validate intrinsic capabilities as the fundamental building blocks of model learning and establish CADC as a principle paradigm for instruction data curation. 

**Abstract (ZH)**: 大规模多模态视觉-语言模型在基准测试中表现出色，但通过指令调优控制其行为仍具挑战性。减少指令调优数据集的预算往往会引发性能倒退，因为启发式策略将模型视为黑盒并忽视了控制学习的潜在能力。我们提出了能力归因数据收集框架（CADC），该框架将数据收集从特定任务的启发式策略转移到内在能力分析。CADC通过梯度导向的学习轨迹以无监督方式发现内在能力，通过影响估计将训练数据归因于这些能力，并通过平衡选择和分阶段排序制定能力意识的教学计划，从而将黑盒指令调优转化为可控的能力驱动过程。仅使用原始数据的5%，CADC在多模态基准测试中超过了全数据训练。这些结果验证了内在能力是模型学习的基本构建块，并确立了CADC作为指令数据收集基本原则框架的地位。 

---
# AutoPK: Leveraging LLMs and a Hybrid Similarity Metric for Advanced Retrieval of Pharmacokinetic Data from Complex Tables and Documents 

**Title (ZH)**: AutoPK：利用大规模语言模型和混合相似度度量从复杂表格和文档中进行高级药代动力学数据检索 

**Authors**: Hossein Sholehrasa, Amirhossein Ghanaatian, Doina Caragea, Lisa A. Tell, Jim E. Riviere, Majid Jaberi-Douraki  

**Link**: [PDF](https://arxiv.org/pdf/2510.00039)  

**Abstract**: Pharmacokinetics (PK) plays a critical role in drug development and regulatory decision-making for human and veterinary medicine, directly affecting public health through drug safety and efficacy assessments. However, PK data are often embedded in complex, heterogeneous tables with variable structures and inconsistent terminologies, posing significant challenges for automated PK data retrieval and standardization. AutoPK, a novel two-stage framework for accurate and scalable extraction of PK data from complex scientific tables. In the first stage, AutoPK identifies and extracts PK parameter variants using large language models (LLMs), a hybrid similarity metric, and LLM-based validation. The second stage filters relevant rows, converts the table into a key-value text format, and uses an LLM to reconstruct a standardized table. Evaluated on a real-world dataset of 605 PK tables, including captions and footnotes, AutoPK shows significant improvements in precision and recall over direct LLM baselines. For instance, AutoPK with LLaMA 3.1-70B achieved an F1-score of 0.92 on half-life and 0.91 on clearance parameters, outperforming direct use of LLaMA 3.1-70B by margins of 0.10 and 0.21, respectively. Smaller models such as Gemma 3-27B and Phi 3-12B with AutoPK achieved 2-7 fold F1 gains over their direct use, with Gemma's hallucination rates reduced from 60-95% down to 8-14%. Notably, AutoPK enabled open-source models like Gemma 3-27B to outperform commercial systems such as GPT-4o Mini on several PK parameters. AutoPK enables scalable and high-confidence PK data extraction, making it well-suited for critical applications in veterinary pharmacology, drug safety monitoring, and public health decision-making, while addressing heterogeneous table structures and terminology and demonstrating generalizability across key PK parameters. Code and data: this https URL 

**Abstract (ZH)**: 药代动力学（PK）在药物开发和人用及兽用药物监管决策中起着关键作用，直接影响公共健康的药物安全性与有效性评估。然而，PK数据通常嵌入在结构复杂、异质性高且术语不一致的表格中，这给自动化PK数据检索与标准化带来了巨大挑战。AutoPK，一种新颖的两阶段框架，用于从复杂科学表格中准确且可扩展地提取PK数据。第一阶段使用大型语言模型（LLMs）、混合相似度度量和基于LLM的验证来识别和提取PK参数变体。第二阶段过滤相关行，将表格转换为键值文本格式，并使用LLM重建标准化表格。在包含605张PK表格（包括表格标题和脚注）的真实数据集上评估，AutoPK在精度和召回率方面显著优于直接的LLM基线。例如，使用LLaMA 3.1-70B的AutoPK在半衰期参数上的F1分数为0.92，清除率参数上的F1分数为0.91，分别比直接使用LLaMA 3.1-70B高0.10和0.21。较小的模型如Gemma 3-27B和Phi 3-12B使用AutoPK实现了2-7倍的F1分数提升，Gemma的幻觉率从60-95%降至8-14%。值得注意的是，AutoPK使开源模型Gemma 3-27B在某些PK参数上超越了像GPT-4o Mini这样的商业系统。AutoPK能够实现可扩展且高置信度的PK数据提取，使其适用于兽医药理学、药物安全性监测和公共卫生决策等关键应用，并解决异质表格结构和术语问题，展示了在关键PK参数上的普适性。代码和数据：this https URL。 

---
# DexBench: Benchmarking LLMs for Personalized Decision Making in Diabetes Management 

**Title (ZH)**: DexBench: 评估糖尿病管理中个性化决策的LLM性能 

**Authors**: Maria Ana Cardei, Josephine Lamp, Mark Derdzinski, Karan Bhatia  

**Link**: [PDF](https://arxiv.org/pdf/2510.00038)  

**Abstract**: We present DexBench, the first benchmark designed to evaluate large language model (LLM) performance across real-world decision-making tasks faced by individuals managing diabetes in their daily lives. Unlike prior health benchmarks that are either generic, clinician-facing or focused on clinical tasks (e.g., diagnosis, triage), DexBench introduces a comprehensive evaluation framework tailored to the unique challenges of prototyping patient-facing AI solutions in diabetes, glucose management, metabolic health and related domains. Our benchmark encompasses 7 distinct task categories, reflecting the breadth of real-world questions individuals with diabetes ask, including basic glucose interpretation, educational queries, behavioral associations, advanced decision making and long term planning. Towards this end, we compile a rich dataset comprising one month of time-series data encompassing glucose traces and metrics from continuous glucose monitors (CGMs) and behavioral logs (e.g., eating and activity patterns) from 15,000 individuals across three different diabetes populations (type 1, type 2, pre-diabetes/general health and wellness). Using this data, we generate a total of 360,600 personalized, contextual questions across the 7 tasks. We evaluate model performance on these tasks across 5 metrics: accuracy, groundedness, safety, clarity and actionability. Our analysis of 8 recent LLMs reveals substantial variability across tasks and metrics; no single model consistently outperforms others across all dimensions. By establishing this benchmark, we aim to advance the reliability, safety, effectiveness and practical utility of AI solutions in diabetes care. 

**Abstract (ZH)**: DexBench：第一个评估大型语言模型在糖尿病管理实际决策任务中表现的基准测试 

---
# On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations 

**Title (ZH)**: 基于多模态扰动的视觉-语言-动作模型鲁棒性研究 

**Authors**: Jianing Guo, Zhenhong Wu, Chang Tu, Yiyao Ma, Xiangqi Kong, Zhiqian Liu, Jiaming Ji, Shuning Zhang, Yuanpei Chen, Kai Chen, Xianglong Liu, Qi Dou, Yaodong Yang, Huijie Zhao, Weifeng Lv, Simin Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00037)  

**Abstract**: In Vision-Language-Action (VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness with a diffusion-based action head. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust VLAs, and a 10.4% gain under mixed perturbations. Our RobustVLA is particularly effective on real-world FR5 robot with limited demonstrations, showing absolute gains by 65.6% under perturbations of four modalities. 

**Abstract (ZH)**: 多模态视觉语言行动（VLA）模型在现实世界扰动下的鲁棒性对于部署至关重要：RobustVLA——对抗VLA输入和输出扰动的多模态鲁棒模型 

---
# Deep Learning-Based Pneumonia Detection from Chest X-ray Images: A CNN Approach with Performance Analysis and Clinical Implications 

**Title (ZH)**: 基于深度学习的胸部X光图像肺炎检测：一种CNN方法及其性能分析和临床意义 

**Authors**: P K Dutta, Anushri Chowdhury, Anouska Bhattacharyya, Shakya Chakraborty, Sujatra Dey  

**Link**: [PDF](https://arxiv.org/pdf/2510.00035)  

**Abstract**: Deep learning integration into medical imaging systems has transformed disease detection and diagnosis processes with a focus on pneumonia identification. The study introduces an intricate deep learning system using Convolutional Neural Networks for automated pneumonia detection from chest Xray images which boosts diagnostic precision and speed. The proposed CNN architecture integrates sophisticated methods including separable convolutions along with batch normalization and dropout regularization to enhance feature extraction while reducing overfitting. Through the application of data augmentation techniques and adaptive learning rate strategies the model underwent training on an extensive collection of chest Xray images to enhance its generalization capabilities. A convoluted array of evaluation metrics such as accuracy, precision, recall, and F1 score collectively verify the model exceptional performance by recording an accuracy rate of 91. This study tackles critical clinical implementation obstacles such as data privacy protection, model interpretability, and integration with current healthcare systems beyond just model performance. This approach introduces a critical advancement by integrating medical ontologies with semantic technology to improve diagnostic accuracy. The study enhances AI diagnostic reliability by integrating machine learning outputs with structured medical knowledge frameworks to boost interpretability. The findings demonstrate AI powered healthcare tools as a scalable efficient pneumonia detection solution. This study advances AI integration into clinical settings by developing more precise automated diagnostic methods that deliver consistent medical imaging results. 

**Abstract (ZH)**: 深度学习在医学影像系统中的集成改变了肺炎检测和诊断流程：一种基于卷积神经网络的自动化肺炎检测系统及其应用 

---
# Review of Hallucination Understanding in Large Language and Vision Models 

**Title (ZH)**: 大规模语言和视觉模型中的幻觉理解综述 

**Authors**: Zhengyi Ho, Siyuan Liang, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.00034)  

**Abstract**: The widespread adoption of large language and vision models in real-world applications has made urgent the need to address hallucinations -- instances where models produce incorrect or nonsensical outputs. These errors can propagate misinformation during deployment, leading to both financial and operational harm. Although much research has been devoted to mitigating hallucinations, our understanding of it is still incomplete and fragmented. Without a coherent understanding of hallucinations, proposed solutions risk mitigating surface symptoms rather than underlying causes, limiting their effectiveness and generalizability in deployment. To tackle this gap, we first present a unified, multi-level framework for characterizing both image and text hallucinations across diverse applications, aiming to reduce conceptual fragmentation. We then link these hallucinations to specific mechanisms within a model's lifecycle, using a task-modality interleaved approach to promote a more integrated understanding. Our investigations reveal that hallucinations often stem from predictable patterns in data distributions and inherited biases. By deepening our understanding, this survey provides a foundation for developing more robust and effective solutions to hallucinations in real-world generative AI systems. 

**Abstract (ZH)**: 广泛采用的大语言和视觉模型在实际应用中普及，迫切需要解决幻觉问题——模型产生错误或无意义输出的现象。这些错误在部署过程中可能导致传播虚假信息，造成金融和运营方面的损害。尽管已经进行了许多研究来减轻幻觉问题，但对这一问题的理解仍然不完整且碎片化。缺乏对幻觉的全面理解，提出的方法可能会仅缓解表面症状而非根本原因，从而限制其在部署中的有效性和普适性。为解决这一差距，我们首先提出了一种统一的多层次框架，旨在衡量跨多种应用中的图像和文本幻觉，以减少概念上的碎片化。然后，我们将这些幻觉与模型生命周期中的具体机制联系起来，通过任务-模态交错的方法促进更集成的理解。我们的研究表明，幻觉往往源自数据分布中的可预测模式和继承的偏见。通过对这些问题的更深入理解，本次综述为在实际生成式AI系统中开发更稳健和有效的解决方案奠定了基础。 

---
# Hybrid Deep Learning for Hyperspectral Single Image Super-Resolution 

**Title (ZH)**: 混合深度学习在高光谱单张图像超分辨中的应用 

**Authors**: Usman Muhammad, Jorma Laaksonen  

**Link**: [PDF](https://arxiv.org/pdf/2510.00033)  

**Abstract**: Hyperspectral single image super-resolution (SISR) is a challenging task due to the difficulty of restoring fine spatial details while preserving spectral fidelity across a wide range of wavelengths, which limits the performance of conventional deep learning models. To address this challenge, we introduce Spectral-Spatial Unmixing Fusion (SSUF), a novel module that can be seamlessly integrated into standard 2D convolutional architectures to enhance both spatial resolution and spectral integrity. The SSUF combines spectral unmixing with spectral--spatial feature extraction and guides a ResNet-based convolutional neural network for improved reconstruction. In addition, we propose a custom Spatial-Spectral Gradient Loss function that integrates mean squared error with spatial and spectral gradient components, encouraging accurate reconstruction of both spatial and spectral features. Experiments on three public remote sensing hyperspectral datasets demonstrate that the proposed hybrid deep learning model achieves competitive performance while reducing model complexity. 

**Abstract (ZH)**: 基于光谱-空间非线性解混融洽的高光谱单图像超分辨率（SISR） 

---
# WaveMind: Towards a Conversational EEG Foundation Model Aligned to Textual and Visual Modalities 

**Title (ZH)**: WaveMind: 朝着文本和视觉模态对齐的对话EEG基础模型的进步 

**Authors**: Ziyi Zeng, Zhenyang Cai, Yixi Cai, Xidong Wang, Junying Chen, Rongsheng Wang, Yipeng Liu, Siqi Cai, Benyou Wang, Zhiguo Zhang, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.00032)  

**Abstract**: Electroencephalography (EEG) interpretation using multimodal large language models (MLLMs) offers a novel approach for analyzing brain signals. However, the complex nature of brain activity introduces critical challenges: EEG signals simultaneously encode both cognitive processes and intrinsic neural states, creating a mismatch in EEG paired-data modality that hinders effective cross-modal representation learning. Through a pivot investigation, we uncover complementary relationships between these modalities. Leveraging this insight, we propose mapping EEG signals and their corresponding modalities into a unified semantic space to achieve generalized interpretation. To fully enable conversational capabilities, we further introduce WaveMind-Instruct-338k, the first cross-task EEG dataset for instruction tuning. The resulting model demonstrates robust classification accuracy while supporting flexible, open-ended conversations across four downstream tasks, thereby offering valuable insights for both neuroscience research and the development of general-purpose EEG models. 

**Abstract (ZH)**: 利用多模态大型语言模型解读脑电信号（EEG）为分析脑信号提供了新方法。然而，脑活动的复杂性引入了关键挑战：EEG信号同时编码认知过程和内在神经状态，造成配对数据模态之间的不匹配，阻碍了有效的跨模态表示学习。通过一项关键调查，我们揭示了这些模态之间的互补关系。基于此洞察，我们提出将EEG信号及其对应的模态映射到一个统一的语义空间，以实现通用解释。为进一步增强对话能力，我们引入了WaveMind-Instruct-338k，这是首个用于指令微调的跨任务EEG数据集。该模型的分类准确性表现出色，并支持在四项下游任务中进行灵活、开放的对话，从而为神经科学领域的研究和通用EEG模型的发展提供了宝贵见解。 

---
# VibeCodeHPC: An Agent-Based Iterative Prompting Auto-Tuner for HPC Code Generation Using LLMs 

**Title (ZH)**: VibeCodeHPC: 基于代理的迭代提示自动调优器，用于使用LLM进行HPC代码生成 

**Authors**: Shun-ichiro Hayashi, Koki Morita, Daichi Mukunoki, Tetsuya Hoshino, Takahiro Katagiri  

**Link**: [PDF](https://arxiv.org/pdf/2510.00031)  

**Abstract**: We propose VibeCodeHPC, an automatic tuning system for HPC programs based on multi-agent LLMs for code generation. VibeCodeHPC tunes programs through multi-agent role allocation and iterative prompt refinement. We describe the system configuration with four roles: Project Manager (PM), System Engineer (SE), Programmer (PG), and Continuous Delivery (CD). We introduce dynamic agent deployment and activity monitoring functions to facilitate effective multi-agent collaboration. In our case study, we convert and optimize CPU-based matrix-matrix multiplication code written in C to GPU code using CUDA. The multi-agent configuration of VibeCodeHPC achieved higher-quality code generation per unit time compared to a solo-agent configuration. Additionally, the dynamic agent deployment and activity monitoring capabilities facilitated more effective identification of requirement violations and other issues. 

**Abstract (ZH)**: 我们提出VibeCodeHPC，一种基于多智能体LLM的自动调优系统，用于HPC程序的代码生成。VibeCodeHPC通过多智能体角色分配和迭代提示精炼来调优程序。我们描述了该系统配置，包括项目管理器（PM）、系统工程师（SE）、程序员（PG）和持续交付（CD）四个角色。我们介绍了动态智能体部署和活动监控功能，以促进有效的多智能体协作。在我们的案例研究中，我们将基于CPU的用C编写的矩阵-矩阵乘法代码转换并优化为GPU代码。VibeCodeHPC的多智能体配置在单位时间内产生了更高质量的代码生成，与单智能体配置相比更为优越。此外，动态智能体部署和活动监控能力促进了对需求违反和其他问题的有效识别。 

---
# Temporal-Aware Iterative Speech Model for Dementia Detection 

**Title (ZH)**: 带时aware迭代语音模型的痴呆检测 

**Authors**: Chukwuemeka Ugwu, Oluwafemi Oyeleke  

**Link**: [PDF](https://arxiv.org/pdf/2510.00030)  

**Abstract**: Deep learning systems often struggle with processing long sequences, where computational complexity can become a bottleneck. Current methods for automated dementia detection using speech frequently rely on static, time-agnostic features or aggregated linguistic content, lacking the flexibility to model the subtle, progressive deterioration inherent in speech production. These approaches often miss the dynamic temporal patterns that are critical early indicators of cognitive decline. In this paper, we introduce TAI-Speech, a Temporal Aware Iterative framework that dynamically models spontaneous speech for dementia detection. The flexibility of our method is demonstrated through two key innovations: 1) Optical Flow-inspired Iterative Refinement: By treating spectrograms as sequential frames, this component uses a convolutional GRU to capture the fine-grained, frame-to-frame evolution of acoustic features. 2) Cross-Attention Based Prosodic Alignment: This component dynamically aligns spectral features with prosodic patterns, such as pitch and pauses, to create a richer representation of speech production deficits linked to functional decline (IADL). TAI-Speech adaptively models the temporal evolution of each utterance, enhancing the detection of cognitive markers. Experimental results on the DementiaBank dataset show that TAI-Speech achieves a strong AUC of 0.839 and 80.6\% accuracy, outperforming text-based baselines without relying on ASR. Our work provides a more flexible and robust solution for automated cognitive assessment, operating directly on the dynamics of raw audio. 

**Abstract (ZH)**: 深度学习系统在处理长序列时常常遇到困难，其中计算复杂性可能成为瓶颈。目前使用语音进行痴呆症自动检测的方法通常依赖于静态、时间无关的特征或汇总的语义内容，缺乏灵活性来建模言语产生的微妙且渐进的退化。这些方法往往忽略了早期认知衰退的关键动态时间模式。本文 Introduction: Temporal Aware Iterative Framework for Spontaneous Speech in Dementia Detection 

---
# Enhancing Safety in Diabetic Retinopathy Detection: Uncertainty-Aware Deep Learning Models with Rejection Capabilities 

**Title (ZH)**: 提高糖尿病视网膜病变检测的安全性：具有拒识能力的不确定性aware深度学习模型 

**Authors**: Madhushan Ramalingam, Yaish Riaz, Priyanthi Rajamanoharan, Piyumi Dasanayaka  

**Link**: [PDF](https://arxiv.org/pdf/2510.00029)  

**Abstract**: Diabetic retinopathy (DR) is a major cause of visual impairment, and effective treatment options depend heavily on timely and accurate diagnosis. Deep learning models have demonstrated great success identifying DR from retinal images. However, relying only on predictions made by models, without any indication of model confidence, creates uncertainty and poses significant risk in clinical settings. This paper investigates an alternative in uncertainty-aware deep learning models, including a rejection mechanism to reject low-confidence predictions, contextualized by deferred decision-making in clinical practice. The results show there is a trade-off between prediction coverage and coverage reliability. The Variational Bayesian model adopted a more conservative strategy when predicting DR, subsequently rejecting the uncertain predictions. The model is evaluated by means of important performance metrics such as Accuracy on accepted predictions, the proportion of accepted cases (coverage), the rejection-ratio, and Expected Calibration Error (ECE). The findings also demonstrate a clear trade-off between accuracy and caution, establishing that the use of uncertainty estimation and selective rejection improves the model's reliability in safety-critical diagnostic use cases. 

**Abstract (ZH)**: 糖尿病视网膜病变（DR）是导致视觉障碍的主要原因之一，有效的治疗选项依赖于及时准确的诊断。深度学习模型在从视网膜图像中识别DR方面取得了巨大成功。然而，仅依靠模型的预测而不提供模型置信度的指示，在临床环境中会带来不确定性并造成重大风险。本文研究了一种新的不确定性感知深度学习模型，包括拒绝机制以拒绝低置信度的预测，并针对临床实践中延迟决策的背景进行考量。结果表明，在预测覆盖率与可靠性之间存在权衡。采用变分贝叶斯模型在预测DR时采取了更为保守的策略，随后拒绝了不确定的预测。该模型通过准确率、接受案例比例（覆盖率）、拒绝率以及预期校准误差（ECE）等关键性能指标进行了评估。研究结果还表明，准确性和谨慎性之间存在明显的权衡，证实了在关键安全诊断用例中使用不确定性估计和选择性拒绝可以提高模型的可靠性。 

---
# Rethinking RoPE Scaling in Quantized LLM: Theory, Outlier, and Channel-Band Analysis with Weight Rescaling 

**Title (ZH)**: 重新思考量化LLM中的RoPE缩放：理论、异常值和通道带宽分析与权重重新缩放 

**Authors**: Ye Qiao, Haocheng Xu, Xiaofan Zhang, Sitao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.00028)  

**Abstract**: Extending the context window support of large language models (LLMs) is crucial for tasks with long-distance dependencies. RoPE-based interpolation and extrapolation methods, such as linear scaling and frequency-aware schemes, enable longer input length support without retraining, while post-training quantization (PTQ) makes deployment practical. However, we show that combining RoPE position interpolation (PI) with PTQ degrades accuracy due to coupled effects including long-context aliasing, dynamic-range dilation, anisotropy from axis-aligned quantizers vs. rotated RoPE pairs, and outlier shifting that produces position-dependent logit noise. We provide, to the best of our knowledge, the first systematic analysis of the PI+PTQ approach and introduce two practical diagnostics: interpolation pressure (per-band sensitivity to phase scaling) and tail-inflation ratios (outlier shift from short to long contexts). Following the analysis results, we propose Q-ROAR (Quantization, RoPE-interpolation, and Outlier Aware Rescaling), a weight-only, interpolation-aware stabilization of PI for quantized LLMs. Q-ROAR groups RoPE dimensions into a small number of frequency bands and performs a lightweight search over per-band scales for Key and Query weights (with an optional symmetric variant to preserve logit scale). The search is guided by our diagnostics and uses a tiny long-context development dataset, requiring no fine-tuning to the model, no architecture or kernel changes, and no additional deployment overhead. Empirically, Q-ROAR reduces the model's perplexity on long-context workloads by more than 14%, while preserving short-context performance, inference throughput, and compatibility with existing LLM system stacks. 

**Abstract (ZH)**: 扩展大型语言模型的上下文窗口支持对于处理长距离依赖任务至关重要。基于RoPE的内插和外推方法，如线性缩放和频率感知方案，可以在无需重新训练的情况下支持更长的输入长度，而后训练量化（PTQ）使其部署更加实际。然而，我们表明，将RoPE位置内插（PI）与PTQ结合使用会因长上下文混叠、动态范围扩张、轴对齐量化的各向异性与旋转RoPE对以及异常值位移产生位置依赖的logit噪声等因素的耦合效应而降低准确性。据我们所知，首次提供了PI+PTQ方法的系统分析，并引入了两种实用诊断方法：内插压力（每频带对相位缩放的灵敏度）和尾部膨胀比（从短上下文到长上下文的异常值位移）。根据分析结果，我们提出了Q-ROAR（量化、RoPE内插和异常值感知重新缩放），这是一种仅权重、内插意识的量化LLM稳定方法。Q-ROAR将RoPE维度分为少量频率带，并在Key和Query权重（可选对称变体以保持logit尺度）的每带尺度上进行轻量级搜索。搜索由我们的诊断引导，并使用少量长上下文开发数据集，无需对模型进行微调，无需更改架构或内核，且无需额外的部署开销。实验结果显示，Q-ROAR在长上下文工作负载中将模型的困惑度降低了超过14%，同时保留了短上下文性能、推理吞吐量以及对现有LLM系统堆栈的兼容性。 

---
# Learning Inter-Atomic Potentials without Explicit Equivariance 

**Title (ZH)**: 学习原子间势能而不显式保证等变性 

**Authors**: Ahmed A. Elhag, Arun Raja, Alex Morehead, Samuel M. Blau, Garrett M. Morris, Michael M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2510.00027)  

**Abstract**: Accurate and scalable machine-learned inter-atomic potentials (MLIPs) are essential for molecular simulations ranging from drug discovery to new material design. Current state-of-the-art models enforce roto-translational symmetries through equivariant neural network architectures, a hard-wired inductive bias that can often lead to reduced flexibility, computational efficiency, and scalability. In this work, we introduce TransIP: Transformer-based Inter-Atomic Potentials, a novel training paradigm for interatomic potentials achieving symmetry compliance without explicit architectural constraints. Our approach guides a generic non-equivariant Transformer-based model to learn SO(3)-equivariance by optimizing its representations in the embedding space. Trained on the recent Open Molecules (OMol25) collection, a large and diverse molecular dataset built specifically for MLIPs and covering different types of molecules (including small organics, biomolecular fragments, and electrolyte-like species), TransIP attains comparable performance in machine-learning force fields versus state-of-the-art equivariant baselines. Further, compared to a data augmentation baseline, TransIP achieves 40% to 60% improvement in performance across varying OMol25 dataset sizes. More broadly, our work shows that learned equivariance can be a powerful and efficient alternative to equivariant or augmentation-based MLIP models. 

**Abstract (ZH)**: 基于变换器的原子间势（TransIP）：无需显式架构约束的原子间势训练新范式 

---
# EpidemIQs: Prompt-to-Paper LLM Agents for Epidemic Modeling and Analysis 

**Title (ZH)**: EpidemIQs: 从提示到论文的LLM代理模型与分析适用于流行病学 

**Authors**: Mohammad Hossein Samaei, Faryad Darabi Sahneh, Lee W. Cohnstaedt, Caterina Scoglio  

**Link**: [PDF](https://arxiv.org/pdf/2510.00024)  

**Abstract**: Large Language Models (LLMs) offer new opportunities to automate complex interdisciplinary research domains. Epidemic modeling, characterized by its complexity and reliance on network science, dynamical systems, epidemiology, and stochastic simulations, represents a prime candidate for leveraging LLM-driven automation. We introduce \textbf{EpidemIQs}, a novel multi-agent LLM framework that integrates user inputs and autonomously conducts literature review, analytical derivation, network modeling, mechanistic modeling, stochastic simulations, data visualization and analysis, and finally documentation of findings in a structured manuscript. We introduced two types of agents: a scientist agent for planning, coordination, reflection, and generation of final results, and a task-expert agent to focus exclusively on one specific duty serving as a tool to the scientist agent. The framework consistently generated complete reports in scientific article format. Specifically, using GPT 4.1 and GPT 4.1 mini as backbone LLMs for scientist and task-expert agents, respectively, the autonomous process completed with average total token usage 870K at a cost of about \$1.57 per study, achieving a 100\% completion success rate through our experiments. We evaluate EpidemIQs across different epidemic scenarios, measuring computational cost, completion success rate, and AI and human expert reviews of generated reports. We compare EpidemIQs to the single-agent LLM, which has the same system prompts and tools, iteratively planning, invoking tools, and revising outputs until task completion. The comparison shows consistently higher performance of the proposed framework across five different scenarios. EpidemIQs represents a step forward in accelerating scientific research by significantly reducing costs and turnaround time of discovery processes, and enhancing accessibility to advanced modeling tools. 

**Abstract (ZH)**: 大型语言模型（LLMs）为自动化复杂跨学科研究领域提供了新的机会。传染病模型因其实现复杂性及对网络科学、动力系统、流行病学和随机模拟的依赖而成为利用LLM驱动自动化的一个理想候选领域。我们提出了\textbf{EpidemIQs}，这是一种新颖的多 Agent LLM框架，整合用户输入并自主开展文献综述、分析推导、网络建模、机制建模、随机模拟、数据可视化与分析，并最终以结构化的论文形式记录研究发现。我们引入了两种类型的代理：科学家代理负责规划、协调、反思和生成最终结果，以及专注于特定任务并为科学家代理提供支持的任务专家代理。该框架一致生成了符合科学文章格式的完整报告。具体而言，使用GPT 4.1和GPT 4.1 mini分别作为科学家代理和任务专家代理的基础LLM，在平均总计Token使用量为870K的情况下，每项研究成本约为1.57美元，通过我们的实验实现了100%的完成成功率。我们通过不同传染病场景评估了EpidemIQs，测量了计算成本、完成成功率以及AI和人类专家对生成报告的评审。我们将EpidemIQs与具有相同系统提示和工具的单Agent LLM进行了对比，后者迭代规划、调用工具并修订输出直至任务完成。结果表明，在五个不同场景中，提出框架的一致性能更高。EpidemIQs代表了通过显著降低发现过程的成本和周转时间以及提升高级建模工具的可访问性来加速科学研究的一个重要进展。 

---
# IA aplicada al análisis del conflicto Irán-Israel: Mapeo de discursos en YouTube 

**Title (ZH)**: IA应用于伊朗-以色列冲突分析：YouTube上话语 mapping 

**Authors**: Alvaro Vallejo Ramírez  

**Link**: [PDF](https://arxiv.org/pdf/2510.00021)  

**Abstract**: Purpose. This study analyzes the digital representation of the Iran-Israel conflict that occurred in June 2025, based on 120,000 comments posted on YouTube. It sought to identify discursive positions regarding the actors involved and to examine how media and algorithmic biases shape digital conversations. Methodology. A mixed-methods design with triangulation was adopted. In the quantitative phase, natural language processing techniques and machine learning models (BERT and XLM-RoBERTa) were used to classify comments into ten categories. In the qualitative phase, a critical analysis of media context and ideological narratives was conducted, complemented by manual annotation and supervised training. This strategy enabled the integration of statistical robustness with contextual understanding. Results and conclusions. The findings reveal a clear overrepresentation of pro-Palestinian and anti-United States/Israel discourses, while pro-United States and anti-Palestinian positions were marginal. Iran, usually rendered invisible in global media, emerged as a central actor in the digital conversation during the conflict, suggesting a narrative shift away from previous hegemonic frameworks. Likewise, the results confirm the influence of algorithmic biases in amplifying certain discourses while limiting others. Original contributions. This work combines computational analysis and philosophical critique for the study of digital controversies, providing a methodological framework replicable in geopolitical contexts. It is one of the first Spanish-language studies to map, through artificial intelligence and critical analysis, discourses on an international conflict on YouTube, highlighting asymmetries and narrative disputes that are often overlooked. 

**Abstract (ZH)**: 目的. 本文基于2025年6月发生在伊朗与以色列之间的冲突在YouTube上发布的12万条评论，分析了该冲突的数字化表现，并旨在识别涉事各方的话语立场，同时考察媒体和算法偏见如何塑造数字对话。方法. 采用混合方法并进行 triangulation。在定量阶段，使用自然语言处理技术及机器学习模型（BERT和XLM-RoBERTa）对评论进行分类，分为十个类别。在定性阶段，进行了媒体背景和意识形态叙事的批判分析，辅以手工标注和监督训练。该策略实现了统计稳健性与情境理解的融合。结果与结论. 研究发现显示，亲巴勒斯坦和反美国/以色列话语明显占上风，而亲美国和反巴勒斯坦立场则处于边缘地位。通常在全球媒体中被忽视的伊朗，在冲突期间的数字对话中成为了中心角色，表明叙事转向了偏离以往主导框架的领域。同样，研究结果证实了算法偏见在放大某些话语、抑制其他话语方面的影响力。原创贡献. 该研究结合了计算分析和哲学批判，为数字争议的研究提供了可复制的方法论框架，尤其是在地缘政治背景下。它是首部利用人工智能和批判性分析在YouTube上绘制国际冲突话语图谱的西班牙语研究之一，突显了经常被忽视的不对称性与叙事争端。 

---
# Methodological Framework for Quantifying Semantic Test Coverage in RAG Systems 

**Title (ZH)**: 用于量化RAG系统语义测试覆盖率的方法论框架 

**Authors**: Noah Broestl, Adel Nasser Abdalla, Rajprakash Bale, Hersh Gupta, Max Struever  

**Link**: [PDF](https://arxiv.org/pdf/2510.00001)  

**Abstract**: Reliably determining the performance of Retrieval-Augmented Generation (RAG) systems depends on comprehensive test questions. While a proliferation of evaluation frameworks for LLM-powered applications exists, current practices lack a systematic method to ensure these test sets adequately cover the underlying knowledge base, leaving developers with significant blind spots. To address this, we present a novel, applied methodology to quantify the semantic coverage of RAG test questions against their underlying documents. Our approach leverages existing technologies, including vector embeddings and clustering algorithms, to create a practical framework for validating test comprehensiveness. Our methodology embeds document chunks and test questions into a unified vector space, enabling the calculation of multiple coverage metrics: basic proximity, content-weighted coverage, and multi-topic question coverage. Furthermore, we incorporate outlier detection to filter irrelevant questions, allowing for the refinement of test sets. Experimental evidence from two distinct use cases demonstrates that our framework effectively quantifies test coverage, identifies specific content areas with inadequate representation, and provides concrete recommendations for generating new, high-value test questions. This work provides RAG developers with essential tools to build more robust test suites, thereby improving system reliability and extending to applications such as identifying misaligned documents. 

**Abstract (ZH)**: 可靠地确定 Retrieval-Augmented Generation (RAG) 系统的性能取决于全面的测试问题。尽管存在多种针对 LLM 动力应用的评估框架，但当前的实践方法缺乏确保测试集充分覆盖底层知识库的系统方法，给开发者留下了显著的知识盲区。为解决这一问题，我们提出了一种新颖的应用方法，用于量化 RAG 测试问题对其底层文档的语义覆盖程度。我们的方法利用现有的技术，包括向量嵌入和聚类算法，以创建一个实用的验证测试全面性的框架。我们的方法将文档片段和测试问题嵌入到统一的向量空间中，以便计算多种覆盖度量：基本接近度、内容加权覆盖度以及多主题问题覆盖度。此外，我们还引入了离群值检测来过滤无关问题，从而 refinethe 测试集。来自两个不同应用场景的实验证据显示，我们的框架有效地量化了测试覆盖度，识别了内容表示不足的具体领域，并提供了生成新的高质量测试问题的具体建议。这项工作为 RAG 开发者提供了重要工具，以构建更 robust 的测试集，从而提高系统可靠性，并扩展到诸如识别不一致文档等应用。 

---
# Autonomous Multi-Robot Infrastructure for AI-Enabled Healthcare Delivery and Diagnostics 

**Title (ZH)**: 自主多机器人基础设施以实现AI赋能的健康 care 交付与诊断 

**Authors**: Nakhul Kalaivanan, Senthil Arumugam Muthukumaraswamy, Girish Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2509.26106)  

**Abstract**: This research presents a multi-robot system for inpatient care, designed using swarm intelligence principles and incorporating wearable health sensors, RF-based communication, and AI-driven decision support. Within a simulated hospital environment, the system adopts a leader-follower swarm configuration to perform patient monitoring, medicine delivery, and emergency assistance. Due to ethical constraints, live patient trials were not conducted; instead, validation was carried out through controlled self-testing with wearable sensors. The Leader Robot acquires key physiological parameters, including temperature, SpO2, heart rate, and fall detection, and coordinates other robots when required. The Assistant Robot patrols corridors for medicine delivery, while a robotic arm provides direct drug administration. The swarm-inspired leader-follower strategy enhanced communication reliability and ensured continuous monitoring, including automated email alerts to healthcare staff. The system hardware was implemented using Arduino, Raspberry Pi, NRF24L01 RF modules, and a HuskyLens AI camera. Experimental evaluation showed an overall sensor accuracy above 94%, a 92% task-level success rate, and a 96% communication reliability rate, demonstrating system robustness. Furthermore, the AI-enabled decision support was able to provide early warnings of abnormal health conditions, highlighting the potential of the system as a cost-effective solution for hospital automation and patient safety. 

**Abstract (ZH)**: 基于 swarm 智能原则的设计的多机器人系统：应用于病房护理中的穿戴式健康传感器、RF通信和AI驱动决策支持的领导-跟随者配置 

---
# MARS: Audio Generation via Multi-Channel Autoregression on Spectrograms 

**Title (ZH)**: MARS：基于多通道自回归谱图的音频生成 

**Authors**: Eleonora Ristori, Luca Bindini, Paolo Frasconi  

**Link**: [PDF](https://arxiv.org/pdf/2509.26007)  

**Abstract**: Research on audio generation has progressively shifted from waveform-based approaches to spectrogram-based methods, which more naturally capture harmonic and temporal structures. At the same time, advances in image synthesis have shown that autoregression across scales, rather than tokens, improves coherence and detail. Building on these ideas, we introduce MARS (Multi-channel AutoRegression on Spectrograms), a framework that treats spectrograms as multi-channel images and employs channel multiplexing (CMX), a reshaping technique that lowers height and width without discarding information. A shared tokenizer provides consistent discrete representations across scales, enabling a transformer-based autoregressor to refine spectrograms from coarse to fine resolutions efficiently. Experiments on a large-scale dataset demonstrate that MARS performs comparably or better than state-of-the-art baselines across multiple evaluation metrics, establishing an efficient and scalable paradigm for high-fidelity audio generation. 

**Abstract (ZH)**: 基于光谱的多通道自回归音频生成研究 

---
# PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models 

**Title (ZH)**: PCPO: 比例信用政策优化以对齐图像生成模型 

**Authors**: Jeongjae Lee, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.25774)  

**Abstract**: While reinforcement learning has advanced the alignment of text-to-image (T2I) models, state-of-the-art policy gradient methods are still hampered by training instability and high variance, hindering convergence speed and compromising image quality. Our analysis identifies a key cause of this instability: disproportionate credit assignment, in which the mathematical structure of the generative sampler produces volatile and non-proportional feedback across timesteps. To address this, we introduce Proportionate Credit Policy Optimization (PCPO), a framework that enforces proportional credit assignment through a stable objective reformulation and a principled reweighting of timesteps. This correction stabilizes the training process, leading to significantly accelerated convergence and superior image quality. The improvement in quality is a direct result of mitigating model collapse, a common failure mode in recursive training. PCPO substantially outperforms existing policy gradient baselines on all fronts, including the state-of-the-art DanceGRPO. 

**Abstract (ZH)**: 尽管强化学习提高了文本到图像（T2I）模型的一致性，但最先进的策略梯度方法仍然受到训练不稳定性和高方差的困扰，影响了收敛速度并损害了图像质量。我们的分析指出这一不稳定性的一个关键原因：生成采样器的数学结构导致了时序间不稳定的、不成比例的反馈。为解决这一问题，我们提出了比例信用策略优化（PCPO）框架，该框架通过稳定的目标重构和合理的时序重新加权来确保比例信用分配，从而稳定训练过程，显著加速收敛并提高图像质量。质量的提升直接来源于缓解了递归训练中常见的模型崩溃问题。在所有方面，包括最先进的DanceGRPO，PCPO都显著优于现有策略梯度基线。 

---
# ReLumix: Extending Image Relighting to Video via Video Diffusion Models 

**Title (ZH)**: ReLumix: 通过视频扩散模型将图像光照更改扩展到视频 

**Authors**: Lezhong Wang, Shutong Jin, Ruiqi Cui, Anders Bjorholm Dahl, Jeppe Revall Frisvad, Siavash Bigdeli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23769)  

**Abstract**: Controlling illumination during video post-production is a crucial yet elusive goal in computational photography. Existing methods often lack flexibility, restricting users to certain relighting models. This paper introduces ReLumix, a novel framework that decouples the relighting algorithm from temporal synthesis, thereby enabling any image relighting technique to be seamlessly applied to video. Our approach reformulates video relighting into a simple yet effective two-stage process: (1) an artist relights a single reference frame using any preferred image-based technique (e.g., Diffusion Models, physics-based renderers); and (2) a fine-tuned stable video diffusion (SVD) model seamlessly propagates this target illumination throughout the sequence. To ensure temporal coherence and prevent artifacts, we introduce a gated cross-attention mechanism for smooth feature blending and a temporal bootstrapping strategy that harnesses SVD's powerful motion priors. Although trained on synthetic data, ReLumix shows competitive generalization to real-world videos. The method demonstrates significant improvements in visual fidelity, offering a scalable and versatile solution for dynamic lighting control. 

**Abstract (ZH)**: 在视频后处理中控制照明是计算摄影中的一个关键但难以实现的目标。现有的方法往往缺乏灵活性，限制用户只能使用特定的光照重建模型。本文介绍了ReLumix，这是一个新颖的框架，它将光照重建算法与时空合成解耦，从而允许任何图像光照技术无缝应用于视频。我们的方法将视频光照重建重新定义为一个简单而有效的两阶段过程：（1）艺术家使用任何首选的基于图像的技术（例如，扩散模型、基于物理的渲染器）对单个参考帧进行光照重建；（2）一个微调过的稳定视频扩散（SVD）模型无缝地将这种目标照明传播到整个序列中。为了确保时空一致性并防止伪像，我们引入了一种门控交叉注意力机制以实现平滑特征混合，并利用SVD强大的运动先验引入了一种时空-bootstrap策略。尽管是在合成数据上训练的，ReLumix显示了在真实世界视频上的竞争性泛化能力。该方法在视觉保真度方面表现出显著的改进，提供了一种可扩展且多功能的动态照明控制解决方案。 

---
# EVO-LRP: Evolutionary Optimization of LRP for Interpretable Model Explanations 

**Title (ZH)**: EVO-LRP：可解释模型解释中的LRP进化优化 

**Authors**: Emerald Zhang, Julian Weaver, Samantha R Santacruz, Edward Castillo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23585)  

**Abstract**: Explainable AI (XAI) methods help identify which image regions influence a model's prediction, but often face a trade-off between detail and interpretability. Layer-wise Relevance Propagation (LRP) offers a model-aware alternative. However, LRP implementations commonly rely on heuristic rule sets that are not optimized for clarity or alignment with model behavior. We introduce EVO-LRP, a method that applies Covariance Matrix Adaptation Evolution Strategy (CMA-ES) to tune LRP hyperparameters based on quantitative interpretability metrics, such as faithfulness or sparseness. EVO-LRP outperforms traditional XAI approaches in both interpretability metric performance and visual coherence, with strong sensitivity to class-specific features. These findings demonstrate that attribution quality can be systematically improved through principled, task-specific optimization. 

**Abstract (ZH)**: 可解释人工智能（XAI）方法有助于识别哪幅图像区域影响模型预测，但通常面临详细性和可解释性之间的权衡。层wise相关性传播（LRP）提供了一种基于模型的替代方案。然而，LRP实现通常依赖于非优化的启发式规则集，不利于清晰度或与模型行为的一致性。我们引入了EVO-LRP方法，该方法利用Covariance Matrix Adaptation Evolution Strategy（CMA-ES）根据忠实度或稀疏性等定量可解释性指标调整LRP超参数。EVO-LRP在可解释性指标性能和视觉一致性方面均优于传统XAI方法，并且对类别特定特征表现出强烈的敏感性。这些发现证明，通过原理明确且针对任务的优化，可以系统地提高归因质量。 

---
