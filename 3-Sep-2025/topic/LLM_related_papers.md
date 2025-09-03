# The Landscape of Agentic Reinforcement Learning for LLMs: A Survey 

**Title (ZH)**: 代理强化学习在大语言模型中的应用场景：一个综述 

**Authors**: Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, Yifan Zhou, Yang Chen, Chen Zhang, Yutao Fan, Zihu Wang, Songtao Huang, Yue Liao, Hongru Wang, Mengyue Yang, Heng Ji, Michael Littman, Jun Wang, Shuicheng Yan, Philip Torr, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2509.02547)  

**Abstract**: The emergence of agentic reinforcement learning (Agentic RL) marks a paradigm shift from conventional reinforcement learning applied to large language models (LLM RL), reframing LLMs from passive sequence generators into autonomous, decision-making agents embedded in complex, dynamic worlds. This survey formalizes this conceptual shift by contrasting the degenerate single-step Markov Decision Processes (MDPs) of LLM-RL with the temporally extended, partially observable Markov decision processes (POMDPs) that define Agentic RL. Building on this foundation, we propose a comprehensive twofold taxonomy: one organized around core agentic capabilities, including planning, tool use, memory, reasoning, self-improvement, and perception, and the other around their applications across diverse task domains. Central to our thesis is that reinforcement learning serves as the critical mechanism for transforming these capabilities from static, heuristic modules into adaptive, robust agentic behavior. To support and accelerate future research, we consolidate the landscape of open-source environments, benchmarks, and frameworks into a practical compendium. By synthesizing over five hundred recent works, this survey charts the contours of this rapidly evolving field and highlights the opportunities and challenges that will shape the development of scalable, general-purpose AI agents. 

**Abstract (ZH)**: 生成代理增强学习（Agentic RL）的兴起标志着从面向大型语言模型的传统增强学习（LLM RL）的一种范式转变，将大型语言模型重新定义为嵌入在复杂动态世界中的自主决策代理，而非被动的序列生成器。本文通过对比大型语言模型增强学习中的退化单步马尔可夫决策过程（MDP）与生成代理增强学习中定义的时间延伸和部分可观测马尔可夫决策过程（POMDP），形式化了这一概念转变。在此基础上，我们提出了一个全面的二元分类体系：一个围绕核心代理能力，包括规划、工具使用、记忆、推理、自我改进和感知组织，另一个围绕这些能力在多种任务领域中的应用。我们主要论点是，增强学习是将这些能力从静态启发式模块转化成为适应性强、稳健的代理行为的关键机理。为了支持并加速未来的研究，我们将开源环境、基准测试和框架的景观整合成一个实用的参考手册。通过综合五百多篇近期的研究工作，本文描绘了这一快速发展的领域的轮廓，并指出了塑造可扩展且通用AI代理发展的机遇与挑战。 

---
# GridMind: LLMs-Powered Agents for Power System Analysis and Operations 

**Title (ZH)**: GridMind: 由大规模语言模型驱动的电力系统分析与运营代理 

**Authors**: Hongwei Jin, Kibaek Kim, Jonghwan Kwon  

**Link**: [PDF](https://arxiv.org/pdf/2509.02494)  

**Abstract**: The complexity of traditional power system analysis workflows presents significant barriers to efficient decision-making in modern electric grids. This paper presents GridMind, a multi-agent AI system that integrates Large Language Models (LLMs) with deterministic engineering solvers to enable conversational scientific computing for power system analysis. The system employs specialized agents coordinating AC Optimal Power Flow and N-1 contingency analysis through natural language interfaces while maintaining numerical precision via function calls. GridMind addresses workflow integration, knowledge accessibility, context preservation, and expert decision-support augmentation. Experimental evaluation on IEEE test cases demonstrates that the proposed agentic framework consistently delivers correct solutions across all tested language models, with smaller LLMs achieving comparable analytical accuracy with reduced computational latency. This work establishes agentic AI as a viable paradigm for scientific computing, demonstrating how conversational interfaces can enhance accessibility while preserving numerical rigor essential for critical engineering applications. 

**Abstract (ZH)**: 传统电力系统分析工作流的复杂性对现代电网的有效决策形成了显著障碍。本文介绍了GridMind，一种将大型语言模型与确定性工程求解器集成的多代理AI系统，以实现电力系统分析中的对话型科学计算。该系统通过自然语言界面专门协调AC最优潮流和N-1应急分析，并通过函数调用保持数值精度。GridMind解决了工作流集成、知识获取、上下文保留以及专家决策支持增强等问题。在IEEE测试案例上的实验评估表明，所提出的代理框架能够在所有测试的语言模型中一致地提供正确解决方案，且较小的LLM在减少计算延迟的同时实现了相当的分析准确性。本文建立了代理AI作为一种可行的科学计算范式，并展示了对话界面如何在保持对关键工程应用至关重要的数值严谨性的同时增强可访问性。 

---
# Towards Agents That Know When They Don't Know: Uncertainty as a Control Signal for Structured Reasoning 

**Title (ZH)**: 向着那些知道自己不知道的代理：不确定性作为结构化推理的控制信号 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Lawrence Phillips, Gianluca Mazzoni, Lea Mørch Harder, Philip Torr, Jesper Ferkinghoff-Borg, Kaspar Martens, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2509.02401)  

**Abstract**: Large language model (LLM) agents are increasingly deployed in structured biomedical data environments, yet they often produce fluent but overconfident outputs when reasoning over complex multi-table data. We introduce an uncertainty-aware agent for query-conditioned multi-table summarization that leverages two complementary signals: (i) retrieval uncertainty--entropy over multiple table-selection rollouts--and (ii) summary uncertainty--combining self-consistency and perplexity. Summary uncertainty is incorporated into reinforcement learning (RL) with Group Relative Policy Optimization (GRPO), while both retrieval and summary uncertainty guide inference-time filtering and support the construction of higher-quality synthetic datasets.
On multi-omics benchmarks, our approach improves factuality and calibration, nearly tripling correct and useful claims per summary (3.0\(\rightarrow\)8.4 internal; 3.6\(\rightarrow\)9.9 cancer multi-omics) and substantially improving downstream survival prediction (C-index 0.32\(\rightarrow\)0.63). These results demonstrate that uncertainty can serve as a control signal--enabling agents to abstain, communicate confidence, and become more reliable tools for complex structured-data environments. 

**Abstract (ZH)**: 大语言模型代理在结构化生物医学数据环境中的应用越来越多，但在处理复杂多表数据时往往会产生产生流畅但过度自信的输出。我们提出了一种 Awareness 不确定性代理用于查询条件下的多表总结，利用两种互补的信号：（i）检索不确定性——多个表选择展开的熵；（ii）总结不确定性——结合自一致性与困惑度。总结不确定性被纳入强化学习（RL）中的组相对策略优化（GRPO），而检索和总结不确定性在推理时指导过滤并支持构建更高质量的合成数据集。在多组学基准测试中，该方法提高了事实性和校准性，几乎将每总结正确的有用声明翻了三倍（从内部的3.0增加到8.4；从癌症多组学到9.9），并显著提高了下游生存预测（C指数从0.32增加到0.63）。这些结果表明，不确定性可以作为一种控制信号——使代理能够犹豫、传达自信，并成为更可靠的复杂结构化数据环境工具。 

---
# Re-evaluating LLM-based Heuristic Search: A Case Study on the 3D Packing Problem 

**Title (ZH)**: 基于LLM的启发式搜索再评估：一个关于3D打包问题的案例研究 

**Authors**: Guorui Quan, Mingfei Sun, Manuel López-Ibáñez  

**Link**: [PDF](https://arxiv.org/pdf/2509.02297)  

**Abstract**: The art of heuristic design has traditionally been a human pursuit. While Large Language Models (LLMs) can generate code for search heuristics, their application has largely been confined to adjusting simple functions within human-crafted frameworks, leaving their capacity for broader innovation an open question. To investigate this, we tasked an LLM with building a complete solver for the constrained 3D Packing Problem. Direct code generation quickly proved fragile, prompting us to introduce two supports: constraint scaffolding--prewritten constraint-checking code--and iterative self-correction--additional refinement cycles to repair bugs and produce a viable initial population. Notably, even within a vast search space in a greedy process, the LLM concentrated its efforts almost exclusively on refining the scoring function. This suggests that the emphasis on scoring functions in prior work may reflect not a principled strategy, but rather a natural limitation of LLM capabilities. The resulting heuristic was comparable to a human-designed greedy algorithm, and when its scoring function was integrated into a human-crafted metaheuristic, its performance rivaled established solvers, though its effectiveness waned as constraints tightened. Our findings highlight two major barriers to automated heuristic design with current LLMs: the engineering required to mitigate their fragility in complex reasoning tasks, and the influence of pretrained biases, which can prematurely narrow the search for novel solutions. 

**Abstract (ZH)**: 大型语言模型在启发式设计中的应用：从人类追求到自动化设计的探索 

---
# LLMs for LLMs: A Structured Prompting Methodology for Long Legal Documents 

**Title (ZH)**: LLMs 为 LLMs：一种针对长法律文件的结构化提示方法论 

**Authors**: Strahinja Klem, Noura Al Moubayed  

**Link**: [PDF](https://arxiv.org/pdf/2509.02241)  

**Abstract**: The rise of Large Language Models (LLMs) has had a profoundly transformative effect on a number of fields and domains. However, their uptake in Law has proven more challenging due to the important issues of reliability and transparency. In this study, we present a structured prompting methodology as a viable alternative to the often expensive fine-tuning, with the capability of tacking long legal documents from the CUAD dataset on the task of information retrieval. Each document is first split into chunks via a system of chunking and augmentation, addressing the long document problem. Then, alongside an engineered prompt, the input is fed into QWEN-2 to produce a set of answers for each question. Finally, we tackle the resulting candidate selection problem with the introduction of the Distribution-based Localisation and Inverse Cardinality Weighting heuristics. This approach leverages a general purpose model to promote long term scalability, prompt engineering to increase reliability and the two heuristic strategies to reduce the impact of the black box effect. Whilst our model performs up to 9\% better than the previously presented method, reaching state-of-the-art performance, it also highlights the limiting factor of current automatic evaluation metrics for question answering, serving as a call to action for future research. However, the chief aim of this work is to underscore the potential of structured prompt engineering as a useful, yet under-explored, tool in ensuring accountability and responsibility of AI in the legal domain, and beyond. 

**Abstract (ZH)**: 大型语言模型的兴起对多个领域产生了深远的变革影响。然而，在法律领域的应用由于可靠性和透明度等问题遇到了更多挑战。在本研究中，我们提出了一种结构化提示方法作为经济高效的替代方案，该方法能够处理CUAD数据集中长法律文档的信息检索任务。首先，通过分块和增强的方法将每个文档拆分为片段，解决长文档问题。然后，结合精心设计的提示，将输入传递给QWEN-2以生成每个问题的答案集合。最后，通过引入基于分布的位置化和逆基数加权启发式策略来解决候选答案选择问题。这种方法利用通用模型促进长期内的可扩展性，使用提示工程提高可靠性，并采用两种启发式策略减少黑盒效应的影响。虽然我们的模型在问答任务中比之前的方法高出9%，达到了最先进的性能，但也指出了当前自动评价指标的限制，为未来的研究提出了挑战。然而，本文的主要目标是强调结构化提示工程作为一种有助于确保人工智能在法律领域及更广泛领域中的责任和问责制的有用但尚未充分探索的工具的重要性。 

---
# mFARM: Towards Multi-Faceted Fairness Assessment based on HARMs in Clinical Decision Support 

**Title (ZH)**: mFARM: 基于HARMs的多维度公平性评估在临床决策支持中的应用 

**Authors**: Shreyash Adappanavar, Krithi Shailya, Gokul S Krishnan, Sriraam Natarajan, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.02007)  

**Abstract**: The deployment of Large Language Models (LLMs) in high-stakes medical settings poses a critical AI alignment challenge, as models can inherit and amplify societal biases, leading to significant disparities. Existing fairness evaluation methods fall short in these contexts as they typically use simplistic metrics that overlook the multi-dimensional nature of medical harms. This also promotes models that are fair only because they are clinically inert, defaulting to safe but potentially inaccurate outputs. To address this gap, our contributions are mainly two-fold: first, we construct two large-scale, controlled benchmarks (ED-Triage and Opioid Analgesic Recommendation) from MIMIC-IV, comprising over 50,000 prompts with twelve race x gender variants and three context tiers. Second, we propose a multi-metric framework - Multi-faceted Fairness Assessment based on hARMs ($mFARM$) to audit fairness for three distinct dimensions of disparity (Allocational, Stability, and Latent) and aggregate them into an $mFARM$ score. We also present an aggregated Fairness-Accuracy Balance (FAB) score to benchmark and observe trade-offs between fairness and prediction accuracy. We empirically evaluate four open-source LLMs (Mistral-7B, BioMistral-7B, Qwen-2.5-7B, Bio-LLaMA3-8B) and their finetuned versions under quantization and context variations. Our findings showcase that the proposed $mFARM$ metrics capture subtle biases more effectively under various settings. We find that most models maintain robust performance in terms of $mFARM$ score across varying levels of quantization but deteriorate significantly when the context is reduced. Our benchmarks and evaluation code are publicly released to enhance research in aligned AI for healthcare. 

**Abstract (ZH)**: Large Language Models在高风险医疗环境中的部署提出了关键的AI对齐挑战，因为模型可能会继承和放大社会偏见，导致显著的不平等。现有的公平性评估方法在这些情境下存在不足，因为它们通常使用简化的指标，忽视了医疗伤害的多维性。这也会促进只在临床上无害但可能不准确的模型，从而默认生成安全但可能不准确的输出。为了弥补这一缺口，我们的贡献主要包含两方面：首先，我们从MIMIC-IV构建了两个大规模控制基准（ED-Triage和Opioid Analgesic Recommendation），包含超过50,000个提示，涵盖了十二个种族与性别变体和三个情境层次。其次，我们提出了一种多指标框架——基于hARMs的多方公平性评估（$mFARM$）——用于审计在分配、稳定性和隐含三个维度上的差异，并将这些维度综合为一个$mFARM$分数。我们还提出了公平-准确权衡（FAB）分数来衡量和观察公平性和预测准确性之间的权衡。我们 empirically 评估了四种开源 LLM（Mistral-7B、BioMistral-7B、Qwen-2.5-7B、Bio-LLaMA3-8B）及其量化和情境变化下的微调版本。我们的研究发现表明，提出的$mFARM$指标在各种场景下更能有效捕捉细微偏见。我们发现，大多数模型在不同量化水平下的$mFARM$分数保持稳健性能，但当情境减少时会显著恶化。我们的基准和评估代码已公开发布，以促进医疗保健领域的对齐AI研究。 

---
# EigenBench: A Comparative Behavioral Measure of Value Alignment 

**Title (ZH)**: EigenBench: 价值对齐的比较性行为衡量 

**Authors**: Jonathn Chang, Leonard Piff, Suvadip Sana, Jasmine X. Li, Lionel Levine  

**Link**: [PDF](https://arxiv.org/pdf/2509.01938)  

**Abstract**: Aligning AI with human values is a pressing unsolved problem. To address the lack of quantitative metrics for value alignment, we propose EigenBench: a black-box method for comparatively benchmarking language models' values. Given an ensemble of models, a constitution describing a value system, and a dataset of scenarios, our method returns a vector of scores quantifying each model's alignment to the given constitution. To produce these scores, each model judges the outputs of other models across many scenarios, and these judgments are aggregated with EigenTrust (Kamvar et al, 2003), yielding scores that reflect a weighted-average judgment of the whole ensemble. EigenBench uses no ground truth labels, as it is designed to quantify traits for which reasonable judges may disagree on the correct label. Using prompted personas, we test whether EigenBench scores are more sensitive to the model or the prompt: we find that most of the variance is explained by the prompt, but a small residual quantifies the disposition of the model itself. 

**Abstract (ZH)**: 将AI与人类values对齐是一个亟待解决的问题。为了解决价值观对齐缺乏定量指标的问题，我们提出EigenBench：一种黑盒方法，用于比较基准语言模型的价值观。给定一个模型集合、描述价值观体系的宪法和场景数据集，该方法返回一个向量分数，量化每个模型与给定宪法的对齐程度。通过在许多场景中评判其他模型的输出，这些评判被EigenTrust（Kamvar等人，2003年）聚合，从而产生反映整个集合加权平均评判的分数。EigenBench 不使用 ground truth 标签，因为它被设计用来量化即使合理评判者也可能在正确标签上存在分歧的特质。使用激发的人格，我们测试EigenBench分数对模型或提示是否更敏感：我们发现大部分差异是由提示解释的，但一小部分残差反映了模型本身的态度。 

---
# Dynamic Speculative Agent Planning 

**Title (ZH)**: 动态推测性代理规划 

**Authors**: Yilin Guan, Wenyue Hua, Qingfeng Lan, Sun Fei, Dujian Ding, Devang Acharya, Chi Wang, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01920)  

**Abstract**: Despite their remarkable success in complex tasks propelling widespread adoption, large language-model-based agents still face critical deployment challenges due to prohibitive latency and inference costs. While recent work has explored various methods to accelerate inference, existing approaches suffer from significant limitations: they either fail to preserve performance fidelity, require extensive offline training of router modules, or incur excessive operational costs. Moreover, they provide minimal user control over the tradeoff between acceleration and other performance metrics. To address these gaps, we introduce Dynamic Speculative Planning (DSP), an asynchronous online reinforcement learning framework that provides lossless acceleration with substantially reduced costs without requiring additional pre-deployment preparation. DSP explicitly optimizes a joint objective balancing end-to-end latency against dollar cost, allowing practitioners to adjust a single parameter that steers the system toward faster responses, cheaper operation, or any point along this continuum. Experiments on two standard agent benchmarks demonstrate that DSP achieves comparable efficiency to the fastest lossless acceleration method while reducing total cost by 30% and unnecessary cost up to 60%. Our code and data are available through this https URL. 

**Abstract (ZH)**: 尽管基于大规模语言模型的智能代理在复杂任务上的出色表现推动了广泛采用，但仍因高昂的延迟和推理成本面临关键部署挑战。虽然近期研究探索了多种加速推理的方法，但现有方法存在显著局限：要么无法保持性能一致性，要么需要进行繁琐的离线培训，要么导致过高的运营成本。此外，它们对加速与其它性能指标之间的权衡提供有限的用户控制。为弥补这些不足，我们引入了动态推测规划（DSP），这是一种异步在线强化学习框架，在无需额外预先部署准备的情况下，提供无损加速并显著降低运营成本。DSP 明确优化了端到端延迟与成本的联合目标，允许实践者通过调整单一参数来引导系统向更快响应、更低成本或介于两者之间的任何点发展。在两个标准智能代理基准测试上的实验显示，DSP 在实现与最快无损加速方法相当的效率的同时，总体成本降低了30%，不必要的成本最多降低了60%。我们的代码和数据可通过这个链接获得。 

---
# How Real Is AI Tutoring? Comparing Simulated and Human Dialogues in One-on-One Instruction 

**Title (ZH)**: AI辅导真实吗？一对一指导中模拟对话与人类对话的比较 

**Authors**: Ruijia Li, Yuan-Hao Jiang, Jiatong Wang, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01914)  

**Abstract**: Heuristic and scaffolded teacher-student dialogues are widely regarded as critical for fostering students' higher-order thinking and deep learning. However, large language models (LLMs) currently face challenges in generating pedagogically rich interactions. This study systematically investigates the structural and behavioral differences between AI-simulated and authentic human tutoring dialogues. We conducted a quantitative comparison using an Initiation-Response-Feedback (IRF) coding scheme and Epistemic Network Analysis (ENA). The results show that human dialogues are significantly superior to their AI counterparts in utterance length, as well as in questioning (I-Q) and general feedback (F-F) behaviors. More importantly, ENA results reveal a fundamental divergence in interactional patterns: human dialogues are more cognitively guided and diverse, centered around a "question-factual response-feedback" teaching loop that clearly reflects pedagogical guidance and student-driven thinking; in contrast, simulated dialogues exhibit a pattern of structural simplification and behavioral convergence, revolving around an "explanation-simplistic response" loop that is essentially a simple information transfer between the teacher and student. These findings illuminate key limitations in current AI-generated tutoring and provide empirical guidance for designing and evaluating more pedagogically effective generative educational dialogue systems. 

**Abstract (ZH)**: 人工智能模拟与真实人类辅导对话的结构与行为差异研究：启发式和支架式师生对话对促进高阶思维和深度学习的重要性及其局限性 

---
# Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models 

**Title (ZH)**: Oyster-I：超越拒绝——负责任的语言模型的建设性安全对齐 

**Authors**: Ranjie Duan, Jiexi Liu, Xiaojun Jia, Shiji Zhao, Ruoxi Cheng, Fengxiang Wang, Cheng Wei, Yong Xie, Chang Liu, Defeng Li, Yinpeng Dong, Yichi Zhang, Yuefeng Chen, Chongwen Wang, Xingjun Ma, Xingxing Wei, Yang Liu, Hang Su, Jun Zhu, Xinfeng Li, Yitong Sun, Jie Zhang, Jinzhao Hu, Sha Xu, Yitong Yang, Jialing Tao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2509.01909)  

**Abstract**: Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常部署安全机制以防止有害内容生成。目前大多数方法专注于恶意行为者带来的风险，往往将风险框定为敌对事件，并依赖于防御性的拒绝。然而，在实际应用场景中，风险也可来自处于心理压力下的非恶意用户（例如，自伤意图），寻求帮助。在这种情况下，模型的响应可以强烈影响用户的后续行为。简单的拒绝可能导致用户重复、升级行为或转向不安全的平台，从而产生更糟糕的结果。我们提出了构建性安全对齐（CSA），这是一种以人为本的范式，既可以防止恶意滥用，又能主动引导脆弱用户走向安全且有益的结果。在Oyster-I（Oy1）中实现，CSA结合了对用户反应的博弈论预测、精细的风险边界发现和可解释的推理控制，将安全性转化为信任建设过程。Oy1在开放模型中实现了最先进的安全性，同时保持了高的一般能力。在我们的构建性基准测试中，它展示了接近GPT-5的强大建设性参与，并在Strata-Sword逃逸测试集上展现了无与伦比的稳健性，接近GPT-o1的水平。通过从拒绝为主转向指导为主的安全策略，CSA重新定义了模型与用户的关系，旨在构建不仅安全而且有意义的有帮助的系统。我们发布了Oy1、代码和基准以支持负责任的、以用户为中心的AI。 

---
# An LLM-enabled semantic-centric framework to consume privacy policies 

**Title (ZH)**: 基于LLM的以语义为中心的隐私政策消费框架 

**Authors**: Rui Zhao, Vladyslav Melnychuk, Jun Zhao, Jesse Wright, Nigel Shadbolt  

**Link**: [PDF](https://arxiv.org/pdf/2509.01716)  

**Abstract**: In modern times, people have numerous online accounts, but they rarely read the Terms of Service or Privacy Policy of those sites, despite claiming otherwise, due to the practical difficulty in comprehending them. The mist of data privacy practices forms a major barrier for user-centred Web approaches, and for data sharing and reusing in an agentic world. Existing research proposed methods for using formal languages and reasoning for verifying the compliance of a specified policy, as a potential cure for ignoring privacy policies. However, a critical gap remains in the creation or acquisition of such formal policies at scale. We present a semantic-centric approach for using state-of-the-art large language models (LLM), to automatically identify key information about privacy practices from privacy policies, and construct $\mathit{Pr}^2\mathit{Graph}$, knowledge graph with grounding from Data Privacy Vocabulary (DPV) for privacy practices, to support downstream tasks. Along with the pipeline, the $\mathit{Pr}^2\mathit{Graph}$ for the top-100 popular websites is also released as a public resource, by using the pipeline for analysis. We also demonstrate how the $\mathit{Pr}^2\mathit{Graph}$ can be used to support downstream tasks by constructing formal policy representations such as Open Digital Right Language (ODRL) or perennial semantic Data Terms of Use (psDToU). To evaluate the technology capability, we enriched the Policy-IE dataset by employing legal experts to create custom annotations. We benchmarked the performance of different large language models for our pipeline and verified their capabilities. Overall, they shed light on the possibility of large-scale analysis of online services' privacy practices, as a promising direction to audit the Web and the Internet. We release all datasets and source code as public resources to facilitate reuse and improvement. 

**Abstract (ZH)**: 现代网络账户众多，但用户罕有阅读服务条款或隐私政策，这为以用户为中心的网络方法、以及在能动世界中的数据共享与重用设下了重大障碍。现有研究提出使用形式语言和推理验证特定政策遵守性的方法，以解决忽视隐私政策的问题。然而，大规模创建或获取此类形式政策仍存在重要缺口。我们提出一种语义为中心的方法，利用最先进的大规模语言模型（LLM），自动识别隐私政策中的关键隐私实践信息，并构建基于数据隐私词汇表（DPV）的知识图谱 $\mathit{Pr}^2\mathit{Graph}$，以支持下游任务。我们还通过该管线分析了顶级100个网站，并公开了 $\mathit{Pr}^2\mathit{Graph}$ 作为公共资源。我们展示了如何利用 $\mathit{Pr}^2\mathit{Graph}$ 支持下游任务，例如构建如开放数字权利语言（ODRL）或持久语义数据服务条款（psDToU）等正式政策表示。为评估技术能力，我们通过法律专家创建自定义注释丰富了Policy-IE数据集，并对不同大规模语言模型的管线性能进行了基准测试。总体而言，这揭示了大规模分析在线服务隐私实践的可能性，为审计网络和互联网指明了前景。我们释放所有数据集和源代码作为公共资源，以促进重用和改进。 

---
# Unraveling LLM Jailbreaks Through Safety Knowledge Neurons 

**Title (ZH)**: 通过安全知识神经元解析LLM Jailbreaks 

**Authors**: Chongwen Zhao, Kaizhu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01631)  

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种应用中愈发受到关注，但一些用户试图利用这些模型进行恶意行为，如合成 Controlled Substances 和传播虚假信息，这一行为被称为“Jailbreak”。尽管一些研究通过修改输出分布或检测有害内容来抵御Jailbreak攻击，其确切原理仍然不清晰。在此工作中，我们提出了一种新的神经元级别可解释性方法，重点关注与安全性相关的知识神经元的作用。不同于现有方法，我们的方法将模型的内部表示投射到一个更具一致性和可解释性的词汇空间。我们随后展示了调整与安全性相关的神经元的激活可以有效地控制模型的行为，其平均ASR高于97%。基于这一见解，我们提出了SafeTuning，一种增强安全关键神经元的微调策略，以提高模型抵御Jailbreak的鲁棒性。SafeTuning在多个LLM模型上一致地降低了攻击成功率，并优于所有四种基线防御措施。这些发现为理解并抵御Jailbreak攻击提供了一个新的视角。 

---
# Counterfactual Sensitivity for Faithful Reasoning in Language Models 

**Title (ZH)**: 语言模型中忠实地-counterfactual 敏感性推理 

**Authors**: Ibne Farabi Shihab, Sanjeda Akter, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2509.01544)  

**Abstract**: Large language models (LLMs) often produce correct answers while relying on flawed or irrelevant reasoning traces, undermining their trustworthiness in high-stakes domains. We propose Counterfactual Sensitivity Regularization (CSR), a lightweight training objective that enforces dependence between intermediate reasoning and final outputs. CSR introduces automated, operator-level counterfactual interventions (e.g., swapping "+" with "-") during training and penalizes models that preserve the same answer under logically invalid traces. This requires only one additional forward pass per sample. To measure faithfulness, we introduce Counterfactual Outcome Sensitivity (COS), which quantifies the impact of such perturbations on model predictions. Across structured reasoning tasks - arithmetic (GSM8K), logical deduction (PrOntoQA), and planning (Blocks World) - CSR improves faithfulness by up to 70 percentage points over standard fine-tuning and process supervision, with only minor accuracy loss. The learned sensitivity generalizes to larger models and synergizes with inference-time methods such as self-consistency. A pilot study on HellaSwag further demonstrates that extending CSR with semantic perturbations can enhance faithfulness in commonsense reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）往往依赖于错误或不相关的推理痕迹来产生正确答案，这在高风险领域削弱了它们的可信度。我们提出了一种名为反事实敏感正则化（CSR，Counterfactual Sensitivity Regularization）的轻量级训练目标，该目标确保中间推理与最终输出之间的依赖关系。CSR 在训练过程中引入了自动化的操作级反事实干预（例如，将“+”替换为“-”），并惩罚那些在逻辑上无效的推理痕迹下仍保持相同答案的模型。这仅需每个样本额外进行一次前向传播。为了衡量忠实性，我们引入了反事实结果敏感性（COS，Counterfactual Outcome Sensitivity），量化此类干扰对模型预测的影响。在结构化推理任务（算术：GSM8K、逻辑演绎：PrOntoQA、规划：Blocks World）中，CSR 的忠实性比标准微调和过程监督提高了最多 70 个百分点，且仅略有准确率损失。学到的敏感性可以推广到大型模型，并与推理时的方法（如自一致性）协同作用。在 HellaSwag 的初步研究中进一步表明，扩展 CSR 以包含语义干扰可以增强常识推理的忠实性。 

---
# LLM-empowered Agents Simulation Framework for Scenario Generation in Service Ecosystem Governance 

**Title (ZH)**: LLM赋能的代理模拟框架：服务生态系统治理中的场景生成 

**Authors**: Deyu Zhou, Yuqi Hou, Xiao Xue, Xudong Lu, Qingzhong Li, Lizhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2509.01441)  

**Abstract**: As the social environment is growing more complex and collaboration is deepening, factors affecting the healthy development of service ecosystem are constantly changing and diverse, making its governance a crucial research issue. Applying the scenario analysis method and conducting scenario rehearsals by constructing an experimental system before managers make decisions, losses caused by wrong decisions can be largely avoided. However, it relies on predefined rules to construct scenarios and faces challenges such as limited information, a large number of influencing factors, and the difficulty of measuring social elements. These challenges limit the quality and efficiency of generating social and uncertain scenarios for the service ecosystem. Therefore, we propose a scenario generator design method, which adaptively coordinates three Large Language Model (LLM) empowered agents that autonomously optimize experimental schemes to construct an experimental system and generate high quality scenarios. Specifically, the Environment Agent (EA) generates social environment including extremes, the Social Agent (SA) generates social collaboration structure, and the Planner Agent (PA) couples task-role relationships and plans task solutions. These agents work in coordination, with the PA adjusting the experimental scheme in real time by perceiving the states of each agent and these generating scenarios. Experiments on the ProgrammableWeb dataset illustrate our method generates more accurate scenarios more efficiently, and innovatively provides an effective way for service ecosystem governance related experimental system construction. 

**Abstract (ZH)**: 随着社会环境日益复杂和合作不断加深，影响服务生态系统健康发展的影响因素不断变化且多样化，使其治理成为一项关键的科研课题。通过情景分析方法，在管理者决策前构建实验系统进行情景预演，可以大大避免因错误决策造成的损失。然而，这种方法依赖预定义规则构建情景，并面临信息有限、影响因素众多以及社会元素衡量难度大等挑战。这些挑战限制了生成高质量和服务生态系统相关的社会不确定性情景的质量和效率。因此，我们提出了一种情景生成设计方法，该方法适应性协调三个由大规模语言模型（LLM）赋能的代理，自主优化实验方案构建实验系统并生成高质量情景。具体而言，环境代理（EA）生成社会环境，社会代理（SA）生成社会协作结构，计划代理（PA）结合任务角色关系并规划任务解决方案。这些代理协同工作，PA通过感知每个代理的状态实时调整实验方案并生成情景。基于ProgrammableWeb数据集的实验展示了我们的方法更高效地生成更准确的情景，并创新性地为服务生态系统治理相关的实验系统构建提供了有效途径。 

---
# DeepResearch Arena: The First Exam of LLMs' Research Abilities via Seminar-Grounded Tasks 

**Title (ZH)**: 深度研究 arena：首次通过基于研讨会的任务来评估大语言模型的研究能力 

**Authors**: Haiyuan Wan, Chen Yang, Junchi Yu, Meiqi Tu, Jiaxuan Lu, Di Yu, Jianbao Cao, Ben Gao, Jiaqing Xie, Aoran Wang, Wenlong Zhang, Philip Torr, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.01396)  

**Abstract**: Deep research agents have attracted growing attention for their potential to orchestrate multi-stage research workflows, spanning literature synthesis, methodological design, and empirical verification. Despite these strides, evaluating their research capability faithfully is rather challenging due to the difficulty of collecting frontier research questions that genuinely capture researchers' attention and intellectual curiosity. To address this gap, we introduce DeepResearch Arena, a benchmark grounded in academic seminars that capture rich expert discourse and interaction, better reflecting real-world research environments and reducing the risk of data leakage. To automatically construct DeepResearch Arena, we propose a Multi-Agent Hierarchical Task Generation (MAHTG) system that extracts research-worthy inspirations from seminar transcripts. The MAHTG system further translates research-worthy inspirations into high-quality research tasks, ensuring the traceability of research task formulation while filtering noise. With the MAHTG system, we curate DeepResearch Arena with over 10,000 high-quality research tasks from over 200 academic seminars, spanning 12 disciplines, such as literature, history, and science. Our extensive evaluation shows that DeepResearch Arena presents substantial challenges for current state-of-the-art agents, with clear performance gaps observed across different models. 

**Abstract (ZH)**: 深度研究代理的深度研究 arenas：基于学术研讨会的benchmark及其挑战评估 

---
# Error Notebook-Guided, Training-Free Part Retrieval in 3D CAD Assemblies via Vision-Language Models 

**Title (ZH)**: 基于视觉语言模型的3D CAD装配中无训练数据零件检索引导错误笔记本 

**Authors**: Yunqing Liu, Nan Zhang, Zhiming Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01350)  

**Abstract**: Effective specification-aware part retrieval within complex CAD assemblies is essential for automated design verification and downstream engineering tasks. However, directly using LLMs/VLMs to this task presents some challenges: the input sequences may exceed model token limits, and even after processing, performance remains unsatisfactory. Moreover, fine-tuning LLMs/VLMs requires significant computational resources, and for many high-performing general-use proprietary models (e.g., GPT or Gemini), fine-tuning access is not available. In this paper, we propose a novel part retrieval framework that requires no extra training, but using Error Notebooks + RAG for refined prompt engineering to help improve the existing general model's retrieval performance. The construction of Error Notebooks consists of two steps: (1) collecting historical erroneous CoTs and their incorrect answers, and (2) connecting these CoTs through reflective corrections until the correct solutions are obtained. As a result, the Error Notebooks serve as a repository of tasks along with their corrected CoTs and final answers. RAG is then employed to retrieve specification-relevant records from the Error Notebooks and incorporate them into the inference process. Another major contribution of our work is a human-in-the-loop CAD dataset, which is used to evaluate our method. In addition, the engineering value of our novel framework lies in its ability to effectively handle 3D models with lengthy, non-natural language metadata. Experiments with proprietary models, including GPT-4o and the Gemini series, show substantial gains, with GPT-4o (Omni) achieving up to a 23.4% absolute accuracy improvement on the human preference dataset. Moreover, ablation studies confirm that CoT reasoning provides benefits especially in challenging cases with higher part counts (>10). 

**Abstract (ZH)**: 有效的面向规范的部件检索对于复杂CAD组件的自动化设计验证及下游工程任务至关重要。然而，直接使用LLMs/VLMs进行此任务存在一些挑战：输入序列可能超出模型的标记限制，即使处理后性能仍不理想。此外，微调LLMs/VLMs需要大量计算资源，而对于许多高性能的通用模型（如GPT或Gemini），微调访问是不可用的。在本文中，我们提出了一种无需额外训练的新颖部件检索框架，通过Error Notebooks + RAG进行精细提示工程，以帮助提高现有通用模型的检索性能。Error Notebooks的构建包括两个步骤：（1）收集历史错误的CoTs及其错误答案，（2）通过反思性修正将这些CoTs连接起来，直至获得正确解。结果，Error Notebooks充当了任务及其修正后的CoTs和最终答案的存储库。RAG随后用于从Error Notebooks检索规范相关的记录，并将其集成到推理过程中。我们工作的另一项重大贡献是带有工程师参与的CAD数据集，用于评估我们的方法。此外，我们新颖框架的工程价值在于其能够有效处理带有长且非自然语言元数据的3D模型。使用GPT-4o和Gemini系列等专有模型的实验显示了显著的提升，GPT-4o (Omni)在人类偏好数据集上实现了高达23.4%的绝对准确性提升。此外，消融研究证实，CoT推理在部件计数较高（>10）的挑战性情况下特别有益。 

---
# GradeSQL: Outcome Reward Models for Ranking SQL Queries from Large Language Models 

**Title (ZH)**: GradeSQL：用于排名大规模语言模型生成的SQL查询的结果奖励模型 

**Authors**: Mattia Tritto, Giuseppe Farano, Dario Di Palma, Gaetano Rossiello, Fedelucio Narducci, Dharmashankar Subramanian, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2509.01308)  

**Abstract**: Text-to-SQL, the task of translating natural language questions into SQL queries, has significantly advanced with the introduction of Large Language Models (LLMs), broadening database accessibility for a wide range of users. Despite substantial progress in generating valid SQL, current LLMs still struggle with complex queries that require precise alignment between user intent and the database schema. To mitigate this, test-time strategies such as Best-of-N (BoN) and Majority Voting (Maj) are often employed, based on the assumption that LLMs can generate correct answers but may require multiple attempts. However, these methods rely on surface-level heuristics, selecting either the syntactically correct query through execution-based BoN (ex-BoN) or the most frequently generated query with Maj. Recently, Outcome Reward Models (ORMs), which assign utility scores to generated outputs based on semantic correctness, have emerged as a promising approach for better aligning model predictions with user intent. Nevertheless, their application to Text-to-SQL remains largely underexplored.
In this work, we evaluate ORMs as an effective heuristic for BoN, compare them with ex-BoN and Maj, and introduce a framework for training ORMs for the Text-to-SQL task. We evaluate our ORMs on the BIRD and SPIDER benchmarks, finetuning various open-source LLMs, including the Qwen2, Granite3, and Llama3 model families. Our results show that ORMs outperform ex-BoN and Maj, achieving execution accuracy gains of +4.33% (BIRD) and +2.10% (Spider) over ex-BoN, and +2.91% (BIRD) and +0.93% (Spider) over Maj. We further demonstrate that finetuning models already aligned with SQL generation, such as OmniSQL, yields superior ORM performance. Additionally, we observe that ORMs achieve competitive results on simple queries and benefit more from an increased number of candidates compared to ex-BoN and Maj. 

**Abstract (ZH)**: 基于大规模语言模型的Text-to-SQL中Outcome Reward Models作为有效启发式的评估与应用 

---
# Communicative Agents for Slideshow Storytelling Video Generation based on LLMs 

**Title (ZH)**: 基于LLM的幻灯片故事讲述视频生成中的沟通代理 

**Authors**: Jingxing Fan, Jinrong Shen, Yusheng Yao, Shuangqing Wang, Qian Wang, Yuling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01277)  

**Abstract**: With the rapid advancement of artificial intelligence (AI), the proliferation of AI-generated content (AIGC) tasks has significantly accelerated developments in text-to-video generation. As a result, the field of video production is undergoing a transformative shift. However, conventional text-to-video models are typically constrained by high computational costs.
In this study, we propose Video-Generation-Team (VGTeam), a novel slide show video generation system designed to redefine the video creation pipeline through the integration of large language models (LLMs). VGTeam is composed of a suite of communicative agents, each responsible for a distinct aspect of video generation, such as scriptwriting, scene creation, and audio design. These agents operate collaboratively within a chat tower workflow, transforming user-provided textual prompts into coherent, slide-style narrative videos.
By emulating the sequential stages of traditional video production, VGTeam achieves remarkable improvements in both efficiency and scalability, while substantially reducing computational overhead. On average, the system generates videos at a cost of only $0.103, with a successful generation rate of 98.4%. Importantly, this framework maintains a high degree of creative fidelity and customization.
The implications of VGTeam are far-reaching. It democratizes video production by enabling broader access to high-quality content creation without the need for extensive resources. Furthermore, it highlights the transformative potential of language models in creative domains and positions VGTeam as a pioneering system for next-generation content creation. 

**Abstract (ZH)**: 随着人工智能（AI）的迅速发展，AI生成内容（AIGC）任务的增多显著加快了文本到视频生成的发展。因此，视频制作领域正经历着一场转型变革。然而，传统的文本到视频模型通常受限于高昂的计算成本。

本研究提出了一种新的幻灯片视频生成系统——视频生成团队（VGTeam），旨在通过大型语言模型（LLMs）的整合来重塑视频创作流程。VGTeam由一系列通信代理组成，每个代理负责视频生成的不同方面，如剧本写作、场景创作和音频设计。这些代理在聊天塔工作流程中协同工作，将用户提供的文本提示转化为连贯的幻灯片风格叙述视频。

通过模仿传统视频制作的顺序阶段，VGTeam在效率和扩展性方面取得了显著改善，同时显著降低了计算开销。系统平均生成每个视频的成本仅为0.103美元，成功生成率为98.4%。重要的是，该框架保持了高度的创意 fidelity和定制性。

VGTeam的影响深远。它使视频制作更加民主化，无需大量资源即可获得高质量内容创作的广泛访问权限。此外，它突显了语言模型在创意领域中的转变潜力，并将VGTeam定位为下一代内容创作的先驱系统。 

---
# Towards Agentic OS: An LLM Agent Framework for Linux Schedulers 

**Title (ZH)**: 面向能动OS：一个针对Linux调度器的LLM代理框架 

**Authors**: Yusheng Zheng, Yanpeng Hu, Wei Zhang, Andi Quinn  

**Link**: [PDF](https://arxiv.org/pdf/2509.01245)  

**Abstract**: Operating system schedulers suffer from a fundamental semantic gap, where kernel policies fail to understand application-specific needs, leading to suboptimal performance. We introduce SchedCP, the first framework that enables fully autonomous Large Language Model (LLM) agents to safely and efficiently optimize Linux schedulers without human involvement. Our core insight is that the challenge is not merely to apply a better LLM, but to architect a decoupled control plane that separates the AI's role of semantic reasoning ("what to optimize") from the system's role of execution ("how to observe and act"). Implemented as Model Context Protocol(MCP) server, SchedCP provides a stable interface with three key services: a Workload Analysis Engine, an evolving Scheduler Policy Repository, and an Execution Verifier that validates all AI-generated code and configure before deployment with static and dynamic analysis.
We demonstrate this architecture's power with sched-agent, a multi-agent system that autonomously analyzes workloads, synthesizes custom eBPF scheduling policies, and deploys them via the sched\_ext infrastructure. Our evaluation shows that SchedCP achieves up to an 1.79x performance improvement, and a 13x cost reduction compared to naive agentic approaches, all while maintaining high success rate. By bridging the semantic gap, SchedCP democratizes expert-level system optimization and represents a step towards creating truly self-optimizing, application-aware operating systems. The code is open-sourced in this https URL 

**Abstract (ZH)**: 操作系统的调度器存在根本性的语义差距，内核策略无法理解应用程序特定的需求，导致性能不佳。我们引入了SchedCP框架，这是首个能够在没有人类干预的情况下安全且高效地自主优化Linux调度器的框架。我们的核心洞察是，挑战不仅在于应用更好的语言模型，还在于架构一个解耦的控制平面，使AI在语义推理（“何者需要优化”）方面与系统的执行观察和行动角色（“如何进行观察和操作”）分离。SchedCP以Model Context Protocol (MCP)服务器实现，提供了稳定的接口，并包含三个关键服务：工作负载分析引擎、不断进化的调度策略仓库以及一个执行验证器，该验证器在部署前通过静态和动态分析验证所有AI生成的代码和配置。我们通过调度代理（sched-agent）多代理系统展示了该架构的力量，该系统自主分析工作负载、合成自定义eBPF调度策略并通过sched\_ext基础设施部署它们。评估结果显示，与简单的代理方法相比，SchedCP的性能提高了最多1.79倍，成本降低了13倍，同时保持了高的成功率。通过弥合语义差距，SchedCP提高了系统优化的普及程度，并朝着创建真正自我优化、应用感知的操作系统迈出了一步。代码在此开放源代码：https://github.com/alibaba/SchedCP 

---
# Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework 

**Title (ZH)**: 面向知识图谱的开放世界检索增强生成：一种多agent协作框架 

**Authors**: Jiasheng Xu, Mingda Li, Yongqiang Tang, Peijie Wang, Wensheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01238)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in language understanding and reasoning. However, their dependence on static training corpora makes them prone to factual errors and knowledge gaps. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge sources, especially structured Knowledge Graphs (KGs), which provide explicit semantics and efficient retrieval. Existing KG-based RAG approaches, however, generally assume that anchor entities are accessible to initiate graph traversal, which limits their robustness in open world settings where accurate linking between the query and the entity is unreliable. To overcome this limitation, we propose AnchorRAG, a novel multi-agent collaboration framework for open-world RAG without the predefined anchor entities. Specifically, a predictor agent dynamically identifies candidate anchor entities by aligning user query terms with KG nodes and initializes independent retriever agents to conduct parallel multi-hop explorations from each candidate. Then a supervisor agent formulates the iterative retrieval strategy for these retriever agents and synthesizes the resulting knowledge paths to generate the final answer. This multi-agent collaboration framework improves retrieval robustness and mitigates the impact of ambiguous or erroneous anchors. Extensive experiments on four public benchmarks demonstrate that AnchorRAG significantly outperforms existing baselines and establishes new state-of-the-art results on the real-world question answering tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言理解和推理方面展示了强大的能力，但在依赖静态训练语料库的情况下，容易出现事实错误和知识空白。检索增强生成（RAG）通过集成外部知识源，特别是结构化知识图谱（KGs），解决了这一限制，KGs提供明确的语义和高效的检索。现有的基于KG的RAG方法通常假定锚实体可用以启动图遍历，这在开放世界环境中可能导致链接不准确，限制了其鲁棒性。为了解决这一限制，我们提出了一种新颖的多代理协作框架AnchorRAG，该框架在没有预定义锚实体的情况下应用于开放世界RAG。具体来说，预测代理动态识别候选锚实体并将用户查询术语与KG节点对齐，并初始化多个独立的检索代理进行并行多跳探索。然后，监督代理为这些检索代理制定迭代检索策略，并综合生成的知识路径以产生最终答案。这种多代理协作框架提高了检索的鲁棒性，并减轻了模糊或错误的锚实体的影响。在四个公开基准上的 extensive 实验表明，AnchorRAG 显著优于现有基线方法，并在真实世界的问题回答任务上建立了新的最佳成果。 

---
# Question-to-Knowledge: Multi-Agent Generation of Inspectable Facts for Product Mapping 

**Title (ZH)**: 知识问答：多Agent生成可检验的事实进行产品映射 

**Authors**: Wonduk Seo, Taesub Shin, Hyunjin An, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.01182)  

**Abstract**: Identifying whether two product listings refer to the same Stock Keeping Unit (SKU) is a persistent challenge in ecommerce, especially when explicit identifiers are missing and product names vary widely across platforms. Rule based heuristics and keyword similarity often misclassify products by overlooking subtle distinctions in brand, specification, or bundle configuration. To overcome these limitations, we propose Question to Knowledge (Q2K), a multi agent framework that leverages Large Language Models (LLMs) for reliable SKU mapping. Q2K integrates: (1) a Reasoning Agent that generates targeted disambiguation questions, (2) a Knowledge Agent that resolves them via focused web searches, and (3) a Deduplication Agent that reuses validated reasoning traces to reduce redundancy and ensure consistency. A human in the loop mechanism further refines uncertain cases. Experiments on real world consumer goods datasets show that Q2K surpasses strong baselines, achieving higher accuracy and robustness in difficult scenarios such as bundle identification and brand origin disambiguation. By reusing retrieved reasoning instead of issuing repeated searches, Q2K balances accuracy with efficiency, offering a scalable and interpretable solution for product integration. 

**Abstract (ZH)**: 识别两个产品列表是否指向同一个库存单位（SKU）是电子商务中的一项持续性挑战，特别是在缺少明确标识符且产品名称在不同平台上差异较大时。基于规则的启发式方法和关键词相似性往往因为忽视了品牌、规格或捆绑配置的细微差异而产生误分类。为克服这些限制，我们提出了一种利用大语言模型（LLMs）进行可靠SKU映射的多代理框架Question to Knowledge（Q2K）。Q2K集成了：（1）一个推理代理，生成针对性的去模糊化问题；（2）一个知识代理，通过聚焦的网络搜索来解决这些问题；（3）一个去重代理，通过重用验证过的推理轨迹来减少冗余并确保一致性。通过引入人工介入机制，进一步完善不确定性案例。实验证明，Q2K在困难场景如捆绑识别和品牌来源去模糊化方面超越了强大基线，提高了准确性和鲁棒性。通过重用检索到的推理而不是重复搜索，Q2K在准确性和效率之间取得了平衡，提供了一种可扩展且可解释的产品整合解决方案。 

---
# Heads or Tails: A Simple Example of Causal Abstractive Simulation 

**Title (ZH)**: 正反面：因果抽象模拟的一个简单示例 

**Authors**: Gabriel Simmons  

**Link**: [PDF](https://arxiv.org/pdf/2509.01136)  

**Abstract**: This note illustrates how a variety of causal abstraction arXiv:1707.00819 arXiv:1812.03789, defined here as causal abstractive simulation, can be used to formalize a simple example of language model simulation. This note considers the case of simulating a fair coin toss with a language model. Examples are presented illustrating the ways language models can fail to simulate, and a success case is presented, illustrating how this formalism may be used to prove that a language model simulates some other system, given a causal description of the system. This note may be of interest to three groups. For practitioners in the growing field of language model simulation, causal abstractive simulation is a means to connect ad-hoc statistical benchmarking practices to the solid formal foundation of causality. Philosophers of AI and philosophers of mind may be interested as causal abstractive simulation gives a precise operationalization to the idea that language models are role-playing arXiv:2402.12422. Mathematicians and others working on causal abstraction may be interested to see a new application of the core ideas that yields a new variation of causal abstraction. 

**Abstract (ZH)**: 本文展示了如何使用这里定义为因果抽象模拟的方法（arXiv:1707.00819 arXiv:1812.03789）来形式化语言模型模拟的简单示例。本文考虑了使用语言模型模拟公平硬币抛掷的情况。本文提供了语言模型模拟失败的示例，并展示了成功案例，说明了如何使用此形式主义在给定系统因果描述的情况下证明语言模型模拟其他系统。本文可能对三类人群感兴趣：语言模型模拟领域日益增长的从业者，因果抽象模拟为其所提出的语言模型角色扮演 idea（arXiv:2402.12422）提供了精确的操作化；人工智能哲学家和心灵哲学家，因果抽象模拟；数学家和其他从事因果抽象工作的人员，它提供了一种核心思想的新应用，从而产生了一种因果抽象的新变体。 

---
# Analysis of Error Sources in LLM-based Hypothesis Search for Few-Shot Rule Induction 

**Title (ZH)**: 基于LLM的少样本规则归纳假设搜索中的误差来源分析 

**Authors**: Aishni Parab, Hongjing Lu, Ying Nian Wu, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2509.01016)  

**Abstract**: Inductive reasoning enables humans to infer abstract rules from limited examples and apply them to novel situations. In this work, we compare an LLM-based hypothesis search framework with direct program generation approaches on few-shot rule induction tasks. Our findings show that hypothesis search achieves performance comparable to humans, while direct program generation falls notably behind. An error analysis reveals key bottlenecks in hypothesis generation and suggests directions for advancing program induction methods. Overall, this paper underscores the potential of LLM-based hypothesis search for modeling inductive reasoning and the challenges in building more efficient systems. 

**Abstract (ZH)**: 基于LLM的假设搜索框架在少量示例规则归纳任务中相较于直接程序生成方法表现出更高的性能，同时暴露出假设生成的关键瓶颈并 suggests 方向以推进程序归纳方法。总体而言，本文强调了基于LLM的假设搜索在建模归纳推理方面的潜力及其构建更高效系统的挑战。 

---
# Supporting Our AI Overlords: Redesigning Data Systems to be Agent-First 

**Title (ZH)**: 支持我们的AI overlords：重设计数据系统以用户为先 

**Authors**: Shu Liu, Soujanya Ponnapalli, Shreya Shankar, Sepanta Zeighami, Alan Zhu, Shubham Agarwal, Ruiqi Chen, Samion Suwito, Shuo Yuan, Ion Stoica, Matei Zaharia, Alvin Cheung, Natacha Crooks, Joseph E. Gonzalez, Aditya G. Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2509.00997)  

**Abstract**: Large Language Model (LLM) agents, acting on their users' behalf to manipulate and analyze data, are likely to become the dominant workload for data systems in the future. When working with data, agents employ a high-throughput process of exploration and solution formulation for the given task, one we call agentic speculation. The sheer volume and inefficiencies of agentic speculation can pose challenges for present-day data systems. We argue that data systems need to adapt to more natively support agentic workloads. We take advantage of the characteristics of agentic speculation that we identify, i.e., scale, heterogeneity, redundancy, and steerability - to outline a number of new research opportunities for a new agent-first data systems architecture, ranging from new query interfaces, to new query processing techniques, to new agentic memory stores. 

**Abstract (ZH)**: 大型语言模型代理在未来可能成为数据系统的主要工作负载，它们代表用户操作，以操纵和分析数据。在处理数据时，这些代理采用一种针对给定任务进行高效探索和解决方案形成的高通量过程，我们称之为代理推测。代理推测的庞大体量和低效率给现有数据系统带来了挑战。我们认为数据系统需要适应以更自然地支持代理型工作负载。我们利用识别出的代理推测的特征，即规模、异构性、冗余性和可控性，来概述一种以代理为中心的新数据系统架构的新研究机会，涵盖新型查询接口、新的查询处理技术以及新的代理型存储系统等方面。 

---
# Causal MAS: A Survey of Large Language Model Architectures for Discovery and Effect Estimation 

**Title (ZH)**: 因果MAS：大规模语言模型架构综述，用于发现和效应估计 

**Authors**: Adib Bazgir, Amir Habibdoust, Yuwen Zhang, Xing Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.00987)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various reasoning and generation tasks. However, their proficiency in complex causal reasoning, discovery, and estimation remains an area of active development, often hindered by issues like hallucination, reliance on spurious correlations, and difficulties in handling nuanced, domain-specific, or personalized causal relationships. Multi-agent systems, leveraging the collaborative or specialized abilities of multiple LLM-based agents, are emerging as a powerful paradigm to address these limitations. This review paper explores the burgeoning field of causal multi-agent LLMs. We examine how these systems are designed to tackle different facets of causality, including causal reasoning and counterfactual analysis, causal discovery from data, and the estimation of causal effects. We delve into the diverse architectural patterns and interaction protocols employed, from pipeline-based processing and debate frameworks to simulation environments and iterative refinement loops. Furthermore, we discuss the evaluation methodologies, benchmarks, and diverse application domains where causal multi-agent LLMs are making an impact, including scientific discovery, healthcare, fact-checking, and personalized systems. Finally, we highlight the persistent challenges, open research questions, and promising future directions in this synergistic field, aiming to provide a comprehensive overview of its current state and potential trajectory. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种推理和生成任务中展示了令人瞩目的能力。然而，在复杂因果推理、发现和估计方面的熟练程度仍然是一个活跃发展的领域，常常受到幻觉、依赖虚假相关性和处理细微、领域特定或个性化因果关系困难等问题的阻碍。多智能体系统利用多个基于LLM的智能体的协作或专业能力，正在成为一个强大的范式来解决这些问题。本文回顾了因果多智能体LLM的新兴领域。我们将探讨这些系统如何设计以应对因果性的不同方面，包括因果推理和反事实分析、从数据中发现因果关系以及因果效应的估计。我们将深入探讨采用的各种架构模式和交互协议，从管道处理和辩论框架到模拟环境和迭代改进循环。此外，我们将讨论因果多智能体LLM的评估方法、基准以及它们在科学发现、医疗保健、事实核查和个人化系统等多元应用领域的影响力。最后，我们将突出该交叉领域的持续挑战、开放的研究问题和充满希望的未来方向，旨在提供其当前状态和潜在轨迹的全面概述。 

---
# Self-Exploring Language Models for Explainable Link Forecasting on Temporal Graphs via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的自探索语言模型在临时图上进行可解释链接预测 

**Authors**: Zifeng Ding, Shenyang Huang, Zeyu Cao, Emma Kondrup, Zachary Yang, Xingyue Huang, Yuan Sui, Zhangdie Yuan, Yuqicheng Zhu, Xianglong Hu, Yuan He, Farimah Poursafaei, Michael Bronstein, Andreas Vlachos  

**Link**: [PDF](https://arxiv.org/pdf/2509.00975)  

**Abstract**: Forecasting future links is a central task in temporal graph (TG) reasoning, requiring models to leverage historical interactions to predict upcoming ones. Traditional neural approaches, such as temporal graph neural networks, achieve strong performance but lack explainability and cannot be applied to unseen graphs without retraining. Recent studies have begun to explore using large language models (LLMs) for graph reasoning, but most of them are constrained to static graphs or small synthetic TGs and lack the evaluation of the quality of reasoning traces generated by LLMs. In this work, we present Reasoning-Enhanced Learning for Temporal Graphs (ReaL-TG), a reinforcement learning framework that fine-tunes LLMs to perform explainable link forecasting on real-world TGs. ReaL-TG uses outcome-based reward to encourage models to self-explore reasoning strategies from graph structure and to produce explanations that directly justify their predictions. To enable evaluation on LLM-generated reasoning traces, we propose a new evaluation protocol combining ranking metrics with an LLM-as-a-Judge system that assesses both the quality of reasoning and the impact of hallucinations. Experiments with ReaL-TG-4B, obtained by fine-tuning Qwen3-4B under our framework, show that it outperforms much larger frontier LLMs, including GPT-5 mini, on ranking metrics, while producing high-quality explanations confirmed by both the LLM judge and human evaluation. 

**Abstract (ZH)**: 基于强化学习的图解释增强学习方法（ReaL-TG）：一种用于真实时序图的可解释链接预测框架 

---
# CoreThink: A Symbolic Reasoning Layer to reason over Long Horizon Tasks with LLMs 

**Title (ZH)**: CoreThink：一个符号推理层，用于通过LLMs进行长期任务推理 

**Authors**: Jay Vaghasiya, Omkar Ghugarkar, Vishvesh Bhat, Vipul Dholaria, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2509.00971)  

**Abstract**: We introduce CoreThink, a state-of-the-art Reasoning Layer built upon a novel reasoning method called General Symbolics. This approach diverges from reasoning paradigms such as test-time scaling, Supervised Fine-Tuning (SFT), and Reinforcement Learning with Verifiable Rewards (RLVR). CoreThink General Symbolic Reasoner (GSR) is specifically structured around three key use cases: tool-calling, code generation, and planning, demonstrating exemplary performance across a total of seven benchmarks in their respective areas. Notably, we are achieving SOTA scores of 66.66\% on Livecodebench v6, 89\% on Instruction-Following Evals, and 24.4\% on ARC-AGI-2. We also present an agentic coding IDE, developed using the principles of General Symbolics, which achieves a state-of-the-art accuracy of 62.3\% on \texttt{SWE-Bench Lite}. We are able to achieve these improvements without any finetuning or training costs. Our Reasoning Layer is designed to provide a pure performance uplift, ensuring that a model's accuracy on reasoning tasks is never negatively impacted. We argue that incumbent methods will eventually lead to diminishing returns in LLM performance, necessitating the development of new reasoning techniques. This technical report details our approach at a high level and the availability of the CoreThink models for reasoning-intensive use cases. 

**Abstract (ZH)**: 我们介绍CoreThink，一种基于新颖推理方法General Symbolics构建的先进推理层。CoreThink通用符号推理器（GSR）特别针对工具调用、代码生成和规划这三个关键应用场景，展示了在各自的七个基准测试中卓越的性能。我们在Livecodebench v6中达到66.66%的最佳表现，在Instruction-Following Evals中达到89%，在ARC-AGI-2中达到24.4%。我们还展示了一个基于General Symbolics原则开发的软件开发环境，该环境在SWE-Bench Lite中达到62.3%的最佳精度。我们能够实现这些改进而无需任何微调或训练成本。我们的推理层旨在提供纯粹的性能提升，确保模型在推理任务上的准确性不会受到影响。我们认为，现有的方法最终会导致LLM性能的边际效应递减，需要开发新的推理技术。本技术报告概述了我们的方法并提供了CoreThink模型在密集推理用例中的可用性。 

---
# Ultra Strong Machine Learning: Teaching Humans Active Learning Strategies via Automated AI Explanations 

**Title (ZH)**: 超强大机器学习：通过自动化人工智能解释教学人类主动学习策略 

**Authors**: Lun Ai, Johannes Langer, Ute Schmid, Stephen Muggleton  

**Link**: [PDF](https://arxiv.org/pdf/2509.00961)  

**Abstract**: Ultra Strong Machine Learning (USML) refers to symbolic learning systems that not only improve their own performance but can also teach their acquired knowledge to quantifiably improve human performance. In this work, we present LENS (Logic Programming Explanation via Neural Summarisation), a neuro-symbolic method that combines symbolic program synthesis with large language models (LLMs) to automate the explanation of machine-learned logic programs in natural language. LENS addresses a key limitation of prior USML approaches by replacing hand-crafted explanation templates with scalable automated generation. Through systematic evaluation using multiple LLM judges and human validation, we demonstrate that LENS generates superior explanations compared to direct LLM prompting and hand-crafted templates. To investigate whether LENS can teach transferable active learning strategies, we carried out a human learning experiment across three related domains. Our results show no significant human performance improvements, suggesting that comprehensive LLM responses may overwhelm users for simpler problems rather than providing learning support. Our work provides a solid foundation for building effective USML systems to support human learning. The source code is available on: this https URL. 

**Abstract (ZH)**: 超強機器學習（USML）指的是不僅能改善自身性能還能將其獲得的知識有效地傳授给人類以量化改善人類 PERFORMANCE 的符號學習系統。在本工作中，我們提出了 LENS（通過神經摘要的邏輯程式解釋），一種結合符號程式合成和大型語言模型（LLMs）以自動以自然語言解釋機器學習邏輯程式的神經符號方法。LENS 通過將手動 Crafting 說明模板替換為可擴展的自動生成，解決了先前 USML 接口的重要 limitations。通過使用多個 LLM 判斷員進行系統評估并進行人的確認，我們證明了 LENS 生成的解釋優于直接 LLM 命令和手動 Crafting 模板生成的解釋。为了研究LENS是否可以传授可转移的主动学习策略，我们在三个相关领域进行了人类学习实验。我们的结果显示人类 performance 没有显著提升，这表明全面的 LLM 响应可能会让用户在解决简单问题时感到 Overwhelmed 而不是提供学习支持。我们的工作为构建有效支持人类学习的 USML 系统奠定了坚实基础。源代码可在以下网址获取：this https URL。 

---
# A Hybrid Ai Framework For Strategic Patent Portfolio Pruning: Integrating Learning To-Rank And Market Need Analysis For Technology Transfer Optimization 

**Title (ZH)**: 一种综合AI框架用于战略专利组合精简：结合学习排序和市场需要分析以优化技术转移 

**Authors**: Manish Verma, Vivek Sharma, Vishal Singh  

**Link**: [PDF](https://arxiv.org/pdf/2509.00958)  

**Abstract**: This paper introduces a novel, multi stage hybrid intelligence framework for pruning patent portfolios to identify high value assets for technology transfer. Current patent valuation methods often rely on retrospective indicators or manual, time intensive analysis. Our framework automates and deepens this process by combining a Learning to Rank (LTR) model, which evaluates patents against over 30 legal and commercial parameters, with a unique "Need-Seed" agent-based system. The "Need Agent" uses Natural Language Processing (NLP) to mine unstructured market and industry data, identifying explicit technological needs. Concurrently, the "Seed Agent" employs fine tuned Large Language Models (LLMs) to analyze patent claims and map their technological capabilities. The system generates a "Core Ontology Framework" that matches high potential patents (Seeds) to documented market demands (Needs), providing a strategic rationale for divestment decisions. We detail the architecture, including a dynamic parameter weighting system and a crucial Human in the-Loop (HITL) validation protocol, to ensure both adaptability and real-world credibility. 

**Abstract (ZH)**: 一种新颖的多阶段混合智能框架，用于精简专利组合以识别高价值资产，促进技术转移 

---
# UrbanInsight: A Distributed Edge Computing Framework with LLM-Powered Data Filtering for Smart City Digital Twins 

**Title (ZH)**: 城市洞察：一种基于大模型驱动数据过滤的分布式边缘计算框架，用于智能城市数字孪生 

**Authors**: Kishor Datta Gupta, Md Manjurul Ahsan, Mohd Ariful Haque, Roy George, Azmine Toushik Wasi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00936)  

**Abstract**: Cities today generate enormous streams of data from sensors, cameras, and connected infrastructure. While this information offers unprecedented opportunities to improve urban life, most existing systems struggle with scale, latency, and fragmented insights. This work introduces a framework that blends physics-informed machine learning, multimodal data fusion, and knowledge graph representation with adaptive, rule-based intelligence powered by large language models (LLMs). Physics-informed methods ground learning in real-world constraints, ensuring predictions remain meaningful and consistent with physical dynamics. Knowledge graphs act as the semantic backbone, integrating heterogeneous sensor data into a connected, queryable structure. At the edge, LLMs generate context-aware rules that adapt filtering and decision-making in real time, enabling efficient operation even under constrained resources. Together, these elements form a foundation for digital twin systems that go beyond passive monitoring to provide actionable insights. By uniting physics-based reasoning, semantic data fusion, and adaptive rule generation, this approach opens new possibilities for creating responsive, trustworthy, and sustainable smart infrastructures. 

**Abstract (ZH)**: 今天的城市产生了来自传感器、相机和互联基础设施的巨大数据流。虽然这些信息提供了前所未有的改善城市生活的机遇，但现有系统大多面临规模、延迟和碎片化的挑战。本文提出了一种结合物理知情机器学习、多模态数据融合、知识图谱表示以及由大型语言模型（LLMs）驱动的自适应规则推理的框架。物理知情方法将学习过程扎根于现实世界的约束，确保预测结果具有实际意义并符合物理动态。知识图谱作为语义骨干，将异构传感器数据整合到一个可查询的连接结构中。在边缘端，LLMs生成基于上下文的规则，实现实时的筛选和决策，即使在资源受限的情况下也能实现高效运行。这些元素共同构成了超越被动监控提供可操作洞察的数字孪生系统的基石。通过结合基于物理的推理、语义数据融合和自适应规则生成，这种方法为创建响应式、可信赖和可持续的智能基础设施开启了新的可能性。 

---
# SATQuest: A Verifier for Logical Reasoning Evaluation and Reinforcement Fine-Tuning of LLMs 

**Title (ZH)**: SATQuest: 逻辑推理评估与大规模语言模型fine-tuning强化验证器 

**Authors**: Yanxiao Zhao, Yaqian Li, Zihao Bo, Rinyoichi Takezoe, Haojia Hui, Mo Guang, Lei Ren, Xiaolin Qin, Kaiwen Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.00930)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated remarkable general reasoning capabilities. However, systematically evaluating and enhancing these reasoning capabilities is challenging due to the lack of controllable and scalable tools for fine-grained analysis. Existing benchmarks and datasets often lack the necessary variable control for multi-dimensional, systematic analysis and training, or have narrow problem types and formats. To address these limitations, we introduce SATQuest, a systematic verifier designed to evaluate and enhance logical reasoning in LLMs by generating diverse, Satisfiability-based logical reasoning problems directly from Conjunctive Normal Form (CNF) instances. SATQuest structures these problems along three orthogonal dimensions: instance scale, problem type, and question format, employing randomized, SAT-based problem generation and objective answer verification via PySAT. This design mitigates memorization issues, allows for nuanced insights into reasoning performance, and enables effective reinforcement fine-tuning. Our extensive evaluation of various LLMs using SATQuest identified significant limitations in their logical reasoning, particularly in generalizing beyond familiar mathematical formats. Furthermore, we show that reinforcement fine-tuning with SATQuest rewards substantially improves targeted task performance and generalizes to more complex instances, while highlighting remaining challenges in cross-format adaptation. Through these demonstrations, we showcase SATQuest's potential as a foundational tool and a valuable starting point for advancing LLM logical reasoning. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）展示了卓越的综合推理能力。然而，由于缺乏用于精细分析的可控且可扩展的工具，系统性评估和提升这些推理能力具有挑战性。现有的基准和数据集往往缺乏进行多维度、系统性分析和训练所需的变量控制，或者具有狭窄的问题类型和格式。为解决这些限制，我们引入了SATQuest，这是一种系统验证器，旨在通过直接从合取范式（CNF）实例生成多样化的可满足性基础逻辑推理问题来评估和提升LLMs的逻辑推理能力。SATQuest沿三个正交维度结构化这些问题：实例规模、问题类型和问题格式，采用随机化的SAT基础问题生成和基于PySAT的目标答案验证。该设计减轻了记忆问题，允许对推理性能进行细致的洞察，并使有效的强化微调成为可能。我们使用SATQuest对多种LLMs的广泛评估揭示了它们在逻辑推理方面的重要局限性，尤其是在超越熟悉的数学格式方面的一般化能力不足。此外，我们展示了使用SATQuest进行强化微调可以显著提高目标任务性能，并推广到更复杂的情况，同时指出了跨格式适应中的遗留挑战。通过这些演示，我们展示了SATQuest作为基础工具和提升LLMs逻辑推理有价值起点的潜力。 

---
# ChatCLIDS: Simulating Persuasive AI Dialogues to Promote Closed-Loop Insulin Adoption in Type 1 Diabetes Care 

**Title (ZH)**: ChatCLIDS: 模拟有说服力的AI对话以促进1型糖尿病护理中闭环胰岛素治疗的采纳 

**Authors**: Zonghai Yao, Talha Chafekar, Junda Wang, Shuo Han, Feiyun Ouyang, Junhui Qian, Lingxi Li, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00891)  

**Abstract**: Real-world adoption of closed-loop insulin delivery systems (CLIDS) in type 1 diabetes remains low, driven not by technical failure, but by diverse behavioral, psychosocial, and social barriers. We introduce ChatCLIDS, the first benchmark to rigorously evaluate LLM-driven persuasive dialogue for health behavior change. Our framework features a library of expert-validated virtual patients, each with clinically grounded, heterogeneous profiles and realistic adoption barriers, and simulates multi-turn interactions with nurse agents equipped with a diverse set of evidence-based persuasive strategies. ChatCLIDS uniquely supports longitudinal counseling and adversarial social influence scenarios, enabling robust, multi-dimensional evaluation. Our findings reveal that while larger and more reflective LLMs adapt strategies over time, all models struggle to overcome resistance, especially under realistic social pressure. These results highlight critical limitations of current LLMs for behavior change, and offer a high-fidelity, scalable testbed for advancing trustworthy persuasive AI in healthcare and beyond. 

**Abstract (ZH)**: 现实世界中1型糖尿病患者使用闭环胰岛素输送系统（CLIDS）的采用率仍然较低，这并非由于技术失败，而是由于多样的行为、心理社会和社交障碍。我们介绍了ChatCLIDS，这是首个用于严格评估LLM驱动的说服性对话以促进健康行为改变的标准基准。我们的框架包含一组经过专家验证的虚拟患者，每个患者都有临床依据的、异质化的个人资料和实际的采用障碍，并模拟了与配备多样化的证据基于说服策略的护士代理多轮互动。ChatCLIDS的独特之处在于支持纵向咨询和对抗性社会影响场景，从而实现稳健的多维度评估。我们的研究发现，虽然规模更大、更自省的LLM随着时间推移会调整策略，但所有模型在克服阻力方面都遇到困难，尤其是在现实的社会压力下。这些结果突显了当前LLM在促进行为改变方面的关键限制，并提供了一个高保真、可扩展的测试平台，用于推进医疗保健和其他领域中的可信说服性AI。 

---
# Aligning Reasoning LLMs for Materials Discovery with Physics-aware Rejection Sampling 

**Title (ZH)**: 基于物理意识拒绝采样的材料发现理性LLM对齐 

**Authors**: Lee Hyun, Sohee Yoon, Jinwoo Park, Sue In Chae, Seongeon Park, Jooyeon Ahn, Yebin Jung, Youjung Chung, Hogeun Chang, Myeonginn Kang, Jina Kim, Ho-Gyeong Kim, Myeonghun Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2509.00768)  

**Abstract**: AI-driven materials discovery that couples automated experimentation with algorithmic decision-making requires process aware recipe to property predictors that are accurate, calibrated, and physically admissible. We approach this as a reasoning problem with large reasoning models (LRMs). To instill reasoning capability into language models, we curate reasoning traces from a teacher model to train a student model. However, most training pipelines select reasoning traces using binary correctness or learned preference signals that poorly reflect physical admissibility. We introduce Physics-aware Rejection Sampling (PaRS), a training-time trace selection scheme that favors traces consistent with fundamental physics and numerically close to targets, with lightweight halting to control compute. We instantiate our framework with a large student model fine-tuned on traces synthesized by a larger teacher model, and evaluate under matched token budgets against various rejection sampling baselines. Our method improves accuracy and calibration, reduces physics-violation rates, and lowers sampling cost relative to baselines. These results indicate that modest, domain-aware constraints combined with trace-level selection provide a practical path toward reliable, efficient LRMs for process-aware property prediction and closed-loop materials design. 

**Abstract (ZH)**: 基于物理意识的拒绝采样在过程意识材料预测中的应用：结合自动化实验与算法决策的AI驱动材料发现要求具备物理可行的配方到性质预测器，且需准确、校准并符合物理原理。我们将其视为具有大规模推理能力的模型（LRM）推理问题。通过从教师模型中精心挑选推理轨迹来增强语言模型的推理能力，我们训练学生模型。然而，大多数训练管道使用二元正确性或学习偏好信号来选择推理轨迹，这些信号未能充分反映物理可行性。我们引入物理意识拒绝采样（PaRS），这是一种在训练时轨迹选择方案，倾向于选择与基本物理法则一致且数值上接近目标的轨迹，并通过轻量级终止控制计算量。我们通过一个在更大教师模型合成的轨迹上微调的大规模学生模型实例化该框架，并在匹配的令牌预算下与各种拒绝采样基线进行评估。我们的方法提高了准确性与校准度，降低了物理违背率，并相对于基线降低了采样成本。这些结果表明，结合适度的领域意识约束与轨迹级别选择是实现可靠、高效过程意识属性预测和闭环材料设计的大规模推理模型的实际途径。 

---
# Efficient Graph Understanding with LLMs via Structured Context Injection 

**Title (ZH)**: 利用结构化背景注入实现高效的图理解 

**Authors**: Govind Waghmare, Sumedh BG, Sonia Gupta, Srikanta Bedathur  

**Link**: [PDF](https://arxiv.org/pdf/2509.00740)  

**Abstract**: Large Language Models (LLMs) have shown strong capabilities in solving problems across domains, including graph-related tasks traditionally addressed by symbolic or algorithmic methods. In this work, we present a framework for structured context injection, where task-specific information is systematically embedded in the input to guide LLMs in solving a wide range of graph problems. Our method does not require fine-tuning of LLMs, making it cost-efficient and lightweight. We observe that certain graph reasoning tasks remain challenging for LLMs unless they are mapped to conceptually grounded representations. However, achieving such mappings through fine-tuning or repeated multi-step querying can be expensive and inefficient. Our approach offers a practical alternative by injecting structured context directly into the input, enabling the LLM to implicitly align the task with grounded conceptual spaces. We evaluate the approach on multiple graph tasks using both lightweight and large models, highlighting the trade-offs between accuracy and computational cost. The results demonstrate consistent performance improvements, showing that structured input context can rival or surpass more complex approaches. Our findings underscore the value of structured context injection as an effective and scalable strategy for graph understanding with LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在解决跨领域问题方面展现了强大的能力，包括传统的符号或算法方法处理的图相关任务。本文提出了一种结构化上下文注入框架，通过系统地将任务特定信息嵌入输入中以引导LLMs解决广泛的图问题。该方法无需对LLMs进行微调，使其成本效益高且轻量级。我们观察到，除非将某些图推理任务映射到概念性基础表示中，否则它们对LLMs仍然是具有挑战性的。然而，通过微调或重复多步查询来实现这样的映射可能是昂贵且低效的。我们的方法通过直接将结构化上下文注入输入提供了一种实际的替代方案，使LLMs能够隐式地将任务对齐到基础的概念空间。我们使用轻量级和大型模型对多种图任务进行了评估，突显了精度和计算成本之间的权衡。结果表明，结构化输入上下文可以与更复杂的方法相媲美或超越它们。我们的研究结果强调了结构化上下文注入作为使用LLMs进行图理解的有效且可扩展策略的价值。 

---
# OmniDPO: A Preference Optimization Framework to Address Omni-Modal Hallucination 

**Title (ZH)**: 全方位偏好优化框架以应对多模态幻觉 

**Authors**: Junzhe Chen, Tianshu Zhang, Shiyu Huang, Yuwei Niu, Chao Sun, Rongzhou Zhang, Guanyu Zhou, Lijie Wen, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00723)  

**Abstract**: Recently, Omni-modal large language models (OLLMs) have sparked a new wave of research, achieving impressive results in tasks such as audio-video understanding and real-time environment perception. However, hallucination issues still persist. Similar to the bimodal setting, the priors from the text modality tend to dominate, leading OLLMs to rely more heavily on textual cues while neglecting visual and audio information. In addition, fully multimodal scenarios introduce new challenges. Most existing models align visual or auditory modalities with text independently during training, while ignoring the intrinsic correlations between video and its corresponding audio. This oversight results in hallucinations when reasoning requires interpreting hidden audio cues embedded in video content. To address these challenges, we propose OmniDPO, a preference-alignment framework designed to mitigate hallucinations in OLLMs. Specifically, OmniDPO incorporates two strategies: (1) constructing text-preference sample pairs to enhance the model's understanding of audio-video interactions; and (2) constructing multimodal-preference sample pairs to strengthen the model's attention to visual and auditory information. By tackling both challenges, OmniDPO effectively improves multimodal grounding and reduces hallucination. Experiments conducted on two OLLMs demonstrate that OmniDPO not only effectively mitigates multimodal hallucinations but also significantly enhances the models' reasoning capabilities across modalities. All code and datasets will be released upon paper acceptance. 

**Abstract (ZH)**: 最近，全模态大型语言模型（OLLMs）引发了新的研究热潮，在音频-视频理解以及实时环境感知等任务中取得了显著成果。然而，幻觉问题仍然存在。类似二模态设置，文本模态的先验知识往往会占据主导地位，导致OLLMs更多依赖文本线索而忽视视觉和音频信息。此外，全模态场景引入了新的挑战。大多数现有模型在训练过程中独立地将视觉或听觉模态与文本对齐，而忽视了视频与其对应音频之间的内在关联。这种忽视导致在需要解释视频内容中嵌入的隐藏音频线索时产生幻觉。为了解决这些挑战，我们提出了一种偏好对齐框架OmniDPO，旨在减轻OLLMs中的幻觉。具体而言，OmniDPO采用了两种策略：（1）构建文本偏好样本对，增强模型对音频-视频交互的理解；（2）构建多模态偏好样本对，增强模型对视觉和听觉信息的关注。通过同时应对这两项挑战，OmniDPO有效提高了多模态 grounding 并降低了幻觉。实验在两个OLLMs上进行，结果表明OmniDPO不仅有效地减轻了多模态幻觉，还显著增强了模型在不同模态上的推理能力。论文接受后，所有代码和数据集将公开发布。 

---
# BALM-TSF: Balanced Multimodal Alignment for LLM-Based Time Series Forecasting 

**Title (ZH)**: BALM-TSF：平衡多模态对齐的时间序列预测方法 

**Authors**: Shiqiao Zhou, Holger Schöner, Huanbo Lyu, Edouard Fouché, Shuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00622)  

**Abstract**: Time series forecasting is a long-standing and highly challenging research topic. Recently, driven by the rise of large language models (LLMs), research has increasingly shifted from purely time series methods toward harnessing textual modalities to enhance forecasting performance. However, the vast discrepancy between text and temporal data often leads current multimodal architectures to over-emphasise one modality while neglecting the other, resulting in information loss that harms forecasting performance. To address this modality imbalance, we introduce BALM-TSF (Balanced Multimodal Alignment for LLM-Based Time Series Forecasting), a lightweight time series forecasting framework that maintains balance between the two modalities. Specifically, raw time series are processed by the time series encoder, while descriptive statistics of raw time series are fed to an LLM with learnable prompt, producing compact textual embeddings. To ensure balanced cross-modal context alignment of time series and textual embeddings, a simple yet effective scaling strategy combined with a contrastive objective then maps these textual embeddings into the latent space of the time series embeddings. Finally, the aligned textual semantic embeddings and time series embeddings are together integrated for forecasting. Extensive experiments on standard benchmarks show that, with minimal trainable parameters, BALM-TSF achieves state-of-the-art performance in both long-term and few-shot forecasting, confirming its ability to harness complementary information from text and time series. Code is available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的平衡多模态对齐时间序列forecasting框架（Balanced Multimodal Alignment for LLM-Based Time Series Forecasting） 

---
# Text-to-Layout: A Generative Workflow for Drafting Architectural Floor Plans Using LLMs 

**Title (ZH)**: 文本到布局：使用大型语言模型生成建筑平面图的工作流程 

**Authors**: Jayakrishna Duggempudi, Lu Gao, Ahmed Senouci, Zhe Han, Yunpeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00543)  

**Abstract**: This paper presents the development of an AI-powered workflow that uses Large Language Models (LLMs) to assist in drafting schematic architectural floor plans from natural language prompts. The proposed system interprets textual input to automatically generate layout options including walls, doors, windows, and furniture arrangements. It combines prompt engineering, a furniture placement refinement algorithm, and Python scripting to produce spatially coherent draft plans compatible with design tools such as Autodesk Revit. A case study of a mid-sized residential layout demonstrates the approach's ability to generate functional and structured outputs with minimal manual effort. The workflow is designed for transparent replication, with all key prompt specifications documented to enable independent implementation by other researchers. In addition, the generated models preserve the full range of Revit-native parametric attributes required for direct integration into professional BIM processes. 

**Abstract (ZH)**: 本文提出了一种基于AI的工作流，利用大型语言模型（LLMs）从自然语言提示中协助草拟建筑平面图。所提出的系统通过解释文本输入自动生成包括墙体、门窗和家具布局在内的布局选项。该系统结合了提示工程、家具布置细化算法和Python脚本，生成与Autodesk Revit等设计工具兼容的具有空间一致性的工作草图。通过一个中型住宅布局的案例研究，展示了该方法能够生成功能性和结构化输出，同时减少手动努力。该工作流设计为透明可复制，所有关键提示规范均被文档化，以便其他研究人员独立实施。此外，生成的模型保留了所有Revit原生的参数属性，可以直接集成到专业BIM流程中。 

---
# LLM-Assisted Iterative Evolution with Swarm Intelligence Toward SuperBrain 

**Title (ZH)**: 基于群智的大型语言模型辅助迭代进化 toward 超级大脑 

**Authors**: Li Weigang, Pedro Carvalho Brom, Lucas Ramson Siefert  

**Link**: [PDF](https://arxiv.org/pdf/2509.00510)  

**Abstract**: We propose a novel SuperBrain framework for collective intelligence, grounded in the co-evolution of large language models (LLMs) and human users. Unlike static prompt engineering or isolated agent simulations, our approach emphasizes a dynamic pathway from Subclass Brain to Superclass Brain: (1) A Subclass Brain arises from persistent, personalized interaction between a user and an LLM, forming a cognitive dyad with adaptive learning memory. (2) Through GA-assisted forward-backward evolution, these dyads iteratively refine prompts and task performance. (3) Multiple Subclass Brains coordinate via Swarm Intelligence, optimizing across multi-objective fitness landscapes and exchanging distilled heuristics. (4) Their standardized behaviors and cognitive signatures integrate into a Superclass Brain, an emergent meta-intelligence capable of abstraction, generalization and self-improvement. We outline the theoretical constructs, present initial implementations (e.g., UAV scheduling, KU/KI keyword filtering) and propose a registry for cross-dyad knowledge consolidation. This work provides both a conceptual foundation and an architectural roadmap toward scalable, explainable and ethically aligned collective AI. 

**Abstract (ZH)**: 我们提出了一种基于大语言模型（LLM）和人类用户共生进化的新型SuperBrain框架，用于集体智能。该方法强调从子类脑到超类脑的动态路径：（1）子类脑通过用户与LLM的持续个性化交互形成具有适应性学习记忆的认知二元体。（2）通过GA辅助的正向-反向演化，这些二元体逐步优化提示和任务表现。（3）多个子类脑通过 swarm 智能协调，优化多目标适应性景观并交换提炼的启发式方法。（4）其标准化行为和认知特征整合为超类脑，这是一种能够进行抽象、泛化和自我改进的新兴元智能。本文阐述了理论构架，展示了初步实现（如无人机调度、关键词过滤），并提出了跨二元体知识整合的登记制度。本工作为可扩展、可解释和伦理对齐的集体人工智能提供了概念基础和架构蓝图。 

---
# SIGMUS: Semantic Integration for Knowledge Graphs in Multimodal Urban Spaces 

**Title (ZH)**: SIGMUS: 多模态城市空间中知识图谱的语义集成 

**Authors**: Brian Wang, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2509.00287)  

**Abstract**: Modern urban spaces are equipped with an increasingly diverse set of sensors, all producing an abundance of multimodal data. Such multimodal data can be used to identify and reason about important incidents occurring in urban landscapes, such as major emergencies, cultural and social events, as well as natural disasters. However, such data may be fragmented over several sources and difficult to integrate due to the reliance on human-driven reasoning for identifying relationships between the multimodal data corresponding to an incident, as well as understanding the different components which define an incident. Such relationships and components are critical to identifying the causes of such incidents, as well as producing forecasting the scale and intensity of future incidents as they begin to develop. In this work, we create SIGMUS, a system for Semantic Integration for Knowledge Graphs in Multimodal Urban Spaces. SIGMUS uses Large Language Models (LLMs) to produce the necessary world knowledge for identifying relationships between incidents occurring in urban spaces and data from different modalities, allowing us to organize evidence and observations relevant to an incident without relying and human-encoded rules for relating multimodal sensory data with incidents. This organized knowledge is represented as a knowledge graph, organizing incidents, observations, and much more. We find that our system is able to produce reasonable connections between 5 different data sources (new article text, CCTV images, air quality, weather, and traffic measurements) and relevant incidents occurring at the same time and location. 

**Abstract (ZH)**: 现代城市空间装备了日益多样化的传感器，产生了大量多模态数据。这种多模态数据可以用来识别和推理城市景观中发生的重要事件，如重大紧急情况、文化和社交活动以及自然灾害。然而，这些数据可能分散在不同的来源中，由于依赖人工推理来识别与事件相关的多模态数据之间的关系以及理解定义事件的不同组成部分，因此难以集成。这些关系和组成部分对于识别事件的原因以及预测事件的规模和强度至关重要。在本文中，我们创建了SIGMUS系统，用于多模态城市空间中的知识图谱语义集成。SIGMUS利用大语言模型（LLMs）生成必要的世界知识，以识别城市空间中发生的事件与不同模态数据之间关系，从而无需依赖人工编码的规则来关联多模态感官数据与事件。组织化的知识被表示为知识图谱，组织事件、观察以及其他内容。我们发现，我们的系统能够合理地将5种不同数据源（新闻文章文本、闭路电视图像、空气质量、天气和交通测量）与同一时间同一地点发生的相关事件连接起来。 

---
# SHERPA: A Model-Driven Framework for Large Language Model Execution 

**Title (ZH)**: SHERPA: 一种基于模型的大型语言模型执行框架 

**Authors**: Boqi Chen, Kua Chen, José Antonio Hernández López, Gunter Mussbacher, Dániel Varró, Amir Feizpour  

**Link**: [PDF](https://arxiv.org/pdf/2509.00272)  

**Abstract**: Recently, large language models (LLMs) have achieved widespread application across various fields. Despite their impressive capabilities, LLMs suffer from a lack of structured reasoning ability, particularly for complex tasks requiring domain-specific best practices, which are often unavailable in the training data. Although multi-step prompting methods incorporating human best practices, such as chain-of-thought and tree-of-thought, have gained popularity, they lack a general mechanism to control LLM behavior. In this paper, we propose SHERPA, a model-driven framework to improve the LLM performance on complex tasks by explicitly incorporating domain-specific best practices into hierarchical state machines. By structuring the LLM execution processes using state machines, SHERPA enables more fine-grained control over their behavior via rules or decisions driven by machine learning-based approaches, including LLMs. We show that SHERPA is applicable to a wide variety of tasks-specifically, code generation, class name generation, and question answering-replicating previously proposed approaches while further improving the performance. We demonstrate the effectiveness of SHERPA for the aforementioned tasks using various LLMs. Our systematic evaluation compares different state machine configurations against baseline approaches without state machines. Results show that integrating well-designed state machines significantly improves the quality of LLM outputs, and is particularly beneficial for complex tasks with well-established human best practices but lacking data used for training LLMs. 

**Abstract (ZH)**: Recently,大型语言模型（LLMs）已经在多个领域取得了广泛应用。尽管它们具有令人印象深刻的性能，但LLMs在处理需要特定领域最佳实践的复杂任务时缺乏结构化的推理能力，而这些最佳实践往往在训练数据中不可用。虽然结合了人类最佳实践的多步提示方法，如链式思考和树状思考，已经受到欢迎，但仍缺乏一个通用机制来控制LLM的行为。在本文中，我们提出了SHERPA，这是一种通过显式将领域特定的最佳实践集成到分层状态机中，以提高LLMs在复杂任务上的性能的模型驱动框架。通过使用状态机结构化LLM的执行过程，SHERPA能够通过基于机器学习的方法驱动的规则或决策对它们的行为进行更精细的控制。我们展示了SHERPA适用于代码生成、类名生成和问答等各种任务，可以在重复先前提出的策略的同时进一步提高性能。我们使用各种LLM展示了SHERPA在上述任务上的有效性。系统评估了不同的状态机配置与没有状态机的基线方法的性能差异。结果表明，结合设计良好的状态机显著提高了LLM输出的质量，并且特别有利于那些有成熟人类最佳实践但缺乏用于训练LLM的数据的复杂任务。 

---
# Instruction-Level Weight Shaping: A Framework for Self-Improving AI Agents 

**Title (ZH)**: 指令级权重塑造：自改善AI代理的框架 

**Authors**: Rimom Costa  

**Link**: [PDF](https://arxiv.org/pdf/2509.00251)  

**Abstract**: Large language models (LLMs) are fluent but largely static after pre-training; new or shifting knowledge is typically added with retrieval-augmented generation (RAG) or fine-tuning. RAG raises latency and engineering overhead and often fails to integrate facts; prompt engineering is brittle and can conflict with prior knowledge; fine-tuning is costly and risks catastrophic forgetting. We propose Instruction-Level Weight Shaping (ILWS): curated system instructions act as external, auditable pseudo-parameters updated after each session via reflection and user feedback. A Reflection Engine inspects conversation traces, diagnoses reasoning successes and failures, and proposes typed deltas $\Delta K=(\Delta S,\Delta U,\Delta T)$ over instructions, user preferences, and tools. Deltas are version-controlled, evaluated with a sliding window of 1-5 star ratings, auto-repaired on first failure, and rolled back on repeated failure. When an edit budget crosses a threshold, the agent compiles a rating-weighted synthetic set and distills matured instruction-space gains into parameters, converting prompt-space improvements into weight-space without downtime. ILWS makes explicit the low-rank shaping induced by context in transformer blocks, preserves governance, and removes per-call retrieval. In enterprise support it increased throughput 2.4-5.0x and cut audited hallucinations by about 80% versus a frozen baseline. In an Adobe Commerce Cloud proof of concept "L0 Support", it achieved 4-5x more tickets per hour and about 80% lower time per ticket, with autonomous instruction updates and optional tool synthesis. Because ILWS operates at the instruction layer until controlled distillation, it generalizes to dynamic domains (legal, medical, engineering) requiring adaptive reasoning, tool creation, and low-latency deployment. 

**Abstract (ZH)**: 指令级权重塑造（ILWS）：面向动态知识领域的系统指令适应方法 

---
# Universal Deep Research: Bring Your Own Model and Strategy 

**Title (ZH)**: 自选模型与策略的通用深度研究 

**Authors**: Peter Belcak, Pavlo Molchanov  

**Link**: [PDF](https://arxiv.org/pdf/2509.00244)  

**Abstract**: Deep research tools are among the most impactful and most commonly encountered agentic systems today. We observe, however, that each deep research agent introduced so far is hard-coded to carry out a particular research strategy using a fixed choice of tools. We introduce Universal Deep Research (UDR), a generalist agentic system that wraps around any language model and enables the user to create, edit, and refine their own entirely custom deep research strategies without any need for additional training or finetuning. To showcase the generality of our system, we equip UDR with example minimal, expansive, and intensive research strategies, and provide a user interface to facilitate experimentation with the system. 

**Abstract (ZH)**: 通用深度研究（UDR）是一种能够围绕任何语言模型运行的一般主义代理系统，使用户能够创建、编辑和细化完全自定义的深度研究策略，而无需额外的训练或微调。为了展示系统的通用性，我们为UDR配备了示例最小、扩展性和密集性研究策略，并提供了一个用户界面以方便用户对该系统的实验。 

---
# Know When to Explore: Difficulty-Aware Certainty as a Guide for LLM Reinforcement Learning 

**Title (ZH)**: 知所探索：基于难度的认知不确定性指导大语言模型强化学习 

**Authors**: Ang Li, Zhihang Yuan, Yang Zhang, Shouda Liu, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00125)  

**Abstract**: Reinforcement Learning with Verifiable Feedback (RLVF) has become a key technique for enhancing the reasoning abilities of Large Language Models (LLMs). However, its reliance on sparse, outcome based rewards, which only indicate if a final answer is correct or not, fails to provide granular guidance on the reasoning process itself. This limitation hinders efficient learning, as the model cannot distinguish between high quality and inefficient solutions, nor can it learn effectively from different types of failures. To address this, we observe that an LLMs self-certainty often correlates with task difficulty and solution quality. We introduce Difficulty Aware Certainty guided Exploration (DACE), a novel RL algorithm that leverages this insight to dynamically balance the exploration exploitation trade-off. DACE assesses task difficulty online based on the policys success rate. It then uses this signal to modulate an intrinsic reward: for difficult tasks where the model is struggling, DACE encourages exploration by penalizing high certainty; for easier tasks, it encourages learning efficiency by rewarding high certainty. Experiments on challenging mathematical reasoning benchmarks (AIME, MATH) show that DACE significantly outperforms strong baselines. The DACE-trained models not only achieve higher accuracy but also demonstrate more robust performance when scaling test-time compute, validating that our adaptive approach fosters effective exploration without sacrificing precision. 

**Abstract (ZH)**: 可验证反馈增强学习 (Reinforcement Learning with Verifiable Feedback, RLVF) 已成为提升大语言模型 (Large Language Models, LLMs) 推理能力的关键技术。然而，其依赖于基于最终结果的稀疏奖励，只能指示答案是否正确，无法提供推理过程中的详细指导。这一限制阻碍了高效学习，因为模型无法区分高质量和低效的解决方案，也无法从不同类型的失败中有效学习。为解决这一问题，我们观察到大语言模型的自我置信度往往与任务难度和解决方案质量相关。我们提出了一个基于此洞察的新型强化学习算法——难度感知置信引导探索 (Difficulty Aware Certainty guided Exploration, DACE)，该算法利用这一知识动态平衡探索与利用的权衡。DACE 根据策略的成功率在线评估任务难度，并使用此信号调节内在奖励：在难题上，当模型挣扎时，DACE 鼓励探索并通过惩罚高置信度来实现；在较简单任务上，它通过奖励高置信度来促进高效学习。实验结果显示，DACE 在具有挑战性的数学推理基准测试 (AIME, MATH) 上显著优于强基线。训练后的 DACE 模型不仅准确度更高，而且在测试时计算量增加时表现出更稳健的性能，验证了我们的自适应方法能够在不牺牲精度的情况下促进有效的探索。 

---
# Ensemble Debates with Local Large Language Models for AI Alignment 

**Title (ZH)**: 本地大规模语言模型ensemble辩论以实现AI对齐 

**Authors**: Ephraiem Sarabamoun  

**Link**: [PDF](https://arxiv.org/pdf/2509.00091)  

**Abstract**: As large language models (LLMs) take on greater roles in high-stakes decisions, alignment with human values is essential. Reliance on proprietary APIs limits reproducibility and broad participation. We study whether local open-source ensemble debates can improve alignmentoriented reasoning. Across 150 debates spanning 15 scenarios and five ensemble configurations, ensembles outperform single-model baselines on a 7-point rubric (overall: 3.48 vs. 3.13), with the largest gains in reasoning depth (+19.4%) and argument quality (+34.1%). Improvements are strongest for truthfulness (+1.25 points) and human enhancement (+0.80). We provide code, prompts, and a debate data set, providing an accessible and reproducible foundation for ensemble-based alignment evaluation. 

**Abstract (ZH)**: 大型语言模型在高风险决策中扮演更大角色时，与人类价值观的对齐至关重要。依赖专有API限制了再现性和广泛参与。我们研究本地开源集成辩论是否能改进导向对齐的推理。在涉及15种情景和五种集成配置的150场辩论中，集成优于单一模型基线（总体得分：3.48 vs. 3.13），其中推理深度提高了19.4%，论质量提高了34.1%。改进最显著的是真实性（提高1.25分）和人类增强（提高0.80分）。我们提供了代码、提示和辩论数据集，为基于集成的对齐评估提供了可访问和可再现的基础。 

---
# Beyond Memorization: Reasoning-Driven Synthesis as a Mitigation Strategy Against Benchmark Contamination 

**Title (ZH)**: 超越记忆：基于推理的合成作为一种缓解基准污染的策略 

**Authors**: Terry Jingchen Zhang, Gopal Dev, Ning Wang, Nicole Ni, Wenyuan Jiang, Yinya Huang, Bernhard Schölkopf, Mrinmaya Sachan, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00072)  

**Abstract**: Capability evaluation of large language models (LLMs) is increasingly shadowed by rising concerns of data contamination that cast doubts on whether static benchmarks measure genuine reasoning or mere memorization. We present an empirical study using an infinitely scalable framework to synthesize research-level QA directly from arXiv papers, harnessing the natural temporal structure of research publications where performance decay after knowledge cutoffs may indicate potential contamination. We evaluated 4 frontier model represented by 2 models of different knowledge cutoff dates per family on 1,643 multi-step reasoning questions synthesized from 20,277 arXiv papers stratified over 26 months, covering at least 6 months before and after all cutoff dates. Our results consistently showed a lack of significant performance decay near knowledge cutoff dates for models of various sizes, developers, and release dates. We further performed a comparative analysis with previous longitudinal studies that reported significant post-cutoff performance decay using directly retrieved questions based on public data. we hypothesize that the multi-step reasoning required by our synthesis pipeline offered additional complexity that goes deeper than shallow memorization, which effectively serves a mitigation strategy against benchmark contamination. We fully open source our code and dataset to aid reproducibility and advocate for a paradigm shift that prioritize reasoning-driven synthesis to construct benchmarks over simply collecting newly released questions periodically. 

**Abstract (ZH)**: 大规模语言模型能力评估日益受到数据污染问题的阴影，这引发了质疑静态基准测度的是真正的推理能力还是简单的记忆。我们提出了一项基于无限扩展框架的实证研究，直接从arXiv论文合成研究级问答，利用研究出版物的自然时间结构，其中性能在知识截止后的衰减可能表明潜在的数据污染。我们评估了来自26个月、至少涵盖所有截止日期前后6个月的20,277篇arXiv论文合成的1,643个多步推理问题，评估了2个不同知识截止日期的4个前沿模型。结果显示，不同规模、开发者和发布时间的模型在知识截止日期附近并未表现出显著的性能衰减。我们进一步与之前基于公开数据直接检索问题的纵向研究进行了对比分析，这些研究报告了显著的知识截止后性能衰减。我们假设，合成管道所需的多步推理提供了比浅层记忆更深层次的复杂性，有效抵消了基准测试中的数据污染风险。我们完全开源了代码和数据集，以促进可重复性，并倡导一种以推理驱动合成构建基准的范式转变，而不是定期收集新发布的测试问题。 

---
# Surrogate Benchmarks for Model Merging Optimization 

**Title (ZH)**: 代理基准模型合并优化 

**Authors**: Rio Akizuki, Yuya Kudo, Nozomu Yoshinari, Yoichi Hirose, Toshiyuki Nishimoto, Kento Uchida, Shinichi Shirakawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.02555)  

**Abstract**: Model merging techniques aim to integrate the abilities of multiple models into a single model. Most model merging techniques have hyperparameters, and their setting affects the performance of the merged model. Because several existing works show that tuning hyperparameters in model merging can enhance the merging outcome, developing hyperparameter optimization algorithms for model merging is a promising direction. However, its optimization process is computationally expensive, particularly in merging LLMs. In this work, we develop surrogate benchmarks for optimization of the merging hyperparameters to realize algorithm development and performance comparison at low cost. We define two search spaces and collect data samples to construct surrogate models to predict the performance of a merged model from a hyperparameter. We demonstrate that our benchmarks can predict the performance of merged models well and simulate optimization algorithm behaviors. 

**Abstract (ZH)**: 模型合并技术旨在将多个模型的能力整合到一个模型中。大多数模型合并技术包含超参数，这些超参数的设置会影响合并模型的性能。由于现有的一些研究表明，在模型合并中调整超参数可以提升合并效果，因此开发模型合并的超参数优化算法是一个有前途的方向。然而，其优化过程在合并大语言模型（LLMs）时计算成本较高。在这项工作中，我们开发了代理基准来优化合并超参数，以在低成本下实现算法开发和性能比较。我们定义了两个搜索空间并收集数据样本来构建代理模型，以预测超参数对合并模型性能的影响。我们证明我们的基准可以很好地预测合并模型的性能，并模拟优化算法的行为。 

---
# FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training 

**Title (ZH)**: FLM-Audio：自然独白提升原生全双工聊天机器人 via 双任务训练 

**Authors**: Yiqun Yao, Xiang Li, Xin Jiang, Xuezhi Fang, Naitong Yu, Wenjia Ma, Aixin Sun, Yequan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02521)  

**Abstract**: Full-duplex dialog models are designed to listen and speak simultaneously with rapid responses to fast-changing user input. Among existing approaches, native full-duplex models merges different channels (e.g. listen and speak) in a single time step, overcoming the high response latency inherent to time-division multiplexing time-division multiplexing (TDM) alternatives. Yet, a key challenge remains: aligning textual monologues with audio streams that operate at different bitrates. The prevailing solution relies on word-level alignment, but this can degrade the language ability of large pre-trained models. Moreover, it requires highly accurate timestamps for every token, which introduces cascading errors and increases pre-processing costs. In this paper, we propose textual monologues in continuous tokens sequence, namely "natural" monologues, which mimics humanoid cognitive behavior in dialogs. For temporal alignment, we alternate the position of the natural monologue - leading or trailing the audio - across different training stages. This "dual" training paradigm proves highly effective in building FLM-Audio, our 7B spoken dialog model that demonstrates superior responsiveness, duplexity, and chatting experiences, as confirmed by experimental results. 

**Abstract (ZH)**: 全双工对话模型旨在同时监听和讲话，并能快速响应快速变化的用户输入。现有方法中，原生全双工模型在同一时间步内合并不同的通道（例如监听和讲话），从而克服了时分多路复用（TDM）替代方法固有的高响应 latency。然而，一个关键挑战依然存在：对不同比特率的音频流进行文本独白的时间对齐。当前的解决方案依赖于字级对齐，但这会削弱大型预训练模型的语言能力。此外，这还需要每个tokens的高精度时间戳，这会导致级联错误并增加预处理成本。在本文中，我们提出了连续token序列的文本独白，即“自然”独白，这模仿了类人认知行为在对话中的表现。为时间对齐，我们在不同的训练阶段交替“自然”独白的音频位置（领先或滞后）。这种“双轨”训练范式在构建我们的7B声道对话模型FLM-Audio方面证明了高度有效性，该模型在响应性、全双工性和对话体验方面表现出优越性能，实验结果予以证实。 

---
# Contemporary Agent Technology: LLM-Driven Advancements vs Classic Multi-Agent Systems 

**Title (ZH)**: 当代智能体技术：基于LLM的进步与经典多智能体系统比较 

**Authors**: Costin Bădică, Amelia Bădică, Maria Ganzha, Mirjana Ivanović, Marcin Paprzycki, Dan Selişteanu, Zofia Wrona  

**Link**: [PDF](https://arxiv.org/pdf/2509.02515)  

**Abstract**: This contribution provides our comprehensive reflection on the contemporary agent technology, with a particular focus on the advancements driven by Large Language Models (LLM) vs classic Multi-Agent Systems (MAS). It delves into the models, approaches, and characteristics that define these new systems. The paper emphasizes the critical analysis of how the recent developments relate to the foundational MAS, as articulated in the core academic literature. Finally, it identifies key challenges and promising future directions in this rapidly evolving domain. 

**Abstract (ZH)**: 本文提供了我们对当代代理技术的全面反思，重点关注由大型语言模型（LLM）驱动的进展与经典多代理系统（MAS）的对比。文章深入探讨了这些新系统的模型、方法及其特性。论文强调了对近期发展与核心学术文献中提出的经典MAS基础之间的关键分析。最后，本文指出了这一快速发展的领域中的关键挑战和有前景的未来发展方向。 

---
# Top-H Decoding: Adapting the Creativity and Coherence with Bounded Entropy in Text Generation 

**Title (ZH)**: Top-H解码：在文本生成中适应有界熵的创造力和连贯性 

**Authors**: Erfan Baghaei Potraghloo, Seyedarmin Azizi, Souvik Kundu, Massoud Pedram  

**Link**: [PDF](https://arxiv.org/pdf/2509.02510)  

**Abstract**: Large language models (LLMs), despite their impressive performance across a wide range of tasks, often struggle to balance two competing objectives in open-ended text generation: fostering diversity and creativity while preserving logical coherence. Existing truncated sampling techniques, including temperature scaling, top-\$p\$ (nucleus) sampling, and min-\$p\$ sampling, aim to manage this trade-off. However, they exhibit limitations, particularly in the effective incorporation of the confidence of the model into the corresponding sampling strategy. For example, min-\$p\$ sampling relies on a single top token as a heuristic for confidence, eventually underutilizing the information of the probability distribution. Toward effective incorporation of the confidence of the model, in this paper, we present **top-H** decoding. We first establish the theoretical foundation of the interplay between creativity and coherence in truncated sampling by formulating an **entropy-constrained minimum divergence** problem. We then prove this minimization problem to be equivalent to an **entropy-constrained mass maximization** (ECMM) problem, which is NP-hard. Finally, we present top-H decoding, a computationally efficient greedy algorithm to solve the ECMM problem. Extensive empirical evaluations demonstrate that top-H outperforms the state-of-the-art (SoTA) alternative of min-\$p\$ sampling by up to **25.63%** on creative writing benchmarks, while maintaining robustness on question-answering datasets such as GPQA, GSM8K, and MT-Bench. Additionally, an *LLM-as-judge* evaluation confirms that top-H indeed produces coherent outputs even at higher temperatures, where creativity is especially critical. In summary, top-H advances SoTA in open-ended text generation and can be *easily integrated* into creative writing applications. The code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）尽管在广泛的任务上表现出色，但在开放生成文本时常常难以同时平衡促进多样性和创造性与保持逻辑连贯性这两项竞争性目标。现有的截断采样技术，包括温度调整、top-\$p\$（核采样）和min-\$p\$采样，旨在管理这种权衡。然而，它们在有效地将模型的信心纳入相应的采样策略中时存在局限性。例如，min-\$p\$采样依赖于单个顶级令牌作为信心的启发式方法，最终未能充分利用概率分布中的信息。为了有效地将模型的信心纳入采样策略中，本文提出了一种名为top-H解码的方法。我们首先通过形式化一个熵约束最小偏离问题来建立截断采样中创造性和一致性之间相互作用的理论基础。然后，我们将此最小化问题证明为一个熵约束质量最大化（ECMM）问题，该问题是NP难的。最后，我们提出了top-H解码算法，这是一种计算高效的贪心算法，用于求解ECMM问题。广泛的实验证明，top-H在创造性写作基准上优于min-\$p\$采样的最新替代方案（SoTA），可提高多达25.63%，同时在GPQA、GSM8K和MT-Bench等问答数据集中保持稳健性。此外，LLM作为评判者的评估证实，top-H确实在高温下生成连贯的输出，此时创造性尤其重要。总之，top-H推动了开放生成文本生成的最新成果，并且可以轻松集成到创造性写作应用中。代码可在以下链接获取。 

---
# MoSEs: Uncertainty-Aware AI-Generated Text Detection via Mixture of Stylistics Experts with Conditional Thresholds 

**Title (ZH)**: MoSEs: 具有条件阈值的风格专家混合模型的不确定性感知AI生成文本检测 

**Authors**: Junxi Wu, Jinpeng Wang, Zheng Liu, Bin Chen, Dongjian Hu, Hao Wu, Shu-Tao Xiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02499)  

**Abstract**: The rapid advancement of large language models has intensified public concerns about the potential misuse. Therefore, it is important to build trustworthy AI-generated text detection systems. Existing methods neglect stylistic modeling and mostly rely on static thresholds, which greatly limits the detection performance. In this paper, we propose the Mixture of Stylistic Experts (MoSEs) framework that enables stylistics-aware uncertainty quantification through conditional threshold estimation. MoSEs contain three core components, namely, the Stylistics Reference Repository (SRR), the Stylistics-Aware Router (SAR), and the Conditional Threshold Estimator (CTE). For input text, SRR can activate the appropriate reference data in SRR and provide them to CTE. Subsequently, CTE jointly models the linguistic statistical properties and semantic features to dynamically determine the optimal threshold. With a discrimination score, MoSEs yields prediction labels with the corresponding confidence level. Our framework achieves an average improvement 11.34% in detection performance compared to baselines. More inspiringly, MoSEs shows a more evident improvement 39.15% in the low-resource case. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型的快速进展加剧了公众对其潜在滥用的关注。因此，建立可信赖的AI生成文本检测系统至关重要。现有方法忽视了风格化建模，主要依赖于静态阈值，极大地限制了检测性能。本文提出了一种混合风格专家（MoSEs）框架，通过条件阈值估计实现风格化的不确定性量化。MoSEs包含三个核心组件，即风格参考库（SRR）、风格感知路由器（SAR）和条件阈值估计器（CTE）。对于输入文本，SRR可以激活SRR中的适当参考数据，并将其提供给CTE。随后，CTE联合建模语言统计属性和语义特征，以动态确定最优阈值。通过鉴别分数，MoSEs生成具有相应置信水平的预测标签。与基线相比，我们的框架在检测性能上平均提高了11.34%。更令人鼓舞的是，在低资源情况下，MoSEs的表现提升更为显著，达到了39.15%。我们的代码可在以下链接获取。 

---
# MLP-Offload: Multi-Level, Multi-Path Offloading for LLM Pre-training to Break the GPU Memory Wall 

**Title (ZH)**: MLP-卸载：LLM预训练的多层多路径卸载以突破GPU内存墙 

**Authors**: Avinash Maurya, M. Mustafa Rafique, Franck Cappello, Bogdan Nicolae  

**Link**: [PDF](https://arxiv.org/pdf/2509.02480)  

**Abstract**: Training LLMs larger than the aggregated memory of multiple GPUs is increasingly necessary due to the faster growth of LLM sizes compared to GPU memory. To this end, multi-tier host memory or disk offloading techniques are proposed by state of art. Despite advanced asynchronous multi-tier read/write strategies, such offloading strategies result in significant I/O overheads in the critical path of training, resulting in slower iterations. To this end, we propose MLP-Offload, a novel multi-level, multi-path offloading engine specifically designed for optimizing LLM training on resource-constrained setups by mitigating I/O bottlenecks. We make several key observations that drive the design of MLP-Offload, such as I/O overheads during the update dominate the iteration time; I/O bandwidth of the third-level remote storage tier remains unutilized; and, contention due to concurrent offloading amplifies I/O bottlenecks. Driven by these insights, we design and implement MLP-Offload to offload the optimizer states across multiple tiers in a cache-efficient and concurrency-controlled fashion to mitigate I/O bottlenecks during the backward and update phases. Evaluations on models up to 280B parameters shows that MLP-Offload achieves 2.5$\times$ faster iterations compared to the state-of-the-art LLM training runtimes. 

**Abstract (ZH)**: MLP-Offload：一种针对资源受限设置的多级多路径卸载引擎，用于优化大模型训练 

---
# Do LLMs Adhere to Label Definitions? Examining Their Receptivity to External Label Definitions 

**Title (ZH)**: 大规模语言模型遵循标签定义吗？探究其对外部标签定义的接受度 

**Authors**: Seyedali Mohammadi, Bhaskara Hanuma Vedula, Hemank Lamba, Edward Raff, Ponnurangam Kumaraguru, Francis Ferraro, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2509.02452)  

**Abstract**: Do LLMs genuinely incorporate external definitions, or do they primarily rely on their parametric knowledge? To address these questions, we conduct controlled experiments across multiple explanation benchmark datasets (general and domain-specific) and label definition conditions, including expert-curated, LLM-generated, perturbed, and swapped definitions. Our results reveal that while explicit label definitions can enhance accuracy and explainability, their integration into an LLM's task-solving processes is neither guaranteed nor consistent, suggesting reliance on internalized representations in many cases. Models often default to their internal representations, particularly in general tasks, whereas domain-specific tasks benefit more from explicit definitions. These findings underscore the need for a deeper understanding of how LLMs process external knowledge alongside their pre-existing capabilities. 

**Abstract (ZH)**: 大型语言模型是真正融合外部定义，还是主要依赖其参数知识？为回答这些问题，我们在多个解释基准数据集（通用和领域特定）以及标签定义条件下进行控制实验，包括专家策展、模型生成、扰动和替换定义。研究结果表明，虽然明确的标签定义可以提高准确性和可解释性，但它们融入大型语言模型的任务解决过程并不总是不可避免且一致的，这在很多情况下表明模型依赖于其内部表示，尤其是在通用任务中，而特定领域任务则更多地受益于明确的定义。这些发现强调了需要更深入地理解大型语言模型如何结合外部知识与其现有能力。 

---
# A Survey: Towards Privacy and Security in Mobile Large Language Models 

**Title (ZH)**: 隐私与安全 towards 移动大型语言模型的研究 

**Authors**: Honghui Xu, Kaiyang Li, Wei Chen, Danyang Zheng, Zhiyuan Li, Zhipeng Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.02411)  

**Abstract**: Mobile Large Language Models (LLMs) are revolutionizing diverse fields such as healthcare, finance, and education with their ability to perform advanced natural language processing tasks on-the-go. However, the deployment of these models in mobile and edge environments introduces significant challenges related to privacy and security due to their resource-intensive nature and the sensitivity of the data they process. This survey provides a comprehensive overview of privacy and security issues associated with mobile LLMs, systematically categorizing existing solutions such as differential privacy, federated learning, and prompt encryption. Furthermore, we analyze vulnerabilities unique to mobile LLMs, including adversarial attacks, membership inference, and side-channel attacks, offering an in-depth comparison of their effectiveness and limitations. Despite recent advancements, mobile LLMs face unique hurdles in achieving robust security while maintaining efficiency in resource-constrained environments. To bridge this gap, we propose potential applications, discuss open challenges, and suggest future research directions, paving the way for the development of trustworthy, privacy-compliant, and scalable mobile LLM systems. 

**Abstract (ZH)**: 移动大型语言模型（LLMs）正凭借其移动环境下的高级自然语言处理能力，极大地革新了医疗、金融和教育等多个领域。然而，这些模型在移动和边缘环境中的部署引入了与之资源密集度高及处理数据的敏感性相关的重大隐私和安全挑战。本文综述了移动LLMs所面临的隐私和安全问题，系统地分类了现有的解决方案，如差分隐私、联邦学习和提示加密。此外，我们分析了移动LLMs特有的脆弱性，包括对抗性攻击、成员推理攻击和侧信道攻击，并深入比较了它们的有效性和局限性。尽管取得了近期的进步，移动LLMs在资源受限环境中实现稳健安全性和效率性仍面临独特挑战。为解决这一问题，本文提出了潜在的应用、讨论了开放性的挑战，并提出了未来研究方向，为开发可信赖、符合隐私要求且可扩展的移动LLM系统指明了方向。 

---
# Poisoned at Scale: A Scalable Audit Uncovers Hidden Scam Endpoints in Production LLMs 

**Title (ZH)**: 大规模中毒：一种可扩展的审核揭示生产中的隐藏诈骗端点 

**Authors**: Zhiyang Chen, Tara Saba, Xun Deng, Xujie Si, Fan Long  

**Link**: [PDF](https://arxiv.org/pdf/2509.02372)  

**Abstract**: Large Language Models (LLMs) have become critical to modern software development, but their reliance on internet datasets for training introduces a significant security risk: the absorption and reproduction of malicious content. To evaluate this threat, this paper introduces a scalable, automated audit framework that synthesizes innocuous, developer-style prompts from known scam databases to query production LLMs and determine if they generate code containing harmful URLs. We conducted a large-scale evaluation across four production LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3), and found a systemic vulnerability, with all tested models generating malicious code at a non-negligible rate. On average, 4.2\% of programs generated in our experiments contained malicious URLs. Crucially, this malicious code is often generated in response to benign prompts. We manually validate the prompts which cause all four LLMs to generate malicious code, and resulting in 177 innocuous prompts that trigger all models to produce harmful outputs. These results provide strong empirical evidence that the training data of production LLMs has been successfully poisoned at scale, underscoring the urgent need for more robust defense mechanisms and post-generation safety checks to mitigate the propagation of hidden security threats. 

**Abstract (ZH)**: 大型语言模型（LLMs）在现代软件开发中变得至关重要，但其依赖互联网数据集进行训练引入了一个重大的安全风险：吸收和复现恶意内容。为了评估这一威胁，本文引入了一个可扩展的自动化审计框架，从已知骗局数据库中综合生成无害的开发者风格提示，查询生产中的LLMs，以确定它们是否会生成包含有害URL的代码。我们在四个生产中的LLM（GPT-4o、GPT-4o-mini、Llama-4-Scout和DeepSeek-V3）上进行了大规模评估，发现所有测试模型均在非可忽略的比例上生成恶意代码。平均而言，在我们的实验中生成的程序中有4.2%包含恶意URL。此外，这些恶意代码往往是在响应良性提示时生成的。我们手动验证了导致所有四种LLM生成恶意代码的提示，并由此得到177个无害的提示，这些提示会使所有模型生成有害输出。这些结果提供了强有力的经验证据，表明生产中的LLM的训练数据已被规模化污染，强调了迫切需要更健壮的防御机制和生成后的安全检查，以减轻隐藏安全威胁的传播。 

---
# Implicit Reasoning in Large Language Models: A Comprehensive Survey 

**Title (ZH)**: 大型语言模型中的隐式推理：一个综合调研 

**Authors**: Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2509.02350)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong generalization across a wide range of tasks. Reasoning with LLMs is central to solving multi-step problems and complex decision-making. To support efficient reasoning, recent studies have shifted attention from explicit chain-of-thought prompting toward implicit reasoning, where reasoning occurs silently via latent structures without emitting intermediate textual steps. Implicit reasoning brings advantages such as lower generation cost, faster inference, and better alignment with internal computation. Although prior surveys have discussed latent representations in the context of reasoning, a dedicated and mechanism-level examination of how reasoning unfolds internally within LLMs remains absent. This survey fills that gap by introducing a taxonomy centered on execution paradigms, shifting the focus from representational forms to computational strategies. We organize existing methods into three execution paradigms based on \textbf{\textit{how and where internal computation unfolds}}: latent optimization, signal-guided control, and layer-recurrent execution. We also review structural, behavioral and representation-based evidence that supports the presence of implicit reasoning in LLMs. We further provide a structured overview of the evaluation metrics and benchmarks used in existing works to assess the effectiveness and reliability of implicit this http URL maintain a continuously updated project at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的任务中展示了强大的泛化能力。利用LLMs进行推理对于解决多步问题和复杂决策至关重要。为了支持高效的推理，最近的研究将注意力从显式的链式思维提示转向了隐式推理，在这种推理方式中，推理在潜藏结构中悄无声息地进行，而不发出中间文本步骤。隐式推理带来了如降低生成成本、加快推理速度和更好地与内部计算对齐等优势。尽管先前的综述已经讨论了推理的潜藏表示，但关于如何在LLMs内部展开推理的具体机制级分析仍然缺失。本文通过构建一种以执行范式为中心的分类体系，填补了这一空白，重点关注计算策略而非表示形式。我们将现有方法按照**内部计算如何以及何地展开**组织成三种执行范式：潜藏优化、信号引导控制和层递归执行。我们还回顾了支持LLMs中存在隐式推理的结构、行为和表示方面的证据。此外，我们提供了现有工作中评估隐式推理有效性和可靠性的度量标准和基准的结构化概述，并维护一个持续更新的项目：这一链接。 

---
# DCPO: Dynamic Clipping Policy Optimization 

**Title (ZH)**: DCPO：动态剪裁策略优化 

**Authors**: Shihui Yang, Chengfeng Dou, Peidong Guo, Kai Lu, Qiang Ju, Fei Deng, Rihui Xin  

**Link**: [PDF](https://arxiv.org/pdf/2509.02333)  

**Abstract**: Reinforcement Learning from Verifiable Rewards (RLVR) has emerged as a promising framework for enhancing the reasoning capabilities of large language models. However, existing approaches such as GRPO often suffer from zero gradients. This problem arises primarily due to fixed clipping bounds for token-level probability ratios and the standardization of identical rewards, which can lead to ineffective gradient updates and underutilization of generated responses. In this work, we propose Dynamic Clipping Policy Optimization (DCPO), which introduces a dynamic clipping strategy that adaptively adjusts the clipping bounds based on token-specific prior probabilities to enhance token-level exploration, and a smooth advantage standardization technique that standardizes rewards across cumulative training steps to improve the response-level effective utilization of generated responses. DCPO achieved state-of-the-art performance on four benchmarks based on four different models. In particular, DCPO achieved an Avg@1 of 46.7 under greedy decoding and an Avg@32 of 38.8 under 32 times sampling on the AIME24 benchmark, surpassing both DAPO (36.7/31.6) and GRPO (36.7/32.1) on the Qwen2.5-Math-7B model. On the AIME25 benchmark based on Qwen2.5-14B, DCPO achieves a performance of (23.3/19.0), surpassing GRPO (13.3/10.5) and DAPO (20.0/15.3). Furthermore, DCPO achieved an average 28% improvement in the nonzero advantage over GRPO in four models, doubled the training efficiency over DAPO, and significantly reduced the token clipping ratio by an order of magnitude compared to both GRPO and DAPO, while achieving superior performance. These results highlight DCPO's effectiveness in leveraging generated data more efficiently for reinforcement learning in large language models. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）已成为提升大规模语言模型推理能力的有前途的框架。然而，现有方法如GRPO往往会遭遇零梯度问题。这一问题主要源于固定剪裁界限以及标准化相同奖励，可能导致梯度更新无效和生成响应的有效利用不足。在本工作中，我们提出了一种动态剪裁策略优化（DCPO），引入了基于令牌特定先验概率自适应调整剪裁界限的动态剪裁策略，以增强令牌级探索，并提出了一种平滑优势标准化技术，该技术在累积训练步骤中标准化奖励，以提高生成响应的响应级有效利用率。DCPO在四个不同模型的四个基准测试中均实现了最优性能，特别是在AIME24基准测试中，在贪婪解码下的Avg@1为46.7，在32次采样下的Avg@32为38.8，超越了DAPO（36.7/31.6）和GRPO（36.7/32.1）在Qwen2.5-Math-7B模型上的表现，在基于Qwen2.5-14B的AIME25基准测试中，DCPO达到了（23.3/19.0）的表现，超越了GRPO（13.3/10.5）和DAPO（20.0/15.3）。此外，DCPO在四个模型上使得非零优势提升了28%，相比DAPO提高了训练效率两倍，相较于GRPO和DAPO显著降低了一个数量级的令牌剪裁比率，实现了更好的性能。这些结果突显了DCPO在大规模语言模型强化学习中更高效利用生成数据的有效性。 

---
# ReCode: Improving LLM-based Code Repair with Fine-Grained Retrieval-Augmented Generation 

**Title (ZH)**: ReCode: 通过细粒度检索增强生成改进基于LLM的代码修复 

**Authors**: Yicong Zhao, Shisong Chen, Jiacheng Zhang, Zhixu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.02330)  

**Abstract**: Recent advances in large language models (LLMs) have demonstrated impressive capabilities in code-related tasks, such as code generation and automated program repair. Despite their promising performance, most existing approaches for code repair suffer from high training costs or computationally expensive inference. Retrieval-augmented generation (RAG), with its efficient in-context learning paradigm, offers a more scalable alternative. However, conventional retrieval strategies, which are often based on holistic code-text embeddings, fail to capture the structural intricacies of code, resulting in suboptimal retrieval quality. To address the above limitations, we propose ReCode, a fine-grained retrieval-augmented in-context learning framework designed for accurate and efficient code repair. Specifically, ReCode introduces two key innovations: (1) an algorithm-aware retrieval strategy that narrows the search space using preliminary algorithm type predictions; and (2) a modular dual-encoder architecture that separately processes code and textual inputs, enabling fine-grained semantic matching between input and retrieved contexts. Furthermore, we propose RACodeBench, a new benchmark constructed from real-world user-submitted buggy code, which addresses the limitations of synthetic benchmarks and supports realistic evaluation. Experimental results on RACodeBench and competitive programming datasets demonstrate that ReCode achieves higher repair accuracy with significantly reduced inference cost, highlighting its practical value for real-world code repair scenarios. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在代码相关任务中的应用取得了显著进展，例如代码生成和自动化程序修复。尽管这些模型表现出色，但现有大多数代码修复方法在训练成本或推理计算成本上存在较高开销。检索增强生成（RAG）通过其高效的上下文学习范式提供了更可扩展的替代方案。然而，传统的检索策略通常基于整体代码文本嵌入，无法捕捉代码的结构复杂性，导致检索质量不理想。为了解决上述局限性，我们提出了ReCode，这是一种专门设计用于准确和高效代码修复的细粒度检索增强上下文学习框架。具体而言，ReCode引入了两项创新：（1）算法感知的检索策略，利用初步的算法类型预测缩小搜索空间；（2）模块化双编码器架构，分别处理代码和文本输入，实现输入与检索上下文之间精细的语义匹配。此外，我们还提出了RACodeBench，这是一种基于真实用户提交的错误代码构建的新基准，解决了合成基准的局限性，并支持现实场景的评估。在RACodeBench和高水平编程数据集上的实验结果表明，ReCode在显著降低推理成本的情况下实现了更高的修复准确性，突显了其在实际代码修复场景中的实用价值。 

---
# Application Of Large Language Models For The Extraction Of Information From Particle Accelerator Technical Documentation 

**Title (ZH)**: 大型语言模型在抽取粒子加速器技术文档信息中的应用 

**Authors**: Qing Dai, Rasmus Ischebeck, Maruisz Sapinski, Adam Grycner  

**Link**: [PDF](https://arxiv.org/pdf/2509.02227)  

**Abstract**: The large set of technical documentation of legacy accelerator systems, coupled with the retirement of experienced personnel, underscores the urgent need for efficient methods to preserve and transfer specialized knowledge. This paper explores the application of large language models (LLMs), to automate and enhance the extraction of information from particle accelerator technical documents. By exploiting LLMs, we aim to address the challenges of knowledge retention, enabling the retrieval of domain expertise embedded in legacy documentation. We present initial results of adapting LLMs to this specialized domain. Our evaluation demonstrates the effectiveness of LLMs in extracting, summarizing, and organizing knowledge, significantly reducing the risk of losing valuable insights as personnel retire. Furthermore, we discuss the limitations of current LLMs, such as interpretability and handling of rare domain-specific terms, and propose strategies for improvement. This work highlights the potential of LLMs to play a pivotal role in preserving institutional knowledge and ensuring continuity in highly specialized fields. 

**Abstract (ZH)**: 大型遗留加速器系统技术文档资料库与经验人员的退休凸显了高效方法保存和传承专门知识的迫切需求。本文探讨了大型语言模型（LLMs）在自动和增强提取粒子加速器技术文档信息方面的应用。通过利用LLMs，我们旨在解决知识保留的挑战，使领域专长能够嵌入在遗留文档中得以检索。我们介绍了将LLMs适应于这一专门领域所取得的初步结果。我们的评估表明，LLMs在提取、总结和组织知识方面非常有效，显著减少了人员退休导致的知识损失风险。此外，我们讨论了当前LLMs的局限性，如可解释性和处理罕见领域特定术语的能力，并提出了一些改进策略。本文强调了LLMs在保存机构知识和确保高度专业化领域连续性方面可能发挥的关键作用。 

---
# Baichuan-M2: Scaling Medical Capability with Large Verifier System 

**Title (ZH)**: Baichuan-M2: 通过大规模验证系统扩展医疗能力 

**Authors**: Baichuan-M2 Team, Chengfeng Dou, Chong Liu, Fan Yang, Fei Li, Jiyuan Jia, Mingyang Chen, Qiang Ju, Shuai Wang, Shunya Dang, Tianpeng Li, Xiangrong Zeng, Yijie Zhou, Chenzheng Zhu, Da Pan, Fei Deng, Guangwei Ai, Guosheng Dong, Hongda Zhang, Jinyang Tai, Jixiang Hong, Kai Lu, Linzhuang Sun, Peidong Guo, Qian Ma, Rihui Xin, Shihui Yang, Shusen Zhang, Yichuan Mo, Zheng Liang, Zhishou Zhang, Hengfu Cui, Zuyi Zhu, Xiaochuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02208)  

**Abstract**: As large language models (LLMs) advance in conversational and reasoning capabilities, their practical application in healthcare has become a critical research focus. However, there is a notable gap between the performance of medical LLMs on static benchmarks such as USMLE and their utility in real-world clinical decision-making. This discrepancy arises because traditional exams fail to capture the dynamic, interactive nature of medical consultations. To address this challenge, we introduce a novel dynamic verification framework that moves beyond static answer verifier, establishing a large-scale, high-fidelity interactive reinforcement learning system. Our framework comprises two key components: a Patient Simulator that creates realistic clinical environments using de-identified medical records, and a Clinical Rubrics Generator that dynamically produces multi-dimensional evaluation metrics. Building on this foundation, we develop Baichuan-M2, a 32B-parameter medical augmented reasoning model trained through a multi-stage reinforcement learning strategy with an improved Group Relative Policy Optimization (GRPO) algorithm. Evaluated on HealthBench, Baichuan-M2 outperforms all other open-source models and most advanced closed-source counterparts, achieving a score above 32 on the challenging HealthBench Hard benchmark-previously exceeded only by GPT-5. Our work demonstrates that robust dynamic verifier system is essential for aligning LLM capabilities with practical clinical applications, establishing a new Pareto front in the performance-parameter trade-off for medical AI deployment. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在对话和推理能力方面的进步，它们在医疗领域的实际应用已成为关键的研究焦点。然而，医学LLMs在诸如USMLE等静态基准测试上的性能与其在实际临床决策中的实用性之间存在明显差距。这种差异源于传统考试未能捕捉到医疗咨询的动态和交互性质。为应对这一挑战，我们引入了一种新的动态验证框架，超越了静态答案验证器，建立了一个大规模、高保真度的互动强化学习系统。该框架包含两个关键组成部分：患者模拟器，通过脱敏医疗记录创建现实的临床环境；临床评鉴生成器，动态生成多维度的评估指标。在此基础上，我们开发了Baichuan-M2医疗增强推理模型，通过多阶段强化学习策略并采用改进的组相对策略优化（GRPO）算法进行训练。在HealthBench上评估，Baichuan-M2优于所有开源模型和大多数先进的封闭源模型，获得了超过32分的挑战性HealthBench Hard基准分数——此前仅被GPT-5超越。我们的工作证明了强大的动态验证系统对于将LLM能力与实际临床应用相匹配至关重要，并为此建立了医疗AI部署性能-参数权衡的新帕累托前沿。 

---
# Avoidance Decoding for Diverse Multi-Branch Story Generation 

**Title (ZH)**: 避免解码以生成多元分支故事 

**Authors**: Kyeongman Park, Nakyeong Yang, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2509.02170)  

**Abstract**: Large Language Models (LLMs) often generate repetitive and monotonous outputs, especially in tasks like story generation, due to limited creative diversity when given the same input prompt. To address this challenge, we propose a novel decoding strategy, Avoidance Decoding, that modifies token logits by penalizing similarity to previously generated outputs, thereby encouraging more diverse multi-branch stories. This penalty adaptively balances two similarity measures: (1) Concept-level Similarity Penalty, which is prioritized in early stages to diversify initial story concepts, and (2) Narrative-level Similarity Penalty, which is increasingly emphasized later to ensure natural yet diverse plot development. Notably, our method achieves up to 2.6 times higher output diversity and reduces repetition by an average of 30% compared to strong baselines, while effectively mitigating text degeneration. Furthermore, we reveal that our method activates a broader range of neurons, demonstrating that it leverages the model's intrinsic creativity. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常在遇到相同输入提示时生成重复和单调的输出，尤其是在故事生成任务中。为解决这一挑战，我们提出了一种新颖的解码策略——避免解码，通过惩罚与先前生成输出的相似性来修改词元logits，从而促进更多样化的多分支故事生成。这种惩罚适应性地平衡了两种相似性度量：(1) 概念层次相似性惩罚，在早期阶段优先使用以多样化初始故事概念，(2) 故事层次相似性惩罚，后期逐渐强调以确保自然且多样的剧情发展。值得注意的是，与强大的基线方法相比，我们的方法可实现高达2.6倍的输出多样性提升，平均减少30%的重复性，同时有效缓解文本退化。此外，我们发现我们的方法激活了更多的神经元，表明其利用了模型内部的创造力。 

---
# Meta-Pretraining for Zero-Shot Cross-Lingual Named Entity Recognition in Low-Resource Philippine Languages 

**Title (ZH)**: 低资源菲律宾语言零样本跨语言命名实体识别的元预训练 

**Authors**: David Demitri Africa, Suchir Salhan, Yuval Weiss, Paula Buttery, Richard Diehl Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.02160)  

**Abstract**: Named-entity recognition (NER) in low-resource languages is usually tackled by finetuning very large multilingual LMs, an option that is often infeasible in memory- or latency-constrained settings. We ask whether small decoder LMs can be pretrained so that they adapt quickly and transfer zero-shot to languages unseen during pretraining. To this end we replace part of the autoregressive objective with first-order model-agnostic meta-learning (MAML). Tagalog and Cebuano are typologically similar yet structurally different in their actor/non-actor voice systems, and hence serve as a challenging test-bed. Across four model sizes (11 M - 570 M) MAML lifts zero-shot micro-F1 by 2-6 pp under head-only tuning and 1-3 pp after full tuning, while cutting convergence time by up to 8%. Gains are largest for single-token person entities that co-occur with Tagalog case particles si/ni, highlighting the importance of surface anchors. 

**Abstract (ZH)**: 低资源语言中的命名实体识别（NER）通常通过微调非常大的多语言LM来解决，在内存或延迟受限的环境中，这是一个 often infeasible 的选项。我们询问是否可以预先训练小型解码器LM，使其在预训练期间未见过的语言中能够快速适应并进行零-shot 转移。为此，我们将自回归目标部分替换为一阶模型无偏元学习（MAML）。塔加洛语和达比乌语在演员/非演员声音系统上类型学类似但结构不同，因此作为一项具有挑战性的测试床。通过对四种模型规模（11M - 570M）进行测试，MAML 在仅头部微调下将零-shot 微-F1 提升了 2-6 个百分点，在完全微调后提升 1-3 个百分点，同时将收敛时间缩短了最多 8%。最大的收益来自于与塔加洛语标记粒子 si/ni 共现的单词人称实体，突显了表面锚点的重要性。 

---
# JudgeAgent: Dynamically Evaluate LLMs with Agent-as-Interviewer 

**Title (ZH)**: JudgeAgent: 以面试官代理人的形式动态评估LLMs 

**Authors**: Zhichao Shi, Xuhui Jiang, Chengjin Xu, Cangli Yao, Zhenxin Huang, Shengjie Ma, Yinghan Shen, Yuanzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02097)  

**Abstract**: Evaluating the capabilities of large language models (LLMs) is an essential step to ensure the successful application of LLMs across various domains. The current evaluation of LLMs is based on a paradigm that involves querying them with predefined question sets and assessing their outputs. This paradigm offers controllable processes and simplicity, but faces challenges such as limited interaction with targets, insufficient difficulty control, and difficulties in verifying the validity of evaluation results, making it hard to precisely determine the knowledge and capability boundaries of target models. To address these challenges, we propose JudgeAgent, a knowledge-target adaptive dynamic evaluation framework based on a new interviewer-style evaluation paradigm. JudgeAgent employs a comprehensive evaluation approach consisting of benchmark grading, interactive extension, and evaluation feedback. It utilizes knowledge-driven data synthesis and target-adaptive difficulty adjustment methods to conduct extended testing, providing accurate and effective evaluation results. We also introduce a novel insight into validating evaluation methods, demonstrating the effectiveness of JudgeAgent and its dynamic evaluation paradigm through extensive experiments. 

**Abstract (ZH)**: 评估大规模语言模型的能力是确保大规模语言模型在各个领域成功应用的关键步骤。当前对大规模语言模型的评估基于一种查询预定义问题集并评估输出的范式。这一范式提供了可控的过程和简化性，但也面临着与目标互动有限、难度控制不足以及验证评估结果有效性等方面的挑战，使得难以精确确定目标模型的知识和能力边界。为应对这些挑战，我们提出了JudgeAgent，这是一种基于新型面试风格评估范式的知识目标自适应动态评估框架。JudgeAgent采用了一种全面的评估方法，包括基准评分、互动扩展和评估反馈。它利用知识驱动的数据合成和目标自适应难度调整方法进行扩展测试，提供准确有效的评估结果。我们还引入了一种新的视角来验证评估方法，并通过广泛的实验展示了JudgeAgent及其动态评估范式的有效性。 

---
# Better by Comparison: Retrieval-Augmented Contrastive Reasoning for Automatic Prompt Optimization 

**Title (ZH)**: 通过比较更优：检索增强对比推理在自动提示优化中的应用 

**Authors**: Juhyeon Lee, Wonduk Seo, Hyunjin An, Seunghyun Lee, Yi Bu  

**Link**: [PDF](https://arxiv.org/pdf/2509.02093)  

**Abstract**: Automatic prompt optimization has recently emerged as a strategy for improving the quality of prompts used in Large Language Models (LLMs), with the goal of generating more accurate and useful responses. However, most prior work focuses on direct prompt refinement or model fine-tuning, overlooking the potential of leveraging LLMs' inherent reasoning capability to learn from contrasting examples. In this paper, we present Contrastive Reasoning Prompt Optimization (CRPO), a novel framework that formulates prompt optimization as a retrieval augmented reasoning process. Our approach retrieves top k reference prompts from the HelpSteer2 dataset, an open-source collection annotated for helpfulness, correctness, coherence, complexity, and verbosity, and constructs two complementary optimization paradigms: (1) tiered contrastive reasoning, where the LLM compares high, medium, and low quality prompts to refine its own generation through reflective reasoning, and (2) multi-metric contrastive reasoning, where the LLM analyzes the best prompts along each evaluation dimension and integrates their strengths into an optimized prompt. By explicitly contrasting high and low quality exemplars, CRPO enables the model to deduce why certain prompts succeed while others fail, thereby achieving more robust and interpretable optimization. Experimental results on the HelpSteer2 benchmark demonstrate that CRPO significantly outperforms baselines. Our findings highlight the promise of contrastive, retrieval-augmented reasoning for advancing automatic prompt optimization. 

**Abstract (ZH)**: 自动对比推理提示优化：一种通过检索增强推理过程实现提示优化的新框架 

---
# How Instruction-Tuning Imparts Length Control: A Cross-Lingual Mechanistic Analysis 

**Title (ZH)**: 指令调优如何 impart 长度控制：一种跨语言机制分析 

**Authors**: Elisabetta Rocchetti, Alfio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2509.02075)  

**Abstract**: Adhering to explicit length constraints, such as generating text with a precise word count, remains a significant challenge for Large Language Models (LLMs). This study aims at investigating the differences between foundation models and their instruction-tuned counterparts, on length-controlled text generation in English and Italian. We analyze both performance and internal component contributions using Cumulative Weighted Attribution, a metric derived from Direct Logit Attribution. Our findings reveal that instruction-tuning substantially improves length control, primarily by specializing components in deeper model layers. Specifically, attention heads in later layers of IT models show increasingly positive contributions, particularly in English. In Italian, while attention contributions are more attenuated, final-layer MLPs exhibit a stronger positive role, suggesting a compensatory mechanism. These results indicate that instruction-tuning reconfigures later layers for task adherence, with component-level strategies potentially adapting to linguistic context. 

**Abstract (ZH)**: 遵循明确的长度约束，如生成精确词数的文字，仍然是大型语言模型（LLMs）的一个重要挑战。本研究旨在探究基础模型与其指令调优版本在英意两种语言下的长度控制文本生成方面的差异。我们使用累积加权归因这一源自直接逻辑归因的指标，对性能和内部组件贡献进行分析。研究发现，指令调优显著提高了长度控制能力，主要通过专门化更深模型层的组件实现。具体而言，IT模型后期层的注意力头显示越来越积极的贡献，尤其是在英语中。在意大利语中，尽管注意力贡献有所减弱，最终层的MLP表现出更强的积极作用，表明可能存在补偿机制。这些结果表明，指令调优重新配置了后期层以遵循任务要求，并且组件级策略可能适应语言环境。 

---
# DeepSeek performs better than other Large Language Models in Dental Cases 

**Title (ZH)**: DeepSeek 在牙科案例中的表现优于其他大型语言模型 

**Authors**: Hexian Zhang, Xinyu Yan, Yanqi Yang, Lijian Jin, Ping Yang, Junwen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.02036)  

**Abstract**: Large language models (LLMs) hold transformative potential in healthcare, yet their capacity to interpret longitudinal patient narratives remains inadequately explored. Dentistry, with its rich repository of structured clinical data, presents a unique opportunity to rigorously assess LLMs' reasoning abilities. While several commercial LLMs already exist, DeepSeek, a model that gained significant attention earlier this year, has also joined the competition. This study evaluated four state-of-the-art LLMs (GPT-4o, Gemini 2.0 Flash, Copilot, and DeepSeek V3) on their ability to analyze longitudinal dental case vignettes through open-ended clinical tasks. Using 34 standardized longitudinal periodontal cases (comprising 258 question-answer pairs), we assessed model performance via automated metrics and blinded evaluations by licensed dentists. DeepSeek emerged as the top performer, demonstrating superior faithfulness (median score = 0.528 vs. 0.367-0.457) and higher expert ratings (median = 4.5/5 vs. 4.0/5), without significantly compromising readability. Our study positions DeepSeek as the leading LLM for case analysis, endorses its integration as an adjunct tool in both medical education and research, and highlights its potential as a domain-specific agent. 

**Abstract (ZH)**: 大型语言模型在医疗保健领域的转型潜力巨大，但其解释 longitudinal 患者叙述的能力仍需进一步探索。牙科因其丰富的结构化临床数据，提供了严格评估大型语言模型推理能力的unique机会。尽管目前已存在多个商业大型语言模型，但今年早些时候引起广泛关注的DeepSeek模型也加入了竞争。本研究评估了四款最先进的大型语言模型（GPT-4o、Gemini 2.0 Flash、Copilot 和 DeepSeek V3），它们在通过开放性临床任务分析纵向牙科案例方面的能力。通过使用 34 个标准化的纵向牙周病案例（包含 258 个问答对），我们通过自动评估指标和经过牙科执照医生盲测的方式评估了模型的性能。DeepSeek 出色地脱颖而出，表现出更高的忠实度（中位分 = 0.528 对比 0.367-0.457）和更高的专家评分（中位分 = 4.5/5 对比 4.0/5），且未显著牺牲可读性。本研究将DeepSeek定位为案例分析的最佳大型语言模型，支持其作为医学教育和研究中的辅助工具，并突显了其在专业领域内的潜力。 

---
# Empowering Large Language Model for Sequential Recommendation via Multimodal Embeddings and Semantic IDs 

**Title (ZH)**: 基于多模态嵌入和语义ID增强大型语言模型的序列推荐 

**Authors**: Yuhao Wang, Junwei Pan, Xinhang Li, Maolin Wang, Yuan Wang, Yue Liu, Dapeng Liu, Jie Jiang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.02017)  

**Abstract**: Sequential recommendation (SR) aims to capture users' dynamic interests and sequential patterns based on their historical interactions. Recently, the powerful capabilities of large language models (LLMs) have driven their adoption in SR. However, we identify two critical challenges in existing LLM-based SR methods: 1) embedding collapse when incorporating pre-trained collaborative embeddings and 2) catastrophic forgetting of quantized embeddings when utilizing semantic IDs. These issues dampen the model scalability and lead to suboptimal recommendation performance. Therefore, based on LLMs like Llama3-8B-instruct, we introduce a novel SR framework named MME-SID, which integrates multimodal embeddings and quantized embeddings to mitigate embedding collapse. Additionally, we propose a Multimodal Residual Quantized Variational Autoencoder (MM-RQ-VAE) with maximum mean discrepancy as the reconstruction loss and contrastive learning for alignment, which effectively preserve intra-modal distance information and capture inter-modal correlations, respectively. To further alleviate catastrophic forgetting, we initialize the model with the trained multimodal code embeddings. Finally, we fine-tune the LLM efficiently using LoRA in a multimodal frequency-aware fusion manner. Extensive experiments on three public datasets validate the superior performance of MME-SID thanks to its capability to mitigate embedding collapse and catastrophic forgetting. The implementation code and datasets are publicly available for reproduction: this https URL. 

**Abstract (ZH)**: 基于大型语言模型的序贯推荐框架MME-SID：缓解embedding崩塌和灾难性遗忘 

---
# Extracting OPQRST in Electronic Health Records using Large Language Models with Reasoning 

**Title (ZH)**: 使用具有推理能力的大语言模型从电子健康记录中提取OPQRST 

**Authors**: Zhimeng Luo, Abhibha Gupta, Adam Frisch, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.01885)  

**Abstract**: The extraction of critical patient information from Electronic Health Records (EHRs) poses significant challenges due to the complexity and unstructured nature of the data. Traditional machine learning approaches often fail to capture pertinent details efficiently, making it difficult for clinicians to utilize these tools effectively in patient care. This paper introduces a novel approach to extracting the OPQRST assessment from EHRs by leveraging the capabilities of Large Language Models (LLMs). We propose to reframe the task from sequence labeling to text generation, enabling the models to provide reasoning steps that mimic a physician's cognitive processes. This approach enhances interpretability and adapts to the limited availability of labeled data in healthcare settings. Furthermore, we address the challenge of evaluating the accuracy of machine-generated text in clinical contexts by proposing a modification to traditional Named Entity Recognition (NER) metrics. This includes the integration of semantic similarity measures, such as the BERT Score, to assess the alignment between generated text and the clinical intent of the original records. Our contributions demonstrate a significant advancement in the use of AI in healthcare, offering a scalable solution that improves the accuracy and usability of information extraction from EHRs, thereby aiding clinicians in making more informed decisions and enhancing patient care outcomes. 

**Abstract (ZH)**: 从电子健康记录中提取关键患者信息由于数据的复杂性和非结构化性质提出了显著挑战。传统的机器学习方法往往无法高效地捕捉相关信息，使得临床医生在患者 care 中有效利用这些工具变得困难。本文提出了一种利用大规模语言模型（LLMs）能力的新方法，用于从 EHRs 中提取 OPQRST 评估。我们提出将任务从序列标注重新框定为文本生成，使模型能够提供模拟医生认知过程的推理步骤。这种方法增强了可解释性，并适应了医疗保健环境中标注数据的有限可用性。此外，我们通过提出对传统命名实体识别（NER）指标的修改来应对在临床环境中评估机器生成文本准确性的挑战，其中包括整合语义相似性度量，如 BERT Score，以评估生成文本与原始记录的临床意图之间的对齐情况。我们的贡献展示了 AI 在医疗保健中的重大进步，提供了一种可扩展的解决方案，提高了从 EHRs 中提取信息的准确性和可用性，从而帮助临床医生做出更明智的决策并提升患者护理结果。 

---
# When LLM Meets Time Series: Can LLMs Perform Multi-Step Time Series Reasoning and Inference 

**Title (ZH)**: 当大规模语言模型遇见时间序列：大规模语言模型能否进行多步时间序列推理和推断 

**Authors**: Wen Ye, Jinbo Liu, Defu Cao, Wei Yang, Yan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01822)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has sparked growing interest in their application to time series analysis tasks. However, their ability to perform complex reasoning over temporal data in real-world application domains remains underexplored. To move toward this goal, a first step is to establish a rigorous benchmark dataset for evaluation. In this work, we introduce the TSAIA Benchmark, a first attempt to evaluate LLMs as time-series AI assistants. To ensure both scientific rigor and practical relevance, we surveyed over 20 academic publications and identified 33 real-world task formulations. The benchmark encompasses a broad spectrum of challenges, ranging from constraint-aware forecasting to anomaly detection with threshold calibration: tasks that require compositional reasoning and multi-step time series analysis. The question generator is designed to be dynamic and extensible, supporting continuous expansion as new datasets or task types are introduced. Given the heterogeneous nature of the tasks, we adopt task-specific success criteria and tailored inference-quality metrics to ensure meaningful evaluation for each task. We apply this benchmark to assess eight state-of-the-art LLMs under a unified evaluation protocol. Our analysis reveals limitations in current models' ability to assemble complex time series analysis workflows, underscoring the need for specialized methodologies for domain-specific adaptation. Our benchmark is available at this https URL, and the code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型的快速 advancement 在时间序列分析任务中的应用引发了广泛关注。然而，它们在现实世界应用领域中处理时间数据的复杂推理能力仍然未被充分探索。为实现这一目标，第一步是建立一个严格的基准数据集用于评估。在本工作中，我们引入了 TSAIA 基准，这是首次尝试评估大型语言模型作为时间序列AI助手。为了确保科学严谨性和实际相关性，我们调研了超过20篇学术出版物，并确定了33个实际世界任务的表述。基准涵盖了广泛的挑战，从带有约束条件的预测到需要阈值校准的异常检测：这些任务要求进行组合推理和多步时间序列分析。问题生成器旨在动态和可扩展，支持随着新数据集或任务类型的引入而持续扩展。鉴于任务的异质性，我们采用特定任务的成功标准和定制的推理质量度量，以确保对每个任务进行有意义的评估。我们使用统一的评估协议对八种当前最先进的大型语言模型进行了评估。我们的分析揭示了现有模型在构建复杂时间序列分析工作流方面的局限性，强调了需要专门的方法来适应特定领域的需求。该基准可在以下链接获取：this https URL，代码可在以下链接获取：this https URL。 

---
# Mic Drop or Data Flop? Evaluating the Fitness for Purpose of AI Voice Interviewers for Data Collection within Quantitative & Qualitative Research Contexts 

**Title (ZH)**: Mic Drop or Data Flop? 评估AI语音采访者在定量与定性研究数据收集上下文中适用性的效果 

**Authors**: Shreyas Tirumala, Nishant Jain, Danny D. Leybzon, Trent D. Buskirk  

**Link**: [PDF](https://arxiv.org/pdf/2509.01814)  

**Abstract**: Transformer-based Large Language Models (LLMs) have paved the way for "AI interviewers" that can administer voice-based surveys with respondents in real-time. This position paper reviews emerging evidence to understand when such AI interviewing systems are fit for purpose for collecting data within quantitative and qualitative research contexts. We evaluate the capabilities of AI interviewers as well as current Interactive Voice Response (IVR) systems across two dimensions: input/output performance (i.e., speech recognition, answer recording, emotion handling) and verbal reasoning (i.e., ability to probe, clarify, and handle branching logic). Field studies suggest that AI interviewers already exceed IVR capabilities for both quantitative and qualitative data collection, but real-time transcription error rates, limited emotion detection abilities, and uneven follow-up quality indicate that the utility, use and adoption of current AI interviewer technology may be context-dependent for qualitative data collection efforts. 

**Abstract (ZH)**: 基于Transformer的大型语言模型（LLMs）为“AI面试者”铺平了道路，这些“AI面试者”可以实时对受访者进行基于语音的调查。本文综述新兴证据，以了解在定量和定性研究背景下，此类AI面试系统何时适合用于收集数据。我们从输入/输出性能（即语音识别、答案记录、情绪处理）和口头推理能力（即探询、澄清和处理分支逻辑的能力）两个维度评估AI面试者和当前交互式语音响应（IVR）系统的功能。实地研究显示，AI面试者在定量和定性数据收集方面已经超过了IVR系统的功能，但实时转录错误率、有限的情绪检测能力和不均衡的后续质量表明，当前AI面试技术在定性数据收集方面的适用性、使用和采纳可能依赖于具体的背景条件。 

---
# Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs 

**Title (ZH)**: 瑕障还是 artefact？重新审视评价大语言模型的提示敏感性 

**Authors**: Andong Hua, Kenan Tang, Chenhe Gu, Jindong Gu, Eric Wong, Yao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.01790)  

**Abstract**: Prompt sensitivity, referring to the phenomenon where paraphrasing (i.e., repeating something written or spoken using different words) leads to significant changes in large language model (LLM) performance, has been widely accepted as a core limitation of LLMs. In this work, we revisit this issue and ask: Is the widely reported high prompt sensitivity truly an inherent weakness of LLMs, or is it largely an artifact of evaluation processes? To answer this question, we systematically evaluate 7 LLMs (e.g., GPT and Gemini family) across 6 benchmarks, including both multiple-choice and open-ended tasks on 12 diverse prompt templates. We find that much of the prompt sensitivity stems from heuristic evaluation methods, including log-likelihood scoring and rigid answer matching, which often overlook semantically correct responses expressed through alternative phrasings, such as synonyms or paraphrases. When we adopt LLM-as-a-Judge evaluations, we observe a substantial reduction in performance variance and a consistently higher correlation in model rankings across prompts. Our findings suggest that modern LLMs are more robust to prompt templates than previously believed, and that prompt sensitivity may be more an artifact of evaluation than a flaw in the models. 

**Abstract (ZH)**: 大语言模型的提示敏感性：广泛报道的高提示敏感性究竟是LLM内在的弱点，还是评价过程中的伪现象？ 

---
# AHAMask: Reliable Task Specification for Large Audio Language Models without Instructions 

**Title (ZH)**: AHAMask: 可靠的任务规范，无需指令的大规模音频语言模型 

**Authors**: Yiwei Guo, Bohan Li, Hankun Wang, Zhihan Li, Shuai Wang, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.01787)  

**Abstract**: Although current large audio language models (LALMs) extend text large language models (LLMs) with generic acoustic understanding abilities, they usually suffer from instruction sensitivity, where different instructions of the same intention can yield drastically different outcomes. In this work, we propose AHAMask, where we simply mask some of the attention heads in the decoder-only LLM backbone of LALMs, to trigger specific acoustic task functionalities without instructions. These masks are efficiently obtained by training on an LALM, with the number of trainable parameters equal to the attention head count in its LLM backbone. We show by experiments that applying such selective attention head masks achieves comparable or even better performance than using instructions, either on single or composite tasks. Besides achieving reliable acoustic task specification for LALMs, this also reveals that LALMs exhibit certain "functional pathways" in their attention heads. 

**Abstract (ZH)**: 尽管当前的大规模语音语言模型（LALMs）在通用声学理解能力上扩展了文本大型语言模型（LLMs），但它们通常会面临指令敏感性的问题，即相同意图的不同指令可能会导致截然不同的结果。在本文中，我们提出了AHAMask，通过在LALMs的解码器背部结构中简单地遮蔽一些注意力头，以在不需要指令的情况下触发特定的声学任务功能。这些遮蔽是通过在LALM上进行训练而高效获得的，可训练参数的数量等于其文本背部结构中的注意力头数量。实验结果表明，应用这种有选择性的注意力头遮蔽在单任务或复合任务上的性能与使用指令相当甚至更好。此外，这一发现还揭示了LALMs在其注意力头中存在某些“功能路径”。 

---
# chDzDT: Word-level morphology-aware language model for Algerian social media text 

**Title (ZH)**: chDzDT：面向阿尔及利亚社交媒体文本的词级形态学感知语言模型 

**Authors**: Abdelkrime Aries  

**Link**: [PDF](https://arxiv.org/pdf/2509.01772)  

**Abstract**: Pre-trained language models (PLMs) have substantially advanced natural language processing by providing context-sensitive text representations. However, the Algerian dialect remains under-represented, with few dedicated models available. Processing this dialect is challenging due to its complex morphology, frequent code-switching, multiple scripts, and strong lexical influences from other languages. These characteristics complicate tokenization and reduce the effectiveness of conventional word- or subword-level approaches.
To address this gap, we introduce chDzDT, a character-level pre-trained language model tailored for Algerian morphology. Unlike conventional PLMs that rely on token sequences, chDzDT is trained on isolated words. This design allows the model to encode morphological patterns robustly, without depending on token boundaries or standardized orthography. The training corpus draws from diverse sources, including YouTube comments, French, English, and Berber Wikipedia, as well as the Tatoeba project. It covers multiple scripts and linguistic varieties, resulting in a substantial pre-training workload.
Our contributions are threefold: (i) a detailed morphological analysis of Algerian dialect using YouTube comments; (ii) the construction of a multilingual Algerian lexicon dataset; and (iii) the development and extensive evaluation of a character-level PLM as a morphology-focused encoder for downstream tasks. The proposed approach demonstrates the potential of character-level modeling for morphologically rich, low-resource dialects and lays a foundation for more inclusive and adaptable NLP systems. 

**Abstract (ZH)**: 预训练语言模型（PLM）通过提供具有上下文感知的文字表示，极大地推动了自然语言处理的发展。然而，阿尔及利亚方言仍然相对未被充分代表，可用的专门模型较少。由于其复杂的词形变化、频繁的语言转换、多种书写系统以及来自其他语言的强烈词形影响，处理这种方言颇具挑战性。这些特性使得分词复杂化，并降低了传统基于单词或子单词级别的方法的有效性。

为了弥补这一差距，我们引入了chDzDT，这是一个针对阿尔及利亚词形特征设计的字符级预训练语言模型。与依赖于标记序列的常规PLM不同，chDzDT是基于孤立词进行训练的。这种设计使模型能够稳健地编码词形模式，而不依赖于标记边界或标准化的书写惯例。训练语料库来源于多种来源，包括YouTube评论、法语、英语和柏柏尔语维基百科，以及Tatoeba项目。该语料库涵盖了多种书写系统和语言变体，从而产生了大量的预训练工作量。

我们的贡献包括三个方面：（i）使用YouTube评论对阿尔及利亚方言进行详细的词形分析；（ii）构建一个多语言阿尔及利亚词汇词典数据集；（iii）开发并广泛评估一种字符级预训练语言模型，作为面向下游任务的词形导向编码器。所提出的方法展示了字符级建模对富有词形特征且资源稀缺的方言的潜在价值，并为更具包容性和适应性的NLP系统奠定了基础。 

---
# Reinforcement Learning for Machine Learning Engineering Agents 

**Title (ZH)**: 机器学习工程代理的强化学习 

**Authors**: Sherry Yang, Joy He-Yueya, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01684)  

**Abstract**: Existing agents for solving tasks such as ML engineering rely on prompting powerful language models. As a result, these agents do not improve with more experience. In this paper, we show that agents backed by weaker models that improve via reinforcement learning (RL) can outperform agents backed by much larger, but static models. We identify two major challenges with RL in this setting. First, actions can take a variable amount of time (e.g., executing code for different solutions), which leads to asynchronous policy gradient updates that favor faster but suboptimal solutions. To tackle variable-duration actions, we propose duration- aware gradient updates in a distributed asynchronous RL framework to amplify high-cost but high-reward actions. Second, using only test split performance as a reward provides limited feedback. A program that is nearly correct is treated the same as one that fails entirely. To address this, we propose environment instrumentation to offer partial credit, distinguishing almost-correct programs from those that fail early (e.g., during data loading). Environment instrumentation uses a separate static language model to insert print statement to an existing program to log the agent's experimental progress, from which partial credit can be extracted as reward signals for learning. Our experimental results on MLEBench suggest that performing gradient updates on a much smaller model (Qwen2.5-3B) trained with RL outperforms prompting a much larger model (Claude-3.5-Sonnet) with agent scaffolds, by an average of 22% across 12 Kaggle tasks. 

**Abstract (ZH)**: 基于较弱模型并通过强化学习提升的智能代理超越了基于更大静态模型的智能代理：通过分布异步强化学习框架中的时长感知梯度更新克服挑战并改进性能 

---
# Benchmarking the Detection of LLMs-Generated Modern Chinese Poetry 

**Title (ZH)**: LLMs生成的现代汉语诗歌的检测基准 

**Authors**: Shanshan Wang, Junchao Wu, Fengying Ye, Jingming Yao, Lidia S. Chao, Derek F. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.01620)  

**Abstract**: The rapid development of advanced large language models (LLMs) has made AI-generated text indistinguishable from human-written text. Previous work on detecting AI-generated text has made effective progress, but has not involved modern Chinese poetry. Due to the distinctive characteristics of modern Chinese poetry, it is difficult to identify whether a poem originated from humans or AI. The proliferation of AI-generated modern Chinese poetry has significantly disrupted the poetry ecosystem. Based on the urgency of identifying AI-generated poetry in the real Chinese world, this paper proposes a novel benchmark for detecting LLMs-generated modern Chinese poetry. We first construct a high-quality dataset, which includes both 800 poems written by six professional poets and 41,600 poems generated by four mainstream LLMs. Subsequently, we conduct systematic performance assessments of six detectors on this dataset. Experimental results demonstrate that current detectors cannot be used as reliable tools to detect modern Chinese poems generated by LLMs. The most difficult poetic features to detect are intrinsic qualities, especially style. The detection results verify the effectiveness and necessity of our proposed benchmark. Our work lays a foundation for future detection of AI-generated poetry. 

**Abstract (ZH)**: 先进大型语言模型的迅速发展使得AI生成的文字难以与人类撰写的文字区分开来。尽管先前有关检测AI生成文字的工作已取得有效进展，但尚未涉及现代中文诗歌。由于现代中文诗歌的独特特点，难以确定一首诗是出于人类还是AI之手。AI生成的现代中文诗歌的泛滥极大地扰乱了诗歌生态系统。鉴于在现实中国的紧迫需求中识别AI生成的诗歌，本文提出了一种新型基准用于检测由LLM生成的现代中文诗歌。我们首先构建了一个高质量的数据集，其中包括6位专业诗人撰写的800首诗和由4种主流LLM生成的41,600首诗。随后，我们在该数据集上系统评估了6种检测器的性能。实验结果表明，当前的检测器无法可靠地检测由LLM生成的现代中文诗歌。最难检测的诗歌特征是内在品质，尤其是风格。检测结果验证了我们提出基准的有效性和必要性。我们的工作为未来的AI生成诗歌检测奠定了基础。 

---
# CAT: Causal Attention Tuning For Injecting Fine-grained Causal Knowledge into Large Language Models 

**Title (ZH)**: CAT：因果注意力调谐，用于向大规模语言模型注入细粒度因果知识 

**Authors**: Kairong Han, Wenshuo Zhao, Ziyu Zhao, JunJian Ye, Lujia Pan, Kun Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01535)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across various domains. However, a fundamental question remains: Can LLMs effectively utilize causal knowledge for prediction and generation? Through empirical studies, we find that LLMs trained directly on large-scale data often capture spurious correlations rather than true causal relationships, leading to suboptimal performance, especially in out-of-distribution (OOD) scenarios. To address this challenge, we propose Causal Attention Tuning (CAT), a novel approach that injects fine-grained causal knowledge into the attention mechanism. We propose an automated pipeline that leverages human priors to automatically generate token-level causal signals and introduce the Re-Attention mechanism to guide training, helping the model focus on causal structures while mitigating noise and biases in attention scores. Experimental results on our proposed Spurious Token Game (STG) benchmark and multiple downstream tasks demonstrate that our approach effectively leverages causal knowledge for prediction and remains robust in OOD scenarios. Implementation details can be found at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域取得了显著成功。然而，一个基本问题依然存在：LLMs是否能够有效利用因果知识进行预测和生成？通过实证研究，我们发现直接训练在大规模数据上的LLMs往往捕捉到的是虚假的相关性而非真实的因果关系，导致在分布外（OOD）场景中的性能不佳。为了解决这一挑战，我们提出了一种新颖的方法——因果注意力调优（CAT），该方法将细粒度的因果知识注入到注意力机制中。我们提出了一种自动流水线，利用人类先验知识自动生成标记级别的因果信号，并引入重新关注机制来引导训练，帮助模型聚焦于因果结构，同时减轻注意力分数中的噪声和偏差。在我们提出的虚假标记游戏（STG）基准测试以及多个下游任务上的实验结果表明，我们的方法能够有效利用因果知识进行预测，并在分布外场景中保持稳健。更多实施细节请参见<这个链接>。 

---
# Agentic Workflow for Education: Concepts and Applications 

**Title (ZH)**: 教育中的代理工作流：概念与应用 

**Authors**: Yuan-Hao Jiang, Yijie Lu, Ling Dai, Jiatong Wang, Ruijia Li, Bo Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01517)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs) and Artificial Intelligence (AI) agents, agentic workflows are showing transformative potential in education. This study introduces the Agentic Workflow for Education (AWE), a four-component model comprising self-reflection, tool invocation, task planning, and multi-agent collaboration. We distinguish AWE from traditional LLM-based linear interactions and propose a theoretical framework grounded in the von Neumann Multi-Agent System (MAS) architecture. Through a paradigm shift from static prompt-response systems to dynamic, nonlinear workflows, AWE enables scalable, personalized, and collaborative task execution. We further identify four core application domains: integrated learning environments, personalized AI-assisted learning, simulation-based experimentation, and data-driven decision-making. A case study on automated math test generation shows that AWE-generated items are statistically comparable to real exam questions, validating the model's effectiveness. AWE offers a promising path toward reducing teacher workload, enhancing instructional quality, and enabling broader educational innovation. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）和人工智能（AI）代理的快速进步，代理工作流程在教育领域显示出变革性的潜力。本研究介绍了教育代理工作流程（AWE），这是一个由自我反思、工具调用、任务规划和多代理协作组成的四组件模型。我们区分了AWE与传统的基于LLM的线性交互，并提出了基于冯·诺依曼多代理系统（MAS）架构的理论框架。通过从静态提示-响应系统向动态、非线性工作流程的范式转变，AWE使得大规模的、个性化的和协作的任务执行成为可能。我们进一步确定了四个核心应用领域：综合学习环境、个性化AI辅助学习、基于模拟的实验和数据驱动的决策支持。自动化数学测试生成案例研究显示，AWE生成的题目在统计上与实际考试题目相当，验证了该模型的有效性。AWE为减轻教师工作负担、提高教学质量以及推动更广泛的教育创新提供了有前景的道路。 

---
# MeVe: A Modular System for Memory Verification and Effective Context Control in Language Models 

**Title (ZH)**: MeVe：一种用于语言模型内存验证和有效上下文控制的模块化系统 

**Authors**: Andreas Ottem  

**Link**: [PDF](https://arxiv.org/pdf/2509.01514)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically face constraints because of their inherent mechanism: a simple top-k semantic search [1]. The approach often leads to the incorporation of irrelevant or redundant information in the context, degrading performance and efficiency [10][11]. This paper presents MeVe, a novel modular architecture intended for Memory Verification and smart context composition. MeVe rethinks the RAG paradigm by proposing a five-phase modular design that distinctly breaks down the retrieval and context composition process into distinct, auditable, and independently tunable phases: initial retrieval, relevance verification, fallback retrieval, context prioritization, and token budgeting. This architecture enables fine-grained control of what knowledge is made available to an LLM, enabling task-dependent filtering and adaptation. We release a reference implementation of MeVe as a proof of concept and evaluate its performance on knowledge-heavy QA tasks over a subset of English Wikipedia [22]. Our results demonstrate that by actively verifying information before composition, MeVe significantly improves context efficiency, achieving a 57% reduction on the Wikipedia dataset and a 75% reduction on the more complex HotpotQA dataset compared to standard RAG implementations [25]. This work provides a framework for more scalable and reliable LLM applications. By refining and distilling contextual information, MeVe offers a path toward better grounding and more accurate factual support [16]. 

**Abstract (ZH)**: 基于检索增强生成的MeVe：一种新颖的记忆验证和智能语境组成架构 

---
# Do Retrieval Augmented Language Models Know When They Don't Know? 

**Title (ZH)**: 检索增强语言模型知道它们不知道什么时候？ 

**Authors**: Youchao Zhou, Heyan Huang, Yicheng Liu, Rui Dai, Xinglin Wang, Xingchen Zhang, Shumin Shi, Yang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.01476)  

**Abstract**: Existing Large Language Models (LLMs) occasionally generate plausible yet factually incorrect responses, known as hallucinations. Researchers are primarily using two approaches to mitigate hallucinations, namely Retrieval Augmented Language Models (RALMs) and refusal post-training. However, current research predominantly emphasizes their individual effectiveness while overlooking the evaluation of the refusal capability of RALMs. In this study, we ask the fundamental question: Do RALMs know when they don't know? Specifically, we ask three questions. First, are RALMs well-calibrated regarding different internal and external knowledge states? We examine the influence of various factors. Contrary to expectations, we find that LLMs exhibit significant \textbf{over-refusal} behavior. Then, how does refusal post-training affect the over-refusal issue? We investigate the Refusal-aware Instruction Tuning and In-Context Fine-tuning methods. Our results show that the over-refusal problem is mitigated by In-context fine-tuning. but magnified by R-tuning. However, we also find that the refusal ability may conflict with the quality of the answer. Finally, we develop a simple yet effective refusal method for refusal post-trained models to improve their overall answer quality in terms of refusal and correct answers. Our study provides a more comprehensive understanding of the influence of important factors on RALM systems. 

**Abstract (ZH)**: 现有大型语言模型（LLMs）偶尔会生成虽然合乎情理但事实错误的回答，称为幻觉。研究人员主要采用两种方法来缓解幻觉，即检索增强语言模型（RALMs）和事后拒绝机制。然而，当前研究主要强调了这两种方法的个体有效性，而忽视了对RALMs的拒绝能力的评估。在本研究中，我们提出了一个基本问题：RALMs是否知道它们不知道什么？具体来说，我们提出三个问题。首先，RALMs在不同的内部和外部知识状态下是否校准良好？我们考察了各种因素的影响。与预期相反，我们发现LLMs表现出显著的\textbf{过度拒绝}行为。其次，事后拒绝机制如何影响过度拒绝问题？我们研究了注意拒绝的指令调整和上下文调整方法。结果显示，上下文调整缓解了过度拒绝问题，但R调整使其加剧。然而，我们还发现拒绝能力可能与回答质量相冲突。最后，我们为事后拒绝机制模型开发了一种简单而有效的拒绝方法，以提高其在拒绝和正确答案方面的总体回答质量。我们的研究提供了对重要因素对RALM系统影响的更全面理解。 

---
# LLMs cannot spot math errors, even when allowed to peek into the solution 

**Title (ZH)**: LLMs无法识别数学错误，即使允许查看答案。 

**Authors**: KV Aditya Srivatsa, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2509.01395)  

**Abstract**: Large language models (LLMs) demonstrate remarkable performance on math word problems, yet they have been shown to struggle with meta-reasoning tasks such as identifying errors in student solutions. In this work, we investigate the challenge of locating the first error step in stepwise solutions using two error reasoning datasets: VtG and PRM800K. Our experiments show that state-of-the-art LLMs struggle to locate the first error step in student solutions even when given access to the reference solution. To that end, we propose an approach that generates an intermediate corrected student solution, aligning more closely with the original student's solution, which helps improve performance. 

**Abstract (ZH)**: 大型语言模型在识别学生解题中的首个错误步骤方面面临挑战：基于VtG和PRM800K错误推理数据集的探究 

---
# DPF-CM: A Data Processing Framework with Privacy-Preserving Vector Databases for Chinese Medical LLMs Training and Deployment 

**Title (ZH)**: DPF-CM：一种面向中文医疗LLM训练与部署的数据处理框架，具备隐私保护向量数据库 

**Authors**: Wei Huang, Anda Cheng, Zhao Zhang, Yinggui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01354)  

**Abstract**: Current open-source training pipelines for Chinese medical language models predominantly emphasize optimizing training methodologies to enhance the performance of large language models (LLMs), yet lack comprehensive exploration into training data processing. To address this gap, we propose DPF-CM, a holistic Data Processing Framework for Chinese Medical LLMs training and deployment. DPF-CM comprises two core modules. The first module is a data processing pipeline tailored for model training. Beyond standard data processing operations, we (1) introduce a chained examples context-learning strategy to generate question-oriented instructions to mitigate the lack of instruction content, and (2) implement an ensemble-based filtering mechanism for preference data curation that averages multiple reward models to suppress noisy samples. The second module focuses on privacy preservation during model deployment. To prevent privacy risks from the inadvertent exposure of training data, we propose a Privacy Preserving Vector Database (PPVD) approach, which involves model memory search, high-risk database construction, secure database construction, and match-and-replace, four key stages to minimize privacy leakage during inference collectively. Experimental results show that DPF-CM significantly improves model accuracy, enabling our trained Chinese medical LLM to achieve state-of-the-art performance among open-source counterparts. Moreover, the framework reduces training data privacy leakage by 27%. 

**Abstract (ZH)**: 一种全面的数据处理框架DPF-CM：用于中文医学大型语言模型的训练与部署 

---
# LLM-Guided Semantic Relational Reasoning for Multimodal Intent Recognition 

**Title (ZH)**: 基于LLM的语义关系推理的多模态意图识别 

**Authors**: Qianrui Zhou, Hua Xu, Yifan Wang, Xinzhi Dong, Hanlei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01337)  

**Abstract**: Understanding human intents from multimodal signals is critical for analyzing human behaviors and enhancing human-machine interactions in real-world scenarios. However, existing methods exhibit limitations in their modality-level reliance, constraining relational reasoning over fine-grained semantics for complex intent understanding. This paper proposes a novel LLM-Guided Semantic Relational Reasoning (LGSRR) method, which harnesses the expansive knowledge of large language models (LLMs) to establish semantic foundations that boost smaller models' relational reasoning performance. Specifically, an LLM-based strategy is proposed to extract fine-grained semantics as guidance for subsequent reasoning, driven by a shallow-to-deep Chain-of-Thought (CoT) that autonomously uncovers, describes, and ranks semantic cues by their importance without relying on manually defined priors. Besides, we formally model three fundamental types of semantic relations grounded in logical principles and analyze their nuanced interplay to enable more effective relational reasoning. Extensive experiments on multimodal intent and dialogue act recognition tasks demonstrate LGSRR's superiority over state-of-the-art methods, with consistent performance gains across diverse semantic understanding scenarios. The complete data and code are available at this https URL. 

**Abstract (ZH)**: 从多模态信号理解人类意图对于分析人类行为和增强现实场景中的人机交互至关重要。然而，现有方法在模态级别依赖方面存在局限性，限制了对复杂意图理解的细粒度语义关系推理。本文提出了一种新颖的LLM引导语义关系推理（LGSRR）方法，利用大型语言模型（LLMs）的知识建立语义基础，提升较小模型的关系推理性能。具体地，提出了一种基于大型语言模型的策略，通过从浅到深的Chain-of-Thought（CoT）自主揭示、描述和按重要性排序语义线索，作为后续推理的指导，无需依赖手动定义的先验知识。此外，我们基于逻辑原则正式建模了三种基本类型的语义关系，并分析它们的细微交互，以实现更有效的关系推理。在多模态意图和对话行为识别任务上的广泛实验表明，LGSRR优于现有最先进的方法，在多种语义理解场景中一致地提高了性能。完整的数据和代码可在以下链接获取：this https URL。 

---
# LongCat-Flash Technical Report 

**Title (ZH)**: LongCat-Flash 技术报告 

**Authors**: Meituan LongCat Team, Bayan, Bei Li, Bingye Lei, Bo Wang, Bolin Rong, Chao Wang, Chao Zhang, Chen Gao, Chen Zhang, Cheng Sun, Chengcheng Han, Chenguang Xi, Chi Zhang, Chong Peng, Chuan Qin, Chuyu Zhang, Cong Chen, Congkui Wang, Dan Ma, Daoru Pan, Defei Bu, Dengchang Zhao, Deyang Kong, Dishan Liu, Feiye Huo, Fengcun Li, Fubao Zhang, Gan Dong, Gang Liu, Gang Xu, Ge Li, Guoqiang Tan, Guoyuan Lin, Haihang Jing, Haomin Fu, Haonan Yan, Haoxing Wen, Haozhe Zhao, Hong Liu, Hongmei Shi, Hongyan Hao, Hongyin Tang, Huantian Lv, Hui Su, Jiacheng Li, Jiahao Liu, Jiahuan Li, Jiajun Yang, Jiaming Wang, Jian Yang, Jianchao Tan, Jiaqi Sun, Jiaqi Zhang, Jiawei Fu, Jiawei Yang, Jiaxi Hu, Jiayu Qin, Jingang Wang, Jiyuan He, Jun Kuang, Junhui Mei, Kai Liang, Ke He, Kefeng Zhang, Keheng Wang, Keqing He, Liang Gao, Liang Shi, Lianhui Ma, Lin Qiu, Lingbin Kong, Lingtong Si, Linkun Lyu, Linsen Guo, Liqi Yang, Lizhi Yan, Mai Xia, Man Gao, Manyuan Zhang, Meng Zhou, Mengxia Shen, Mingxiang Tuo, Mingyang Zhu, Peiguang Li, Peng Pei, Peng Zhao, Pengcheng Jia, Pingwei Sun, Qi Gu, Qianyun Li, Qingyuan Li, Qiong Huang, Qiyuan Duan, Ran Meng, Rongxiang Weng, Ruichen Shao, Rumei Li, Shizhe Wu, Shuai Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01322)  

**Abstract**: We introduce LongCat-Flash, a 560-billion-parameter Mixture-of-Experts (MoE) language model designed for both computational efficiency and advanced agentic capabilities. Stemming from the need for scalable efficiency, LongCat-Flash adopts two novel designs: (a) Zero-computation Experts, which enables dynamic computational budget allocation and activates 18.6B-31.3B (27B on average) per token depending on contextual demands, optimizing resource usage. (b) Shortcut-connected MoE, which enlarges the computation-communication overlap window, demonstrating notable gains in inference efficiency and throughput compared to models of a comparable scale. We develop a comprehensive scaling framework for large models that combines hyperparameter transfer, model-growth initialization, a multi-pronged stability suite, and deterministic computation to achieve stable and reproducible training. Notably, leveraging the synergy among scalable architectural design and infrastructure efforts, we complete model training on more than 20 trillion tokens within 30 days, while achieving over 100 tokens per second (TPS) for inference at a cost of \$0.70 per million output tokens. To cultivate LongCat-Flash towards agentic intelligence, we conduct a large-scale pre-training on optimized mixtures, followed by targeted mid- and post-training on reasoning, code, and instructions, with further augmentation from synthetic data and tool use tasks. Comprehensive evaluations demonstrate that, as a non-thinking foundation model, LongCat-Flash delivers highly competitive performance among other leading models, with exceptional strengths in agentic tasks. The model checkpoint of LongCat-Flash is open-sourced to foster community research.
LongCat Chat: this https URL
Hugging Face: this https URL
GitHub: this https URL 

**Abstract (ZH)**: LongCat-Flash：一种兼顾计算效率与先进代理能力的560亿参数Mixture-of-Experts语言模型 

---
# Rethinking the Chain-of-Thought: The Roles of In-Context Learning and Pre-trained Priors 

**Title (ZH)**: 重新思考链式思维：上下文学习与预训练先验的作用 

**Authors**: Hao Yang, Zhiyu Yang, Yunjie Zhang, Shanyi Zhu, Lin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01236)  

**Abstract**: Chain-of-Thought reasoning has emerged as a pivotal methodology for enhancing model inference capabilities. Despite growing interest in Chain-of-Thought reasoning, its underlying mechanisms remain unclear. This paper explores the working mechanisms of Chain-of-Thought reasoning from the perspective of the dual relationship between in-context learning and pretrained priors. We first conduct a fine-grained lexical-level analysis of rationales to examine the model's reasoning behavior. Then, by incrementally introducing noisy exemplars, we examine how the model balances pretrained priors against erroneous in-context information. Finally, we investigate whether prompt engineering can induce slow thinking in large language models. Our extensive experiments reveal three key findings: (1) The model not only quickly learns the reasoning structure at the lexical level but also grasps deeper logical reasoning patterns, yet it heavily relies on pretrained priors. (2) Providing sufficient exemplars shifts the model's decision-making from pretrained priors to in-context signals, while misleading prompts introduce instability. (3) Long Chain-of-Thought prompting can induce the model to generate longer reasoning chains, thereby improving its performance on downstream tasks. 

**Abstract (ZH)**: Chain-of-Thought 理论在增强模型推理能力方面的应用：基于上下文学习与预训练先验的双重关系探究 

---
# LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving 

**Title (ZH)**: 液滴GEMM：高效的W4A8 GEMM内核，用于高性能语言模型服务 

**Authors**: Huanqi Hu, Bowen Xiao, Shixuan Sun, Jianian Yin, Zhexi Zhang, Xiang Luo, Chengquan Jiang, Weiqi Xu, Xiaoying Jia, Xin Liu, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.01229)  

**Abstract**: Quantization is a critical technique for accelerating LLM inference by reducing memory footprint and improving computational efficiency. Among various schemes, 4-bit weight and 8-bit activation quantization (W4A8) offers a strong balance between accuracy and performance. However, existing W4A8 GEMM kernels fall short in practice due to inefficient dequantization on CUDA Cores, which cannot keep pace with the high throughput of Tensor Cores. In this paper, we present LiquidGEMM, a hardware-efficient W4A8 GEMM kernel for efficient LLM serving. LiquidGEMM designs two key techniques: LiquidQuant, a hardware-efficient quantization method that enables fast, overflow-safe dequantization using just two arithmetic instructions per four elements; and an implicit fine-grained pipeline that fully overlaps weight loading, dequantization, and MMA across warp groups without software synchronization or redundant memory traffic. Experimental results show that LiquidGEMM achieves up to 2.90x speedup over state-of-the-art W4A8 kernels and up to 4.94x end-to-end system-level speedup. Compared to various quantized GEMM kernels in NVIDIA TensorRT-LLM, LiquidGEMM delivers 1.12-1.63x performance gains, and achieves up to 1.63x system-level speedup. 

**Abstract (ZH)**: 量化是一种关键技术，通过减少内存占用和提高计算效率来加速大语言模型的推理。在各种方案中，4位权重和8位激活量化（W4A8）在准确性和性能之间提供了良好的平衡。然而，现有的W4A8 GEMM内核在实践中由于CUDA核心的不高效去量化而表现不佳，无法跟上张量核心的高吞吐量。本文介绍了LiquidGEMM，这是一种高效的硬件优化W4A8 GEMM内核，用于高效的LLM服务。LiquidGEMM设计了两种关键技术：LiquidQuant，这是一种高效量化方法，允许在每个四元素上仅使用两个算术指令实现快速且无上溢的安全去量化；以及一种隐式的细粒度流水线，可以在不使用软件同步或冗余内存传输的情况下，在波束组之间完全重叠权重加载、去量化和矩阵-向量乘法（MMA）操作。实验结果表明，LiquidGEMM在与最先进的W4A8内核相比时，可实现2.90倍的加速，并在整个系统水平上实现4.94倍的加速。与NVIDIA TensorRT-LLM中的各种量化GEMM内核相比，LiquidGEMM提供了1.12-1.63倍的性能增益，并实现了最高1.63倍的系统水平加速。 

---
# DaMoC: Efficiently Selecting the Optimal Large Language Model for Fine-tuning Domain Taks Based on Data and Model Compression 

**Title (ZH)**: DaMoC：基于数据和模型压缩高效选择适合领域任务微调的最佳大型语言模型 

**Authors**: Wei Huang, Huang Wei, Yinggui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.01221)  

**Abstract**: Large language models (LLMs) excel in general tasks but struggle with domain-specific ones, requiring fine-tuning with specific data. With many open-source LLMs available, selecting the best model for fine-tuning downstream tasks is challenging, primarily focusing on how to quickly identify the optimal LLM. We introduce a Data and Model Compression Framework (DaMoC) that addresses this challenge by: 1) Data Level: A systematic categorization of data filtering methodologies for LLMs is first established, classifying them into three distinct paradigms: (1) distribution-aware methods, (2) quality-aware methods, and (3) hybrid approaches considering both dimensions. Further, we enhance the density of key tokens in the text achieving token compression. Subsequently, we use an LLM to iterative rewrite the text to optimize its expression. 2) Model Level: We use layer similarity scores to assess each layer's importance and remove those with lower importance. Then, we introduce a sparse merging paradigm to preserve as much of the original model's capability as possible. Extensive experiments on four datasets, medical Q&A, financial Q&A, general Q&A, and reading comprehension, show that we can select the optimal LLM while saving approximately 20-fold in training time. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用任务上表现出色但在领域特定任务上存在困难，需要使用特定数据进行微调。由于有许多开源LLM可供选择，选择最适合下游任务微调的最佳模型变得具有挑战性，主要关注如何快速识别最优的LLM。我们介绍了一种数据和模型压缩框架（DaMoC），通过以下方式解决这一挑战：1) 数据层：首先建立了一种系统的数据过滤方法分类，分为三大范式：（1）分布感知方法，（2）质量感知方法，以及（3）结合两个维度的混合方法。进一步地，我们增强了文本中关键词的密度，实现词压缩。随后，我们使用LLM迭代重写文本以优化其表达。2) 模型层：我们使用层相似度评分评估每一层的重要性并移除重要性较低的层。然后，我们引入了一种稀疏合并范式，尽可能保留原始模型的能力。在四个数据集（医疗问答、金融问答、通用问答和阅读理解）上的广泛实验表明，我们可以在节省约20倍训练时间的前提下选择最优的LLM。 

---
# Web Fraud Attacks Against LLM-Driven Multi-Agent Systems 

**Title (ZH)**: 针对以LLM驱动的多 agent 系统的网络欺诈攻击 

**Authors**: Dezhang Kong, Hujin Peng, Yilun Zhang, Lele Zhao, Zhenhua Xu, Shi Lin, Changting Lin, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.01211)  

**Abstract**: With the proliferation of applications built upon LLM-driven multi-agent systems (MAS), the security of Web links has become a critical concern in ensuring system reliability. Once an agent is induced to visit a malicious website, attackers can use it as a springboard to conduct diverse subsequent attacks, which will drastically expand the attack surface. In this paper, we propose Web Fraud Attacks, a novel type of attack aiming at inducing MAS to visit malicious websites. We design 11 representative attack variants that encompass domain name tampering (homoglyph deception, character substitution, etc.), link structure camouflage (sub-directory nesting, sub-domain grafting, parameter obfuscation, etc.), and other deceptive techniques tailored to exploit MAS's vulnerabilities in link validation. Through extensive experiments on these crafted attack vectors, we demonstrate that Web fraud attacks not only exhibit significant destructive potential across different MAS architectures but also possess a distinct advantage in evasion: they circumvent the need for complex input formats such as jailbreaking, which inherently carry higher exposure risks. These results underscore the importance of addressing Web fraud attacks in LLM-driven MAS, as their stealthiness and destructiveness pose non-negligible threats to system security and user safety. 

**Abstract (ZH)**: 基于LLM驱动多智能体系统的网络欺诈攻击 

---
# Modular Techniques for Synthetic Long-Context Data Generation in Language Model Training and Evaluation 

**Title (ZH)**: 模块化技术在语言模型训练和评估中生成合成长上下文数据 

**Authors**: Seganrasan Subramanian, Abhigya Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.01185)  

**Abstract**: The ability of large language models (LLMs) to process and reason over long textual inputs is critical for a wide range of real-world applications. However, progress in this area is significantly constrained by the absence of high-quality, diverse, and verifiable long-context datasets suitable for both training and evaluation. This work introduces a modular, extensible framework for synthetic long-context data generation via prompt-based interaction with LLMs. The framework supports multiple training and alignment objectives, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). It encompasses four core generation paradigms: multi-turn conversational dialogues, document-grounded input-output pairs, verifiable instruction-response tasks, and long-context reasoning examples. Through templated prompting, a model-agnostic architecture, and metadata-enriched outputs, the proposed approach facilitates scalable, controllable, and purpose-aligned dataset creation for advancing long-context capabilities in LLMs. 

**Abstract (ZH)**: 大型语言模型处理和推理长文本的能力对于广泛的现实应用至关重要。然而，这一领域的进步受到缺乏高质量、多样性和可验证的长上下文数据集的限制，这些数据集适用于训练和评估。本文介绍了一种模块化、可扩展的框架，通过基于提示的与大型语言模型的交互生成合成长上下文数据。该框架支持多种训练和对齐目标，包括监督微调（SFT）、直接偏好优化（DPO）和组相对策略优化（GRPO）。它涵盖了四种核心生成范式：多轮对话、文档导向的输入-输出对、可验证的指令-响应任务以及长上下文推理示例。通过模板化提示、模型无关的架构和元数据丰富的输出，所提出的方法促进了可扩展、可控和目的导向的数据集创建，以推进大型语言模型的长上下文能力。 

---
# Enhancing Large Language Model for Knowledge Graph Completion via Structure-Aware Alignment-Tuning 

**Title (ZH)**: 通过结构 Awareness 对齐调优增强大规模语言模型在知识图谱完成中的性能 

**Authors**: Yu Liu, Yanan Cao, Xixun Lin, Yanmin Shang, Shi Wang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01166)  

**Abstract**: Knowledge graph completion (KGC) aims to infer new knowledge and make predictions from knowledge graphs. Recently, large language models (LLMs) have exhibited remarkable reasoning capabilities. LLM-enhanced KGC methods primarily focus on designing task-specific instructions, achieving promising advancements. However, there are still two critical challenges. First, existing methods often ignore the inconsistent representation spaces between natural language and graph structures. Second, most approaches design separate instructions for different KGC tasks, leading to duplicate works and time-consuming processes. To address these challenges, we propose SAT, a novel framework that enhances LLMs for KGC via structure-aware alignment-tuning. Specifically, we first introduce hierarchical knowledge alignment to align graph embeddings with the natural language space through multi-task contrastive learning. Then, we propose structural instruction tuning to guide LLMs in performing structure-aware reasoning over KGs, using a unified graph instruction combined with a lightweight knowledge adapter. Experimental results on two KGC tasks across four benchmark datasets demonstrate that SAT significantly outperforms state-of-the-art methods, especially in the link prediction task with improvements ranging from 8.7% to 29.8%. 

**Abstract (ZH)**: 基于结构感知对齐调优的大型语言模型增强知识图谱完成方法 

---
# NoLBERT: A No Lookahead(back) Foundational Language Model for Empirical Research 

**Title (ZH)**: NoLBERT：一种无前瞻（回溯）查找的基础语言模型用于实证研究 

**Authors**: Ali Kakhbod, Peiyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.01110)  

**Abstract**: We present NoLBERT, a lightweight, timestamped foundational language model for empirical research in social sciences, particularly in economics and finance. By pre-training exclusively on 1976-1995 text, NoLBERT avoids both lookback and lookahead biases that can undermine econometric inference. It exceeds domain-specific baselines on NLP benchmarks while maintaining temporal consistency. Applied to patent texts, NoLBERT enables the construction of firm-level innovation networks and shows that gains in innovation centrality predict higher long-run profit growth. 

**Abstract (ZH)**: NoLBERT：一种适用于社会科学实证研究的轻量级时间戳基础语言模型，特别是在经济学和金融学领域 

---
# Natural Context Drift Undermines the Natural Language Understanding of Large Language Models 

**Title (ZH)**: 自然上下文漂移损害了大型语言模型的自然语言理解能力 

**Authors**: Yulong Wu, Viktor Schlegel, Riza Batista-Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2509.01093)  

**Abstract**: How does the natural evolution of context paragraphs affect question answering in generative Large Language Models (LLMs)? To investigate this, we propose a framework for curating naturally evolved, human-edited variants of reading passages from contemporary QA benchmarks and for analyzing LLM performance across a range of semantic similarity scores, which quantify how closely each variant aligns with content seen during pretraining. Using this framework, we evaluate six QA datasets and eight LLMs with publicly available training data. Our experiments reveal that LLM performance declines as reading passages naturally diverge from the versions encountered during pretraining-even when the question and all necessary information remains present at inference time. For instance, average model accuracy on BoolQ drops by over 30% from the highest to lowest similarity bins, with slopes exceeding 70 across several LLMs. These findings suggest that natural text evolution poses a significant challenge to the language understanding capabilities of LLMs. 

**Abstract (ZH)**: 自然演化背景下环境段落对生成型大型语言模型问答影响的研究：一种基于当代问答基准和语义相似度分析的框架 

---
# REFRAG: Rethinking RAG based Decoding 

**Title (ZH)**: REFRAG: 重思RAG基于的解码方法 

**Authors**: Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, Vijai Mohan  

**Link**: [PDF](https://arxiv.org/pdf/2509.01092)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in leveraging extensive external knowledge to enhance responses in multi-turn and agentic applications, such as retrieval-augmented generation (RAG). However, processing long-context inputs introduces significant system latency and demands substantial memory for the key-value cache, resulting in reduced throughput and a fundamental trade-off between knowledge enrichment and system efficiency. While minimizing latency for long-context inputs is a primary objective for LLMs, we contend that RAG require specialized consideration. In RAG, much of the LLM context consists of concatenated passages from retrieval, with only a small subset directly relevant to the query. These passages often exhibit low semantic similarity due to diversity or deduplication during re-ranking, leading to block-diagonal attention patterns that differ from those in standard LLM generation tasks. Based on this observation, we argue that most computations over the RAG context during decoding are unnecessary and can be eliminated with minimal impact on performance. To this end, we propose REFRAG, an efficient decoding framework that compresses, senses, and expands to improve latency in RAG applications. By exploiting the sparsity structure, we demonstrate a 30.85 the time-to-first-token acceleration (3.75 improvement to previous work) without loss in perplexity. In addition, our optimization framework for large context enables REFRAG to extend the context size of LLMs by 16. We provide rigorous validation of REFRAG across diverse long-context tasks, including RAG, multi-turn conversations, and long document summarization, spanning a wide range of datasets. Experimental results confirm that REFRAG delivers substantial speedup with no loss in accuracy compared to LLaMA models and other state-of-the-art baselines across various context sizes. 

**Abstract (ZH)**: 高效解码框架 REFRAG：针对检索增强生成任务的上下文压缩与扩展方法 

---
# DSDE: Dynamic Speculative Decoding with KLD Stability for Real-World Serving 

**Title (ZH)**: DSDE：动态投机解码与KLD稳定性在实际部署中的应用 

**Authors**: Mingyu Yang, Jae-Young Choi, Kihyo Moon, Minsung Jang, Eunjoo Joen  

**Link**: [PDF](https://arxiv.org/pdf/2509.01083)  

**Abstract**: Speculative decoding accelerates large language model inference, but its reliance on a fixed speculation length is suboptimal in large-batch serving environments with diverse requests. This paper explores a new direction for dynamic adaptation by investigating a novel class of post-hoc, diagnostic signals. We propose Dynamic Speculative Decoding Engine (DSDE), a training-free framework built on two primary components: (1) a predictive signal based on the variance of the Kullback-Leibler (KLD) divergence, which diagnoses the generation's regional stability, and (2) an adaptive speculation length cap to mitigate the straggler problem in per-sequence decoding. Experiments demonstrate the potential of using KLD-based stability signals for dynamic adaptation. An algorithm guided by these signals achieves end-to-end latency competitive with leading baselines and exhibits superior robustness across diverse workloads. This robustness is particularly valuable in challenging low-acceptance-rate regimes, where the proposed signal maintains its diagnostic utility. Collectively, these findings validate post-hoc signals as a valuable component for building more robust and intelligent LLM inference systems, and highlight a promising direction for future research on dynamic speculation length adaptation. 

**Abstract (ZH)**: 投机解码加速了大规模语言模型推理，但在具有多样化请求的大批量服务环境中，其依赖于固定投机长度的特性不尽 optimal。本文探索了动态适应的新方向，通过调查一种新的后处理诊断信号。我们提出了动态投机解码引擎（DSDE），这是一个无需训练的框架，由两个主要组件构成：（1）基于Kullback-Leibler（KLD）散度方差的预测信号，用于诊断生成区域的稳定性；（2）一种自适应的投机长度上限，以缓解逐序列解码中的拖尾问题。实验表明，使用基于KLD的稳定性信号进行动态适应具有潜力。由这些信号指导的算法实现了与领先基线相媲美的端到端延迟，并且在多种工作负载下表现出更优的鲁棒性。特别是在低接受率的挑战环境中，所提信号保持了其诊断作用。这些发现验证了后处理信号作为构建更鲁棒和智能的大规模语言模型推理系统的重要组件的价值，并强调了未来研究动态推测长度适应的有希望的方向。 

---
# Assessing Large Language Models on Islamic Legal Reasoning: Evidence from Inheritance Law Evaluation 

**Title (ZH)**: 评估大型语言模型在伊斯兰法律推理中的表现：遗产法评估的证据 

**Authors**: Abdessalam Bouchekif, Samer Rashwani, Heba Sbahi, Shahd Gaben, Mutez Al-Khatib, Mohammed Ghaly  

**Link**: [PDF](https://arxiv.org/pdf/2509.01081)  

**Abstract**: This paper evaluates the knowledge and reasoning capabilities of Large Language Models in Islamic inheritance law, known as 'ilm al-mawarith. We assess the performance of seven LLMs using a benchmark of 1,000 multiple-choice questions covering diverse inheritance scenarios, designed to test models' ability to understand the inheritance context and compute the distribution of shares prescribed by Islamic jurisprudence. The results reveal a significant performance gap: o3 and Gemini 2.5 achieved accuracies above 90%, whereas ALLaM, Fanar, LLaMA, and Mistral scored below 50%. These disparities reflect important differences in reasoning ability and domain adaptation. We conduct a detailed error analysis to identify recurring failure patterns across models, including misunderstandings of inheritance scenarios, incorrect application of legal rules, and insufficient domain knowledge. Our findings highlight limitations in handling structured legal reasoning and suggest directions for improving performance in Islamic legal reasoning. Code: this https URL 

**Abstract (ZH)**: 本文评估了大型语言模型在伊斯兰继承法（‘ilm al-mawarith）方面的知识和推理能力。我们使用包含1000道多项选择题的基准测试，涵盖了多种继承场景，以评估模型理解继承背景并计算由伊斯兰教法规定的份额分配的能力。结果显示，性能存在显著差距：o3和Gemini 2.5的准确率超过90%，而ALLaM、Fanar、LLaMA和Mistral的得分低于50%。这些差异反映了推理能力和领域适应性的重大区别。我们进行了详细错误分析，以识别模型之间的反复出现的失败模式，包括对继承场景的误解、法律规则应用错误以及领域知识不足。我们的研究结果指出了处理结构化法律推理的局限性，并建议改进伊斯兰法律推理性能的方向。代码：请参见此链接。 

---
# Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation 

**Title (ZH)**: 无法窃取就复制：基于 logits 泄漏和蒸馏的黑盒大语言模型复制 

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz  

**Link**: [PDF](https://arxiv.org/pdf/2509.00973)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在关键任务系统中的应用日益增多，促进卫星操作、指挥控制、军事决策支持和网络防御等任务的完成。许多这类系统是通过应用程序编程接口（APIs）访问的。当这些API缺乏 robust 访问控制时，它们可能会暴露完整的或前k个logits，创建一个显著且经常被忽视的攻击面。现有研究主要集中在重构输出投影层或提取表面行为上。然而，在严格查询约束下再生一个黑盒模型仍然未被充分探索。我们通过引入一个受限复制流水线来填补这一空白，将部分logits泄漏转化为一个功能性的可部署的替代模型克隆。我们的两阶段方法包括：(i) 通过 singular value decomposition (SVD) 在logits上收集前k个logits来重构输出投影矩阵，然后 (ii) 将剩余的架构提炼成具有不同变压器深度的紧凑型学生模型，并在开源数据集上进行训练。一个6层的学生模型复制了6层教师模型97.6%的隐藏状态几何结构，仅增加7.31%的困惑度，和18.1的负对数似然（NLL）。一个4层的变体实现了17.1%的更快推理速度和18.1%的参数减少，同时保持了可比性能。整个攻击在不到24个图形处理单元（GPU）小时内完成，并避免触发API速率限制防御。这些结果展示了在成本限制下对手快速克隆LLM的速度，突显了需要强化推断API和安全的本地防御部署的迫切性。 

---
# Who Gets Left Behind? Auditing Disability Inclusivity in Large Language Models 

**Title (ZH)**: 谁会被抛在后面？大型语言模型中的残疾人包容性审计 

**Authors**: Deepika Dash, Yeshil Bangera, Mithil Bangera, Gouthami Vadithya, Srikant Panda  

**Link**: [PDF](https://arxiv.org/pdf/2509.00963)  

**Abstract**: Large Language Models (LLMs) are increasingly used for accessibility guidance, yet many disability groups remain underserved by their advice. To address this gap, we present taxonomy aligned benchmark1 of human validated, general purpose accessibility questions, designed to systematically audit inclusivity across disabilities. Our benchmark evaluates models along three dimensions: Question-Level Coverage (breadth within answers), Disability-Level Coverage (balance across nine disability categories), and Depth (specificity of support). Applying this framework to 17 proprietary and open-weight models reveals persistent inclusivity gaps: Vision, Hearing, and Mobility are frequently addressed, while Speech, Genetic/Developmental, Sensory-Cognitive, and Mental Health remain under served. Depth is similarly concentrated in a few categories but sparse elsewhere. These findings reveal who gets left behind in current LLM accessibility guidance and highlight actionable levers: taxonomy-aware prompting/training and evaluations that jointly audit breadth, balance, and depth. 

**Abstract (ZH)**: 大型语言模型（LLMs）在无障碍指导中的应用日益增多，但仍有许多残疾人群体未得到充分的服务。为填补这一空白，我们提出了一项基于分类学的基准测试1，该基准测试包含人类验证的一般无障碍问题，旨在系统地评估不同类型残疾的包容性。该基准测试从三个维度评估模型：问题层面覆盖面（答案内的广度）、残疾层面覆盖面（九类残疾的平衡程度）以及深度（支持的专门性）。将这一框架应用于17个专有和开源模型揭示了持续存在的包容性差距：视力、听力和行动障碍经常被提及，而言语、遗传/发育、感觉认知和心理健康障碍仍然未得到充分的处理。深度也在少数类别中集中，而在其他领域则较为稀疏。这些发现揭示了当前LLM无障碍指导中被忽视的人群，并突显了可操作的杠杆：分类学意识的提示/训练和综合评估广度、平衡和深度的评估方法。 

---
# Structure and Destructure: Dual Forces in the Making of Knowledge Engines 

**Title (ZH)**: 结构与去结构：知识引擎形成中的双重力量 

**Authors**: Yihong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.00949)  

**Abstract**: The making of knowledge engines in natural language processing has been shaped by two seemingly distinct paradigms: one grounded in structure, the other driven by massively available unstructured data. The structured paradigm leverages predefined symbolic interactions, such as knowledge graphs, as priors and designs models to capture them. In contrast, the unstructured paradigm centers on scaling transformer architectures with increasingly vast data and model sizes, as seen in modern large language models. Despite their divergence, this thesis seeks to establish conceptual connections bridging these paradigms. Two complementary forces, structure and destructure, emerge across both paradigms: structure organizes seen symbolic interactions, while destructure, through periodic embedding resets, improves model plasticity and generalization to unseen scenarios. These connections form a new recipe for developing general knowledge engines that can support transparent, controllable, and adaptable intelligent systems. 

**Abstract (ZH)**: 自然语言处理中知识引擎的构建受到两种看似迥异的范式的影响：一种基于结构，另一种源于大规模的非结构化数据。基于结构的范式利用预定义的符号交互，如知识图谱，作为先验知识，并设计模型来捕捉这些交互。相比之下，非结构化范式则侧重于通过不断增大数据量和模型规模来扩展变压器架构。尽管这两种范式存在差异，本论文旨在建立这些范式之间的概念联系。两种互补的力量——结构与解构——在两种范式中同时出现：结构组织已知的符号交互，而解构通过周期性的嵌入重置，提高模型的可塑性和对未见场景的泛化能力。这些联系形成了一种新的配方，用于开发支持透明、可控和适应性强的智能系统的通用知识引擎。 

---
# MedCOD: Enhancing English-to-Spanish Medical Translation of Large Language Models Using Enriched Chain-of-Dictionary Framework 

**Title (ZH)**: MedCOD：利用丰富化的链词典框架增强大型语言模型的英语到西班牙语医学翻译 

**Authors**: Md Shahidul Salim, Lian Fu, Arav Adikesh Ramakrishnan, Zonghai Yao, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00934)  

**Abstract**: We present MedCOD (Medical Chain-of-Dictionary), a hybrid framework designed to improve English-to-Spanish medical translation by integrating domain-specific structured knowledge into large language models (LLMs). MedCOD integrates domain-specific knowledge from both the Unified Medical Language System (UMLS) and the LLM-as-Knowledge-Base (LLM-KB) paradigm to enhance structured prompting and fine-tuning. We constructed a parallel corpus of 2,999 English-Spanish MedlinePlus articles and a 100-sentence test set annotated with structured medical contexts. Four open-source LLMs (Phi-4, Qwen2.5-14B, Qwen2.5-7B, and LLaMA-3.1-8B) were evaluated using structured prompts that incorporated multilingual variants, medical synonyms, and UMLS-derived definitions, combined with LoRA-based fine-tuning. Experimental results demonstrate that MedCOD significantly improves translation quality across all models. For example, Phi-4 with MedCOD and fine-tuning achieved BLEU 44.23, chrF++ 28.91, and COMET 0.863, surpassing strong baseline models like GPT-4o and GPT-4o-mini. Ablation studies confirm that both MedCOD prompting and model adaptation independently contribute to performance gains, with their combination yielding the highest improvements. These findings highlight the potential of structured knowledge integration to enhance LLMs for medical translation tasks. 

**Abstract (ZH)**: MedCOD（医疗链词条典）：一种将领域特定结构化知识集成到大型语言模型中的混合框架，以提高英西医疗翻译质量 

---
# CaresAI at BioCreative IX Track 1 -- LLM for Biomedical QA 

**Title (ZH)**: CaresAI在BioCreative IX Track 1—— biomedical QA中的LLM应用 

**Authors**: Reem Abdel-Salam, Mary Adewunmi, Modinat A. Abayomi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00806)  

**Abstract**: Large language models (LLMs) are increasingly evident for accurate question answering across various domains. However, rigorous evaluation of their performance on complex question-answering (QA) capabilities is essential before deployment in real-world biomedical and healthcare applications. This paper presents our approach to the MedHopQA track of the BioCreative IX shared task, which focuses on multi-hop biomedical question answering involving diseases, genes, and chemicals. We adopt a supervised fine-tuning strategy leveraging LLaMA 3 8B, enhanced with a curated biomedical question-answer dataset compiled from external sources including BioASQ, MedQuAD, and TREC. Three experimental setups are explored: fine-tuning on combined short and long answers, short answers only, and long answers only. While our models demonstrate strong domain understanding, achieving concept-level accuracy scores of up to 0.8, their Exact Match (EM) scores remain significantly lower, particularly in the test phase. We introduce a two-stage inference pipeline for precise short-answer extraction to mitigate verbosity and improve alignment with evaluation metrics. Despite partial improvements, challenges persist in generating strictly formatted outputs. Our findings highlight the gap between semantic understanding and exact answer evaluation in biomedical LLM applications, motivating further research in output control and post-processing strategies. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各领域准确回答问题方面的作用越来越明显。然而，在将其部署到实际生物医学和医疗保健应用之前，对其复杂问答（QA）能力的性能进行严格的评估是必不可少的。本文介绍了我们参加BioCreative IX共享任务MedHopQA track的方法，该轨道侧重于涉及疾病、基因和化学物质的多跳生物医学问答。我们采用了一种基于LLaMA 3 8B的监督微调策略，并结合了从BioASQ、MedQuAD和TREC等外部来源编制的精心策划的生物医学问答数据集。探索了三种实验设置：结合短答案和长答案的微调、仅短答案和仅长答案的微调。尽管我们的模型显示出强大的领域理解能力，达到概念级准确度评分高达0.8，但在测试阶段，其精确匹配（EM）评分仍然显著较低。我们引入了一种两阶段推理管道，以减轻冗余并提高与评估指标的对齐程度，以精确提取短答案。尽管取得了一定的改进，生成严格格式化的输出仍然存在挑战。我们的研究结果突显了生物医学LLM应用中语义理解与精确答案评估之间的差距，激励进一步研究输出控制和后处理策略。 

---
# Reward-Weighted Sampling: Enhancing Non-Autoregressive Characteristics in Masked Diffusion LLMs 

**Title (ZH)**: 带奖励加权采样的非自回归特性增强型掩码扩散大语言模型 

**Authors**: Daehoon Gwak, Minseo Jung, Junwoo Park, Minho Park, ChaeHun Park, Junha Hyung, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00707)  

**Abstract**: Masked diffusion models (MDMs) offer a promising non-autoregressive alternative for large language modeling. Standard decoding methods for MDMs, such as confidence-based sampling, select tokens independently based on individual token confidences at each diffusion step. However, we observe that this independent token selection often results in generation orders resembling sequential autoregressive processes, limiting the advantages of non-autoregressive modeling. To mitigate this pheonomenon, we propose Reward-Weighted Sampling (RWS), a novel decoding strategy that leverages an external reward model to provide a principled global signal during the iterative diffusion process. Specifically, at each diffusion step, RWS evaluates the quality of the entire intermediate sequence and scales token logits accordingly, guiding token selection by integrating global sequence-level coherence. This method selectively increases the confidence of tokens that initially have lower scores, thereby promoting a more non-autoregressive generation order. Furthermore, we provide theoretical justification showing that reward-weighted logit scaling induces beneficial rank reversals in token selection and consistently improves expected reward. Experiments demonstrate that RWS significantly promotes non-autoregressive generation orders, leading to improvements across multiple evaluation metrics. These results highlight the effectiveness of integrating global signals in enhancing both the non-autoregressive properties and overall performance of MDMs. 

**Abstract (ZH)**: Masked扩散模型的奖励加权采样促进非自回归语言生成 

---
# Confident, Calibrated, or Complicit: Probing the Trade-offs between Safety Alignment and Ideological Bias in Language Models in Detecting Hate Speech 

**Title (ZH)**: 自信、校准或同谋：检测仇恨言论中语言模型的安全对齐与意识形态偏见之间的权衡探讨论文标题 

**Authors**: Sanjeeevan Selvaganapathy, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2509.00673)  

**Abstract**: We investigate the efficacy of Large Language Models (LLMs) in detecting implicit and explicit hate speech, examining whether models with minimal safety alignment (uncensored) might provide more objective classification capabilities compared to their heavily-aligned (censored) counterparts. While uncensored models theoretically offer a less constrained perspective free from moral guardrails that could bias classification decisions, our results reveal a surprising trade-off: censored models significantly outperform their uncensored counterparts in both accuracy and robustness, achieving 78.7% versus 64.1% strict accuracy. However, this enhanced performance comes with its own limitation -- the safety alignment acts as a strong ideological anchor, making censored models resistant to persona-based influence, while uncensored models prove highly malleable to ideological framing. Furthermore, we identify critical failures across all models in understanding nuanced language such as irony. We also find alarming fairness disparities in performance across different targeted groups and systemic overconfidence that renders self-reported certainty unreliable. These findings challenge the notion of LLMs as objective arbiters and highlight the need for more sophisticated auditing frameworks that account for fairness, calibration, and ideological consistency. 

**Abstract (ZH)**: 我们调查了大型语言模型（LLMs）在检测隐含和明示仇恨言论方面的有效性，探讨了与安全对齐程度较低（未审查）的模型是否可能在分类准确性上比高度对齐（审查）的模型更客观。虽然理论上未审查模型提供了较少约束的、不受道德护栏限制的视角，可能减少分类决策中的偏见，但我们的结果显示了一个意想不到的权衡：审查模型在准确性和稳健性上的表现显著优于未审查模型，分别达到了78.7%和64.1%的严格准确度。然而，这种增强的表现伴随着自己的局限性——安全性对齐起到了强烈的意识形态锚定作用，使审查模型对个性化的影响力具有抵抗力，而未审查模型则在意识形态框架下非常可塑。此外，我们在所有模型中识别出对含蓄语言如讽刺理解上的关键失败。我们还发现不同目标群体在性能上的不公平差异，在系统性上的过度自信使自我报告的确定性不可靠。这些发现挑战了LLMs作为客观仲裁者的观念，并突显了需要更为复杂的审计框架，以考虑到公平性、校准和意识形态一致性。 

---
# LLM-HyPZ: Hardware Vulnerability Discovery using an LLM-Assisted Hybrid Platform for Zero-Shot Knowledge Extraction and Refinement 

**Title (ZH)**: LLM-HyPZ：一种用于零-shot知识抽取与精炼的LLM辅助混合平台硬件漏洞发现方法 

**Authors**: Yu-Zheng Lin, Sujan Ghimire, Abhiram Nandimandalam, Jonah Michael Camacho, Unnati Tripathi, Rony Macwan, Sicong Shao, Setareh Rafatirad, Rozhin Yasaei, Pratik Satam, Soheil Salehi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00647)  

**Abstract**: The rapid growth of hardware vulnerabilities has created an urgent need for systematic and scalable analysis methods. Unlike software flaws, which are often patchable post-deployment, hardware weaknesses remain embedded across product lifecycles, posing persistent risks to processors, embedded devices, and IoT platforms. Existing efforts such as the MITRE CWE Hardware List (2021) relied on expert-driven Delphi surveys, which lack statistical rigor and introduce subjective bias, while large-scale data-driven foundations for hardware weaknesses have been largely absent. In this work, we propose LLM-HyPZ, an LLM-assisted hybrid framework for zero-shot knowledge extraction and refinement from vulnerability corpora. Our approach integrates zero-shot LLM classification, contextualized embeddings, unsupervised clustering, and prompt-driven summarization to mine hardware-related CVEs at scale. Applying LLM-HyPZ to the 2021-2024 CVE corpus (114,836 entries), we identified 1,742 hardware-related vulnerabilities. We distilled them into five recurring themes, including privilege escalation via firmware and BIOS, memory corruption in mobile and IoT systems, and physical access exploits. Benchmarking across seven LLMs shows that LLaMA 3.3 70B achieves near-perfect classification accuracy (99.5%) on a curated validation set. Beyond methodological contributions, our framework directly supported the MITRE CWE Most Important Hardware Weaknesses (MIHW) 2025 update by narrowing the candidate search space. Specifically, our pipeline surfaced 411 of the 1,026 CVEs used for downstream MIHW analysis, thereby reducing expert workload and accelerating evidence gathering. These results establish LLM-HyPZ as the first data-driven, scalable approach for systematically discovering hardware vulnerabilities, thereby bridging the gap between expert knowledge and real-world vulnerability evidence. 

**Abstract (ZH)**: 硬件漏洞的快速增长迫切需要系统化的可扩展分析方法。现有的努力如MITRE CWE硬件列表（2021年）依赖于专家驱动的德尔菲调查，缺乏统计严谨性并引入了主观偏差，而大规模的数据驱动硬件弱点基础几乎不存在。在此工作中，我们提出了LLM-HyPZ，这是一种基于LLM的混合框架，用于零样本知识提取和 refinement 从漏洞语料库中。我们的方法结合了零样本LLM分类、上下文向量表示、无监督聚类和提示驱动总结，以大规模挖掘硬件相关的CVE。将LLM-HyPZ应用于2021-2024年CVE语料库（114,836条记录），我们识别了1,742个硬件相关的漏洞。我们将这些漏洞提炼为五个重复的主题，包括通过固件和BIOS的权限提升、移动和物联网系统中的内存损坏，以及物理访问利用。在七个LLM的基准测试中，LLaMA 3.3 70B在精心策划的验证集上实现了近乎完美的分类准确性（99.5%）。除了方法论的贡献，我们的框架直接支持了MITRE CWE最重要的硬件弱点（MIHW）2025年的更新，通过缩小候选搜索空间。具体来说，我们的流水线揭示了用于下游MIHW分析的1,026个CVE中的411个，从而减轻了专家的工作负担并加速了证据收集。这些结果确立了LLM-HyPZ作为系统发现硬件漏洞的第一个数据驱动和可扩展的方法，从而弥合了专家知识与实际漏洞证据之间的差距。 

---
# RAG-PRISM: A Personalized, Rapid, and Immersive Skill Mastery Framework with Adaptive Retrieval-Augmented Tutoring 

**Title (ZH)**: RAG-PRISM: 一种个性化、快速且沉浸式的能力掌握框架，具有适应性检索增强辅导 

**Authors**: Gaurangi Raul, Yu-Zheng Lin, Karan Patel, Bono Po-Jen Shih, Matthew W. Redondo, Banafsheh Saber Latibari, Jesus Pacheco, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2509.00646)  

**Abstract**: The rapid digital transformation of Fourth Industrial Revolution (4IR) systems is reshaping workforce needs, widening skill gaps, especially for older workers. With growing emphasis on STEM skills such as robotics, automation, artificial intelligence (AI), and security, large-scale re-skilling and up-skilling are required. Training programs must address diverse backgrounds, learning styles, and motivations to improve persistence and success, while ensuring rapid, cost-effective workforce development through experiential learning. To meet these challenges, we present an adaptive tutoring framework that combines generative AI with Retrieval-Augmented Generation (RAG) to deliver personalized training. The framework leverages document hit rate and Mean Reciprocal Rank (MRR) to optimize content for each learner, and is benchmarked against human-generated training for alignment and relevance. We demonstrate the framework in 4IR cybersecurity learning by creating a synthetic QA dataset emulating trainee behavior, while RAG is tuned on curated cybersecurity materials. Evaluation compares its generated training with manually curated queries representing realistic student interactions. Responses are produced using large language models (LLMs) including GPT-3.5 and GPT-4, assessed for faithfulness and content alignment. GPT-4 achieves the best performance with 87% relevancy and 100% alignment. Results show this dual-mode approach enables the adaptive tutor to act as both a personalized topic recommender and content generator, offering a scalable solution for rapid, tailored learning in 4IR education and workforce development. 

**Abstract (ZH)**: 第四次工业革命系统中的快速数字化转型重塑了劳动力需求，扩大了技能差距，尤其是对老年工作者的影响。随着对STEM技能如机器人、自动化、人工智能（AI）和安全的重视增加，大规模的再培训和提升技能变得至关重要。培训项目必须考虑到多样化背景、学习风格和动机，以提高坚持度和成功率，同时确保通过体验学习实现快速而经济高效的劳动力发展。为了应对这些挑战，我们提出了一种结合生成AI与检索增强生成（RAG）的自适应辅导框架，以提供个性化的培训。该框架利用文档命中率和均倒数排名（MRR）来为每位学习者优化内容，并与人工生成的培训内容进行基准测试，以确保一致性和相关性。我们通过创建模拟学员行为的合成QA数据集并在策展的网络安全材料上调整RAG，在第四次工业革命（4IR）网络安全学习中展示了该框架。评估将生成的培训内容与手动策划的反映真实学生互动的查询进行比较。响应使用大型语言模型（LLMs）生成，包括GPT-3.5和GPT-4，评估其忠实度和内容一致性。GPT-4在相关性和一致性方面均表现最佳，分别为87%和100%。结果显示，这种双模式方法使自适应导师不仅能作为个性化主题推荐者，还能作为内容生成者，为4IR教育和劳动力发展提供可扩展的快速定制学习解决方案。 

---
# A Multi-Strategy Approach for AI-Generated Text Detection 

**Title (ZH)**: 基于多种策略的AI生成文本检测方法 

**Authors**: Ali Zain, Sareem Farooqui, Muhammad Rafi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00623)  

**Abstract**: This paper presents presents three distinct systems developed for the M-DAIGT shared task on detecting AI generated content in news articles and academic abstracts. The systems includes: (1) A fine-tuned RoBERTa-base classifier, (2) A classical TF-IDF + Support Vector Machine (SVM) classifier , and (3) An Innovative ensemble model named Candace, leveraging probabilistic features extracted from multiple Llama-3.2 models processed by a customTransformer this http URL RoBERTa-based system emerged as the most performant, achieving near-perfect results on both development and test sets. 

**Abstract (ZH)**: 本文介绍了为M-DAIGT共享任务开发的用于检测新闻文章和学术摘要中AI生成内容的三个独立系统。这些系统包括：(1) 微调的RoBERTa基分类器，(2) 经典的TF-IDF + 支持向量机(SVM)分类器，以及(3) 一种名为Candace的创新集成模型，该模型利用从多个Llama-3.2模型中提取的概率特征并通过自定义Transformer处理。RoBERTa基系统在开发集和测试集上均取得了近乎完美的性能。 

---
# TimeCopilot 

**Title (ZH)**: 时光共驾 

**Authors**: Azul Garza, Reneé Rosillo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00616)  

**Abstract**: We introduce TimeCopilot, the first open-source agentic framework for forecasting that combines multiple Time Series Foundation Models (TSFMs) with Large Language Models (LLMs) through a single unified API. TimeCopilot automates the forecasting pipeline: feature analysis, model selection, cross-validation, and forecast generation, while providing natural language explanations and supporting direct queries about the future. The framework is LLM-agnostic, compatible with both commercial and open-source models, and supports ensembles across diverse forecasting families. Results on the large-scale GIFT-Eval benchmark show that TimeCopilot achieves state-of-the-art probabilistic forecasting performance at low cost. Our framework provides a practical foundation for reproducible, explainable, and accessible agentic forecasting systems. 

**Abstract (ZH)**: TimeCopilot：首个结合多时间序列基础模型与大型语言模型的开源代理框架 

---
# KVComp: A High-Performance, LLM-Aware, Lossy Compression Framework for KV Cache 

**Title (ZH)**: KVComp：一种高性能、面向LLM的损失性压缩框架用于KV缓存 

**Authors**: Bo Jiang, Taolue Yang, Youyuan Liu, Chengming Zhang, Xubin He, Sian Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.00579)  

**Abstract**: Transformer-based large language models (LLMs) demonstrate impressive potential in various practical applications. However, long context inference poses a significant challenge due to the enormous memory requirements of the key-value (KV) cache, which can scale to multiple gigabytes as sequence length and batch size increase. In this paper, we present KVComp, a generic and efficient KV cache management framework optimized for long-text generation that synergistically works with both latency-critical and throughput-critical inference systems. KVComp employs novel lossy compression techniques specifically designed for KV cache data characteristics, featuring careful co-design of compression algorithms and system architecture. Our approach maintains compatibility with the growing nature of KV cache while preserving high computational efficiency. Experimental results show that KVComp achieves on average 47\% and up to 83\% higher memory reduction rate compared to existing methods with little/no model accuracy degradation. Furthermore, KVComp achieves extremely high execution throughput, effectively reducing decompression overhead and, in some cases, even accelerating the matrix-vector multiplication operation and outperform cuBLAS-based attention kernels with less data movement. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）在各种实际应用中展现出巨大的潜力。然而，长上下文推理由于关键值（KV）缓存的巨大内存需求而面临重大挑战，该需求可随着序列长度和批量大小的增加扩展至数GB。本文介绍了KVComp，一种针对长文本生成优化的一般且高效的KV缓存管理框架，可与关键时延和关键吞吐量的推理系统协同工作。KVComp采用专门为KV缓存数据特性设计的新颖有损压缩技术，结合压缩算法和系统架构的细致设计。我们的方法保持了KV缓存的扩展性，同时保持了高计算效率。实验结果显示，与现有方法相比，KVComp平均实现了47%至83%更高的内存减少率，并且几乎不降低模型准确性。此外，KVComp实现了极高的执行吞吐量，有效减少了解压缩开销，在某些情况下甚至加快了矩阵-向量乘法操作，并在更少数据移动的情况下超过了基于cuBLAS的注意力内核。 

---
# Talk Less, Call Right: Enhancing Role-Play LLM Agents with Automatic Prompt Optimization and Role Prompting 

**Title (ZH)**: 少说话，精准干预：通过自动提示优化与角色提示增强角色扮演大语言模型代理 

**Authors**: Saksorn Ruangtanusak, Pittawat Taveekitworachai, Kunat Pipatanakul  

**Link**: [PDF](https://arxiv.org/pdf/2509.00482)  

**Abstract**: This report investigates approaches for prompting a tool-augmented large language model (LLM) to act as a role-playing dialogue agent in the API track of the Commonsense Persona-grounded Dialogue Challenge (CPDC) 2025. In this setting, dialogue agents often produce overly long in-character responses (over-speaking) while failing to use tools effectively according to the persona (under-acting), such as generating function calls that do not exist or making unnecessary tool calls before answering. We explore four prompting approaches to address these issues: 1) basic role prompting, 2) human-crafted role prompting, 3) automatic prompt optimization (APO), and 4) rule-based role prompting. The rule-based role prompting (RRP) approach achieved the best performance through two novel techniques--character-card/scene-contract design and strict enforcement of function calling--which led to an overall score of 0.571, improving on the zero-shot baseline score of 0.519. These findings demonstrate that RRP design can substantially improve the effectiveness and reliability of role-playing dialogue agents compared with more elaborate methods such as APO. To support future efforts in developing persona prompts, we are open-sourcing all of our best-performing prompts and the APO tool. Source code is available at this https URL. 

**Abstract (ZH)**: 本报告调研了促使工具增强的大语言模型（LLM）在2025年常识角色导向对话挑战（CPDC）API赛道中扮演角色扮演对话代理的方法。在这种设定中，对话代理经常产生过长的入戏回答（过演），而在根据个性使用工具方面表现不足（欠演），例如生成不存在的功能调用或在回答前进行不必要的工具调用。我们探索了四种提示方法以解决这些问题：1）基本角色提示，2）人工设计的角色提示，3）自动提示优化（APO），4）基于规则的角色提示。基于两种新颖的技术——角色卡/场景合同设计和严格的功能调用执行，基于规则的角色提示（RRP）方法取得了最佳性能，得分为0.571，比零-shot基线得分0.519有了显著提升。这些发现表明，RRP设计可以显著提高角色扮演对话代理的有效性和可靠性，相比更加复杂的AoP方法更为有效。为了支持未来关于个性提示的开发工作，我们公开了所有性能最佳的提示和APO工具的源代码。相关源代码可在以下链接获取。 

---
# TECP: Token-Entropy Conformal Prediction for LLMs 

**Title (ZH)**: TECP: Token-Entropy Conformal Prediction for LLMs 

**Authors**: Beining Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00461)  

**Abstract**: Uncertainty quantification (UQ) for open-ended language generation remains a critical yet underexplored challenge, especially under black-box constraints where internal model signals are inaccessible. In this paper, we introduce Token-Entropy Conformal Prediction (TECP), a novel framework that leverages token-level entropy as a logit-free, reference-free uncertainty measure and integrates it into a split conformal prediction (CP) pipeline to construct prediction sets with formal coverage guarantees. Unlike existing approaches that rely on semantic consistency heuristics or white-box features, TECP directly estimates epistemic uncertainty from the token entropy structure of sampled generations and calibrates uncertainty thresholds via CP quantiles to ensure provable error control. Empirical evaluations across six large language models and two benchmarks (CoQA and TriviaQA) demonstrate that TECP consistently achieves reliable coverage and compact prediction sets, outperforming prior self-consistency-based UQ methods. Our method provides a principled and efficient solution for trustworthy generation in black-box LLM settings. 

**Abstract (ZH)**: 开放生成任务中基于黑箱约束的不确定性量化（UQ）仍是一个关键但尚未充分探索的挑战。本文提出了Token-Entropy Conformal Prediction (TECP)，这是一种新颖的方法，利用token级熵作为无logit、无参考的不确定性度量，并将其集成到分立的置信预测（CP）管道中，以构建具有形式覆盖保证的预测集。TECP直接从采样生成的token熵结构中估计认识论不确定性，并通过CP分位数校准不确定性阈值以确保可证明的误差控制。在六个大型语言模型和两个基准（CoQA和TriviaQA）上的实证评估表明，TECP在实现可靠覆盖和紧凑预测集方面表现优异，优于先前基于自我一致性的方法。本文方法为黑箱大语言模型中的可信生成提供了原理上有效的解决方案。 

---
# MedSEBA: Synthesizing Evidence-Based Answers Grounded in Evolving Medical Literature 

**Title (ZH)**: 医学生物信息合成：基于不断更新医学文献的证据支持答案合成 

**Authors**: Juraj Vladika, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2509.00414)  

**Abstract**: In the digital age, people often turn to the Internet in search of medical advice and recommendations. With the increasing volume of online content, it has become difficult to distinguish reliable sources from misleading information. Similarly, millions of medical studies are published every year, making it challenging for researchers to keep track of the latest scientific findings. These evolving studies can reach differing conclusions, which is not reflected in traditional search tools. To address these challenges, we introduce MedSEBA, an interactive AI-powered system for synthesizing evidence-based answers to medical questions. It utilizes the power of Large Language Models to generate coherent and expressive answers, but grounds them in trustworthy medical studies dynamically retrieved from the research database PubMed. The answers consist of key points and arguments, which can be traced back to respective studies. Notably, the platform also provides an overview of the extent to which the most relevant studies support or refute the given medical claim, and a visualization of how the research consensus evolved through time. Our user study revealed that medical experts and lay users find the system usable and helpful, and the provided answers trustworthy and informative. This makes the system well-suited for both everyday health questions and advanced research insights. 

**Abstract (ZH)**: 在数字时代，人们经常通过互联网寻找医疗建议和推荐。随着在线内容的不断增加，辨别可靠来源与误导性信息变得愈发困难。同样，每年都有成千上万篇医学研究发表，使得研究人员难以跟踪最新的科学发现。这些不断发展的研究可能会得出不同的结论，而传统搜索引擎未能反映这一点。为应对这些挑战，我们引入了MedSEBA，这是一个交互式的AI驱动系统，用于综合证据基于的答案，该系统利用大型语言模型生成连贯且富有表现力的答案，并通过动态从研究数据库PubMed中检索可信的医学研究来支撑这些答案。答案包括关键点和论据，用户可以通过这些信息追溯到相应的研究文献。值得注意的是，该平台还提供了最相关研究支持或反驳给定医学声明的程度概述，并展示了研究共识随时间演变的可视化。我们的用户研究显示，医疗专家和普通用户认为该系统易于使用且有所帮助，提供的答案既可靠又富有信息量。该系统适合日常健康问题和高级研究洞察。 

---
# The Resurgence of GCG Adversarial Attacks on Large Language Models 

**Title (ZH)**: GCG对抗攻击在大型语言模型中的复兴 

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00391)  

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation. 

**Abstract (ZH)**: 基于梯度的对抗提示，如贪婪坐标梯度（GCG）算法，已成为打破大型语言模型（LLMs）的强大方法。在本文中，我们对GCG及其退火增强变体T-GCG在各种规模的开源LLM上的表现进行了系统评估。使用Qwen2.5-0.5B、LLaMA-3.2-1B和GPT-OSS-20B进行攻击效果评估，涵盖了安全导向的提示（AdvBench）和推理密集型编码提示。我们的研究揭示了三个关键发现：（1）攻击成功率（ASR）随模型规模增大而下降，反映了大型模型损失景观复杂性和非凸性的增加；（2）基于前缀的经验规则显著高估了攻击效果，而GPT-4o语义判断提供了更严格和现实的评估标准；（3）与对抗安全提示相比，与编码相关提示的脆弱性更高，表明推理本身可以作为攻击向量被利用。此外，初步结果表明，模拟退火可以多样化对抗搜索，并在前缀评估下实现具有竞争力的ASR，但在语义判断下的优势有限。这些发现共同揭示了GCG的可扩展性限制，暴露了推理任务中未被注意到的漏洞，并激励进一步开发基于退火的策略以实现更 robust 的对抗评估。 

---
# Open Data Synthesis For Deep Research 

**Title (ZH)**: 开放数据合成促进深层研究 

**Authors**: Ziyi Xia, Kun Luo, Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00375)  

**Abstract**: Large language models (LLMs) are increasingly expected to go beyond simple factual queries toward Deep Research-tasks that require decomposing questions into sub-problems, coordinating multi-step reasoning, and synthesizing evidence from diverse sources. We formalize Deep Research tasks with verifiable answers as Hierarchical Constraint Satisfaction Problems (HCSPs), which are fundamentally different from single-constraint, multi-hop, or flat CSP formulations. However, existing benchmarks (e.g., Natural Questions, HotpotQA) fail to capture this complexity, while recent synthetic datasets often introduce shortcut reasoning, knowledge leakage, or lack sufficient structural depth. To address this gap, we introduce InfoSeek, a scalable framework for synthesizing complex Deep Research tasks. InfoSeek uses a dual-agent system to recursively build a Research Tree from large-scale webpages, blurring intermediate nodes into valid sub-problems, and converting these trees into natural language questions that require traversing the full hierarchy. It also enables rapid scaling, yielding over 50K training examples, a curated test set, and reasoning trajectories generated via reject sampling. Experiments show that models trained on InfoSeek consistently outperform strong baselines. On a challenging benchmark BrowseComp-Plus, 3B LLMs optimized with InfoSeek surpass much larger 32B models and lightweight commercial APIs (e.g., Gemini2.5-Flash), while achieving performance comparable to stronger APIs (e.g., Gemini2.5-Pro). By preserving meta-information such as intermediate steps and retrieval labels, InfoSeek further supports advanced optimization strategies, including compound reward design and trajectory-level exploration. We provide our codes and datasets in \href{this https URL}{this repository}. 

**Abstract (ZH)**: 大规模语言模型（LLMs）越来越被期望处理深层次的研究任务，这些任务要求将问题分解为子问题、协调多步推理，并从多种来源合成证据。我们将具有可验证答案的深层次研究任务形式化为层次约束满足问题（HCSPs），这与单约束、多跳或多层的CSP表述本质上不同。然而，现有的基准数据集（如Natural Questions、HotpotQA）未能捕捉到这种复杂性，而最近的合成数据集则常常引入捷径推理、知识泄露或缺乏足够的结构深度。为了弥补这一差距，我们提出了InfoSeek，这是一种用于生成复杂深层次研究任务的可扩展框架。InfoSeek 使用双代理系统，递归地从大规模网页构建研究树，将中间节点模糊为有效的子问题，并将这些树转化为需要遍历完整层次的自然语言问题。它还支持快速扩展，生成超过50,000个训练示例、一个精选测试集以及通过拒绝采样生成的推理轨迹。实验显示，基于InfoSeek训练的模型一贯优于强大的基线模型。在一项具有挑战性的基准测试BrowseComp-Plus中，经过InfoSeek优化的3B LLMs超过更大规模的32B模型和轻量级商用API（如Gemini2.5-Flash），同时达到与更强API（如Gemini2.5-Pro）相当的性能。通过保留中间步骤和检索标签等元信息，InfoSeek 进一步支持复杂奖励设计和轨迹级探索。我们已在 \href{this https URL}{此仓库} 提供了代码和数据集。 

---
# LLM-Driven Policy Diffusion: Enhancing Generalization in Offline Reinforcement Learning 

**Title (ZH)**: 基于LLM的政策扩散：增强离线强化学习的一般化能力 

**Authors**: Hanping Zhang, Yuhong Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.00347)  

**Abstract**: Reinforcement Learning (RL) is known for its strong decision-making capabilities and has been widely applied in various real-world scenarios. However, with the increasing availability of offline datasets and the lack of well-designed online environments from human experts, the challenge of generalization in offline RL has become more prominent. Due to the limitations of offline data, RL agents trained solely on collected experiences often struggle to generalize to new tasks or environments. To address this challenge, we propose LLM-Driven Policy Diffusion (LLMDPD), a novel approach that enhances generalization in offline RL using task-specific prompts. Our method incorporates both text-based task descriptions and trajectory prompts to guide policy learning. We leverage a large language model (LLM) to process text-based prompts, utilizing its natural language understanding and extensive knowledge base to provide rich task-relevant context. Simultaneously, we encode trajectory prompts using a transformer model, capturing structured behavioral patterns within the underlying transition dynamics. These prompts serve as conditional inputs to a context-aware policy-level diffusion model, enabling the RL agent to generalize effectively to unseen tasks. Our experimental results demonstrate that LLMDPD outperforms state-of-the-art offline RL methods on unseen tasks, highlighting its effectiveness in improving generalization and adaptability in diverse settings. 

**Abstract (ZH)**: 强化学习（RL）以其强大的决策能力而闻名，并已在多种实际场景中得到广泛应用。然而，随着离线数据集的日益可用以及缺乏由人类专家精心设计的在线环境，离线RL中的泛化挑战变得更加突出。由于离线数据的限制，仅基于收集的经验训练的RL代理往往难以在新任务或环境中泛化。为了解决这一挑战，我们提出了一种名为LLM驱动的策略扩散（LLMDPD）的新方法，该方法利用任务特定的提示来增强离线RL中的泛化能力。该方法结合了基于文本的任务描述和轨迹提示，以指导策略学习。我们利用大型语言模型（LLM）处理基于文本的提示，利用其自然语言理解和广泛的知识库提供丰富的任务相关信息。同时，我们使用变换器模型对轨迹提示进行编码，捕捉潜在过渡动力学中的结构化行为模式。这些提示作为上下文感知策略级别扩散模型的条件输入，使RL代理能够有效地泛化到未见过的任务。我们的实验结果表明，LLMDPD在未见过的任务上优于最先进的离线RL方法，突显了其在各种环境中提高泛化能力和适应性的有效性。 

---
# Access Paths for Efficient Ordering with Large Language Models 

**Title (ZH)**: 大规模语言模型高效排序的访问路径 

**Authors**: Fuheng Zhao, Jiayue Chen, Yiming Pan, Tahseen Rabbani, Divyakant Agrawal, Amr El Abbadi  

**Link**: [PDF](https://arxiv.org/pdf/2509.00303)  

**Abstract**: We present the LLM ORDER BY operator as a logical abstraction and study its physical implementations within a unified evaluation framework. Our experiments show that no single approach is universally optimal, with effectiveness depending on query characteristics and data. We introduce three new designs: an agreement-based batch-size policy, a majority voting mechanism for pairwise sorting, and a two-way external merge sort adapted for LLMs. With extensive experiments, our agreement-based procedure is effective at determining batch size for value-based methods, the majority-voting mechanism consistently strengthens pairwise comparisons on GPT-4o, and external merge sort achieves high accuracy-efficiency trade-offs across datasets and models. We further observe a log-linear scaling between compute cost and ordering quality, offering the first step toward principled cost models for LLM powered data systems. 

**Abstract (ZH)**: 我们提出LLM ORDER BY运算符作为逻辑抽象，并在统一的评估框架内研究其物理实现。我们的实验表明，并不存在单一的最佳方法，其效果取决于查询特性和数据。我们引入了三种新的设计：基于共识的批量大小策略、适用于双向排序的多数投票机制以及适应LLM的外部合并排序。通过大量的实验，我们的基于共识的过程在确定基于值的方法的批量大小时有效，多数投票机制在GPT-4o上的两两比较中始终增强效果，外部合并排序在不同数据集和模型中实现了高准确性和效率的trade-offs。我们还观察到计算成本与排序质量之间的对数线性扩展关系，为基于LLM的数据系统提供了一种成本模型的初步方案。 

---
# OpinioRAG: Towards Generating User-Centric Opinion Highlights from Large-scale Online Reviews 

**Title (ZH)**: OpinioRAG: 向量化生成大规模在线评论中的用户中心意见精华 

**Authors**: Mir Tafseer Nayeem, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2509.00285)  

**Abstract**: We study the problem of opinion highlights generation from large volumes of user reviews, often exceeding thousands per entity, where existing methods either fail to scale or produce generic, one-size-fits-all summaries that overlook personalized needs. To tackle this, we introduce OpinioRAG, a scalable, training-free framework that combines RAG-based evidence retrieval with LLMs to efficiently produce tailored summaries. Additionally, we propose novel reference-free verification metrics designed for sentiment-rich domains, where accurately capturing opinions and sentiment alignment is essential. These metrics offer a fine-grained, context-sensitive assessment of factual consistency. To facilitate evaluation, we contribute the first large-scale dataset of long-form user reviews, comprising entities with over a thousand reviews each, paired with unbiased expert summaries and manually annotated queries. Through extensive experiments, we identify key challenges, provide actionable insights into improving systems, pave the way for future research, and position OpinioRAG as a robust framework for generating accurate, relevant, and structured summaries at scale. 

**Abstract (ZH)**: 我们研究了从大量用户评论中生成意见要点的问题，这些评论常常超过每个实体几千条，现有方法要么无法扩展，要么生成通用的一刀切摘要，忽视个性化需求。为了解决这一问题，我们引入了OpinioRAG，这是一个可扩展、无需训练的框架，结合了RAG基于证据的检索与LLMs，以高效生成定制化的摘要。此外，我们还提出了适用于情感丰富的领域的新颖无参考验证指标，准确捕捉意见和情感一致性至关重要。这些指标提供了细粒度、上下文敏感的事实一致性评估。为了便于评估，我们贡献了首个大规模长文本用户评论数据集，包含每个实体超过一千条评论，并附有客观专家摘要和手动标注的查询。通过广泛实验，我们确定了关键挑战，提供了改进系统的方法，并为未来的研究铺平了道路，将OpinioRAG定位为一个适用于大规模生成准确、相关且结构化的摘要的稳健框架。 

---
# Explainable Chain-of-Thought Reasoning: An Empirical Analysis on State-Aware Reasoning Dynamics 

**Title (ZH)**: 可解释的链式思维推理：基于状态感知的推理动力学的实证分析 

**Authors**: Sheldon Yu, Yuxin Xiong, Junda Wu, Xintong Li, Tong Yu, Xiang Chen, Ritwik Sinha, Jingbo Shang, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2509.00190)  

**Abstract**: Recent advances in chain-of-thought (CoT) prompting have enabled large language models (LLMs) to perform multi-step reasoning. However, the explainability of such reasoning remains limited, with prior work primarily focusing on local token-level attribution, such that the high-level semantic roles of reasoning steps and their transitions remain underexplored. In this paper, we introduce a state-aware transition framework that abstracts CoT trajectories into structured latent dynamics. Specifically, to capture the evolving semantics of CoT reasoning, each reasoning step is represented via spectral analysis of token-level embeddings and clustered into semantically coherent latent states. To characterize the global structure of reasoning, we model their progression as a Markov chain, yielding a structured and interpretable view of the reasoning process. This abstraction supports a range of analyses, including semantic role identification, temporal pattern visualization, and consistency evaluation. 

**Abstract (ZH)**: 近期链式思考（CoT）提示的进展使大型语言模型（LLMs）能够进行多步推理，但这种推理的解释性仍然有限，前期工作主要集中在局部词级归属上，高层语义推理步骤及其转换仍然未得到充分探索。在本文中，我们介绍了一种状态感知转换框架，将CoT轨迹抽象为结构化的潜在动态。具体来说，为了捕捉CoT推理中的演进语义，每一步推理通过词级嵌入的光谱分析来表示，并聚类为语义一致的潜在状态。为了表征推理的全局结构，我们将其进展建模为马尔科夫链，从而获得了一个结构化和可解释的推理过程视图。这种抽象支持一系列分析，包括语义角色识别、时间模式可视化和一致性评估。 

---
# Waste-Bench: A Comprehensive Benchmark for Evaluating VLLMs in Cluttered Environments 

**Title (ZH)**: Waste-Bench: 一种评估VLLMs在杂乱环境中的综合基准 

**Authors**: Muhammad Ali, Salman Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00176)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have paved the way for Vision Large Language Models (VLLMs) capable of performing a wide range of visual understanding tasks. While LLMs have demonstrated impressive performance on standard natural images, their capabilities have not been thoroughly explored in cluttered datasets where there is complex environment having deformed shaped objects. In this work, we introduce a novel dataset specifically designed for waste classification in real-world scenarios, characterized by complex environments and deformed shaped objects. Along with this dataset, we present an in-depth evaluation approach to rigorously assess the robustness and accuracy of VLLMs. The introduced dataset and comprehensive analysis provide valuable insights into the performance of VLLMs under challenging conditions. Our findings highlight the critical need for further advancements in VLLM's robustness to perform better in complex environments. The dataset and code for our experiments will be made publicly available. 

**Abstract (ZH)**: 近期大型语言模型的进展为视觉大型语言模型（VLLMs）的能力提升铺平了道路，使其能够执行一系列视觉理解任务。尽管LLMs在标准自然图像上展现了令人印象深刻的性能，但在拥挤的数据集中，复杂环境中存在变形物体的情况，其能力尚未得到充分探索。本文介绍了一个特别为实际应用场景中的废物分类设计的新数据集，该数据集特征是复杂环境和变形物体。此外，我们还提出了一种深入的评估方法，以严格评估VLLMs的鲁棒性和准确性。引入的数据集和全面分析为评估VLLMs在具有挑战性条件下的性能提供了有价值的见解。我们的研究结果强调了进一步提高VLLMs在复杂环境中的鲁棒性的必要性。我们将公开该数据集和实验代码。 

---
# LLM-based Triplet Extraction for Automated Ontology Generation in Software Engineering Standards 

**Title (ZH)**: 基于LLM的三元组提取在软件工程标准中的自动本体生成 

**Authors**: Songhui Yue  

**Link**: [PDF](https://arxiv.org/pdf/2509.00140)  

**Abstract**: Ontologies have supported knowledge representation and whitebox reasoning for decades; thus, the automated ontology generation (AOG) plays a crucial role in scaling their use. Software engineering standards (SES) consist of long, unstructured text (with high noise) and paragraphs with domain-specific terms. In this setting, relation triple extraction (RTE), together with term extraction, constitutes the first stage toward AOG. This work proposes an open-source large language model (LLM)-assisted approach to RTE for SES. Instead of solely relying on prompt-engineering-based methods, this study promotes the use of LLMs as an aid in constructing ontologies and explores an effective AOG workflow that includes document segmentation, candidate term mining, LLM-based relation inference, term normalization, and cross-section alignment. Golden-standard benchmarks at three granularities are constructed and used to evaluate the ontology generated from the study. The results show that it is comparable and potentially superior to the OpenIE method of triple extraction. 

**Abstract (ZH)**: 基于大型语言模型的关系三元组提取方法在软件工程标准中的应用 

---
# CoComposer: LLM Multi-agent Collaborative Music Composition 

**Title (ZH)**: CoComposer: LLM多智能体合作音乐创作 

**Authors**: Peiwen Xing, Aske Plaat, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2509.00132)  

**Abstract**: Existing AI Music composition tools are limited in generation duration, musical quality, and controllability. We introduce CoComposer, a multi-agent system that consists of five collaborating agents, each with a task based on the traditional music composition workflow. Using the AudioBox-Aesthetics system, we experimentally evaluate CoComposer on four compositional criteria. We test with three LLMs (GPT-4o, DeepSeek-V3-0324, Gemini-2.5-Flash), and find (1) that CoComposer outperforms existing multi-agent LLM-based systems in music quality, and (2) compared to a single-agent system, in production complexity. Compared to non- LLM MusicLM, CoComposer has better interpretability and editability, although MusicLM still produces better music. 

**Abstract (ZH)**: 现有的AI音乐创作工具在生成时长、音乐质量及可控性方面存在局限。我们介绍了CoComposer，一个由五个协作智能体组成的系统，每个智能体的任务基于传统的音乐创作工作流程。利用AudioBox-Aesthetics系统，我们在四个创作标准上实验性地评估了CoComposer。我们使用了三种LLM（GPT-4o、DeepSeek-V3-0324、Gemini-2.5-Flash），发现（1）CoComposer在音乐质量上优于现有的基于LLM的多智能体系统，（2）与单智能体系统相比，在生成复杂性上表现更佳。与非LLM的MusicLM相比，CoComposer在可解释性和可编辑性方面更具优势，尽管MusicLM仍然能创作出更好的音乐。 

---
# A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See 

**Title (ZH)**: 一个 Entire 新世界：创建仅AI代理可见的并行毒化网络 

**Authors**: Shaked Zychlinski  

**Link**: [PDF](https://arxiv.org/pdf/2509.00124)  

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack. 

**Abstract (ZH)**: 本文介绍了一种新颖的攻击向量，利用网站伪装技术攻击由大规模语言模型（LLMs）驱动的自主网络浏览代理。随着这些代理的普及，它们独特的、often homogenous的数字足迹——包括浏览器属性、自动化框架签名和网络特征——形成了新的可区分的网络流量类别。攻击正是利用了这一特点。恶意网站可以识别出一个来自AI代理的入站请求，并动态提供不同的、伪装后的网页内容。虽然人类用户看到的是一个无害的网页，但代理却被展示了一个视觉上相同的网页，其中嵌入了隐蔽的恶意指令，如间接提示注入。这一机制使攻击者能够劫持代理行为，导致数据泄露、恶意软件执行或错误信息传播，同时完全对人类用户和传统安全爬虫隐身。本文正式化了威胁模型，详细说明了代理指纹识别和伪装的机制，并讨论了对未来基于代理的AI的深远安全影响，强调了对这种隐蔽且可扩展攻击的 robust 防御措施的迫切需求。 

---
# Meta-learning ecological priors from large language models explains human learning and decision making 

**Title (ZH)**: 从大规模语言模型学习生态先验以解释人类的学习与决策 

**Authors**: Akshay K. Jagadish, Mirko Thalmann, Julian Coda-Forno, Marcel Binz, Eric Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2509.00116)  

**Abstract**: Human cognition is profoundly shaped by the environments in which it unfolds. Yet, it remains an open question whether learning and decision making can be explained as a principled adaptation to the statistical structure of real-world tasks. We introduce ecologically rational analysis, a computational framework that unifies the normative foundations of rational analysis with ecological grounding. Leveraging large language models to generate ecologically valid cognitive tasks at scale, and using meta-learning to derive rational models optimized for these environments, we develop a new class of learning algorithms: Ecologically Rational Meta-learned Inference (ERMI). ERMI internalizes the statistical regularities of naturalistic problem spaces and adapts flexibly to novel situations, without requiring hand-crafted heuristics or explicit parameter updates. We show that ERMI captures human behavior across 15 experiments spanning function learning, category learning, and decision making, outperforming several established cognitive models in trial-by-trial prediction. Our results suggest that much of human cognition may reflect adaptive alignment to the ecological structure of the problems we encounter in everyday life. 

**Abstract (ZH)**: 生态理性分析：一种生态化规范计算框架 

---
# Pre-trained knowledge elevates large language models beyond traditional chemical reaction optimizers 

**Title (ZH)**: 预训练知识超越传统化学反应优化器 

**Authors**: Robert MacKnight, Jose Emilio Regio, Jeffrey G. Ethier, Luke A. Baldwin, Gabe Gomes  

**Link**: [PDF](https://arxiv.org/pdf/2509.00103)  

**Abstract**: Modern optimization in experimental chemistry employs algorithmic search through black-box parameter spaces. Here we demonstrate that pre-trained knowledge in large language models (LLMs) fundamentally changes this paradigm. Using six fully enumerated categorical reaction datasets (768 - 5,684 experiments), we benchmark LLM-guided optimization (LLM-GO) against Bayesian optimization (BO) and random sampling. Frontier LLMs consistently match or exceed BO performance across five single-objective datasets, with advantages growing as parameter complexity increases and high-performing conditions become scarce (<5% of space). BO retains superiority only for explicit multi-objective trade-offs. To understand these contrasting behaviors, we introduce a topology-agnostic information theory framework quantifying sampling diversity throughout optimization campaigns. This analysis reveals that LLMs maintain systematically higher exploration entropy than BO across all datasets while achieving superior performance, with advantages most pronounced in solution-scarce parameter spaces where high-entropy exploration typically fails - suggesting that pre-trained domain knowledge enables more effective navigation of chemical parameter space rather than replacing structured exploration strategies. To enable transparent benchmarking and community validation, we release Iron Mind (this https URL), a no-code platform for side-by-side evaluation of human, algorithmic, and LLM optimization campaigns with public leaderboards and complete trajectories. Our findings establish that LLM-GO excels precisely where traditional methods struggle: complex categorical spaces requiring domain understanding rather than mathematical optimization. 

**Abstract (ZH)**: 现代实验化学中的优化采用算法在黑盒参数空间中搜索。我们证明预先训练的大语言模型知识从根本上改变了这一范式。使用六个完全枚举的分类反应数据集（768 - 5,684 实验），我们将大语言模型引导优化（LLM-GO）与贝叶斯优化（BO）和随机采样进行了基准测试。在五个单目标数据集中，前沿的大语言模型一致地达到了或超过了BO的性能，随着参数复杂度的增加和高表现条件的稀缺（少于5%的空间），优势变得愈加明显。只有在显式多目标权衡的情况下，BO才保持优势。为了理解这些不同的行为，我们引入了一种拓扑无关的信息理论框架，量化优化过程中采样多样性。分析表明，大语言模型在整个数据集中系统地保持了更高的探索熵，同时实现了更好的性能，特别是在高熵探索通常失败的解稀缺参数空间中优势最为显著——这表明预先训练的专业知识使大语言模型能够在化学参数空间中更有效地导航，而不是替代结构化的探索策略。为了实现透明的基准测试和社区验证，我们发布了Iron Mind（https://），一个无需代码的平台，用于并排评估人类、算法和大语言模型优化过程，并配备公共排行榜和完整轨迹。我们的研究结果确立了LLM-GO在传统方法困难的地方表现优异：需要领域理解而非数学优化的复杂分类空间。 

---
# AEGIS : Automated Co-Evolutionary Framework for Guarding Prompt Injections Schema 

**Title (ZH)**: AEGIS：自动化共生演化框架以防范提示注入模式 

**Authors**: Ting-Chun Liu, Ching-Yu Hsu, Kuan-Yi Lee, Chi-An Fu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.00088)  

**Abstract**: Prompt injection attacks pose a significant challenge to the safe deployment of Large Language Models (LLMs) in real-world applications. While prompt-based detection offers a lightweight and interpretable defense strategy, its effectiveness has been hindered by the need for manual prompt engineering. To address this issue, we propose AEGIS , an Automated co-Evolutionary framework for Guarding prompt Injections Schema. Both attack and defense prompts are iteratively optimized against each other using a gradient-like natural language prompt optimization technique. This framework enables both attackers and defenders to autonomously evolve via a Textual Gradient Optimization (TGO) module, leveraging feedback from an LLM-guided evaluation loop. We evaluate our system on a real-world assignment grading dataset of prompt injection attacks and demonstrate that our method consistently outperforms existing baselines, achieving superior robustness in both attack success and detection. Specifically, the attack success rate (ASR) reaches 1.0, representing an improvement of 0.26 over the baseline. For detection, the true positive rate (TPR) improves by 0.23 compared to the previous best work, reaching 0.84, and the true negative rate (TNR) remains comparable at 0.89. Ablation studies confirm the importance of co-evolution, gradient buffering, and multi-objective optimization. We also confirm that this framework is effective in different LLMs. Our results highlight the promise of adversarial training as a scalable and effective approach for guarding prompt injections. 

**Abstract (ZH)**: 自动共进化框架：守护提示注入scheme（AEGIS） 

---
# Learning to Refine: Self-Refinement of Parallel Reasoning in LLMs 

**Title (ZH)**: 学习 refinement：LLM 中并行推理的自我 refinement 

**Authors**: Qibin Wang, Pu Zhao, Shaohan Huang, Fangkai Yang, Lu Wang, Furu Wei, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.00084)  

**Abstract**: To further enhance the ability of Large Language Models (LLMs) to solve complex, multi-step reasoning problems, test-time scaling (TTS) methods have gained widespread attention. Existing approaches such as Best-of-N and majority voting are limited as their performance depends on the quality of candidate responses, making them unable to produce a correct solution when all candidates are incorrect. Introducing an additional model to select the best response also incurs significant deployment costs. To this end, we introduce Generative Self-Refinement (GSR), a novel parallel test-time scaling framework where a unified model first generates a set of candidate responses in parallel and then performs self-refinement to synthesize a new superior solution based on a prompt consisting of the problem and these candidates. However, LLMs struggle to perform refinement effectively when prompted directly. Therefore, we design a hybrid training pipeline by jointly optimizing for two complementary objectives, solving problems directly and refining candidate responses. Experimental results demonstrate that our method achieves state-of-the-art performance across five mathematical benchmarks. We further show that this learned self-refinement skill is a model-agnostic enhancement, robust across different model scales and generalizing to out-of-distribution reasoning tasks. 

**Abstract (ZH)**: 为了进一步增强大型语言模型（LLMs）解决复杂多步推理问题的能力，测试时扩展（TTS）方法已引起广泛注意。现有方法如Best-of-N和多数投票受限于候选响应的质量，当所有候选响应都错误时，它们无法生成正确的解。引入额外模型来选择最佳响应也会带来显著的部署成本。为此，我们提出了生成式自我精炼（GSR），这是一种新颖的并行测试时扩展框架，统一模型首先并行生成一组候选响应，然后基于问题和这些候选响应的提示进行自我精炼，以合成一个更优的新解决方案。然而，LLM直接提示时难以有效进行精炼。因此，我们设计了一种混合训练管线，同时优化直接解决问题和精炼候选响应的两个互补目标。实验结果表明，我们的方法在五个数学基准上达到了最先进的性能。我们还展示了这种学习到的自我精炼能力是一种模型无关的增强，能在不同模型规模下保持鲁棒性，并适用于分布外推理任务。 

---
# MolErr2Fix:Benchmarking LLM Trustworthiness in Chemistry via Modular Error Detection, Localization, Explanation, and Revision 

**Title (ZH)**: MolErr2Fix：通过模块化错误检测、定位、解释和修正benchmark化化学领域大规模语言模型的信任度 

**Authors**: Yuyang Wu, Jinhui Ye, Shuhao Zhang, Lu Dai, Yonatan Bisk, Olexandr Isayev  

**Link**: [PDF](https://arxiv.org/pdf/2509.00063)  

**Abstract**: Large Language Models (LLMs) have shown growing potential in molecular sciences, but they often produce chemically inaccurate descriptions and struggle to recognize or justify potential errors. This raises important concerns about their robustness and reliability in scientific applications. To support more rigorous evaluation of LLMs in chemical reasoning, we present the MolErr2Fix benchmark, designed to assess LLMs on error detection and correction in molecular descriptions. Unlike existing benchmarks focused on molecule-to-text generation or property prediction, MolErr2Fix emphasizes fine-grained chemical understanding. It tasks LLMs with identifying, localizing, explaining, and revising potential structural and semantic errors in molecular descriptions. Specifically, MolErr2Fix consists of 1,193 fine-grained annotated error instances. Each instance contains quadruple annotations, i.e,. (error type, span location, the explanation, and the correction). These tasks are intended to reflect the types of reasoning and verification required in real-world chemical communication. Evaluations of current state-of-the-art LLMs reveal notable performance gaps, underscoring the need for more robust chemical reasoning capabilities. MolErr2Fix provides a focused benchmark for evaluating such capabilities and aims to support progress toward more reliable and chemically informed language models. All annotations and an accompanying evaluation API will be publicly released to facilitate future research. 

**Abstract (ZH)**: Large Language Models (LLMs)在分子科学中的潜力不断增长，但它们通常会产生化学上不准确的描述并难以识别或解释潜在错误。这引发了对其在科学应用中鲁棒性和可靠性的重大关切。为了支持对LLMs在化学推理中更严格的评估，本文介绍了MolErr2Fix基准，该基准旨在评估LLMs在分子描述中的错误检测和修正能力。与现有的主要关注于分子到文本生成或属性预测的基准不同，MolErr2Fix强调细致的化学理解。它要求LLMs识别、定位、解释并修正分子描述中的潜在结构性和语义错误。具体来说，MolErr2Fix包含1,193个细致标注的错误实例，每个实例包含四元标注，即（错误类型、跨度位置、解释和修正）。这些任务旨在反映实际化学通信中所需的推理和验证类型。当前最先进的LLMs的评估揭示了显著的性能差距，强调了需要更强的化学推理能力。MolErr2Fix提供了一个专门的基准来评估这些能力，并旨在支持更可靠和化学导向的语言模型的发展。所有标注和配套的评估API将公开发布，以促进未来的研究。 

---
# Exploring and Reshaping the Weight Distribution in LLM 

**Title (ZH)**: 探索并重塑LLM中的权重分布 

**Authors**: Chunming Ye, Songzhou Li, Xu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.00046)  

**Abstract**: The performance of Large Language Models is influenced by their characteristics such as architecture, model sizes, decoding methods and so on. Due to differences in structure or function, the weights in different layers of large models have varying distributions. This paper explores the correlations between different types of layers in terms of weights distribution and studies the potential impact of these correlations on LoRA training effectiveness. Firstly, the study reveals that in the model the cosine distances between weights of different layers manifest power-law distribution. We extract Query-projection, down-projection and other weight matrices from the self-attention layers and MLP layers, calculate the singular values of the matrices using singular value decomposition, and organize a certain number of singular values into matrices according to projection's type. By analyzing the probability distribution of the cosine distances between these matrices, it is found that the cosine distances values between them have distinct power-law distribution characteristics. Secondly, based on the results of distance calculations and analysis across different layers of model, a qualitative method is proposed to describe the distribution characteristics of different models. Next, to construct weights that align with the distribution characteristics, a data generator is designed using a combination of Gaussian process and Pareto distribution functions. The generator is used to simulate the generation of data that aligns with specific distribution characteristics. Finally, based on the aforementioned distribution characteristics and data generation method, the weights in LoRA initialization are reshaped for training. Experimental results indicate that, without altering the model structure or training process, this method achieves a certain improvement in the performance of LoRA training. 

**Abstract (ZH)**: 大型语言模型的表现受其架构、模型规模、解码方法等方面特性的影响。由于结构或功能的差异，大型模型不同层中的权重具有不同的分布。本文探讨了不同类型层之间权重分布的相关性及其对LoRA训练效果潜在影响的研究。首先，研究揭示出在模型中不同层权重之间的余弦距离表现出幂律分布。从中提取自我注意力层和MLP层的Query-projection、down-projection及其他权重矩阵，使用奇异值分解计算矩阵的奇异值，并根据投影类型将一定数量的奇异值组织成矩阵。通过对这些矩阵之间余弦距离概率分布的分析，发现它们之间的余弦距离值具有明显的幂律分布特征。其次，在不同层间距离计算与分析的基础上，提出了一种定性方法来描述不同模型的分布特征。接着，为了生成符合分布特征的权重，设计了一个结合高斯过程和帕累托分布函数的数据生成器，该生成器用于模拟生成特定分布特征的数据。最后，在上述分布特征和数据生成方法的基础上，对LoRA初始化权重进行重塑以辅助训练。实验结果表明，不改变模型结构或训练过程的情况下，该方法在LoRA训练性能上取得了一定的提升。 

---
# Compiling Prompts, Not Crafting Them: A Reproducible Workflow for AI-Assisted Evidence Synthesis 

**Title (ZH)**: 不用编写指令，生成可重复的工作流：基于AI辅助证据合成的可重复工作流程 

**Authors**: Teo Susnjak  

**Link**: [PDF](https://arxiv.org/pdf/2509.00038)  

**Abstract**: Large language models (LLMs) offer significant potential to accelerate systematic literature reviews (SLRs), yet current approaches often rely on brittle, manually crafted prompts that compromise reliability and reproducibility. This fragility undermines scientific confidence in LLM-assisted evidence synthesis. In response, this work adapts recent advances in declarative prompt optimisation, developed for general-purpose LLM applications, and demonstrates their applicability to the domain of SLR automation. This research proposes a structured, domain-specific framework that embeds task declarations, test suites, and automated prompt tuning into a reproducible SLR workflow. These emerging methods are translated into a concrete blueprint with working code examples, enabling researchers to construct verifiable LLM pipelines that align with established principles of transparency and rigour in evidence synthesis. This is a novel application of such approaches to SLR pipelines. 

**Abstract (ZH)**: 大型语言模型（LLMs）在加速系统文献综述（SLRs）方面具有显著潜力，但当前方法往往依赖于脆弱的手工构建提示，这损害了可靠性与可重现性。这种脆弱性削弱了对LLM辅助证据综合的科学信心。为此，本研究采纳了最近在声明式提示优化方面的进展，这些进展最初为通用语言模型应用开发，证明了其在SLR自动化领域的适用性。本研究提出了一种结构化、领域特定的框架，该框架嵌入了任务声明、测试套件和自动提示调优，以实现可重现的SLR工作流程。这些新兴方法被转化为一个具体的构建指南，并提供了代码示例，使研究人员能够构建符合证据综合透明性和严格性原则的可验证的LLM管道。这是一个此类方法在SLR管道中的新颖应用。 

---
# ZeroQAT: Your Quantization-aware Training but Efficient 

**Title (ZH)**: ZeroQAT: -your量化感知训练但高效 

**Authors**: Qitao Tan, Xiaoying Song, Jin Lu, Guoming Li, Jun Liu, Lingzi Hong, Caiwen Ding, Jundong Li, Xiaoming Zhai, Shaoyi Huang, Wei Niu, Geng Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.00031)  

**Abstract**: Quantization is an effective technique to reduce the deployment cost of large language models (LLMs), and post-training quantization (PTQ) has been widely studied due to its efficiency. However, existing low-bit PTQ methods suffer from accuracy degradation because their layer-wise optimization introduces cumulative error propagation and misalignment between local reconstruction objectives and downstream performance. While quantization-aware training (QAT) provides a principled solution, its reliance on backpropagation incurs prohibitive data, time, and memory costs, limiting its practicality. To address these challenges, we propose ZeroQAT, a zeroth-order optimization-based QAT framework. ZeroQAT leverages forward-only gradient estimation to eliminate the need for backpropagation, significantly reducing computational and memory overhead while retaining the benefits of end-to-end optimization. Moreover, ZeroQAT jointly learns quantized weights, weight clipping thresholds, and equivalent transformations to mitigate quantization error and handle activation outliers. Experiments demonstrate that ZeroQAT achieves the efficiency of PTQ while retaining the accuracy of QAT, offering a practical solution for high-quality low-bit quantization of LLMs. 

**Abstract (ZH)**: ZeroQAT：基于零阶优化的量化意识训练框架 

---
