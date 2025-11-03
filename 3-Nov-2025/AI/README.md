# MolChord: Structure-Sequence Alignment for Protein-Guided Drug Design 

**Title (ZH)**: MolChord: 结构-序列对齐方法在蛋白质导向药物设计中的应用 

**Authors**: Wei Zhang, Zekun Guo, Yingce Xia, Peiran Jin, Shufang Xie, Tao Qin, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27671)  

**Abstract**: Structure-based drug design (SBDD), which maps target proteins to candidate molecular ligands, is a fundamental task in drug discovery. Effectively aligning protein structural representations with molecular representations, and ensuring alignment between generated drugs and their pharmacological properties, remains a critical challenge. To address these challenges, we propose MolChord, which integrates two key techniques: (1) to align protein and molecule structures with their textual descriptions and sequential representations (e.g., FASTA for proteins and SMILES for molecules), we leverage NatureLM, an autoregressive model unifying text, small molecules, and proteins, as the molecule generator, alongside a diffusion-based structure encoder; and (2) to guide molecules toward desired properties, we curate a property-aware dataset by integrating preference data and refine the alignment process using Direct Preference Optimization (DPO). Experimental results on CrossDocked2020 demonstrate that our approach achieves state-of-the-art performance on key evaluation metrics, highlighting its potential as a practical tool for SBDD. 

**Abstract (ZH)**: 基于结构的药物设计（SBDD），即将目标蛋白质映射到候选分子配体，是药物发现中的一个基本任务。有效地对蛋白质结构表示与分子表示进行对齐，并确保生成的药物与其药理学性质之间的对齐，仍然是一个关键挑战。为了解决这些挑战，我们提出MolChord，该方法整合了两个关键技术：（1）使用NatureLM，一种统一了文本、小分子和蛋白质的自回归模型，作为分子生成器，并结合基于扩散的结构编码器，以对准蛋白质和分子结构与其文本描述和序列表示（例如，蛋白质的FASTA和分子的SMILES）；（2）为了引导分子向期望的性质发展，我们通过整合偏好数据整理了一个具有性质意识的数据集，并使用直接偏好优化（DPO）细化对齐过程。CrossDocked2020上的实验结果表明，我们的方法在关键评估指标上达到了最先进的性能，突显了其作为SBDD实用工具的潜力。 

---
# Interaction as Intelligence Part II: Asynchronous Human-Agent Rollout for Long-Horizon Task Training 

**Title (ZH)**: 交互即智能 Part II：异步人类-代理滚动训练长时域任务 

**Authors**: Dayuan Fu, Yunze Wu, Xiaojie Cai, Lyumanshan Ye, Shijie Xia, Zhen Huang, Weiye Si, Tianze Xu, Jie Sun, Keyu Li, Mohan Jiang, Junfei Wang, Qishuo Hua, Pengrui Lu, Yang Xiao, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27630)  

**Abstract**: Large Language Model (LLM) agents have recently shown strong potential in domains such as automated coding, deep research, and graphical user interface manipulation. However, training them to succeed on long-horizon, domain-specialized tasks remains challenging. Current methods primarily fall into two categories. The first relies on dense human annotations through behavior cloning, which is prohibitively expensive for long-horizon tasks that can take days or months. The second depends on outcome-driven sampling, which often collapses due to the rarity of valid positive trajectories on domain-specialized tasks. We introduce Apollo, a sampling framework that integrates asynchronous human guidance with action-level data filtering. Instead of requiring annotators to shadow every step, Apollo allows them to intervene only when the agent drifts from a promising trajectory, by providing prior knowledge, strategic advice, etc. This lightweight design makes it possible to sustain interactions for over 30 hours and produces valuable trajectories at a lower cost. Apollo then applies supervision control to filter out sub-optimal actions and prevent error propagation. Together, these components enable reliable and effective data collection in long-horizon environments. To demonstrate the effectiveness of Apollo, we evaluate it using InnovatorBench. Our experiments show that when applied to train the GLM-4.5 model on InnovatorBench, Apollo achieves more than a 50% improvement over the untrained baseline and a 28% improvement over a variant trained without human interaction. These results highlight the critical role of human-in-the-loop sampling and the robustness of Apollo's design in handling long-horizon, domain-specialized tasks. 

**Abstract (ZH)**: 大型语言模型（LLM）代理在自动化编码、深度研究和图形用户界面操作等领域展示了强大的潜力。然而，训练它们在长期、领域专业化任务上成功仍然具有挑战性。当前方法主要分为两类。第一类依赖行为克隆的密集人类标注，这由于长期任务可能需要几天或几个月的时间而变得代价高昂。第二类依赖目标驱动的采样，但在领域专业化任务中，由于有效正轨迹的稀疏性，这种方法往往难以有效进行。我们介绍了Apollo，一种结合异步人类指导与动作级别数据过滤的采样框架。Apollo不要求标注者跟进每个步骤，而是仅在代理偏离有希望的轨迹时进行干预，通过提供先验知识、策略性建议等方式。这种轻量级设计使得长时间维持交互成为可能，并在较低成本下产生有价值的数据轨迹。Apollo随后应用监督控制过滤掉非最优动作，防止错误传播。这些组件共同使Apollo能够在长期任务环境中可靠有效地收集数据。为了展示Apollo的有效性，我们使用InnovatorBench对其进行评估。我们的实验表明，当应用于在InnovatorBench上训练GLM-4.5模型时，Apollo相比未训练基线提高了超过50%的表现，相比未与人类交互训练的变体提高了28%。这些结果突显了循环人类干预采样的重要性，以及Apollo设计在处理长期、领域专业化任务方面的鲁棒性。 

---
# Validity Is What You Need 

**Title (ZH)**: 你需要的是有效性。 

**Authors**: Sebastian Benthall, Andrew Clark  

**Link**: [PDF](https://arxiv.org/pdf/2510.27628)  

**Abstract**: While AI agents have long been discussed and studied in computer science, today's Agentic AI systems are something new. We consider other definitions of Agentic AI and propose a new realist definition. Agentic AI is a software delivery mechanism, comparable to software as a service (SaaS), which puts an application to work autonomously in a complex enterprise setting. Recent advances in large language models (LLMs) as foundation models have driven excitement in Agentic AI. We note, however, that Agentic AI systems are primarily applications, not foundations, and so their success depends on validation by end users and principal stakeholders. The tools and techniques needed by the principal users to validate their applications are quite different from the tools and techniques used to evaluate foundation models. Ironically, with good validation measures in place, in many cases the foundation models can be replaced with much simpler, faster, and more interpretable models that handle core logic. When it comes to Agentic AI, validity is what you need. LLMs are one option that might achieve it. 

**Abstract (ZH)**: 虽然人工智能代理在计算机科学中早有讨论和研究，但当今的代理型AI系统却是新的事物。我们考虑其他代理型AI的定义，并提出一个新的现实主义定义。代理型AI是一种软件交付机制，类似于软件即服务（SaaS），它能够让应用程序在复杂的企业环境中自主工作。近年来，以大型语言模型（LLMs）为基础模型的进展激发了对代理型AI的兴趣。然而，我们注意到，代理型AI系统主要是应用程序，而不是基础模型，因此它们的成功依赖于最终用户和主要利害关系人的验证。主要用户验证其应用程序所需的方法和技术与评估基础模型的方法和技术截然不同。讽刺的是，有了良好的验证措施，很多时候基础模型可以被更简单、更快、更具解释性的模型所取代，以处理核心逻辑。对于代理型AI而言，有效性才是关键。大型语言模型是可能实现这一点的一种选择。 

---
# Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning 

**Title (ZH)**: 视觉后门攻击：基于对比触发学习的MLLM结构化决策making中的 embodied 决策攻击 

**Authors**: Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment. 

**Abstract (ZH)**: 多模态大语言模型中的视觉后门攻击：基于环境物体的BEAT框架 

---
# VeriMoA: A Mixture-of-Agents Framework for Spec-to-HDL Generation 

**Title (ZH)**: VeriMoA：一种基于代理混合的从规格到HDL生成框架 

**Authors**: Heng Ping, Arijit Bhattacharjee, Peiyu Zhang, Shixuan Li, Wei Yang, Anzhe Cheng, Xiaole Zhang, Jesse Thomason, Ali Jannesari, Nesreen Ahmed, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2510.27617)  

**Abstract**: Automation of Register Transfer Level (RTL) design can help developers meet increasing computational demands. Large Language Models (LLMs) show promise for Hardware Description Language (HDL) generation, but face challenges due to limited parametric knowledge and domain-specific constraints. While prompt engineering and fine-tuning have limitations in knowledge coverage and training costs, multi-agent architectures offer a training-free paradigm to enhance reasoning through collaborative generation. However, current multi-agent approaches suffer from two critical deficiencies: susceptibility to noise propagation and constrained reasoning space exploration. We propose VeriMoA, a training-free mixture-of-agents (MoA) framework with two synergistic innovations. First, a quality-guided caching mechanism to maintain all intermediate HDL outputs and enables quality-based ranking and selection across the entire generation process, encouraging knowledge accumulation over layers of reasoning. Second, a multi-path generation strategy that leverages C++ and Python as intermediate representations, decomposing specification-to-HDL translation into two-stage processes that exploit LLM fluency in high-resource languages while promoting solution diversity. Comprehensive experiments on VerilogEval 2.0 and RTLLM 2.0 benchmarks demonstrate that VeriMoA achieves 15--30% improvements in Pass@1 across diverse LLM backbones, especially enabling smaller models to match larger models and fine-tuned alternatives without requiring costly training. 

**Abstract (ZH)**: 自动化注册传输级（RTL）设计可以帮助开发者满足日益增长的计算需求。大规模语言模型（LLMs）在硬件描述语言（HDL）生成中展现出潜力，但由于参数知识有限和领域特定约束，面临挑战。尽管提示工程和微调在知识覆盖和训练成本方面存在限制，多智能体架构提供了一种无需训练的范式，通过协作生成增强推理。然而，当前的多智能体方法存在两大关键缺陷：容易传播噪声和受限的推理空间探索。我们提出了VeriMoA，一种无需训练的混合多智能体（MoA）框架，并结合了两项协同创新。首先，一种质量指导的缓存机制，以保持所有中间HDL输出，并在整个生成过程中基于质量进行排名和选择，促进多层推理的知识积累。其次，一种多路径生成策略，利用C++和Python作为中间表示，将规格化到HDL的翻译分解为两个阶段的过程，利用大规模语言模型在高资源语言中的流畅性，同时促进解决方案的多样性。在VerilogEval 2.0和RTLLM 2.0基准测试上的全面实验表明，VeriMoA在不同大规模语言模型架构下实现了15-30%的Pass@1改进，特别是使较小的模型能够达到与较大模型和微调替代方案相当的表现，而无需进行昂贵的训练。 

---
# InnovatorBench: Evaluating Agents' Ability to Conduct Innovative LLM Research 

**Title (ZH)**: InnovatorBench: 评估智能体进行创新性大语言模型研究的能力 

**Authors**: Yunze Wu, Dayuan Fu, Weiye Si, Zhen Huang, Mohan Jiang, Keyu Li, Shijie Xia, Jie Sun, Tianze Xu, Xiangkun Hu, Pengrui Lu, Xiaojie Cai, Lyumanshan Ye, Wenhong Zhu, Yang Xiao, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27598)  

**Abstract**: AI agents could accelerate scientific discovery by automating hypothesis formation, experiment design, coding, execution, and analysis, yet existing benchmarks probe narrow skills in simplified settings. To address this gap, we introduce InnovatorBench, a benchmark-platform pair for realistic, end-to-end assessment of agents performing Large Language Model (LLM) research. It comprises 20 tasks spanning Data Construction, Filtering, Augmentation, Loss Design, Reward Design, and Scaffold Construction, which require runnable artifacts and assessment of correctness, performance, output quality, and uncertainty. To support agent operation, we develop ResearchGym, a research environment offering rich action spaces, distributed and long-horizon execution, asynchronous monitoring, and snapshot saving. We also implement a lightweight ReAct agent that couples explicit reasoning with executable planning using frontier models such as Claude-4, GPT-5, GLM-4.5, and Kimi-K2. Our experiments demonstrate that while frontier models show promise in code-driven research tasks, they struggle with fragile algorithm-related tasks and long-horizon decision making, such as impatience, poor resource management, and overreliance on template-based reasoning. Furthermore, agents require over 11 hours to achieve their best performance on InnovatorBench, underscoring the benchmark's difficulty and showing the potential of InnovatorBench to be the next generation of code-based research benchmark. 

**Abstract (ZH)**: AI代理可以通过自动化假设形成、实验设计、编码、执行和分析来加速科学发现，然而现有的基准测试仅在简化场景中测试狭隘技能。为解决这一问题，我们引入了InnovatorBench，这是一个基准平台对，用于评估代理进行大规模语言模型研究的全流程和真实性。该平台包含20项任务，涵盖数据构建、过滤、增强、损失设计、奖励设计和支架构建，要求可运行的成果，并评估正确性、性能、输出质量和不确定性。为了支持代理操作，我们开发了ResearchGym，这是一个研究环境，提供丰富的动作空间、分布式和长周期执行、异步监控和快照保存。我们还实现了一个轻量级的ReAct代理，将显式推理与可执行规划结合，使用前沿模型如Claude-4、GPT-5、GLM-4.5和Kimi-K2。实验显示，尽管前沿模型在编码驱动的研究任务上表现出前景，但在易碎的算法相关任务和长期决策任务（如缺乏耐心、资源管理不当和过度依赖模板推理）上存在问题。此外，代理在InnovatorBench上达到最佳性能需要超过11小时，表明InnovatorBench难度较大，并显示出InnovatorBench作为下一代代码基准的潜力。 

---
# SIGMA: Search-Augmented On-Demand Knowledge Integration for Agentic Mathematical Reasoning 

**Title (ZH)**: SIGMA：增强搜索的动态知识集成以促进能动数学推理 

**Authors**: Ali Asgarov, Umid Suleymanov, Aadyant Khatri  

**Link**: [PDF](https://arxiv.org/pdf/2510.27568)  

**Abstract**: Solving mathematical reasoning problems requires not only accurate access to relevant knowledge but also careful, multi-step thinking. However, current retrieval-augmented models often rely on a single perspective, follow inflexible search strategies, and struggle to effectively combine information from multiple sources. We introduce SIGMA (Search-Augmented On-Demand Knowledge Integration for AGentic Mathematical reAsoning), a unified framework that orchestrates specialized agents to independently reason, perform targeted searches, and synthesize findings through a moderator mechanism. Each agent generates hypothetical passages to optimize retrieval for its analytic perspective, ensuring knowledge integration is both context-sensitive and computation-efficient. When evaluated on challenging benchmarks such as MATH500, AIME, and PhD-level science QA GPQA, SIGMA consistently outperforms both open- and closed-source systems, achieving an absolute performance improvement of 7.4%. Our results demonstrate that multi-agent, on-demand knowledge integration significantly enhances both reasoning accuracy and efficiency, offering a scalable approach for complex, knowledge-intensive problem-solving. We will release the code upon publication. 

**Abstract (ZH)**: 解决数学推理问题不仅需要准确访问相关知识，还需要进行仔细的多步思考。当前的检索增强模型往往依赖单一视角，遵循僵化的搜索策略，并难以有效地综合多个来源的信息。我们提出了SIGMA（搜索增强的按需知识集成以实现代理数学推理），这是一种统一框架，通过调解机制协调专门的代理分别进行推理、执行针对性搜索并综合发现结果。每个代理生成假设性段落以优化其分析视角的检索，确保知识集成既具有上下文敏感性又具有计算效率。在MATH500、AIME和博士级科学问答GPQA等具有挑战性的基准测试中，SIGMA在开放源代码和闭源系统中均表现出色，绝对性能提升7.4%。我们的结果表明，多代理、按需知识集成显著提高了推理的准确性和效率，提供了一种解决复杂、知识密集型问题的可扩展方法。论文发表后我们将发布代码。 

---
# Mechanics of Learned Reasoning 1: TempoBench, A Benchmark for Interpretable Deconstruction of Reasoning System Performance 

**Title (ZH)**: 学习推理的机理研究 1: TempoBench，一种可解释的推理系统性能分解基准测试 

**Authors**: Nikolaus Holzer, William Fishell, Baishakhi Ray, Mark Santolucito  

**Link**: [PDF](https://arxiv.org/pdf/2510.27544)  

**Abstract**: Large Language Models (LLMs) are increasingly excelling and outpacing human performance on many tasks. However, to improve LLM reasoning, researchers either rely on ad-hoc generated datasets or formal mathematical proof systems such as the Lean proof assistant. Whilst ad-hoc generated methods can capture the decision chains of real-world reasoning processes, they may encode some inadvertent bias in the space of reasoning they cover; they also cannot be formally verified. On the other hand, systems like Lean can guarantee verifiability, but are not well-suited to capture the nature of agentic decision chain-based tasks. This creates a gap both in performance for functions such as business agents or code assistants, and in the usefulness of LLM reasoning benchmarks, whereby these fall short in reasoning structure or real-world alignment. We introduce TempoBench, the first formally grounded and verifiable diagnostic benchmark that parametrizes difficulty to systematically analyze how LLMs perform reasoning. TempoBench uses two evaluation benchmarks to break down reasoning ability. First, temporal trace evaluation (TTE) tests the ability of an LLM to understand and simulate the execution of a given multi-step reasoning system. Subsequently, temporal causal evaluation (TCE) tests an LLM's ability to perform multi-step causal reasoning and to distill cause-and-effect relations from complex systems. We find that models score 65.6% on TCE-normal, and 7.5% on TCE-hard. This shows that state-of-the-art LLMs clearly understand the TCE task but perform poorly as system complexity increases. Our code is available at our \href{this https URL}{GitHub repository}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多任务上逐渐超越人类表现。然而，为了提高LLM推理能力，研究人员要么依赖于手工生成的数据集，要么依赖形式化的数学证明系统，如Lean证明助手。虽然手工生成的方法可以捕捉现实世界的推理链，但它们可能会在覆盖的推理空间中编码一些不自觉的偏见；它们也无法被形式验证。另一方面，诸如Lean这样的系统可以确保可验证性，但不太适合捕捉基于代理决策链的任务的本质。这在性能上产生了差距，特别是在商务代理或代码助手等功能方面，并且影响了LLM推理基准的实际有用性，因为这些基准在推理结构或现实世界对齐方面不足。我们提出了TempoBench，这是首个基于形式验证的诊断基准，参数化难度以系统分析LLM的推理能力。TempoBench使用两个评估基准来分解推理能力。首先，时间轨迹评估（TTE）测试LLM理解并模拟给定多步推理系统执行的能力。随后，时间因果评估（TCE）测试LLM进行多步因果推理以及从复杂系统中提炼因果关系的能力。我们发现，模型在TCE-normal上的得分为65.6%，在TCE-hard上的得分仅为7.5%。这表明最先进的LLM显然理解TCE任务，但在系统复杂度增加时表现较差。我们的代码可在我们的GitHub仓库（[这个链接](this https URL)）中获得。 

---
# GeoFM: Enhancing Geometric Reasoning of MLLMs via Synthetic Data Generation through Formal Language 

**Title (ZH)**: GeoFM: 通过形式语言生成合成数据增强MLLMs的几何推理能力 

**Authors**: Yuhao Zhang, Dingxin Hu, Tinghao Yu, Hao Liu, Yiting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27448)  

**Abstract**: Multi-modal Large Language Models (MLLMs) have gained significant attention in both academia and industry for their capabilities in handling multi-modal tasks. However, these models face challenges in mathematical geometric reasoning due to the scarcity of high-quality geometric data. To address this issue, synthetic geometric data has become an essential strategy. Current methods for generating synthetic geometric data involve rephrasing or expanding existing problems and utilizing predefined rules and templates to create geometric images and problems. However, these approaches often produce data that lacks diversity or is prone to noise. Additionally, the geometric images synthesized by existing methods tend to exhibit limited variation and deviate significantly from authentic geometric diagrams. To overcome these limitations, we propose GeoFM, a novel method for synthesizing geometric data. GeoFM uses formal languages to explore combinations of conditions within metric space, generating high-fidelity geometric problems that differ from the originals while ensuring correctness through a symbolic engine. Experimental results show that our synthetic data significantly outperforms existing methods. The model trained with our data surpass the proprietary GPT-4o model by 18.7\% on geometry problem-solving tasks in MathVista and by 16.5\% on GeoQA. Additionally, it exceeds the performance of a leading open-source model by 5.7\% on MathVista and by 2.7\% on GeoQA. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）在学术界和工业界因其处理多模态任务的能力而引起了广泛关注。然而，这些模型在数学几何推理方面面临挑战，因为高质量几何数据稀缺。为解决这一问题，合成几何数据已成为一项关键技术。当前生成合成几何数据的方法涉及重新表述或扩展现有问题，并利用预定义的规则和模板来创建几何图像和问题。然而，这些方法往往生成的数据缺乏多样性或容易产生噪声。此外，现有方法合成的几何图像往往表现出有限的变化，并且与真实的几何图明显不同。为克服这些限制，我们提出了一种新颖的合成几何数据方法——GeoFM。GeoFM 使用形式语言探索度量空间内的条件组合，生成高保真度的几何问题，通过符号引擎确保正确性。实验结果表明，我们生成的数据显著优于现有方法。使用我们数据训练的模型在MathVista的几何问题解决任务中比专有模型GPT-4o高出了18.7%，在GeoQA中高出了16.5%。此外，在MathVista和GeoQA上，它分别超过了领先开源模型5.7%和2.7%。 

---
# DeepCompress: A Dual Reward Strategy for Dynamically Exploring and Compressing Reasoning Chains 

**Title (ZH)**: DeepCompress: 动态探索和压缩推理链的双重奖励策略 

**Authors**: Tian Liang, Wenxiang Jiao, Zhiwei He, Jiahao Xu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27419)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive capabilities but suffer from cognitive inefficiencies like ``overthinking'' simple problems and ``underthinking'' complex ones. While existing methods that use supervised fine-tuning~(SFT) or reinforcement learning~(RL) with token-length rewards can improve efficiency, they often do so at the cost of accuracy. This paper introduces \textbf{DeepCompress}, a novel framework that simultaneously enhances both the accuracy and efficiency of LRMs. We challenge the prevailing approach of consistently favoring shorter reasoning paths, showing that longer responses can contain a broader range of correct solutions for difficult problems. DeepCompress employs an adaptive length reward mechanism that dynamically classifies problems as ``Simple'' or ``Hard'' in real-time based on the model's evolving capability. It encourages shorter, more efficient reasoning for ``Simple'' problems while promoting longer, more exploratory thought chains for ``Hard'' problems. This dual-reward strategy enables the model to autonomously adjust its Chain-of-Thought (CoT) length, compressing reasoning for well-mastered problems and extending it for those it finds challenging. Experimental results on challenging mathematical benchmarks show that DeepCompress consistently outperforms baseline methods, achieving superior accuracy while significantly improving token efficiency. 

**Abstract (ZH)**: Large Reasoning Models (LRMs)表现出色但存在认知效率问题，如“复杂问题上的过度思考”和“简单问题上的欠考虑”。虽然现有的使用监督微调(SFT)或带标记长度奖励的强化学习(RL)的方法可以提高效率，但往往会以牺牲准确性为代价。本文提出了**DeepCompress**，一种同时提高LRMs准确性和效率的新框架。我们挑战了倾向于始终选择较短推理路径的做法，表明较长的回答可以包含更多困难问题的正确解决方案。DeepCompress采用了一个自适应长度奖励机制，能够根据模型能力的演变实时将问题分类为“简单”或“困难”。对于“简单”问题，它鼓励更短更高效的推理；而对于“困难”问题，它促进更长、更具探索性的思维链。这种双重奖励策略使模型能够自主调整其思维链（CoT）的长度，对于已掌握的问题进行推理压缩，而对于找到困难的问题则延长推理。实验结果表明，DeepCompress在具有挑战性的数学基准测试中优于基线方法，不仅实现了更高的准确性，还显著提高了标记效率。 

---
# Dialogue as Discovery: Navigating Human Intent Through Principled Inquiry 

**Title (ZH)**: 对话即发现：通过原则性的询问导航人类意图 

**Authors**: Jianwen Sun, Yukang Feng, Yifan Chang, Chuanhao Li, Zizhen Li, Jiaxin Ai, Fanrui Zhang, Yu Dai, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27410)  

**Abstract**: A fundamental bottleneck in human-AI collaboration is the "intention expression gap," the difficulty for humans to effectively convey complex, high-dimensional thoughts to AI. This challenge often traps users in inefficient trial-and-error loops and is exacerbated by the diverse expertise levels of users. We reframe this problem from passive instruction following to a Socratic collaboration paradigm, proposing an agent that actively probes for information to resolve its uncertainty about user intent. we name the proposed agent Nous, trained to acquire proficiency in this inquiry policy. The core mechanism of Nous is a training framework grounded in the first principles of information theory. Within this framework, we define the information gain from dialogue as an intrinsic reward signal, which is fundamentally equivalent to the reduction of Shannon entropy over a structured task space. This reward design enables us to avoid reliance on costly human preference annotations or external reward models. To validate our framework, we develop an automated simulation pipeline to generate a large-scale, preference-based dataset for the challenging task of scientific diagram generation. Comprehensive experiments, including ablations, subjective and objective evaluations, and tests across user expertise levels, demonstrate the effectiveness of our proposed framework. Nous achieves leading efficiency and output quality, while remaining robust to varying user expertise. Moreover, its design is domain-agnostic, and we show evidence of generalization beyond diagram generation. Experimental results prove that our work offers a principled, scalable, and adaptive paradigm for resolving uncertainty about user intent in complex human-AI collaboration. 

**Abstract (ZH)**: 人类与AI协作中的一个根本瓶颈是“意图表达差距”，即人类难以有效地将复杂的、高维度的思想传达给AI。这一挑战通常将用户困在低效的试错循环中，并因用户专业知识水平的多样性而加剧。我们从被动的指令遵循重新定义这一问题，转向苏格拉底式协作范式，提出一个能够主动探询信息、解决其对用户意图不确定性的代理。我们称这个提出的代理为Nous，并训练其掌握这种查询策略。Nous的核心机制是基于信息论基本原理的训练框架，在这个框架中，我们将对话的信息增益定义为内在的奖励信号，本质上等同于在结构化任务空间中香农熵的减少。这一奖励设计使我们能够避免依赖昂贵的人类偏好注释或外部奖励模型。为了验证我们的框架，我们开发了一个自动化的仿真管道，用于生成大规模的基于偏好的数据集，用于科学图表生成这一具有挑战性的任务。包括消融实验、主观评价、客观评价和跨用户专业知识水平的测试在内的全面实验显示了我们提出的框架的有效性。Nous在效率和输出质量方面达到了领先水平，并且在不同的用户专业知识水平下保持了鲁棒性。此外，其设计具有领域通用性，并展示了其在图表生成之外的泛化能力。实验结果证明，我们的工作提供了一种解决复杂人类与AI协作中用户意图不确定性问题的原则性、可扩展和适应性范式。 

---
# Realistic pedestrian-driver interaction modelling using multi-agent RL with human perceptual-motor constraints 

**Title (ZH)**: 基于人类感知-运动约束的多智能体RL的现实行人-驾驶人交互建模 

**Authors**: Yueyang Wang, Mehmet Dogar, Gustav Markkula  

**Link**: [PDF](https://arxiv.org/pdf/2510.27383)  

**Abstract**: Modelling pedestrian-driver interactions is critical for understanding human road user behaviour and developing safe autonomous vehicle systems. Existing approaches often rely on rule-based logic, game-theoretic models, or 'black-box' machine learning methods. However, these models typically lack flexibility or overlook the underlying mechanisms, such as sensory and motor constraints, which shape how pedestrians and drivers perceive and act in interactive scenarios. In this study, we propose a multi-agent reinforcement learning (RL) framework that integrates both visual and motor constraints of pedestrian and driver agents. Using a real-world dataset from an unsignalised pedestrian crossing, we evaluate four model variants, one without constraints, two with either motor or visual constraints, and one with both, across behavioural metrics of interaction realism. Results show that the combined model with both visual and motor constraints performs best. Motor constraints lead to smoother movements that resemble human speed adjustments during crossing interactions. The addition of visual constraints introduces perceptual uncertainty and field-of-view limitations, leading the agents to exhibit more cautious and variable behaviour, such as less abrupt deceleration. In this data-limited setting, our model outperforms a supervised behavioural cloning model, demonstrating that our approach can be effective without large training datasets. Finally, our framework accounts for individual differences by modelling parameters controlling the human constraints as population-level distributions, a perspective that has not been explored in previous work on pedestrian-vehicle interaction modelling. Overall, our work demonstrates that multi-agent RL with human constraints is a promising modelling approach for simulating realistic road user interactions. 

**Abstract (ZH)**: 基于人类约束的多智能体强化学习建模对于理解行人与驾驶员行为及开发安全的自动驾驶车辆系统至关重要。 

---
# ToolScope: An Agentic Framework for Vision-Guided and Long-Horizon Tool Use 

**Title (ZH)**: ToolScope：一种自主框架，用于视觉导向和长时 horizon 工具使用 

**Authors**: Mengjie Deng, Guanting Dong, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2510.27363)  

**Abstract**: Recently, large language models (LLMs) have demonstrated remarkable problem-solving capabilities by autonomously integrating with external tools for collaborative reasoning. However, due to the inherently complex and diverse nature of multimodal information, enabling multimodal large language models (MLLMs) to flexibly and efficiently utilize external tools during reasoning remains an underexplored challenge. In this work, we introduce ToolScope, an agentic framework designed to unify global planning with local multimodal perception, adopting a specialized Perceive tool to mitigates visual context degradation in long-horizon VQA task. ToolScope comprises three primary components: the Global Navigator, the Agentic Executor, and the Response Synthesizer. The Global Navigator functions as a "telescope", offering high-level strategic guidance. The Agentic Executor operates iteratively to augment MLLM with local perception through the integration of external tools-Search, Code, and Perceive. Finally, the Response Synthesizer consolidates and organizes the reasoning process into a coherent, user-friendly output. We evaluate ToolScope on four VQA benchmarks across diverse domains, including VQA 2.0, ScienceQA, MAT-Search and MathVista. It demonstrates strong generalization capabilities, achieving an average performance improvement of up to +6.69% across all datasets. 

**Abstract (ZH)**: 最近，大型语言模型（LLMs）通过自主集成外部工具进行协作推理，展示了显著的问题解决能力。然而，由于多模态信息本质上复杂多样，使多模态大型语言模型（MLLMs）在推理过程中灵活高效地利用外部工具仍是一个待探索的挑战。本文介绍了ToolScope，该框架旨在统一全局规划与局部多模态感知，采用专门的Perceive工具来缓解长时距VQA任务中的视觉上下文退化。ToolScope包含三个主要组件：全局导航器、能动执行器和响应合成器。全局导航器作为“望远镜”，提供高层次的战略指导。能动执行器通过集成外部工具（搜索、代码和感知）迭代地增强MLLM的局部感知能力。最后，响应合成器将推理过程整合成一个连贯且用户友好的输出。我们在四个跨学科的VQA基准测试中评估了ToolScope，包括VQA 2.0、ScienceQA、MAT-Search和MathVista，它显示了强大的泛化能力，在所有数据集上平均性能提升高达6.69%。 

---
# An In-depth Study of LLM Contributions to the Bin Packing Problem 

**Title (ZH)**: LLM在Bin Packing问题研究中的深度贡献 

**Authors**: Julien Herrmann, Guillaume Pallez  

**Link**: [PDF](https://arxiv.org/pdf/2510.27353)  

**Abstract**: Recent studies have suggested that Large Language Models (LLMs) could provide interesting ideas contributing to mathematical discovery. This claim was motivated by reports that LLM-based genetic algorithms produced heuristics offering new insights into the online bin packing problem under uniform and Weibull distributions. In this work, we reassess this claim through a detailed analysis of the heuristics produced by LLMs, examining both their behavior and interpretability. Despite being human-readable, these heuristics remain largely opaque even to domain experts. Building on this analysis, we propose a new class of algorithms tailored to these specific bin packing instances. The derived algorithms are significantly simpler, more efficient, more interpretable, and more generalizable, suggesting that the considered instances are themselves relatively simple. We then discuss the limitations of the claim regarding LLMs' contribution to this problem, which appears to rest on the mistaken assumption that the instances had previously been studied. Our findings instead emphasize the need for rigorous validation and contextualization when assessing the scientific value of LLM-generated outputs. 

**Abstract (ZH)**: 近期研究表明，大型语言模型（LLMs）能够为数学发现提供有趣的想法。这一论点是由基于LLMs的遗传算法在均匀分布和威布尔分布下对在线箱填充问题产生的启发式方法提供新见解的报告所激发的。在此项工作中，我们通过详细分析LLMs产生的启发式方法，研究了它们的行为和可解释性。尽管这些启发式方法具有人类可读性，但即使是领域专家也无法充分理解它们。基于此分析，我们提出了一类新的算法，专门针对这些特定的箱填充实例。所获得的算法显著简化、更高效、更可解释、更具通用性，这表明所考虑的实例本身相对简单。随后，我们讨论了关于LLMs对此问题贡献的论点的局限性，该论点似乎基于一个错误的前提，即这些实例之前已被研究过。我们的发现强调，在评估LLM生成输出的科学价值时，需要严格的验证和上下文分析。 

---
# Discriminative Rule Learning for Outcome-Guided Process Model Discovery 

**Title (ZH)**: 基于结果导向的过程模型发现的区分性规则学习 

**Authors**: Ali Norouzifar, Wil van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2510.27343)  

**Abstract**: Event logs extracted from information systems offer a rich foundation for understanding and improving business processes. In many real-world applications, it is possible to distinguish between desirable and undesirable process executions, where desirable traces reflect efficient or compliant behavior, and undesirable ones may involve inefficiencies, rule violations, delays, or resource waste. This distinction presents an opportunity to guide process discovery in a more outcome-aware manner. Discovering a single process model without considering outcomes can yield representations poorly suited for conformance checking and performance analysis, as they fail to capture critical behavioral differences. Moreover, prioritizing one behavior over the other may obscure structural distinctions vital for understanding process outcomes. By learning interpretable discriminative rules over control-flow features, we group traces with similar desirability profiles and apply process discovery separately within each group. This results in focused and interpretable models that reveal the drivers of both desirable and undesirable executions. The approach is implemented as a publicly available tool and it is evaluated on multiple real-life event logs, demonstrating its effectiveness in isolating and visualizing critical process patterns. 

**Abstract (ZH)**: 来自信息系统提取的事件日志为理解和改进业务流程提供了丰富的基础。在许多实际应用中，可以区分期望和非期望的流程执行，其中期望的痕迹反映了高效或合规的行为，而不期望的痕迹可能涉及低效率、规则违反、延迟或资源浪费。这种区分为以更注重结果的方式指导流程发现提供了机会。不考虑结果发现单一的流程模型可能会导致不合适的表示，因为它们未能捕捉到关键的行为差异。此外，优先考虑一种行为而忽视另一种行为可能会模糊对理解流程结果至关重要的结构差异。通过学习控制流特征上的可解释区分规则，我们将具有相似期望性概况的痕迹分组，并在每个组内独立地进行流程发现。这导致了专注于结果并具有解释性的模型，揭示了期望和非期望执行的驱动因素。该方法作为开源工具实现，并在多个实际事件日志上进行评估，证明了其在隔离和可视化关键流程模式方面的有效性。 

---
# Reinforcement Learning for Long-Horizon Unordered Tasks: From Boolean to Coupled Reward Machines 

**Title (ZH)**: 长时限序任务的强化学习：从布尔型到耦合奖励机器 

**Authors**: Kristina Levina, Nikolaos Pappas, Athanasios Karapantelakis, Aneta Vulgarakis Feljan, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2510.27329)  

**Abstract**: Reward machines (RMs) inform reinforcement learning agents about the reward structure of the environment. This is particularly advantageous for complex non-Markovian tasks because agents with access to RMs can learn more efficiently from fewer samples. However, learning with RMs is ill-suited for long-horizon problems in which a set of subtasks can be executed in any order. In such cases, the amount of information to learn increases exponentially with the number of unordered subtasks. In this work, we address this limitation by introducing three generalisations of RMs: (1) Numeric RMs allow users to express complex tasks in a compact form. (2) In Agenda RMs, states are associated with an agenda that tracks the remaining subtasks to complete. (3) Coupled RMs have coupled states associated with each subtask in the agenda. Furthermore, we introduce a new compositional learning algorithm that leverages coupled RMs: Q-learning with coupled RMs (CoRM). Our experiments show that CoRM scales better than state-of-the-art RM algorithms for long-horizon problems with unordered subtasks. 

**Abstract (ZH)**: 奖励机器（RMs）为强化学习代理提供了环境奖励结构的信息。这对于复杂的非马尔可夫任务尤其有利，因为具有RMs访问权限的代理可以从更少的数据中更高效地学习。然而，使用RMs学习不适合处理子任务可以以任意顺序执行的长期问题。在这种情况下，需要学习的信息量随无序子任务数量的增加而指数级增长。在本文中，我们通过引入三种RMs的通用形式来解决这一限制：（1）数字RMs允许用户以紧凑的形式表示复杂任务。（2）议程RMs中，状态与一个跟踪剩余需完成子任务的议程相关联。（3）耦合RMs中，每个议程中的子任务关联有耦合状态。此外，我们还引入了一种新的组合学习算法，该算法利用耦合RMs：带有耦合RMs的Q学习（CoRM）。我们的实验表明，CoRM在处理具有无序子任务的长期问题时比最先进的RM算法更具可扩展性。 

---
# GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation 

**Title (ZH)**: GUI-Rise: 结构化推理与历史总结在GUI导航中的应用 

**Authors**: Tao Liu, Chongyu Wang, Rongjie Li, Yingchen Yu, Xuming He, Bai Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.27210)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have advanced GUI navigation agents, current approaches face limitations in cross-domain generalization and effective history utilization. We present a reasoning-enhanced framework that systematically integrates structured reasoning, action prediction, and history summarization. The structured reasoning component generates coherent Chain-of-Thought analyses combining progress estimation and decision reasoning, which inform both immediate action predictions and compact history summaries for future steps. Based on this framework, we train a GUI agent, \textbf{GUI-Rise}, through supervised fine-tuning on pseudo-labeled trajectories and reinforcement learning with Group Relative Policy Optimization (GRPO). This framework employs specialized rewards, including a history-aware objective, directly linking summary quality to subsequent action performance. Comprehensive evaluations on standard benchmarks demonstrate state-of-the-art results under identical training data conditions, with particularly strong performance in out-of-domain scenarios. These findings validate our framework's ability to maintain robust reasoning and generalization across diverse GUI navigation tasks. Code is available at this https URL. 

**Abstract (ZH)**: 尽管多模态大型语言模型（MLLMs）已推动了GUI导航代理的发展，当前的方法在跨域泛化和有效历史利用方面仍存在局限。我们提出了一种增强推理的框架，系统地整合了结构化推理、动作预测和历史总结。结构化推理组件生成结合进度估计和决策推理的连贯思维链分析，这些分析既指导即时动作预测，又生成紧凑的历史总结用于未来步骤。基于此框架，我们通过伪标签轨迹的监督微调和基于组相对策略优化（GRPO）的强化学习训练了一个GUI代理——GUI-Rise。该框架采用专门的奖励，包括一个历史意识目标，直接将总结质量与后续动作表现联系起来。在标准基准上的全面评估显示，在相同的训练数据条件下达到最先进的结果，特别是在跨域场景中的表现尤为出色。这些发现验证了该框架在各种GUI导航任务中保持稳健推理和泛化的能力。代码可在以下链接获取：this https URL。 

---
# Fints: Efficient Inference-Time Personalization for LLMs with Fine-Grained Instance-Tailored Steering 

**Title (ZH)**: Fints：高效 inference 时个人化调整的细粒度实例导向导航 

**Authors**: Kounianhua Du, Jianxing Liu, Kangning Zhang, Wenxiang Jiao, Yuan Lu, Jiarui Jin, Weiwen Liu, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27206)  

**Abstract**: The rapid evolution of large language models (LLMs) has intensified the demand for effective personalization techniques that can adapt model behavior to individual user preferences. Despite the non-parametric methods utilizing the in-context learning ability of LLMs, recent parametric adaptation methods, including personalized parameter-efficient fine-tuning and reward modeling emerge. However, these methods face limitations in handling dynamic user patterns and high data sparsity scenarios, due to low adaptability and data efficiency. To address these challenges, we propose a fine-grained and instance-tailored steering framework that dynamically generates sample-level interference vectors from user data and injects them into the model's forward pass for personalized adaptation. Our approach introduces two key technical innovations: a fine-grained steering component that captures nuanced signals by hooking activations from attention and MLP layers, and an input-aware aggregation module that synthesizes these signals into contextually relevant enhancements. The method demonstrates high flexibility and data efficiency, excelling in fast-changing distribution and high data sparsity scenarios. In addition, the proposed method is orthogonal to existing methods and operates as a plug-in component compatible with different personalization techniques. Extensive experiments across diverse scenarios--including short-to-long text generation, and web function calling--validate the effectiveness and compatibility of our approach. Results show that our method significantly enhances personalization performance in fast-shifting environments while maintaining robustness across varying interaction modes and context lengths. Implementation is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进化加剧了对有效个性化技术的需求，这些技术可以将模型行为适应个性化用户偏好。尽管非参数方法利用了LLMs的在上下文学习能力，但最近的参数适配方法，包括个性化参数高效微调和奖励建模，应运而生。然而，这些方法在处理动态用户模式和高数据稀疏场景时面临局限性，由于其适应能力和数据效率较低。为此，我们提出了一种细粒度和实例定制的控制框架，动态生成用户数据级别的干扰向量，并将其注入模型的前向传播过程以实现个性化适配。我们的方法引入了两项关键技术创新：细粒度的控制组件，通过连接注意力和MLP层的激活来捕捉细腻信号，以及输入感知聚合模块，将这些信号综合为上下文相关增强。该方法展示了高灵活性和数据效率，在快速变化的分布和高数据稀疏场景中表现出色。此外，所提方法与现有方法正交，并充当与不同个性化技术兼容的插件组件。通过多种场景下的广泛实验——包括从短文本到长文本生成和网页函数调用——验证了我们方法的有效性和兼容性。结果显示，在快速变化的环境中，我们的方法显著提高了个性化性能，同时在各种交互模式和上下文长度下保持了鲁棒性。实现可供此链接访问。 

---
# From product to system network challenges in system of systems lifecycle management 

**Title (ZH)**: 从产品到系统网络在系统体系生命周期管理中的挑战 

**Authors**: Vahid Salehi, Josef Vilsmeier, Shirui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27194)  

**Abstract**: Today, products are no longer isolated artifacts, but nodes in networked systems. This means that traditional, linearly conceived life cycle models are reaching their limits: Interoperability across disciplines, variant and configuration management, traceability, and governance across organizational boundaries are becoming key factors. This collective contribution classifies the state of the art and proposes a practical frame of reference for SoS lifecycle management, model-based systems engineering (MBSE) as the semantic backbone, product lifecycle management (PLM) as the governance and configuration level, CAD-CAE as model-derived domains, and digital thread and digital twin as continuous feedback. Based on current literature and industry experience, mobility, healthcare, and the public sector, we identify four principles: (1) referenced architecture and data models, (2) end-to-end configuration sovereignty instead of tool silos, (3) curated models with clear review gates, and (4) measurable value contributions along time, quality, cost, and sustainability. A three-step roadmap shows the transition from product- to network- centric development: piloting with reference architecture, scaling across variant and supply chain spaces, organizational anchoring (roles, training, compliance). The results are increased change robustness, shorter throughput times, improved reuse, and informed sustainability decisions. This article is aimed at decision-makers and practitioners who want to make complexity manageable and design SoS value streams to be scalable. 

**Abstract (ZH)**: 当今的产品不再是孤立的实体，而是网络系统中的节点。这表明传统的线性生命周期模型已达到其极限：跨学科的互操作性、变体和配置管理、可追溯性以及跨越组织边界的治理已成为关键因素。本文综述了该领域的现状，并基于基于模型的系统工程（MBSE）语义骨干、产品生命周期管理（PLM）的治理和配置层级、CAD-CAE的模型派生领域、以及数字主线和数字孪生的持续反馈，提出了一种实际的框架参考，适用于SoS生命周期管理。根据当前文献和行业经验，我们确定了四个原则：（1）引用的架构和数据模型，（2）端到端的配置主权而非工具孤岛，（3）经过精心筛选的模型，并设有明确的评审关卡，（4）按照时间、质量、成本和可持续性衡量的价值贡献。三个阶段的路线图展示了从产品导向到网络导向的发展转变：以参考架构为试点，跨不同变体和供应链空间进行扩展，组织嵌入（角色、培训、合规性）。结果包括增强的变更稳健性、缩短的流动时间、提高的重用性以及知情的可持续性决策。本文旨在为希望管理复杂性和设计可扩展SoS价值流的决策者和实践者提供指导。 

---
# Glia: A Human-Inspired AI for Automated Systems Design and Optimization 

**Title (ZH)**: 胶质体：一种受人脑启发的自动化系统设计与优化的人工智能 

**Authors**: Pouya Hamadanian, Pantea Karimi, Arash Nasr-Esfahany, Kimia Noorbakhsh, Joseph Chandler, Ali ParandehGheibi, Mohammad Alizadeh, Hari Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2510.27176)  

**Abstract**: Can an AI autonomously design mechanisms for computer systems on par with the creativity and reasoning of human experts? We present Glia, an AI architecture for networked systems design that uses large language models (LLMs) in a human-inspired, multi-agent workflow. Each agent specializes in reasoning, experimentation, and analysis, collaborating through an evaluation framework that grounds abstract reasoning in empirical feedback. Unlike prior ML-for-systems methods that optimize black-box policies, Glia generates interpretable designs and exposes its reasoning process. When applied to a distributed GPU cluster for LLM inference, it produces new algorithms for request routing, scheduling, and auto-scaling that perform at human-expert levels in significantly less time, while yielding novel insights into workload behavior. Our results suggest that by combining reasoning LLMs with structured experimentation, an AI can produce creative and understandable designs for complex systems problems. 

**Abstract (ZH)**: 人工智能能否自主设计出与人类专家创造力和推理能力相匹敌的计算机系统机制？我们介绍了Glia，一种灵感来源于人类的多智能体网络系统设计架构，使用大型语言模型。每个智能体专门负责推理、实验和分析，并通过一个评估框架进行协作，将抽象推理与经验反馈结合。与以往针对系统的黑盒策略优化方法不同，Glia生成可解释的设计并揭示其推理过程。当应用于分布式GPU集群进行大型语言模型推理时，它产生了新型请求路由、调度和自动扩展算法，在显著减少时间的同时实现与人类专家相当的性能，并揭示了工作负载行为的新见解。我们的结果表明，通过将推理型大型语言模型与结构化实验相结合，人工智能可以为复杂系统问题生成创造性和可理解的设计。 

---
# CombiGraph-Vis: A Curated Multimodal Olympiad Benchmark for Discrete Mathematical Reasoning 

**Title (ZH)**: CombiGraph-Vis: 一个精选的多模态奥林匹克基准数据集用于离散数学推理 

**Authors**: Hamed Mahdavi, Pouria Mahdavinia, Alireza Farhadi, Pegah Mohammadipour, Samira Malek, Majid Daliri, Pedram Mohammadipour, Alireza Hashemi, Amir Khasahmadi, Vasant Honavar  

**Link**: [PDF](https://arxiv.org/pdf/2510.27094)  

**Abstract**: State-of-the-art (SOTA) LLMs have progressed from struggling on proof-based Olympiad problems to solving most of the IMO 2025 problems, with leading systems reportedly handling 5 of 6 problems. Given this progress, we assess how well these models can grade proofs: detecting errors, judging their severity, and assigning fair scores beyond binary correctness. We study proof-analysis capabilities using a corpus of 90 Gemini 2.5 Pro-generated solutions that we grade on a 1-4 scale with detailed error annotations, and on MathArena solution sets for IMO/USAMO 2025 scored on a 0-7 scale. Our analysis shows that models can reliably flag incorrect (including subtly incorrect) solutions but exhibit calibration gaps in how partial credit is assigned. To address this, we introduce agentic workflows that extract and analyze reference solutions and automatically derive problem-specific rubrics for a multi-step grading process. We instantiate and compare different design choices for the grading workflows, and evaluate their trade-offs. Across our annotated corpus and MathArena, our proposed workflows achieve higher agreement with human grades and more consistent handling of partial credit across metrics. We release all code, data, and prompts/logs to facilitate future research. 

**Abstract (ZH)**: 最先进的LLM从难以应对基于证明的奥林匹克问题进步到解决2025年国际数学奥林匹克(IMO)大部分问题，领先的系统据报道解决了6道题中的5道。鉴于这一进展，我们评估这些模型在评分证明方面的能力：检测错误、判断严重程度以及在二元正确性之外分配公平分数。我们使用90个Gemini 2.5生成的解答集进行研究，这些解答集按1-4分尺度评分，并附有详细的错误注释；同时，我们还在MathArena的2025年IMO/USAMO解答集上进行评分，采用0-7分尺度。我们的分析显示，模型可以可靠地标记出错误（包括细微错误）的解答，但在如何分配部分分数方面存在校准差距。为了解决这一问题，我们引入了代理工作流，提取并分析参考解答，自动为多步评分过程推导问题特定的评分标准。我们实例化并比较了不同评分工作流的设计选择，并评估了它们的权衡。在我们的注释语料库和MathArena上，我们提出的这些工作流在与人类评分的共识度及在各个评分标准下对部分分数的一致处理上表现更优。我们发布了所有代码、数据和提示/日志，以促进未来的研究。 

---
# Adaptive Data Flywheel: Applying MAPE Control Loops to AI Agent Improvement 

**Title (ZH)**: 自适应数据飞轮：将MAPE控制环应用于AI代理改进 

**Authors**: Aaditya Shukla, Sidney Knowles, Meenakshi Madugula, Dave Farris, Ryan Angilly, Santiago Pombo, Anbang Xu, Lu An, Abhinav Balasubramanian, Tan Yu, Jiaxiang Ren, Rama Akkiraju  

**Link**: [PDF](https://arxiv.org/pdf/2510.27051)  

**Abstract**: Enterprise AI agents must continuously adapt to maintain accuracy, reduce latency, and remain aligned with user needs. We present a practical implementation of a data flywheel in NVInfo AI, NVIDIA's Mixture-of-Experts (MoE) Knowledge Assistant serving over 30,000 employees. By operationalizing a MAPE-driven data flywheel, we built a closed-loop system that systematically addresses failures in retrieval-augmented generation (RAG) pipelines and enables continuous learning. Over a 3-month post-deployment period, we monitored feedback and collected 495 negative samples. Analysis revealed two major failure modes: routing errors (5.25\%) and query rephrasal errors (3.2\%). Using NVIDIA NeMo microservices, we implemented targeted improvements through fine-tuning. For routing, we replaced a Llama 3.1 70B model with a fine-tuned 8B variant, achieving 96\% accuracy, a 10x reduction in model size, and 70\% latency improvement. For query rephrasal, fine-tuning yielded a 3.7\% gain in accuracy and a 40\% latency reduction. Our approach demonstrates how human-in-the-loop (HITL) feedback, when structured within a data flywheel, transforms enterprise AI agents into self-improving systems. Key learnings include approaches to ensure agent robustness despite limited user feedback, navigating privacy constraints, and executing staged rollouts in production. This work offers a repeatable blueprint for building robust, adaptive enterprise AI agents capable of learning from real-world usage at scale. 

**Abstract (ZH)**: 企业AI代理必须持续适应以维持准确性、降低延迟并保持与用户需求一致。我们提出了NVInfo AI中的一种实用数据飞轮实现，这是一种运行在NVIDIA Mixture-of-Experts（MoE）知识助手上的系统，该助手服务于超过30,000名员工。通过在MAPE驱动的数据飞轮中实现这一目标，我们构建了一个闭环系统，系统性地解决了检索增强生成（RAG）管道中的故障，并实现了持续学习。在部署后的3个月内，我们监测反馈并收集了495个负样本。分析显示了两种主要的故障模式：路由错误（5.25%）和查询重表述错误（3.2%）。利用NVIDIA NeMo微服务，我们通过微调实现了有针对性的改进。对于路由问题，我们将一个Llama 3.1 70B模型替换为一个微调后的8B变体，实现了96%的准确率，模型大小减少了10倍，延迟降低了70%。对于查询重表述问题，微调带来了3.7%的准确率提升和40%的延迟减少。我们的方法展示了如何通过数据飞轮中的环路反馈机制，将人类在环（HITL）反馈转化为自我改进的企业级AI代理系统。关键经验教训包括确保代理稳健性的方法，尽管用户反馈有限；处理隐私限制的方法；以及在生产环境中执行分阶段滚动部署的方法。这项工作为企业级AI代理构建可重复的稳健、自适应系统提供了蓝图，这些系统能够大规模地从实际使用中学习。 

---
# e1: Learning Adaptive Control of Reasoning Effort 

**Title (ZH)**: 学习自适应推理努力控制 

**Authors**: Michael Kleinman, Matthew Trager, Alessandro Achille, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2510.27042)  

**Abstract**: Increasing the thinking budget of AI models can significantly improve accuracy, but not all questions warrant the same amount of reasoning. Users may prefer to allocate different amounts of reasoning effort depending on how they value output quality versus latency and cost. To leverage this tradeoff effectively, users need fine-grained control over the amount of thinking used for a particular query, but few approaches enable such control. Existing methods require users to specify the absolute number of desired tokens, but this requires knowing the difficulty of the problem beforehand to appropriately set the token budget for a query. To address these issues, we propose Adaptive Effort Control, a self-adaptive reinforcement learning method that trains models to use a user-specified fraction of tokens relative to the current average chain-of-thought length for each query. This approach eliminates dataset- and phase-specific tuning while producing better cost-accuracy tradeoff curves compared to standard methods. Users can dynamically adjust the cost-accuracy trade-off through a continuous effort parameter specified at inference time. We observe that the model automatically learns to allocate resources proportionally to the task difficulty and, across model scales ranging from 1.5B to 32B parameters, our approach enables approximately 3x reduction in chain-of-thought length while maintaining or improving performance relative to the base model used for RL training. 

**Abstract (ZH)**: 自适应努力控制：一种基于自适应强化学习的细粒度计算控制方法 

---
# Causal Masking on Spatial Data: An Information-Theoretic Case for Learning Spatial Datasets with Unimodal Language Models 

**Title (ZH)**: 基于因果掩蔽的空间数据：使用单模语言模型学习空间数据的信息论案例 

**Authors**: Jared Junkin, Samuel Nathanson  

**Link**: [PDF](https://arxiv.org/pdf/2510.27009)  

**Abstract**: Language models are traditionally designed around causal masking. In domains with spatial or relational structure, causal masking is often viewed as inappropriate, and sequential linearizations are instead used. Yet the question of whether it is viable to accept the information loss introduced by causal masking on nonsequential data has received little direct study, in part because few domains offer both spatial and sequential representations of the same dataset. In this work, we investigate this issue in the domain of chess, which naturally supports both representations. We train language models with bidirectional and causal self-attention mechanisms on both spatial (board-based) and sequential (move-based) data. Our results show that models trained on spatial board states - \textit{even with causal masking} - consistently achieve stronger playing strength than models trained on sequential data. While our experiments are conducted on chess, our results are methodological and may have broader implications: applying causal masking to spatial data is a viable procedure for training unimodal LLMs on spatial data, and in some domains is even preferable to sequentialization. 

**Abstract (ZH)**: 语言模型传统上围绕因果掩码设计。在具有空间或关系结构的领域中，因果掩码通常被视为不恰当的，因此使用序列线性化代替。然而，关于因果掩码在非序列数据中引入信息损失的可接受性在很大程度上没有直接研究，部分原因是很少有领域同时提供空间和序列表示的数据集。在本研究中，我们在支持这两种表示方式的象棋领域中探讨了这一问题。我们使用双向和因果注意力机制分别在基于棋盘的空间数据和基于移动的序列数据上训练语言模型。我们发现，即使使用因果掩码，在空间棋盘状态上训练的模型——其表现 consistently 比在序列数据上训练的模型更强。虽然我们的实验是在象棋上进行的，但我们的结果具有方法论意义，并可能具有更广泛的影响：对空间数据训练单模LLM时，使用因果掩码是可行的程序，甚至在某些领域中，其效果优于序列化。 

---
# SUSTAINABLE Platform: Seamless Smart Farming Integration Towards Agronomy Automation 

**Title (ZH)**: 可持续平台：向农作自动化无缝集成的智能 farming 解决方案 

**Authors**: Agorakis Bompotas, Konstantinos Koutras, Nikitas Rigas Kalogeropoulos, Panagiotis Kechagias, Dimitra Gariza, Athanasios P. Kalogeras, Christos Alexakos  

**Link**: [PDF](https://arxiv.org/pdf/2510.26989)  

**Abstract**: The global agricultural sector is undergoing a transformative shift, driven by increasing food demands, climate variability and the need for sustainable practices. SUSTAINABLE is a smart farming platform designed to integrate IoT, AI, satellite imaging, and role-based task orchestration to enable efficient, traceable, and sustainable agriculture with a pilot usecase in viticulture. This paper explores current smart agriculture solutions, presents a comparative evaluation, and introduces SUSTAINABLE's key features, including satellite index integration, real-time environmental data, and role-aware task management tailored to Mediterranean vineyards. 

**Abstract (ZH)**: 全球农业领域正经历一场转型变革，受到不断增长的粮食需求、气候变化以及可持续实践的需要的驱动。SUSTAINABLE是一个智能 farming 平台，旨在集成物联网、人工智能、卫星成像和基于角色的任务编排，以实现高效、可追溯和可持续的农业，并在葡萄种植业中进行了试点应用。本文探讨了当前的智能农业解决方案，进行了比较评估，并介绍了SUSTAINABLE的关键功能，包括卫星指数整合、实时环境数据和针对地中海葡萄园的角色感知任务管理。 

---
# Cognition Envelopes for Bounded AI Reasoning in Autonomous UAS Operations 

**Title (ZH)**: 认知边界下的受限人工智能推理在自主UAS操作中 

**Authors**: Pedro Antonio Alarcón Granadeno, Arturo Miguel Bernal Russell, Sofia Nelson, Demetrius Hernandez, Maureen Petterson, Michael Murphy, Walter J. Scheirer, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26905)  

**Abstract**: Cyber-physical systems increasingly rely on Foundational Models such as Large Language Models (LLMs) and Vision-Language Models (VLMs) to increase autonomy through enhanced perception, inference, and planning. However, these models also introduce new types of errors, such as hallucinations, overgeneralizations, and context misalignments, resulting in incorrect and flawed decisions. To address this, we introduce the concept of Cognition Envelopes, designed to establish reasoning boundaries that constrain AI-generated decisions while complementing the use of meta-cognition and traditional safety envelopes. As with safety envelopes, Cognition Envelopes require practical guidelines and systematic processes for their definition, validation, and assurance. 

**Abstract (ZH)**: 基于物理系统的认知包络：大型语言模型和视觉-语言模型引入的新错误及其应对方法 

---
# The Denario project: Deep knowledge AI agents for scientific discovery 

**Title (ZH)**: Denary项目：面向科学发现的深度知识AI代理 

**Authors**: Francisco Villaescusa-Navarro, Boris Bolliet, Pablo Villanueva-Domingo, Adrian E. Bayer, Aidan Acquah, Chetana Amancharla, Almog Barzilay-Siegal, Pablo Bermejo, Camille Bilodeau, Pablo Cárdenas Ramírez, Miles Cranmer, Urbano L. França, ChangHoon Hahn, Yan-Fei Jiang, Raul Jimenez, Jun-Young Lee, Antonio Lerario, Osman Mamun, Thomas Meier, Anupam A. Ojha, Pavlos Protopapas, Shimanto Roy, David N. Spergel, Pedro Tarancón-Álvarez, Ujjwal Tiwari, Matteo Viel, Digvijay Wadekar, Chi Wang, Bonny Y. Wang, Licong Xu, Yossi Yovel, Shuwen Yue, Wen-Han Zhou, Qiyao Zhu, Jiajun Zou, Íñigo Zubeldia  

**Link**: [PDF](https://arxiv.org/pdf/2510.26887)  

**Abstract**: We present Denario, an AI multi-agent system designed to serve as a scientific research assistant. Denario can perform many different tasks, such as generating ideas, checking the literature, developing research plans, writing and executing code, making plots, and drafting and reviewing a scientific paper. The system has a modular architecture, allowing it to handle specific tasks, such as generating an idea, or carrying out end-to-end scientific analysis using Cmbagent as a deep-research backend. In this work, we describe in detail Denario and its modules, and illustrate its capabilities by presenting multiple AI-generated papers generated by it in many different scientific disciplines such as astrophysics, biology, biophysics, biomedical informatics, chemistry, material science, mathematical physics, medicine, neuroscience and planetary science. Denario also excels at combining ideas from different disciplines, and we illustrate this by showing a paper that applies methods from quantum physics and machine learning to astrophysical data. We report the evaluations performed on these papers by domain experts, who provided both numerical scores and review-like feedback. We then highlight the strengths, weaknesses, and limitations of the current system. Finally, we discuss the ethical implications of AI-driven research and reflect on how such technology relates to the philosophy of science. We publicly release the code at this https URL. A Denario demo can also be run directly on the web at this https URL, and the full app will be deployed on the cloud. 

**Abstract (ZH)**: 我们介绍Denario，一个设计用于担任科学研究助理的人工智能多agents系统。Denario能够执行多种任务，如生成想法、查阅文献、制定研究计划、编写和执行代码、制作图表以及起草和审查科研论文。该系统具有模块化架构，允许它处理特定任务，如生成想法或使用Cmbagent作为深度研究后端进行端到端的科学研究分析。在本文中，我们详细描述了Denario及其模块，并通过展示它在多个科学学科（如天体物理、生物学、生物物理学、生物医学信息学、化学、材料科学、数学物理、医学、神经科学和行星科学）中生成的多篇AI论文，来阐述其功能。Denario还擅长将不同学科的想法结合起来，我们通过展示一篇应用量子物理和机器学习方法处理天体物理学数据的论文来说明这一点。我们报告了这些论文领域专家的评估结果，他们提供了数值评分和类似审稿的反馈。然后，我们强调当前系统的优势、弱点和限制。最后，我们讨论了由AI驱动的科研的伦理影响，并反思这种技术与科学哲学的关系。我们已将代码公开发布于此httpsURL。同时，您也可以直接在网页上运行Denario的演示版，完整应用程序将部署在云上。 

---
# Inverse Knowledge Search over Verifiable Reasoning: Synthesizing a Scientific Encyclopedia from a Long Chains-of-Thought Knowledge Base 

**Title (ZH)**: 可验证推理中的逆向知识搜索：从长链条思考知识库合成科学百科全书 

**Authors**: Yu Li, Yuan Huang, Tao Wang, Caiyu Fan, Xiansheng Cai, Sihan Hu, Xinzijian Liu, Cheng Shi, Mingjun Xu, Zhen Wang, Yan Wang, Xiangqi Jin, Tianhan Zhang, Linfeng Zhang, Lei Wang, Youjin Deng, Pan Zhang, Weijie Sun, Xingyu Li, Weinan E, Linfeng Zhang, Zhiyuan Yao, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26854)  

**Abstract**: Most scientific materials compress reasoning, presenting conclusions while omitting the derivational chains that justify them. This compression hinders verification by lacking explicit, step-wise justifications and inhibits cross-domain links by collapsing the very pathways that establish the logical and causal connections between concepts. We introduce a scalable framework that decompresses scientific reasoning, constructing a verifiable Long Chain-of-Thought (LCoT) knowledge base and projecting it into an emergent encyclopedia, SciencePedia. Our pipeline operationalizes an endpoint-driven, reductionist strategy: a Socratic agent, guided by a curriculum of around 200 courses, generates approximately 3 million first-principles questions. To ensure high fidelity, multiple independent solver models generate LCoTs, which are then rigorously filtered by prompt sanitization and cross-model answer consensus, retaining only those with verifiable endpoints. This verified corpus powers the Brainstorm Search Engine, which performs inverse knowledge search -- retrieving diverse, first-principles derivations that culminate in a target concept. This engine, in turn, feeds the Plato synthesizer, which narrates these verified chains into coherent articles. The initial SciencePedia comprises approximately 200,000 fine-grained entries spanning mathematics, physics, chemistry, biology, engineering, and computation. In evaluations across six disciplines, Plato-synthesized articles (conditioned on retrieved LCoTs) exhibit substantially higher knowledge-point density and significantly lower factual error rates than an equally-prompted baseline without retrieval (as judged by an external LLM). Built on this verifiable LCoT knowledge base, this reasoning-centric approach enables trustworthy, cross-domain scientific synthesis at scale and establishes the foundation for an ever-expanding encyclopedia. 

**Abstract (ZH)**: 一种可扩展的框架：分解科学推理，构建可验证的长推理链知识库并投影至科学百宗百科 

---
# CATArena: Evaluation of LLM Agents through Iterative Tournament Competitions 

**Title (ZH)**: CATArena：通过迭代锦标赛竞争评估LLM代理 

**Authors**: Lingyue Fu, Xin Ding, Yaoming Zhu, Shao Zhang, Lin Qiu, Weiwen Liu, Weinan Zhang, Xuezhi Cao, Xunliang Cai, Jiaxin Ding, Yong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26852)  

**Abstract**: Large Language Model (LLM) agents have evolved from basic text generation to autonomously completing complex tasks through interaction with external tools. However, current benchmarks mainly assess end-to-end performance in fixed scenarios, restricting evaluation to specific skills and suffering from score saturation and growing dependence on expert annotation as agent capabilities improve. In this work, we emphasize the importance of learning ability, including both self-improvement and peer-learning, as a core driver for agent evolution toward human-level intelligence. We propose an iterative, competitive peer-learning framework, which allows agents to refine and optimize their strategies through repeated interactions and feedback, thereby systematically evaluating their learning capabilities. To address the score saturation issue in current benchmarks, we introduce CATArena, a tournament-style evaluation platform featuring four diverse board and card games with open-ended scoring. By providing tasks without explicit upper score limits, CATArena enables continuous and dynamic evaluation of rapidly advancing agent capabilities. Experimental results and analyses involving both minimal and commercial code agents demonstrate that CATArena provides reliable, stable, and scalable benchmarking for core agent abilities, particularly learning ability and strategy coding. 

**Abstract (ZH)**: 大型语言模型代理已经从基本文本生成演进为通过与外部工具的交互自主完成复杂任务。然而，当前的基准测试主要评估固定场景中的端到端性能，这限制了对特定技能的评估，并且随着代理能力的提高，分数饱和和对专家标注的依赖性也在增加。在本文中，我们强调学习能力，包括自我改进和同伴学习，作为代理向人类水平智能进化的核心驱动力。我们提出了一种迭代的、竞争性的同伴学习框架，允许代理通过反复的交互和反馈来完善和优化其策略，从而系统地评估其学习能力。为了解决当前基准测试中的分数饱和问题，我们引入了CATArena，一个采用四种不同棋盘和纸牌游戏的锦标赛式评估平台，具有开放性得分机制。通过提供没有明确分数上限的任务，CATArena使得能够持续和动态地评估快速进步的代理能力。实验结果和分析表明，CATArena为核心代理能力，尤其是学习能力和策略编码，提供了可靠、稳定和可扩展的基准测试。 

---
# Continuous Autoregressive Language Models 

**Title (ZH)**: 连续自回归语言模型 

**Authors**: Chenze Shao, Darren Li, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.27688)  

**Abstract**: The efficiency of large language models (LLMs) is fundamentally limited by their sequential, token-by-token generation process. We argue that overcoming this bottleneck requires a new design axis for LLM scaling: increasing the semantic bandwidth of each generative step. To this end, we introduce Continuous Autoregressive Language Models (CALM), a paradigm shift from discrete next-token prediction to continuous next-vector prediction. CALM uses a high-fidelity autoencoder to compress a chunk of K tokens into a single continuous vector, from which the original tokens can be reconstructed with over 99.9\% accuracy. This allows us to model language as a sequence of continuous vectors instead of discrete tokens, which reduces the number of generative steps by a factor of K. The paradigm shift necessitates a new modeling toolkit; therefore, we develop a comprehensive likelihood-free framework that enables robust training, evaluation, and controllable sampling in the continuous domain. Experiments show that CALM significantly improves the performance-compute trade-off, achieving the performance of strong discrete baselines at a significantly lower computational cost. More importantly, these findings establish next-vector prediction as a powerful and scalable pathway towards ultra-efficient language models. Code: this https URL. Project: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的效率从根本上受限于其逐 token 生成的过程。我们argue认为突破这一瓶颈需要为LLM扩展引入一个新的设计轴心：增加每一步生成过程中的语义带宽。为此，我们提出了连续自回归语言模型（CALM），这是一种从离散的下一个 token 预测转变为连续的下一个向量预测的范式转变。CALM 使用高保真度自编码器将一段 K 个 token 压缩成一个连续向量，从该向量可以以超过 99.9% 的准确率重构原始 token。这使得我们可以将语言建模为连续向量序列而不是离散 token 序列，从而使生成步骤的数量减少 K 倍。这种范式转变需要一个新的建模工具包；因此，我们开发了一个全面的无似然性框架，可以在连续域中实现稳健的训练、评估和可控采样。实验表明，CALM 显著改善了性能-计算权衡，以显著较低的计算成本实现了强大离散基线的性能。更重要的是，这些发现确立了下一个向量预测是通向超高效语言模型的强大且可扩展的途径。代码：this https URL. 项目：this https URL。 

---
# PETAR: Localized Findings Generation with Mask-Aware Vision-Language Modeling for PET Automated Reporting 

**Title (ZH)**: PETAR：基于掩码意识的视觉-语言模型在PET自动化报告中的局部发现生成 

**Authors**: Danyal Maqbool, Changhee Lee, Zachary Huemann, Samuel D. Church, Matthew E. Larson, Scott B. Perlman, Tomas A. Romero, Joshua D. Warner, Meghan Lubner, Xin Tie, Jameson Merkow, Junjie Hu, Steve Y. Cho, Tyler J. Bradshaw  

**Link**: [PDF](https://arxiv.org/pdf/2510.27680)  

**Abstract**: Recent advances in vision-language models (VLMs) have enabled impressive multimodal reasoning, yet most medical applications remain limited to 2D imaging. In this work, we extend VLMs to 3D positron emission tomography and computed tomography (PET/CT), a domain characterized by large volumetric data, small and dispersed lesions, and lengthy radiology reports. We introduce a large-scale dataset comprising over 11,000 lesion-level descriptions paired with 3D segmentations from more than 5,000 PET/CT exams, extracted via a hybrid rule-based and large language model (LLM) pipeline. Building upon this dataset, we propose PETAR-4B, a 3D mask-aware vision-language model that integrates PET, CT, and lesion contours for spatially grounded report generation. PETAR bridges global contextual reasoning with fine-grained lesion awareness, producing clinically coherent and localized findings. Comprehensive automated and human evaluations demonstrate that PETAR substantially improves PET/CT report generation quality, advancing 3D medical vision-language understanding. 

**Abstract (ZH)**: 近期视觉语言模型的发展已在多模态推理方面取得了显著进展，但大多数医疗应用仍然局限于2D成像。本文将视觉语言模型扩展至正电子发射断层扫描和计算机断层扫描(PET/CT)，该领域以大型体数据、小且分散的病灶以及冗长的放射学报告为特征。我们引入了一个大型数据集，包含超过11,000个病灶级别的描述，以及来自超过5,000次PET/CT检查的3D分割，通过混合基于规则和大型语言模型的管道提取。基于该数据集，我们提出了PETAR-4B，这是一种3D掩码感知的视觉语言模型，能够融合PET、CT和病灶轮廓，以实现空间定位的报告生成。PETAR将全局上下文推理与细粒度的病灶意识相结合，生成临床一致且局部化的发现。全面的自动化和人工评估表明，PETAR显著提高了PET/CT报告生成的质量，推动了3D医疗视觉语言理解的进步。 

---
# Challenges in Credit Assignment for Multi-Agent Reinforcement Learning in Open Agent Systems 

**Title (ZH)**: 开放代理系统中多智能体强化学习中的信用分配挑战 

**Authors**: Alireza Saleh Abadi, Leen-Kiat Soh  

**Link**: [PDF](https://arxiv.org/pdf/2510.27659)  

**Abstract**: In the rapidly evolving field of multi-agent reinforcement learning (MARL), understanding the dynamics of open systems is crucial. Openness in MARL refers to the dynam-ic nature of agent populations, tasks, and agent types with-in a system. Specifically, there are three types of openness as reported in (Eck et al. 2023) [2]: agent openness, where agents can enter or leave the system at any time; task openness, where new tasks emerge, and existing ones evolve or disappear; and type openness, where the capabil-ities and behaviors of agents change over time. This report provides a conceptual and empirical review, focusing on the interplay between openness and the credit assignment problem (CAP). CAP involves determining the contribution of individual agents to the overall system performance, a task that becomes increasingly complex in open environ-ments. Traditional credit assignment (CA) methods often assume static agent populations, fixed and pre-defined tasks, and stationary types, making them inadequate for open systems. We first conduct a conceptual analysis, in-troducing new sub-categories of openness to detail how events like agent turnover or task cancellation break the assumptions of environmental stationarity and fixed team composition that underpin existing CAP methods. We then present an empirical study using representative temporal and structural algorithms in an open environment. The results demonstrate that openness directly causes credit misattribution, evidenced by unstable loss functions and significant performance degradation. 

**Abstract (ZH)**: 多代理强化学习（MARL）领域中的开放系统动态研究：开放性与信用分配问题的关系 

---
# Community Detection on Model Explanation Graphs for Explainable AI 

**Title (ZH)**: 模型解释图上的社区检测 for 可解释人工智能 

**Authors**: Ehsan Moradi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27655)  

**Abstract**: Feature-attribution methods (e.g., SHAP, LIME) explain individual predictions but often miss higher-order structure: sets of features that act in concert. We propose Modules of Influence (MoI), a framework that (i) constructs a model explanation graph from per-instance attributions, (ii) applies community detection to find feature modules that jointly affect predictions, and (iii) quantifies how these modules relate to bias, redundancy, and causality patterns. Across synthetic and real datasets, MoI uncovers correlated feature groups, improves model debugging via module-level ablations, and localizes bias exposure to specific modules. We release stability and synergy metrics, a reference implementation, and evaluation protocols to benchmark module discovery in XAI. 

**Abstract (ZH)**: 基于模块的影响特征归因方法（MoI）：构建模型解释图、发现特征模块并量化其与偏见、冗余和因果关系模式的相关性 

---
# Information-Theoretic Greedy Layer-wise Training for Traffic Sign Recognition 

**Title (ZH)**: 基于信息论的贪婪逐层训练在交通标志识别中的应用 

**Authors**: Shuyan Lyu, Zhanzimo Wu, Junliang Du  

**Link**: [PDF](https://arxiv.org/pdf/2510.27651)  

**Abstract**: Modern deep neural networks (DNNs) are typically trained with a global cross-entropy loss in a supervised end-to-end manner: neurons need to store their outgoing weights; training alternates between a forward pass (computation) and a top-down backward pass (learning) which is biologically implausible. Alternatively, greedy layer-wise training eliminates the need for cross-entropy loss and backpropagation. By avoiding the computation of intermediate gradients and the storage of intermediate outputs, it reduces memory usage and helps mitigate issues such as vanishing or exploding gradients. However, most existing layer-wise training approaches have been evaluated only on relatively small datasets with simple deep architectures. In this paper, we first systematically analyze the training dynamics of popular convolutional neural networks (CNNs) trained by stochastic gradient descent (SGD) through an information-theoretic lens. Our findings reveal that networks converge layer-by-layer from bottom to top and that the flow of information adheres to a Markov information bottleneck principle. Building on these observations, we propose a novel layer-wise training approach based on the recently developed deterministic information bottleneck (DIB) and the matrix-based Rényi's $\alpha$-order entropy functional. Specifically, each layer is trained jointly with an auxiliary classifier that connects directly to the output layer, enabling the learning of minimal sufficient task-relevant representations. We empirically validate the effectiveness of our training procedure on CIFAR-10 and CIFAR-100 using modern deep CNNs and further demonstrate its applicability to a practical task involving traffic sign recognition. Our approach not only outperforms existing layer-wise training baselines but also achieves performance comparable to SGD. 

**Abstract (ZH)**: 现代深度神经网络（DNNs）通常以监督端到端的方式使用全局交叉熵损失进行训练：神经元需要存储其权重；训练交替进行前向传播（计算）和自上而下的反向传播（学习），这在生物学上是不可行的。 alternatively，贪婪分层训练消除了交叉熵损失和反向传播的需要。通过避免中间梯度的计算和中间输出的存储，它减少了内存使用，并有助于缓解梯度消失或爆炸等问题。然而，现有的大多数分层训练方法仅在相对较小的数据集和简单的深度架构上进行了评估。在本文中，我们首先从信息论的视角系统地分析了通过随机梯度下降（SGD）训练的流行卷积神经网络（CNNs）的训练动态。我们的发现揭示了网络自下而上逐层收敛，并且信息流遵循马尔可夫信息瓶颈原则。基于这些观察，我们提出了一个新的分层训练方法，该方法基于最近开发的确定性信息瓶颈（DIB）和基于矩阵的Rényi’s α阶熵函數。具体地，每个层与直接连接到输出层的辅助分类器联合训练，使学习最小充分的任务相关表示成为可能。我们在CIFAR-10和CIFAR-100上使用现代深度CNNs实验证明了我们训练过程的有效性，并进一步证明了其在涉及交通标志识别的实际任务中的适用性。我们的方法不仅优于现有的分层训练基线，而且在性能上与SGD相当。 

---
# VessShape: Few-shot 2D blood vessel segmentation by leveraging shape priors from synthetic images 

**Title (ZH)**: VessShape: 通过利用合成图像中的形状先验实现少量样本的二维血 vessels分割 

**Authors**: Cesar H. Comin, Wesley N. Galvão  

**Link**: [PDF](https://arxiv.org/pdf/2510.27646)  

**Abstract**: Semantic segmentation of blood vessels is an important task in medical image analysis, but its progress is often hindered by the scarcity of large annotated datasets and the poor generalization of models across different imaging modalities. A key aspect is the tendency of Convolutional Neural Networks (CNNs) to learn texture-based features, which limits their performance when applied to new domains with different visual characteristics. We hypothesize that leveraging geometric priors of vessel shapes, such as their tubular and branching nature, can lead to more robust and data-efficient models. To investigate this, we introduce VessShape, a methodology for generating large-scale 2D synthetic datasets designed to instill a shape bias in segmentation models. VessShape images contain procedurally generated tubular geometries combined with a wide variety of foreground and background textures, encouraging models to learn shape cues rather than textures. We demonstrate that a model pre-trained on VessShape images achieves strong few-shot segmentation performance on two real-world datasets from different domains, requiring only four to ten samples for fine-tuning. Furthermore, the model exhibits notable zero-shot capabilities, effectively segmenting vessels in unseen domains without any target-specific training. Our results indicate that pre-training with a strong shape bias can be an effective strategy to overcome data scarcity and improve model generalization in blood vessel segmentation. 

**Abstract (ZH)**: 基于几何先验的血管语义分割：一种克服数据稀缺性和提高模型泛化能力的新方法 

---
# Sketch-to-Layout: Sketch-Guided Multimodal Layout Generation 

**Title (ZH)**: 草图引导的多模态布局生成 

**Authors**: Riccardo Brioschi, Aleksandr Alekseev, Emanuele Nevali, Berkay Döner, Omar El Malki, Blagoj Mitrevski, Leandro Kieliger, Mark Collier, Andrii Maksai, Jesse Berent, Claudiu Musat, Efi Kokiopoulou  

**Link**: [PDF](https://arxiv.org/pdf/2510.27632)  

**Abstract**: Graphic layout generation is a growing research area focusing on generating aesthetically pleasing layouts ranging from poster designs to documents. While recent research has explored ways to incorporate user constraints to guide the layout generation, these constraints often require complex specifications which reduce usability. We introduce an innovative approach exploiting user-provided sketches as intuitive constraints and we demonstrate empirically the effectiveness of this new guidance method, establishing the sketch-to-layout problem as a promising research direction, which is currently under-explored. To tackle the sketch-to-layout problem, we propose a multimodal transformer-based solution using the sketch and the content assets as inputs to produce high quality layouts. Since collecting sketch training data from human annotators to train our model is very costly, we introduce a novel and efficient method to synthetically generate training sketches at scale. We train and evaluate our model on three publicly available datasets: PubLayNet, DocLayNet and SlidesVQA, demonstrating that it outperforms state-of-the-art constraint-based methods, while offering a more intuitive design experience. In order to facilitate future sketch-to-layout research, we release O(200k) synthetically-generated sketches for the public datasets above. The datasets are available at this https URL. 

**Abstract (ZH)**: 图形布局生成是一个快速增长的研究领域，专注于从海报设计到文档等各种 aesthetically pleasing 布局的生成。尽管近期的研究探讨了通过引入用户约束来指导布局生成的方法，但这些约束往往需要复杂的规格说明，从而降低了 usability。我们提出了一种创新的方法，利用用户提供草图作为直观的约束条件，并通过实验证明了这种方法的有效性，将草图转换为布局的问题确立为一个有前景的研究方向，目前这一方向尚未得到充分探索。为了应对草图到布局的问题，我们提出了一种基于多模态变压器的解决方案，使用草图和内容资产作为输入以生成高质量的布局。由于从人类标注者收集用于训练模型的草图训练数据成本高昂，我们引入了一种新颖且高效的方法来大规模合成训练草图。我们使用公开可用的三个数据集：PubLayNet、DocLayNet 和 SlidesVQA，对模型进行训练和评估，证明该方法比现有的基于约束的方法表现更优，同时提供了更直观的设计体验。为了便于未来的草图到布局研究，我们发布了超过20万张合成生成的草图，供上述公开数据集使用。数据集可通过以下链接获取。 

---
# Best Practices for Biorisk Evaluations on Open-Weight Bio-Foundation Models 

**Title (ZH)**: 开放源代码生物基础模型的生物风险评估最佳实践 

**Authors**: Boyi Wei, Zora Che, Nathaniel Li, Udari Madhushani Sehwag, Jasper Götting, Samira Nedungadi, Julian Michael, Summer Yue, Dan Hendrycks, Peter Henderson, Zifan Wang, Seth Donoughe, Mantas Mazeika  

**Link**: [PDF](https://arxiv.org/pdf/2510.27629)  

**Abstract**: Open-weight bio-foundation models present a dual-use dilemma. While holding great promise for accelerating scientific research and drug development, they could also enable bad actors to develop more deadly bioweapons. To mitigate the risk posed by these models, current approaches focus on filtering biohazardous data during pre-training. However, the effectiveness of such an approach remains unclear, particularly against determined actors who might fine-tune these models for malicious use. To address this gap, we propose \eval, a framework to evaluate the robustness of procedures that are intended to reduce the dual-use capabilities of bio-foundation models. \eval assesses models' virus understanding through three lenses, including sequence modeling, mutational effects prediction, and virulence prediction. Our results show that current filtering practices may not be particularly effective: Excluded knowledge can be rapidly recovered in some cases via fine-tuning, and exhibits broader generalizability in sequence modeling. Furthermore, dual-use signals may already reside in the pretrained representations, and can be elicited via simple linear probing. These findings highlight the challenges of data filtering as a standalone procedure, underscoring the need for further research into robust safety and security strategies for open-weight bio-foundation models. 

**Abstract (ZH)**: 开放权重生物基础模型存在双重用途难题。尽管这些模型有望加速科学研究和药物开发，但它们也可能被恶意行为者用于开发更具杀伤力的生物武器。为减轻这些模型带来的风险，当前的方法主要集中在预训练过程中过滤生物危害数据。然而，这种方法的有效性尚不明确，尤其对于那些可能精细调整这些模型用于恶意用途的坚定行为者。为填补这一空白，我们提出了一种名为\eval的框架，用于评估旨在降低生物基础模型双重用途能力的程序的鲁棒性。\eval通过序列建模、突变效应预测和致病性预测三个视角评估模型对病毒的理解。我们的结果显示，当前的过滤实践可能并不特别有效：在某些情况下，通过精细调整可以迅速恢复被排除的知识，并且在序列建模中表现出更广泛的泛化能力。此外，双重用途信号可能已经存在于预训练表示中，并可以通过简单的线性探测被引发。这些发现突显了单独依赖数据过滤的挑战，强调了对未来开放权重生物基础模型稳健的安全与安全策略研究的迫切需求。 

---
# Spatial-SSRL: Enhancing Spatial Understanding via Self-Supervised Reinforcement Learning 

**Title (ZH)**: 基于空间的SSRL：通过自我监督强化学习增强空间理解 

**Authors**: Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, Jiaqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27606)  

**Abstract**: Spatial understanding remains a weakness of Large Vision-Language Models (LVLMs). Existing supervised fine-tuning (SFT) and recent reinforcement learning with verifiable rewards (RLVR) pipelines depend on costly supervision, specialized tools, or constrained environments that limit scale. We introduce Spatial-SSRL, a self-supervised RL paradigm that derives verifiable signals directly from ordinary RGB or RGB-D images. Spatial-SSRL automatically formulates five pretext tasks that capture 2D and 3D spatial structure: shuffled patch reordering, flipped patch recognition, cropped patch inpainting, regional depth ordering, and relative 3D position prediction. These tasks provide ground-truth answers that are easy to verify and require no human or LVLM annotation. Training on our tasks substantially improves spatial reasoning while preserving general visual capabilities. On seven spatial understanding benchmarks in both image and video settings, Spatial-SSRL delivers average accuracy gains of 4.63% (3B) and 3.89% (7B) over the Qwen2.5-VL baselines. Our results show that simple, intrinsic supervision enables RLVR at scale and provides a practical route to stronger spatial intelligence in LVLMs. 

**Abstract (ZH)**: Spatial理解仍是一大难题，宏观视觉-语言模型的短板。现有的监督细调（SFT）和最近的可验证奖励强化学习（RLVR）管道依赖于昂贵的监督、专门的工具或受限的环境，从而限制了模型的规模。我们介绍了一种自我监督的RL范式Spatial-SSRL，该范式直接从普通的RGB或RGB-D图像中提取可验证的信号。Spatial-SSRL自动制定了五个预训练任务，捕捉二维和三维空间结构：打乱贴图的重排序、翻转贴图识别、裁剪贴图修复、区域深度排序以及相对三维位置预测。这些任务提供了易于验证的真实答案，无需人类或LVLM注释。通过我们的任务进行训练，显著改善了空间推理能力，同时保持了通用视觉能力。在七个空间理解基准测试中的图像和视频设置下，Spatial-SSRL分别在Qwen2.5-VL基线上取得了4.63%（3B）和3.89%（7B）的平均准确率提升。我们的结果表明，简单的、内在的监督能够实现大规模的RLVR，并提供了一条通往更强空间智能的实用途径。 

---
# Towards Universal Video Retrieval: Generalizing Video Embedding via Synthesized Multimodal Pyramid Curriculum 

**Title (ZH)**: 面向通用视频检索：通过合成多模态分层课程泛化视频嵌入 

**Authors**: Zhuoning Guo, Mingxin Li, Yanzhao Zhang, Dingkun Long, Pengjun Xie, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27571)  

**Abstract**: The prevailing video retrieval paradigm is structurally misaligned, as narrow benchmarks incentivize correspondingly limited data and single-task training. Therefore, universal capability is suppressed due to the absence of a diagnostic evaluation that defines and demands multi-dimensional generalization. To break this cycle, we introduce a framework built on the co-design of evaluation, data, and modeling. First, we establish the Universal Video Retrieval Benchmark (UVRB), a suite of 16 datasets designed not only to measure performance but also to diagnose critical capability gaps across tasks and domains. Second, guided by UVRB's diagnostics, we introduce a scalable synthesis workflow that generates 1.55 million high-quality pairs to populate the semantic space required for universality. Finally, we devise the Modality Pyramid, a curriculum that trains our General Video Embedder (GVE) by explicitly leveraging the latent interconnections within our diverse data. Extensive experiments show GVE achieves state-of-the-art zero-shot generalization on UVRB. In particular, our analysis reveals that popular benchmarks are poor predictors of general ability and that partially relevant retrieval is a dominant but overlooked scenario. Overall, our co-designed framework provides a practical path to escape the limited scope and advance toward truly universal video retrieval. 

**Abstract (ZH)**: 普遍视频检索范式结构上存在偏差，窄化的基准激励对应受限的数据和单任务训练。因此，由于缺乏诊断性评估来定义和要求多维度通用性，普遍能力受到抑制。为打破这一循环，我们提出了一种基于评估、数据和建模联合设计的框架。首先，我们建立了通用视频检索基准（UVRB），一套包含16个数据集，不仅用于衡量性能，还用于诊断任务和领域间的关键能力差距。其次，根据UVRB的诊断结果，我们引入了一个可扩展的合成工作流，生成155万个高质量的数据对，以填充实现通用性所需的语义空间。最后，我们设计了模态金字塔，通过明确利用我们多样化数据中的潜在关联来训练我们的通用视频嵌入器（GVE）。 extensive 实验表明，GVE 在 UVRB 上实现了最先进的零样本泛化能力。特别地，我们的分析显示流行基准对于预测通用能力效果不佳，部分相关检索是一种占主导但被忽视的场景。总体而言，我们联合设计的框架提供了一条实用的道路，以超越有限范围并朝着真正具有通用性的视频检索迈进。 

---
# CodeAlignBench: Assessing Code Generation Models on Developer-Preferred Code Adjustments 

**Title (ZH)**: CodeAlignBench: 评估代码生成模型在开发人员优选代码调整方面的表现 

**Authors**: Forough Mehralian, Ryan Shar, James R. Rae, Alireza Hashemi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27565)  

**Abstract**: As large language models become increasingly capable of generating code, evaluating their performance remains a complex and evolving challenge. Existing benchmarks primarily focus on functional correctness, overlooking the diversity of real-world coding tasks and developer expectations. To this end, we introduce a multi-language benchmark that evaluates LLM instruction-following capabilities and is extensible to operate on any set of standalone coding problems. Our benchmark evaluates instruction following in two key settings: adherence to pre-defined constraints specified with the initial problem, and the ability to perform refinements based on follow-up instructions. For this paper's analysis, we empirically evaluated our benchmarking pipeline with programming tasks from LiveBench, that are also automatically translated from Python into Java and JavaScript. Our automated benchmark reveals that models exhibit differing levels of performance across multiple dimensions of instruction-following. Our benchmarking pipeline provides a more comprehensive evaluation of code generation models, highlighting their strengths and limitations across languages and generation goals. 

**Abstract (ZH)**: 随着大型语言模型在生成代码方面的能力不断增强，评估其性能仍然是一项复杂且不断发展的挑战。现有的基准测试主要侧重于功能正确性，忽视了现实世界编码任务的多样性和开发者期望。为此，我们引入了一个多语言基准测试，评估LLM的指令跟随能力，并且可以扩展到任何独立编码问题的集合。我们的基准测试在两个关键设置下评估指令跟随能力：遵守初始问题中预定义的约束，以及根据后续指令进行改进的能力。对于本文的分析，我们使用LiveBench中的编程任务来实证评估基准测试流程，这些任务还可以自动从Python翻译成Java和JavaScript。我们的自动化基准测试显示，模型在指令跟随的不同维度上表现出不同的性能水平。我们的基准测试流程提供了对代码生成模型更为全面的评估，突显了它们在多种语言和生成目标下的优缺点。 

---
# Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs 

**Title (ZH)**: 基于场景图的语言到动作转换：通过基础模型实现长期 horizon 机器人操作的准确执行 

**Authors**: Sushil Samuel Dinesh, Shinkyu Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.27558)  

**Abstract**: This paper presents a framework that leverages pre-trained foundation models for robotic manipulation without domain-specific training. The framework integrates off-the-shelf models, combining multimodal perception from foundation models with a general-purpose reasoning model capable of robust task sequencing. Scene graphs, dynamically maintained within the framework, provide spatial awareness and enable consistent reasoning about the environment. The framework is evaluated through a series of tabletop robotic manipulation experiments, and the results highlight its potential for building robotic manipulation systems directly on top of off-the-shelf foundation models. 

**Abstract (ZH)**: 本文提出了一种框架，该框架利用预训练基础模型实现机器人操作，无需领域特定训练。该框架结合了来自基础模型的多模态感知和一个通用推理模型，该模型能够实现稳健的任务序列。场景图在框架中动态维护，提供空间意识并使对环境的一致性推理成为可能。该框架通过一系列桌面机器人操作实验进行了评估，结果突显了其直接基于现成基础模型构建机器人操作系统的能力。 

---
# Sybil-Resistant Service Discovery for Agent Economies 

**Title (ZH)**: Sybil攻击抵御的服务发现机制在代理经济中 

**Authors**: David Shi, Kevin Joo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27554)  

**Abstract**: x402 enables Hypertext Transfer Protocol (HTTP) services like application programming interfaces (APIs), data feeds, and inference providers to accept cryptocurrency payments for access. As agents increasingly consume these services, discovery becomes critical: which swap interface should an agent trust? Which data provider is the most reliable? We introduce TraceRank, a reputation-weighted ranking algorithm where payment transactions serve as endorsements. TraceRank seeds addresses with precomputed reputation metrics and propagates reputation through payment flows weighted by transaction value and temporal recency. Applied to x402's payment graph, this surfaces services preferred by high-reputation users rather than those with high transaction volume. Our system combines TraceRank with semantic search to respond to natural language queries with high quality results. We argue that reputation propagation resists Sybil attacks by making spam services with many low-reputation payers rank below legitimate services with few high-reputation payers. Ultimately, we aim to construct a search method for x402 enabled services that avoids infrastructure bias and has better performance than purely volume based or semantic methods. 

**Abstract (ZH)**: x402使Hyper文本传输协议（HTTP）服务如应用程序编程接口（APIs）、数据流和推理提供商能够接受加密货币支付以获取访问权限。随着代理越来越多地消费这些服务，发现变得至关重要：代理应该信任哪一个兑换接口？哪个数据提供商最为可靠？我们引入了TraceRank，这是一种基于声誉加权的排名算法，其中支付交易作为推荐。TraceRank通过支付流传播由交易价值和时间最近性加权的声誉，并预先计算地址的声誉度量指标进行初始化。应用于x402的支付图，这会呈现由高声誉用户偏好的服务，而不是高交易量的服务。我们的系统结合了TraceRank和语义搜索，以用高质量的结果应答自然语言查询。我们认为，声誉传播能够抵御Sybil攻击，因为众多低声誉支付者的仿冒服务排名会低于少数高声誉支付者的合法服务。最终，我们旨在构建一种避免基础设施偏见且性能优于基于纯粹交易量或语义方法的x402使能服务的搜索方法。 

---
# EBT-Policy: Energy Unlocks Emergent Physical Reasoning Capabilities 

**Title (ZH)**: EBT-Policy：能量解锁 emergent 物理推理能力 

**Authors**: Travis Davies, Yiqi Huang, Alexi Gladstone, Yunxin Liu, Xiang Chen, Heng Ji, Huxian Liu, Luhui Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27545)  

**Abstract**: Implicit policies parameterized by generative models, such as Diffusion Policy, have become the standard for policy learning and Vision-Language-Action (VLA) models in robotics. However, these approaches often suffer from high computational cost, exposure bias, and unstable inference dynamics, which lead to divergence under distribution shifts. Energy-Based Models (EBMs) address these issues by learning energy landscapes end-to-end and modeling equilibrium dynamics, offering improved robustness and reduced exposure bias. Yet, policies parameterized by EBMs have historically struggled to scale effectively. Recent work on Energy-Based Transformers (EBTs) demonstrates the scalability of EBMs to high-dimensional spaces, but their potential for solving core challenges in physically embodied models remains underexplored. We introduce a new energy-based architecture, EBT-Policy, that solves core issues in robotic and real-world settings. Across simulated and real-world tasks, EBT-Policy consistently outperforms diffusion-based policies, while requiring less training and inference computation. Remarkably, on some tasks it converges within just two inference steps, a 50x reduction compared to Diffusion Policy's 100. Moreover, EBT-Policy exhibits emergent capabilities not seen in prior models, such as zero-shot recovery from failed action sequences using only behavior cloning and without explicit retry training. By leveraging its scalar energy for uncertainty-aware inference and dynamic compute allocation, EBT-Policy offers a promising path toward robust, generalizable robot behavior under distribution shifts. 

**Abstract (ZH)**: 基于能量模型的Implicit策略：解决机器人和现实世界核心问题的新架构 

---
# DialectalArabicMMLU: Benchmarking Dialectal Capabilities in Arabic and Multilingual Language Models 

**Title (ZH)**: 方言阿拉伯MMLU：评估阿拉伯语和多语言语言模型的方言能力 

**Authors**: Malik H. Altakrori, Nizar Habash, Abdelhakim Freihat, Younes Samih, Kirill Chirkunov, Muhammed AbuOdeh, Radu Florian, Teresa Lynn, Preslav Nakov, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2510.27543)  

**Abstract**: We present DialectalArabicMMLU, a new benchmark for evaluating the performance of large language models (LLMs) across Arabic dialects. While recently developed Arabic and multilingual benchmarks have advanced LLM evaluation for Modern Standard Arabic (MSA), dialectal varieties remain underrepresented despite their prevalence in everyday communication. DialectalArabicMMLU extends the MMLU-Redux framework through manual translation and adaptation of 3K multiple-choice question-answer pairs into five major dialects (Syrian, Egyptian, Emirati, Saudi, and Moroccan), yielding a total of 15K QA pairs across 32 academic and professional domains (22K QA pairs when also including English and MSA). The benchmark enables systematic assessment of LLM reasoning and comprehension beyond MSA, supporting both task-based and linguistic analysis. We evaluate 19 open-weight Arabic and multilingual LLMs (1B-13B parameters) and report substantial performance variation across dialects, revealing persistent gaps in dialectal generalization. DialectalArabicMMLU provides the first unified, human-curated resource for measuring dialectal understanding in Arabic, thus promoting more inclusive evaluation and future model development. 

**Abstract (ZH)**: DialectalArabicMMLU：一种新的评估大型语言模型在阿拉伯方言表现的基准 

---
# TetraJet-v2: Accurate NVFP4 Training for Large Language Models with Oscillation Suppression and Outlier Control 

**Title (ZH)**: TetraJet-v2: 准确的NVFP4训练方法，包含振荡抑制和异常值控制 

**Authors**: Yuxiang Chen, Xiaoming Xu, Pengle Zhang, Michael Beyer, Martin Rapp, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.27527)  

**Abstract**: Large Language Models (LLMs) training is prohibitively expensive, driving interest in low-precision fully-quantized training (FQT). While novel 4-bit formats like NVFP4 offer substantial efficiency gains, achieving near-lossless training at such low precision remains challenging. We introduce TetraJet-v2, an end-to-end 4-bit FQT method that leverages NVFP4 for activations, weights, and gradients in all linear layers. We identify two critical issues hindering low-precision LLM training: weight oscillation and outliers. To address these, we propose: 1) an unbiased double-block quantization method for NVFP4 linear layers, 2) OsciReset, an algorithm to suppress weight oscillation, and 3) OutControl, an algorithm to retain outlier accuracy. TetraJet-v2 consistently outperforms prior FP4 training methods on pre-training LLMs across varying model sizes up to 370M and data sizes up to 200B tokens, reducing the performance gap to full-precision training by an average of 51.3%. 

**Abstract (ZH)**: 大型语言模型（LLMs）训练代价高昂，推动了低精度全量化训练（FQT）的兴趣。虽然像NVFP4这样的新型4位格式提供了显著的效率提升，但在如此低精度下实现近似无损训练仍然颇具挑战。我们引入了TetraJet-v2，这是一种端到端的4位FQT方法，利用NVFP4对所有线性层的激活、权重和梯度进行量化。我们识别出了阻碍低精度LLM训练的两个关键问题：权重振荡和异常值。为了解决这些问题，我们提出：1）用于NVFP4线性层的无偏双块量化方法，2）抑制权重振荡的OsciReset算法，以及3）保留异常值精度的OutControl算法。TetraJet-v2在不同模型规模（最大370M）和数据规模（最大200B令牌）的预训练LLM上均优于先前的FP4训练方法，平均将性能差距减少51.3%。 

---
# Leveraging Generic Time Series Foundation Models for EEG Classification 

**Title (ZH)**: 利用通用时间序列基础模型进行EEG分类 

**Authors**: Théo Gnassounou, Yessin Moakher, Shifeng Xie, Vasilii Feofanov, Ievgen Redko  

**Link**: [PDF](https://arxiv.org/pdf/2510.27522)  

**Abstract**: Foundation models for time series are emerging as powerful general-purpose backbones, yet their potential for domain-specific biomedical signals such as electroencephalography (EEG) remains rather unexplored. In this work, we investigate the applicability a recently proposed time series classification foundation model, to a different EEG tasks such as motor imagery classification and sleep stage prediction. We test two pretraining regimes: (a) pretraining on heterogeneous real-world time series from multiple domains, and (b) pretraining on purely synthetic data. We find that both variants yield strong performance, consistently outperforming EEGNet, a widely used convolutional baseline, and CBraMod, the most recent EEG-specific foundation model. These results suggest that generalist time series foundation models, even when pretrained on data of non-neural origin or on synthetic signals, can transfer effectively to EEG. Our findings highlight the promise of leveraging cross-domain pretrained models for brain signal analysis, suggesting that EEG may benefit from advances in the broader time series literature. 

**Abstract (ZH)**: 基于时间序列的基础模型正逐步成为强有力的通用骨干模型，但在如脑电图（EEG）等特定生物医学信号领域的应用潜力尚未被充分探索。在本文中，我们研究了一种 recently 提出的时间序列分类基础模型在不同EEG任务中的适用性，如运动想象分类和睡眠阶段预测。我们测试了两种预训练方式：（a）基于多个领域的真实世界异构时间序列数据进行预训练；（b）基于纯合成数据进行预训练。我们发现这两种方式均表现优异，一致性地超过了广泛使用的卷积基线EEGNet和最新的特定于EEG的基础模型CBraMod。这些结果表明，即使基于非神经源数据或合成信号进行预训练，通用时间序列基础模型也能有效转移应用于EEG。我们的研究结果强调了交叉领域预训练模型在脑电信号分析中的潜力，表明EEG可能从更广泛的时间序列文献进展中受益。 

---
# Context-Gated Cross-Modal Perception with Visual Mamba for PET-CT Lung Tumor Segmentation 

**Title (ZH)**: 基于上下文门控跨模态感知的视觉Mamba PET-CT肺肿瘤分割 

**Authors**: Elena Mulero Ayllón, Linlin Shen, Pierangelo Veltri, Fabrizia Gelardi, Arturo Chiti, Paolo Soda, Matteo Tortora  

**Link**: [PDF](https://arxiv.org/pdf/2510.27508)  

**Abstract**: Accurate lung tumor segmentation is vital for improving diagnosis and treatment planning, and effectively combining anatomical and functional information from PET and CT remains a major challenge. In this study, we propose vMambaX, a lightweight multimodal framework integrating PET and CT scan images through a Context-Gated Cross-Modal Perception Module (CGM). Built on the Visual Mamba architecture, vMambaX adaptively enhances inter-modality feature interaction, emphasizing informative regions while suppressing noise. Evaluated on the PCLT20K dataset, the model outperforms baseline models while maintaining lower computational complexity. These results highlight the effectiveness of adaptive cross-modal gating for multimodal tumor segmentation and demonstrate the potential of vMambaX as an efficient and scalable framework for advanced lung cancer analysis. The code is available at this https URL. 

**Abstract (ZH)**: 准确的肺部肿瘤分割对于提高诊断和治疗规划至关重要，有效结合PET和CT的解剖与功能信息仍然是一个主要挑战。本研究提出了一种轻量级多模态框架vMambaX，该框架通过上下文门控跨模态感知模块(CGM)整合PET和CT扫描图像。基于Visual Mamba架构，vMambaX自适应地增强跨模态特征交互，强调信息丰富的区域并抑制噪声。在PCLT20K数据集上的评估结果显示，该模型在保持较低计算复杂度的同时优于基准模型。这些结果突显了自适应跨模态门控在多模态肿瘤分割中的有效性，并展示了vMambaX作为高级肺癌分析高效且可扩展框架的潜力。代码可在以下链接获取：this https URL。 

---
# DP-FedPGN: Finding Global Flat Minima for Differentially Private Federated Learning via Penalizing Gradient Norm 

**Title (ZH)**: DP-FedPGN: 通过惩罚梯度范数寻找差分隐私联邦学习中的全局平坦最小值 

**Authors**: Junkang Liu, Yuxuan Tian, Fanhua Shang, Yuanyuan Liu, Hongying Liu, Junchao Zhou, Daorui Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.27504)  

**Abstract**: To prevent inference attacks in Federated Learning (FL) and reduce the leakage of sensitive information, Client-level Differentially Private Federated Learning (CL-DPFL) is widely used. However, current CL-DPFL methods usually result in sharper loss landscapes, which leads to a decrease in model generalization after differential privacy protection. By using Sharpness Aware Minimization (SAM), the current popular federated learning methods are to find a local flat minimum value to alleviate this problem. However, the local flatness may not reflect the global flatness in CL-DPFL. Therefore, to address this issue and seek global flat minima of models, we propose a new CL-DPFL algorithm, DP-FedPGN, in which we introduce a global gradient norm penalty to the local loss to find the global flat minimum. Moreover, by using our global gradient norm penalty, we not only find a flatter global minimum but also reduce the locally updated norm, which means that we further reduce the error of gradient clipping. From a theoretical perspective, we analyze how DP-FedPGN mitigates the performance degradation caused by DP. Meanwhile, the proposed DP-FedPGN algorithm eliminates the impact of data heterogeneity and achieves fast convergence. We also use Rényi DP to provide strict privacy guarantees and provide sensitivity analysis for local updates. Finally, we conduct effectiveness tests on both ResNet and Transformer models, and achieve significant improvements in six visual and natural language processing tasks compared to existing state-of-the-art algorithms. The code is available at this https URL 

**Abstract (ZH)**: 防止联邦学习中推理攻击并减少敏感信息泄露的客户端差分隐私联邦学习：通过全局梯度范数惩罚寻找全局平坦极小值 

---
# InertialAR: Autoregressive 3D Molecule Generation with Inertial Frames 

**Title (ZH)**: 基于惯性参考系的自回归三维分子生成 

**Authors**: Haorui Li, Weitao Du, Yuqiang Li, Hongyu Guo, Shengchao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27497)  

**Abstract**: Transformer-based autoregressive models have emerged as a unifying paradigm across modalities such as text and images, but their extension to 3D molecule generation remains underexplored. The gap stems from two fundamental challenges: (1) tokenizing molecules into a canonical 1D sequence of tokens that is invariant to both SE(3) transformations and atom index permutations, and (2) designing an architecture capable of modeling hybrid atom-based tokens that couple discrete atom types with continuous 3D coordinates. To address these challenges, we introduce InertialAR. InertialAR devises a canonical tokenization that aligns molecules to their inertial frames and reorders atoms to ensure SE(3) and permutation invariance. Moreover, InertialAR equips the attention mechanism with geometric awareness via geometric rotary positional encoding (GeoRoPE). In addition, it utilizes a hierarchical autoregressive paradigm to predict the next atom-based token, predicting the atom type first and then its 3D coordinates via Diffusion loss. Experimentally, InertialAR achieves state-of-the-art performance on 7 of the 10 evaluation metrics for unconditional molecule generation across QM9, GEOM-Drugs, and B3LYP. Moreover, it significantly outperforms strong baselines in controllable generation for targeted chemical functionality, attaining state-of-the-art results across all 5 metrics. 

**Abstract (ZH)**: 基于Transformer的自回归模型在文本和图像等多种模态中脱颖而出，但将其扩展到3D分子生成领域仍鲜有探索。InertialAR：基于惯性框架的3D分子自回归生成模型 

---
# FedAdamW: A Communication-Efficient Optimizer with Convergence and Generalization Guarantees for Federated Large Models 

**Title (ZH)**: FedAdamW：具有收敛性和泛化保证的大规模联邦模型通信高效优化器 

**Authors**: Junkang Liu, Fanhua Shang, Kewen Zhu, Hongying Liu, Yuanyuan Liu, Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27486)  

**Abstract**: AdamW has become one of the most effective optimizers for training large-scale models. We have also observed its effectiveness in the context of federated learning (FL). However, directly applying AdamW in federated learning settings poses significant challenges: (1) due to data heterogeneity, AdamW often yields high variance in the second-moment estimate $\boldsymbol{v}$; (2) the local overfitting of AdamW may cause client drift; and (3) Reinitializing moment estimates ($\boldsymbol{v}$, $\boldsymbol{m}$) at each round slows down convergence. To address these challenges, we propose the first \underline{Fed}erated \underline{AdamW} algorithm, called \texttt{FedAdamW}, for training and fine-tuning various large models. \texttt{FedAdamW} aligns local updates with the global update using both a \textbf{local correction mechanism} and decoupled weight decay to mitigate local overfitting. \texttt{FedAdamW} efficiently aggregates the \texttt{mean} of the second-moment estimates to reduce their variance and reinitialize them. Theoretically, we prove that \texttt{FedAdamW} achieves a linear speedup convergence rate of $\mathcal{O}(\sqrt{(L \Delta \sigma_l^2)/(S K R \epsilon^2)}+(L \Delta)/R)$ without \textbf{heterogeneity assumption}, where $S$ is the number of participating clients per round, $K$ is the number of local iterations, and $R$ is the total number of communication rounds. We also employ PAC-Bayesian generalization analysis to explain the effectiveness of decoupled weight decay in local training. Empirically, we validate the effectiveness of \texttt{FedAdamW} on language and vision Transformer models. Compared to several baselines, \texttt{FedAdamW} significantly reduces communication rounds and improves test accuracy. The code is available in this https URL. 

**Abstract (ZH)**: AdamW已成为训练大规模模型最有效的优化器之一。我们还发现它在联邦学习（FL）环境中也非常有效。然而，直接在联邦学习环境中应用AdamW会面临重大挑战：（1）由于数据异质性，AdamW在二阶矩估计$\boldsymbol{v}$中往往产生高方差；（2）AdamW的局部过拟合可能会导致客户端漂移；（3）每轮重新初始化动量估计值（$\boldsymbol{v}$，$\boldsymbol{m}$）会减缓收敛速度。为了解决这些挑战，我们提出了第一个用于训练和微调各种大型模型的联邦AdamW算法，称为\texttt{FedAdamW}。\texttt{FedAdamW}通过本地校正机制和解耦的权重衰减来对齐局部更新和全局更新，以减轻局部过拟合。\texttt{FedAdamW}有效地聚合二阶矩估计值的均值，以减少其方差并重新初始化它们。从理论上讲，我们证明了\texttt{FedAdamW}在没有异质性假设的情况下实现了线性加速收敛率$\mathcal{O}(\sqrt{(L \Delta \sigma_l^2)/(S K R \epsilon^2)}+(L \Delta)/R)$，其中$S$是每轮参与的客户端数量，$K$是局部迭代次数，$R$是总通信轮数。我们还使用PAC-贝叶斯泛化分析来解释解耦权重衰减在局部训练中的有效性。实验上，我们在语言和视觉变换器模型上验证了\texttt{FedAdamW}的有效性。与几种基线方法相比，\texttt{FedAdamW}显著减少了通信轮数并提高了测试精度。相关代码可在以下链接获取。 

---
# Thought Branches: Interpreting LLM Reasoning Requires Resampling 

**Title (ZH)**: 思维分支：解释大模型推理需要重采样 

**Authors**: Uzay Macar, Paul C. Bogdan, Senthooran Rajamanoharan, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2510.27484)  

**Abstract**: Most work interpreting reasoning models studies only a single chain-of-thought (CoT), yet these models define distributions over many possible CoTs. We argue that studying a single sample is inadequate for understanding causal influence and the underlying computation. Though fully specifying this distribution is intractable, it can be understood by sampling. We present case studies using resampling to investigate model decisions. First, when a model states a reason for its action, does that reason actually cause the action? In "agentic misalignment" scenarios, we resample specific sentences to measure their downstream effects. Self-preservation sentences have small causal impact, suggesting they do not meaningfully drive blackmail. Second, are artificial edits to CoT sufficient for steering reasoning? These are common in literature, yet take the model off-policy. Resampling and selecting a completion with the desired property is a principled on-policy alternative. We find off-policy interventions yield small and unstable effects compared to resampling in decision-making tasks. Third, how do we understand the effect of removing a reasoning step when the model may repeat it post-edit? We introduce a resilience metric that repeatedly resamples to prevent similar content from reappearing downstream. Critical planning statements resist removal but have large effects when eliminated. Fourth, since CoT is sometimes "unfaithful", can our methods teach us anything in these settings? Adapting causal mediation analysis, we find that hints that have a causal effect on the output without being explicitly mentioned exert a subtle and cumulative influence on the CoT that persists even if the hint is removed. Overall, studying distributions via resampling enables reliable causal analysis, clearer narratives of model reasoning, and principled CoT interventions. 

**Abstract (ZH)**: 通过重采样研究推理模型的因果影响和底层计算 

---
# VCORE: Variance-Controlled Optimization-based Reweighting for Chain-of-Thought Supervision 

**Title (ZH)**: VCORE: 基于链式思维监督的方差控制优化加权方法 

**Authors**: Xuan Gong, Senmiao Wang, Hanbo Huang, Ruoyu Sun, Shiyu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27462)  

**Abstract**: Supervised fine-tuning (SFT) on long chain-of-thought (CoT) trajectories has emerged as a crucial technique for enhancing the reasoning abilities of large language models (LLMs). However, the standard cross-entropy loss treats all tokens equally, ignoring their heterogeneous contributions across a reasoning trajectory. This uniform treatment leads to misallocated supervision and weak generalization, especially in complex, long-form reasoning tasks. To address this, we introduce \textbf{V}ariance-\textbf{C}ontrolled \textbf{O}ptimization-based \textbf{RE}weighting (VCORE), a principled framework that reformulates CoT supervision as a constrained optimization problem. By adopting an optimization-theoretic perspective, VCORE enables a principled and adaptive allocation of supervision across tokens, thereby aligning the training objective more closely with the goal of robust reasoning generalization. Empirical evaluations demonstrate that VCORE consistently outperforms existing token reweighting methods. Across both in-domain and out-of-domain settings, VCORE achieves substantial performance gains on mathematical and coding benchmarks, using models from the Qwen3 series (4B, 8B, 32B) and LLaMA-3.1-8B-Instruct. Moreover, we show that VCORE serves as a more effective initialization for subsequent reinforcement learning, establishing a stronger foundation for advancing the reasoning capabilities of LLMs. The Code will be released at this https URL. 

**Abstract (ZH)**: 基于方差控制优化的Chain-of-Thought监督重权方法 

---
# CoMViT: An Efficient Vision Backbone for Supervised Classification in Medical Imaging 

**Title (ZH)**: CoMViT：一种用于医学影像监督分类的高效视觉骨干网络 

**Authors**: Aon Safdar, Mohamed Saadeldin  

**Link**: [PDF](https://arxiv.org/pdf/2510.27442)  

**Abstract**: Vision Transformers (ViTs) have demonstrated strong potential in medical imaging; however, their high computational demands and tendency to overfit on small datasets limit their applicability in real-world clinical scenarios. In this paper, we present CoMViT, a compact and generalizable Vision Transformer architecture optimized for resource-constrained medical image analysis. CoMViT integrates a convolutional tokenizer, diagonal masking, dynamic temperature scaling, and pooling-based sequence aggregation to improve performance and generalization. Through systematic architectural optimization, CoMViT achieves robust performance across twelve MedMNIST datasets while maintaining a lightweight design with only ~4.5M parameters. It matches or outperforms deeper CNN and ViT variants, offering up to 5-20x parameter reduction without sacrificing accuracy. Qualitative Grad-CAM analyses show that CoMViT consistently attends to clinically relevant regions despite its compact size. These results highlight the potential of principled ViT redesign for developing efficient and interpretable models in low-resource medical imaging settings. 

**Abstract (ZH)**: Vision Transformers (ViTs)在医疗成像领域展现出了强大的潜力；然而，它们的高计算需求和对小数据集的过拟合倾向限制了其在实际临床场景中的应用。本文提出了一种名为CoMViT的紧凑且通用的Vision Transformer架构，该架构旨在针对资源受限的医疗图像分析进行优化。CoMViT结合了卷积分词器、对角掩码、动态温度缩放以及基于池化的序列聚合，以提高性能和泛化能力。通过系统性的架构优化，CoMViT在十二个MedMNIST数据集上实现了稳健的性能，同时保持了轻量级的设计，仅包含约450万参数。CoMViT在参数量上较深层的卷积神经网络（CNN）和ViT变体最多可减少5-20倍，而不牺牲准确性。定性的Grad-CAM分析结果显示，尽管尺寸紧凑，CoMViT仍一致地关注临床相关的区域。这些结果突显了在资源受限的医疗成像环境中设计原理明确的ViT的重要性，以开发有效的可解释模型。 

---
# Mitigating Semantic Collapse in Partially Relevant Video Retrieval 

**Title (ZH)**: 缓解部分相关视频检索中的语义坍塌 

**Authors**: WonJun Moon, MinSeok Jung, Gilhan Park, Tae-Young Kim, Cheol-Ho Cho, Woojin Jun, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27432)  

**Abstract**: Partially Relevant Video Retrieval (PRVR) seeks videos where only part of the content matches a text query. Existing methods treat every annotated text-video pair as a positive and all others as negatives, ignoring the rich semantic variation both within a single video and across different videos. Consequently, embeddings of both queries and their corresponding video-clip segments for distinct events within the same video collapse together, while embeddings of semantically similar queries and segments from different videos are driven apart. This limits retrieval performance when videos contain multiple, diverse events. This paper addresses the aforementioned problems, termed as semantic collapse, in both the text and video embedding spaces. We first introduce Text Correlation Preservation Learning, which preserves the semantic relationships encoded by the foundation model across text queries. To address collapse in video embeddings, we propose Cross-Branch Video Alignment (CBVA), a contrastive alignment method that disentangles hierarchical video representations across temporal scales. Subsequently, we introduce order-preserving token merging and adaptive CBVA to enhance alignment by producing video segments that are internally coherent yet mutually distinctive. Extensive experiments on PRVR benchmarks demonstrate that our framework effectively prevents semantic collapse and substantially improves retrieval accuracy. 

**Abstract (ZH)**: 部分相关视频检索中的语义坍塌及解决方法 

---
# Learning Soft Robotic Dynamics with Active Exploration 

**Title (ZH)**: 基于主动探索学习软体机器人力学模型 

**Authors**: Hehui Zheng, Bhavya Sukhija, Chenhao Li, Klemens Iten, Andreas Krause, Robert K. Katzschmann  

**Link**: [PDF](https://arxiv.org/pdf/2510.27428)  

**Abstract**: Soft robots offer unmatched adaptability and safety in unstructured environments, yet their compliant, high-dimensional, and nonlinear dynamics make modeling for control notoriously difficult. Existing data-driven approaches often fail to generalize, constrained by narrowly focused task demonstrations or inefficient random exploration. We introduce SoftAE, an uncertainty-aware active exploration framework that autonomously learns task-agnostic and generalizable dynamics models of soft robotic systems. SoftAE employs probabilistic ensemble models to estimate epistemic uncertainty and actively guides exploration toward underrepresented regions of the state-action space, achieving efficient coverage of diverse behaviors without task-specific supervision. We evaluate SoftAE on three simulated soft robotic platforms -- a continuum arm, an articulated fish in fluid, and a musculoskeletal leg with hybrid actuation -- and on a pneumatically actuated continuum soft arm in the real world. Compared with random exploration and task-specific model-based reinforcement learning, SoftAE produces more accurate dynamics models, enables superior zero-shot control on unseen tasks, and maintains robustness under sensing noise, actuation delays, and nonlinear material effects. These results demonstrate that uncertainty-driven active exploration can yield scalable, reusable dynamics models across diverse soft robotic morphologies, representing a step toward more autonomous, adaptable, and data-efficient control in compliant robots. 

**Abstract (ZH)**: 软体机器人在非结构化环境中的不可替代的适应性和安全性使其动力学模型难以建模，但现有的数据驱动方法往往由于窄范围的任务演示或低效的随机探索而无法泛化。我们提出了一种不确定性感知的主动探索框架SoftAE，该框架能够自主学习软体机器人系统的任务无关且可泛化的动力学模型。SoftAE 使用概率集成模型来估计认知不确定性，并主动引导探索未充分代表的状态-动作空间区域，从而实现对多种行为的有效覆盖，无需特定任务监督。我们在三种模拟软体机器人平台——连续臂、流体中的铰接鱼和具有混合驱动的肌腱骨骼腿——以及一个气动驱动的连续软臂的现实世界环境中评估了SoftAE。相较于随机探索和特定任务的数据驱动强化学习，SoftAE 生成了更为准确的动力学模型，能够实现更好的零样本控制，并在感知噪声、驱动延迟和非线性材料效应下保持鲁棒性。这些结果表明，以不确定性驱动的主动探索可产生适用于多种软体机器人形态的可扩展且可重用的动力学模型，代表着朝向更自主、更适应、更数据高效的软体柔顺机器人控制迈出的一步。 

---
# Who Does Your Algorithm Fail? Investigating Age and Ethnic Bias in the MAMA-MIA Dataset 

**Title (ZH)**: 你的算法对哪些人群失效？MAMA-MIA数据集中的年龄和种族偏见探究 

**Authors**: Aditya Parikh, Sneha Das, Aasa Feragen  

**Link**: [PDF](https://arxiv.org/pdf/2510.27421)  

**Abstract**: Deep learning models aim to improve diagnostic workflows, but fairness evaluation remains underexplored beyond classification, e.g., in image segmentation. Unaddressed segmentation bias can lead to disparities in the quality of care for certain populations, potentially compounded across clinical decision points and amplified through iterative model development. Here, we audit the fairness of the automated segmentation labels provided in the breast cancer tumor segmentation dataset MAMA-MIA. We evaluate automated segmentation quality across age, ethnicity, and data source. Our analysis reveals an intrinsic age-related bias against younger patients that continues to persist even after controlling for confounding factors, such as data source. We hypothesize that this bias may be linked to physiological factors, a known challenge for both radiologists and automated systems. Finally, we show how aggregating data from multiple data sources influences site-specific ethnic biases, underscoring the necessity of investigating data at a granular level. 

**Abstract (ZH)**: 深度学习模型旨在改进诊断流程，但公平性评估在分割等任务中仍被忽视。未解决的分割偏差可能导致某些人群的医疗服务质量差距，这种差距可能在临床决策点上加剧，并在迭代模型发展中放大。在这里，我们审查了MAMA-MIA乳腺癌肿瘤分割数据集中自动分割标签的公平性。我们评估了不同年龄、种族和地区来源的数据集中的自动分割质量。分析结果显示，即使在控制混杂因素（如数据来源）后，仍然存在与年龄相关的内在偏差，对年轻患者不利。我们推测这种偏差可能与生理因素有关，这是放射科医生和自动系统面临的已知挑战之一。最后，我们展示了从多个数据源聚合数据如何影响特定站点的种族偏差，突显了在细粒度层面上调查数据的必要性。 

---
# Atlas-Alignment: Making Interpretability Transferable Across Language Models 

**Title (ZH)**: Atlas-Alignment: 让可解释性在语言模型之间可传递 

**Authors**: Bruno Puri, Jim Berend, Sebastian Lapuschkin, Wojciech Samek  

**Link**: [PDF](https://arxiv.org/pdf/2510.27413)  

**Abstract**: Interpretability is crucial for building safe, reliable, and controllable language models, yet existing interpretability pipelines remain costly and difficult to scale. Interpreting a new model typically requires costly training of model-specific sparse autoencoders, manual or semi-automated labeling of SAE components, and their subsequent validation. We introduce Atlas-Alignment, a framework for transferring interpretability across language models by aligning unknown latent spaces to a Concept Atlas - a labeled, human-interpretable latent space - using only shared inputs and lightweight representational alignment techniques. Once aligned, this enables two key capabilities in previously opaque models: (1) semantic feature search and retrieval, and (2) steering generation along human-interpretable atlas concepts. Through quantitative and qualitative evaluations, we show that simple representational alignment methods enable robust semantic retrieval and steerable generation without the need for labeled concept data. Atlas-Alignment thus amortizes the cost of explainable AI and mechanistic interpretability: by investing in one high-quality Concept Atlas, we can make many new models transparent and controllable at minimal marginal cost. 

**Abstract (ZH)**: Atlas-Alignment: 跨语言模型的知识迁移以实现解释性和可操控性 

---
# FedMuon: Accelerating Federated Learning with Matrix Orthogonalization 

**Title (ZH)**: FedMuon: 用矩阵正交化加速 federated learning 

**Authors**: Junkang Liu, Fanhua Shang, Junchao Zhou, Hongying Liu, Yuanyuan Liu, Jin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27403)  

**Abstract**: The core bottleneck of Federated Learning (FL) lies in the communication rounds. That is, how to achieve more effective local updates is crucial for reducing communication rounds. Existing FL methods still primarily use element-wise local optimizers (Adam/SGD), neglecting the geometric structure of the weight matrices. This often leads to the amplification of pathological directions in the weights during local updates, leading deterioration in the condition number and slow convergence. Therefore, we introduce the Muon optimizer in local, which has matrix orthogonalization to optimize matrix-structured parameters. Experimental results show that, in IID setting, Local Muon significantly accelerates the convergence of FL and reduces communication rounds compared to Local SGD and Local AdamW. However, in non-IID setting, independent matrix orthogonalization based on the local distributions of each client induces strong client drift. Applying Muon in non-IID FL poses significant challenges: (1) client preconditioner leading to client drift; (2) moment reinitialization. To address these challenges, we propose a novel Federated Muon optimizer (FedMuon), which incorporates two key techniques: (1) momentum aggregation, where clients use the aggregated momentum for local initialization; (2) local-global alignment, where the local gradients are aligned with the global update direction to significantly reduce client drift. Theoretically, we prove that \texttt{FedMuon} achieves a linear speedup convergence rate without the heterogeneity assumption, where $S$ is the number of participating clients per round, $K$ is the number of local iterations, and $R$ is the total number of communication rounds. Empirically, we validate the effectiveness of FedMuon on language and vision models. Compared to several baselines, FedMuon significantly reduces communication rounds and improves test accuracy. 

**Abstract (ZH)**: 联邦学习中联邦学习的核心瓶颈在于通信轮次。具体来说，如何实现更有效的局部更新对于减少通信轮次至关重要。现有的联邦学习方法仍然主要使用元素-wise的局部优化器（如Adam/SGD），忽略了权重矩阵的几何结构。这往往导致在局部更新中权重的病态方向被放大，从而恶化条件数并导致收敛速度变慢。因此，我们引入了在局部使用的Muon优化器，该优化器对矩阵结构的参数进行矩阵正交化优化。实验结果表明，在IID设置下，局部Muon显著加速了联邦学习的收敛速度并减少了通信轮次，相较于Local SGD和Local AdamW。然而，在非IID设置下，基于每个客户端局部分布的独立矩阵正交化会导致强烈的客户端漂移。将Muon应用于非IID联邦学习提出了重大挑战：（1）客户端预条件化导致客户端漂移；（2）动量再初始化。为了解决这些挑战，我们提出了一个新颖的联邦Muon优化器（FedMuon），结合了两种关键技术：（1）动量聚合，客户端使用聚合后的动量进行局部初始化；（2）局部-全局对齐，将局部梯度与全局更新方向对齐，从而显著减少客户端漂移。理论上，我们证明在没有异质性假设的情况下，FedMuon实现了线性加速的收敛速率，其中$S$是每轮参与的客户端数目，$K$是局部迭代次数，$R$是总的通信轮次。实验上，我们验证了FedMuon在语言和视觉模型上的有效性。相比于几种基线方法，FedMuon显著减少了通信轮次并提高了测试精度。 

---
# Balancing Knowledge Updates: Toward Unified Modular Editing in LLMs 

**Title (ZH)**: 平衡知识更新：向着LLMs中统一模块化编辑的方向 

**Authors**: Jiahao Liu, Zijian Wang, Kuo Zhao, Dong Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27400)  

**Abstract**: Knowledge editing has emerged as an efficient approach for updating factual knowledge in large language models (LLMs). It typically locates knowledge storage modules and then modifies their parameters. However, most existing methods focus on the weights of multilayer perceptron (MLP) modules, which are often identified as the main repositories of factual information. Other components, such as attention (Attn) modules, are often ignored during editing. This imbalance can leave residual outdated knowledge and limit editing effectiveness. We perform comprehensive knowledge localization experiments on advanced LLMs and find that Attn modules play a substantial role in factual knowledge storage and retrieval, especially in earlier layers. Based on these insights, we propose IntAttn-Edit, a method that extends the associative memory paradigm to jointly update both MLP and Attn modules. Our approach uses a knowledge balancing strategy that allocates update magnitudes in proportion to each module's measured contribution to knowledge storage. Experiments on standard benchmarks show that IntAttn-Edit achieves higher edit success, better generalization, and stronger knowledge preservation than prior methods. Further analysis shows that the balancing strategy keeps editing performance within an optimal range across diverse settings. 

**Abstract (ZH)**: 知识编辑已成为大型语言模型（LLMs）更新事实性知识的一种有效方法。它通常定位知识存储模块并随后修改其参数。然而，现有大多数方法集中于多层感知器（MLP）模块的权重，这些模块常被视为事实信息的主要仓库。其他组件，如注意（Attn）模块，在编辑过程中经常被忽略。这种不平衡可能会留下残余的过时知识，限制编辑效果。我们对高级LLMs进行了一系列全面的知识定位实验，发现注意（Attn）模块在事实性知识存储与检索中发挥着重要作用，尤其是在早期层中。基于这些见解，我们提出了一种名为IntAttn-Edit的方法，该方法扩展了关联记忆范式，以联合更新MLP和Attn模块。我们的方法采用了一种知识平衡策略，根据每个模块对知识存储的实际贡献分配更新幅度。标准基准上的实验表明，IntAttn-Edit在编辑成功率、泛化能力和知识保持方面都优于先前方法。进一步的分析表明，平衡策略在多种场景下保持了编辑性能在最佳范围内。 

---
# Spiking Neural Networks: The Future of Brain-Inspired Computing 

**Title (ZH)**: 脉冲神经网络：脑启发计算的未来 

**Authors**: Sales G. Aribe Jr  

**Link**: [PDF](https://arxiv.org/pdf/2510.27379)  

**Abstract**: Spiking Neural Networks (SNNs) represent the latest generation of neural computation, offering a brain-inspired alternative to conventional Artificial Neural Networks (ANNs). Unlike ANNs, which depend on continuous-valued signals, SNNs operate using distinct spike events, making them inherently more energy-efficient and temporally dynamic. This study presents a comprehensive analysis of SNN design models, training algorithms, and multi-dimensional performance metrics, including accuracy, energy consumption, latency, spike count, and convergence behavior. Key neuron models such as the Leaky Integrate-and-Fire (LIF) and training strategies, including surrogate gradient descent, ANN-to-SNN conversion, and Spike-Timing Dependent Plasticity (STDP), are examined in depth. Results show that surrogate gradient-trained SNNs closely approximate ANN accuracy (within 1-2%), with faster convergence by the 20th epoch and latency as low as 10 milliseconds. Converted SNNs also achieve competitive performance but require higher spike counts and longer simulation windows. STDP-based SNNs, though slower to converge, exhibit the lowest spike counts and energy consumption (as low as 5 millijoules per inference), making them optimal for unsupervised and low-power tasks. These findings reinforce the suitability of SNNs for energy-constrained, latency-sensitive, and adaptive applications such as robotics, neuromorphic vision, and edge AI systems. While promising, challenges persist in hardware standardization and scalable training. This study concludes that SNNs, with further refinement, are poised to propel the next phase of neuromorphic computing. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）代表了新一代神经计算，提供了与传统人工神经网络（ANNs）相媲美的脑启发替代方案。与依赖连续信号的ANNs不同，SNNs通过独特的脉冲事件进行操作，使其更加固能并具有时间动态性。本研究全面分析了SNN设计模型、训练算法以及包括准确率、能耗、延迟、脉冲计数和收敛行为在内的多维性能指标。深入探讨了诸如泄漏积分并触发（LIF）等关键神经元模型以及替代梯度下降、ANN转SNN转换和突触定时依赖可塑性（STDP）等训练策略。结果显示，经过替代梯度训练的SNNs在准确率上与ANNs相差1-2%，在第20个epoch后收敛速度加快，延迟低至10毫秒。转换得到的SNNs也表现出竞争力，但需要更高的脉冲计数和更长的模拟窗口。基于STDP的SNNs虽然收敛速度较慢，但脉冲计数和能量消耗最低（低至5毫焦耳每推理），使其适用于无监督和低功耗任务。这些发现强调了SNNs在受能源限制、延迟敏感和自适应应用，如机器人技术、类神经视觉和边缘AI系统中的适用性。尽管充满前景，但ハード件标准化和可扩展训练仍面临挑战。本研究结论认为，通过进一步完善，SNNs有可能推动神经形态计算的下一次飞跃。 

---
# Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity 

**Title (ZH)**: 通过忠实度和啰嗦度衡量思维链可监控性 

**Authors**: Austin Meek, Eitan Sprejer, Iván Arcuschin, Austin J. Brockmeier, Steven Basart  

**Link**: [PDF](https://arxiv.org/pdf/2510.27378)  

**Abstract**: Chain-of-thought (CoT) outputs let us read a model's step-by-step reasoning. Since any long, serial reasoning process must pass through this textual trace, the quality of the CoT is a direct window into what the model is thinking. This visibility could help us spot unsafe or misaligned behavior (monitorability), but only if the CoT is transparent about its internal reasoning (faithfulness). Fully measuring faithfulness is difficult, so researchers often focus on examining the CoT in cases where the model changes its answer after adding a cue to the input. This proxy finds some instances of unfaithfulness but loses information when the model maintains its answer, and does not investigate aspects of reasoning not tied to the cue. We extend these results to a more holistic sense of monitorability by introducing verbosity: whether the CoT lists every factor needed to solve the task. We combine faithfulness and verbosity into a single monitorability score that shows how well the CoT serves as the model's external `working memory', a property that many safety schemes based on CoT monitoring depend on. We evaluate instruction-tuned and reasoning models on BBH, GPQA, and MMLU. Our results show that models can appear faithful yet remain hard to monitor when they leave out key factors, and that monitorability differs sharply across model families. We release our evaluation code using the Inspect library to support reproducible future work. 

**Abstract (ZH)**: 链式思考（CoT）输出使我们能够阅读模型的逐步推理过程。由于任何长序列的推理过程都必须通过这一文本痕迹，CoT的质量直接反映了模型的思考过程。这种透明性有助于我们识别不安全或对齐错误的行为（可观测性），但前提是CoT必须忠实反映其内部推理过程。完全衡量忠实性是困难的，因此研究人员通常专注于检查模型在输入中加入提示后改变答案的情况。这种代理方法可以发现一些不忠实的情况，但在模型保持其答案时会丢失信息，并不调查与提示无关的推理方面。我们通过引入冗余性（即CoT是否列出了完成任务所需的所有因素）将这些结果扩展到更全面的可观测性概念。我们将忠实性和冗余性结合成一个单一的可观测性评分，以显示CoT作为模型的外部“工作记忆”表现如何，这是许多基于CoT监控的安全方案所依赖的特性。我们对指令调整和推理模型在BBH、GPQA和MMLU上的表现进行了评估。我们的结果显示，当模型忽略关键因素时，模型可以显得忠实但难以监控，且不同模型家族的可观测性存在明显差异。我们使用Inspect库发布我们的评估代码，以支持未来工作的可重复性。 

---
# Fine-Tuning Open Video Generators for Cinematic Scene Synthesis: A Small-Data Pipeline with LoRA and Wan2.1 I2V 

**Title (ZH)**: 基于LoRA和Wan2.1 I2V的小数据量微调管线优化开放视频生成器以实现电影级场景合成 

**Authors**: Meftun Akarsu, Kerem Catay, Sedat Bin Vedat, Enes Kutay Yarkan, Ilke Senturk, Arda Sar, Dafne Eksioglu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27364)  

**Abstract**: We present a practical pipeline for fine-tuning open-source video diffusion transformers to synthesize cinematic scenes for television and film production from small datasets. The proposed two-stage process decouples visual style learning from motion generation. In the first stage, Low-Rank Adaptation (LoRA) modules are integrated into the cross-attention layers of the Wan2.1 I2V-14B model to adapt its visual representations using a compact dataset of short clips from Ay Yapim's historical television film El Turco. This enables efficient domain transfer within hours on a single GPU. In the second stage, the fine-tuned model produces stylistically consistent keyframes that preserve costume, lighting, and color grading, which are then temporally expanded into coherent 720p sequences through the model's video decoder. We further apply lightweight parallelization and sequence partitioning strategies to accelerate inference without quality degradation. Quantitative and qualitative evaluations using FVD, CLIP-SIM, and LPIPS metrics, supported by a small expert user study, demonstrate measurable improvements in cinematic fidelity and temporal stability over the base model. The complete training and inference pipeline is released to support reproducibility and adaptation across cinematic domains. 

**Abstract (ZH)**: 我们提出了一种实用的流水线，用于从小型数据集 fine-tune 开源视频扩散变换器，以合成电视和电影制作中的电影级场景。提出的两阶段过程将视觉风格学习与运动生成解耦。在第一阶段，将低秩适应（LoRA）模块整合到 Wan2.1 I2V-14B 模型的交叉注意力层中，使用来自 Ay Yapim 历史电视电影《埃尔图克》的短片段紧凑数据集来适应其视觉表示。这使得在单个 GPU 上几小时内高效地实现领域迁移成为可能。在第二阶段，fine-tuned 模型生成风格一致的关键帧，保留服装、照明和色调，然后通过模型的视频解码器按时间扩展为连续的 720p 序列。我们进一步应用轻量级并行化和序列分割策略以在不降质的情况下加速推理。使用 FVD、CLIP-SIM 和 LPIPS 度量，并辅以小型专家用户研究进行定量和定性评估，证明了与基模型相比在电影保真度和时间稳定性方面的可测量改进。完整的训练和推理流水线被发布以支持重现性和跨电影领域调整。 

---
# Generative Semantic Coding for Ultra-Low Bitrate Visual Communication and Analysis 

**Title (ZH)**: 超低比特率视觉通信与分析的生成式语义编码 

**Authors**: Weiming Chen, Yijia Wang, Zhihan Zhu, Zhihai He  

**Link**: [PDF](https://arxiv.org/pdf/2510.27324)  

**Abstract**: We consider the problem of ultra-low bit rate visual communication for remote vision analysis, human interactions and control in challenging scenarios with very low communication bandwidth, such as deep space exploration, battlefield intelligence, and robot navigation in complex environments. In this paper, we ask the following important question: can we accurately reconstruct the visual scene using only a very small portion of the bit rate in existing coding methods while not sacrificing the accuracy of vision analysis and performance of human interactions? Existing text-to-image generation models offer a new approach for ultra-low bitrate image description. However, they can only achieve a semantic-level approximation of the visual scene, which is far insufficient for the purpose of visual communication and remote vision analysis and human interactions. To address this important issue, we propose to seamlessly integrate image generation with deep image compression, using joint text and coding latent to guide the rectified flow models for precise generation of the visual scene. The semantic text description and coding latent are both encoded and transmitted to the decoder at a very small bit rate. Experimental results demonstrate that our method can achieve the same image reconstruction quality and vision analysis accuracy as existing methods while using much less bandwidth. The code will be released upon paper acceptance. 

**Abstract (ZH)**: 超低比特率视觉通信中挑战场景下视觉场景的准确重构：将图像生成与深度图像压缩无缝集成以实现精确的视觉场景生成 

---
# CASR-Net: An Image Processing-focused Deep Learning-based Coronary Artery Segmentation and Refinement Network for X-ray Coronary Angiogram 

**Title (ZH)**: CASR-Net：一种基于深度学习的心脏冠状动脉成像聚焦分割与精修网络 

**Authors**: Alvee Hassan, Rusab Sarmun, Muhammad E. H. Chowdhury, M. Murugappan, Md. Sakib Abrar Hossain, Sakib Mahmud, Abdulrahman Alqahtani, Sohaib Bassam Zoghoul, Amith Khandakar, Susu M. Zughaier, Somaya Al-Maadeed, Anwarul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2510.27315)  

**Abstract**: Early detection of coronary artery disease (CAD) is critical for reducing mortality and improving patient treatment planning. While angiographic image analysis from X-rays is a common and cost-effective method for identifying cardiac abnormalities, including stenotic coronary arteries, poor image quality can significantly impede clinical diagnosis. We present the Coronary Artery Segmentation and Refinement Network (CASR-Net), a three-stage pipeline comprising image preprocessing, segmentation, and refinement. A novel multichannel preprocessing strategy combining CLAHE and an improved Ben Graham method provides incremental gains, increasing Dice Score Coefficient (DSC) by 0.31-0.89% and Intersection over Union (IoU) by 0.40-1.16% compared with using the techniques individually. The core innovation is a segmentation network built on a UNet with a DenseNet121 encoder and a Self-organized Operational Neural Network (Self-ONN) based decoder, which preserves the continuity of narrow and stenotic vessel branches. A final contour refinement module further suppresses false positives. Evaluated with 5-fold cross-validation on a combination of two public datasets that contain both healthy and stenotic arteries, CASR-Net outperformed several state-of-the-art models, achieving an IoU of 61.43%, a DSC of 76.10%, and clDice of 79.36%. These results highlight a robust approach to automated coronary artery segmentation, offering a valuable tool to support clinicians in diagnosis and treatment planning. 

**Abstract (ZH)**: 早期冠状动脉疾病(CAD)的检测对于减少 mortality 和改善患者治疗规划至关重要。虽然从 X 射线图像中进行冠状动脉造影图像分析是一种常见且经济有效的方法，用于识别心脏异常，包括狭窄的冠状动脉，但图像质量差会显著阻碍临床诊断。我们提出了冠状动脉分割和细化网络（CASR-Net），这是一个由图像预处理、分割和细化三个阶段组成的管道。一种新颖的多通道预处理策略结合了 CLAHE 和改进的 Ben Graham 方法，提供了增量收益，与单独使用这些技术相比，Dice 斐比特系数 (DSC) 提高了 0.31-0.89%，交并比 (IoU) 提高了 0.40-1.16%。核心创新在于一个基于 DenseNet121 编码器和 Self-组织操作神经网络 (Self-ONN) 基解码器的分割网络，该网络保留了狭窄和狭窄血管分支的连续性。最终的轮廓细化模块进一步抑制了假阳性。在两个公共数据集的组合上进行 5 折交叉验证，CASR-Net 在 IoU、DSC 和 clDice 上分别达到了 61.43%、76.10% 和 79.36%，展示了自动冠状动脉分割的稳健方法，为临床诊断和治疗规划提供了有价值的工具。 

---
# Un-Attributability: Computing Novelty From Retrieval & Semantic Similarity 

**Title (ZH)**: 不可归因性：从检索与语义相似性计算新颖性 

**Authors**: Philipp Davydov, Ameya Prabhu, Matthias Bethge, Elisa Nguyen, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.27313)  

**Abstract**: Understanding how language-model outputs relate to the pretraining corpus is central to studying model behavior. Most training data attribution (TDA) methods ask which training examples causally influence a given output, often using leave-one-out tests. We invert the question: which outputs cannot be attributed to any pretraining example? We introduce un-attributability as an operational measure of semantic novelty: an output is novel if the pretraining corpus contains no semantically similar context. We approximate this with a simple two-stage retrieval pipeline: index the corpus with lightweight GIST embeddings, retrieve the top-n candidates, then rerank with ColBERTv2. If the nearest corpus item is less attributable than a human-generated text reference, we consider the output of the model as novel. We evaluate on SmolLM and SmolLM2 and report three findings: (1) models draw on pretraining data across much longer spans than previously reported; (2) some domains systematically promote or suppress novelty; and (3) instruction tuning not only alters style but also increases novelty. Reframing novelty assessment around un-attributability enables efficient analysis at pretraining scale. We release ~20 TB of corpus chunks and index artifacts to support replication and large-scale extension of our analysis at this https URL 

**Abstract (ZH)**: 理解语言模型输出与预训练语料的关系是研究模型行为的关键。大多数训练数据归属（TDA）方法询问哪些训练示例因果影响给定的输出，常常使用删一法测试。我们反转了这个问题：哪些输出不能归属到任何预训练示例？我们引入不可归属性作为语义新颖性的操作性度量：如果预训练语料中不存在语义相似的上下文，则输出被认为是新颖的。我们通过一个简单的两阶段检索管道进行近似：使用轻量级GIST嵌入标注语料，检索前n个候选项，然后使用ColBERTv2重新排序。如果最近的语料库项比人类生成的文本参考更难以归属，我们则认为模型的输出是新颖的。我们在SmolLM和SmolLM2上进行了评估，并记录了三项发现：（1）模型跨更长的跨度利用预训练数据；（2）某些领域系统地促进或抑制新颖性；（3）指令调优不仅改变风格，还增加了新颖性。将新颖性评估重新构想为不可归属性为特征，可以高效地在预训练规模下进行分析。我们发布约20 TB的语料片段和索引制品，以支持复制和大规模扩展我们的分析，详见这里。 

---
# Can LLMs Help You at Work? A Sandbox for Evaluating LLM Agents in Enterprise Environments 

**Title (ZH)**: LLM能帮助企业工作者吗？一种评估企业环境中LLM代理的实验平台 

**Authors**: Harsh Vishwakarma, Ankush Agarwal, Ojas Patil, Chaitanya Devaguptapu, Mahesh Chandran  

**Link**: [PDF](https://arxiv.org/pdf/2510.27287)  

**Abstract**: Enterprise systems are crucial for enhancing productivity and decision-making among employees and customers. Integrating LLM based systems into enterprise systems enables intelligent automation, personalized experiences, and efficient information retrieval, driving operational efficiency and strategic growth. However, developing and evaluating such systems is challenging due to the inherent complexity of enterprise environments, where data is fragmented across multiple sources and governed by sophisticated access controls. We present EnterpriseBench, a comprehensive benchmark that simulates enterprise settings, featuring 500 diverse tasks across software engineering, HR, finance, and administrative domains. Our benchmark uniquely captures key enterprise characteristics including data source fragmentation, access control hierarchies, and cross-functional workflows. Additionally, we provide a novel data generation pipeline that creates internally consistent enterprise tasks from organizational metadata. Experiments with state-of-the-art LLM agents demonstrate that even the most capable models achieve only 41.8% task completion, highlighting significant opportunities for improvement in enterprise-focused AI systems. 

**Abstract (ZH)**: 企业系统对于增强员工和客户的产品ivity和决策至关重要。将基于LLM的系统集成到企业系统中能够实现智能化自动化、个性化体验和高效信息检索，推动运营效率和战略增长。然而，由于企业环境固有的复杂性，数据分散在多个来源并受到复杂的访问控制管理，因此开发和评估此类系统颇具挑战。我们提出EnterpriseBench，一个全面的基准测试，模拟企业环境，涵盖500项跨软件工程、人力资源、财务和行政领域的多样化任务。我们的基准测试独特地捕捉了关键的企业特性，包括数据源碎片化、访问控制层次结构和跨职能工作流程。此外，我们提供一种新颖的数据生成管道，从组织元数据中生成内部一致的企业任务。最新的LLM代理实验表明，最优秀的模型也只能完成41.8%的任务，这突显了企业导向的AI系统改进的巨大机会。 

---
# HiF-DTA: Hierarchical Feature Learning Network for Drug-Target Affinity Prediction 

**Title (ZH)**: HiF-DTA：药物-靶标亲和力预测的分层特征学习网络 

**Authors**: Minghui Li, Yuanhang Wang, Peijin Guo, Wei Wan, Shengshan Hu, Shengqing Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27281)  

**Abstract**: Accurate prediction of Drug-Target Affinity (DTA) is crucial for reducing experimental costs and accelerating early screening in computational drug discovery. While sequence-based deep learning methods avoid reliance on costly 3D structures, they still overlook simultaneous modeling of global sequence semantic features and local topological structural features within drugs and proteins, and represent drugs as flat sequences without atomic-level, substructural-level, and molecular-level multi-scale features. We propose HiF-DTA, a hierarchical network that adopts a dual-pathway strategy to extract both global sequence semantic and local topological features from drug and protein sequences, and models drugs multi-scale to learn atomic, substructural, and molecular representations fused via a multi-scale bilinear attention module. Experiments on Davis, KIBA, and Metz datasets show HiF-DTA outperforms state-of-the-art baselines, with ablations confirming the importance of global-local extraction and multi-scale fusion. 

**Abstract (ZH)**: 准确预测药物-靶标亲和力（DTA）对于降低实验成本并加速计算药物发现中的早期筛选至关重要。虽然基于序列的深度学习方法避免了依赖昂贵的三维结构，但仍忽视了同时建模药物和蛋白质中全局序列语义特征和局部拓扑结构特征，以及将药物表示为扁平序列而未考虑原子级、亚结构级和分子级的多尺度特征。我们提出了一种分级网络HiF-DTA，采用双路径策略从药物和蛋白质序列中提取全局序列语义和局部拓扑特征，并多尺度建模药物以通过多尺度双线性注意力模块学习原子、亚结构和分子表示。在Davis、KIBA和Metz数据集上的实验表明，HiF-DTA优于现有基线方法，消融实验进一步证实了全局-局部提取和多尺度融合的重要性。 

---
# FOCUS: Efficient Keyframe Selection for Long Video Understanding 

**Title (ZH)**: FOCUS: 长视频理解中的高效关键帧选择 

**Authors**: Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2510.27280)  

**Abstract**: Multimodal large language models (MLLMs) represent images and video frames as visual tokens. Scaling from single images to hour-long videos, however, inflates the token budget far beyond practical limits. Popular pipelines therefore either uniformly subsample or apply keyframe selection with retrieval-style scoring using smaller vision-language models. However, these keyframe selection methods still rely on pre-filtering before selection to reduce the inference cost and can miss the most informative moments.
We propose FOCUS, Frame-Optimistic Confidence Upper-bound Selection, a training-free, model-agnostic keyframe selection module that selects query-relevant frames under a strict token budget. FOCUS formulates keyframe selection as a combinatorial pure-exploration (CPE) problem in multi-armed bandits: it treats short temporal clips as arms, and uses empirical means and Bernstein confidence radius to identify informative regions while preserving exploration of uncertain areas. The resulting two-stage exploration-exploitation procedure reduces from a sequential policy with theoretical guarantees, first identifying high-value temporal regions, then selecting top-scoring frames within each region On two long-video question-answering benchmarks, FOCUS delivers substantial accuracy improvements while processing less than 2% of video frames. For videos longer than 20 minutes, it achieves an 11.9% gain in accuracy on LongVideoBench, demonstrating its effectiveness as a keyframe selection method and providing a simple and general solution for scalable long-video understanding with MLLMs. 

**Abstract (ZH)**: 基于帧优化置信上界选择的多模态大型语言模型关键帧选择模块 

---
# Why Do Multilingual Reasoning Gaps Emerge in Reasoning Language Models? 

**Title (ZH)**: 为什么多语言推理差距会在推理语言模型中出现？ 

**Authors**: Deokhyung Kang, Seonjeong Hwang, Daehui Kim, Hyounghun Kim, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27269)  

**Abstract**: Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, yet they still suffer from a multilingual reasoning gap, performing better in high-resource languages than in low-resource ones. While recent efforts have reduced this gap, its underlying causes remain largely unexplored. In this paper, we address this by showing that the multilingual reasoning gap largely stems from failures in language understanding-the model's inability to represent the multilingual input meaning into the dominant language (i.e., English) within its reasoning trace. This motivates us to examine whether understanding failures can be detected, as this ability could help mitigate the multilingual reasoning gap. To this end, we evaluate a range of detection methods and find that understanding failures can indeed be identified, with supervised approaches performing best. Building on this, we propose Selective Translation, a simple yet effective strategy that translates the multilingual input into English only when an understanding failure is detected. Experimental results show that Selective Translation bridges the multilingual reasoning gap, achieving near full-translation performance while using translation for only about 20% of inputs. Together, our work demonstrates that understanding failures are the primary cause of the multilingual reasoning gap and can be detected and selectively mitigated, providing key insight into its origin and a promising path toward more equitable multilingual reasoning. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 多语言推理差距主要源于语言理解失败：检测与缓解策略 

---
# MedCalc-Eval and MedCalc-Env: Advancing Medical Calculation Capabilities of Large Language Models 

**Title (ZH)**: MedCalc-Eval和MedCalc-Env: 提升大型语言模型的医疗计算能力 

**Authors**: Kangkun Mao, Jinru Ding, Jiayuan Chen, Mouxiao Bian, Ruiyao Chen, Xinwei Peng, Sijie Ren, Linyang Li, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27267)  

**Abstract**: As large language models (LLMs) enter the medical domain, most benchmarks evaluate them on question answering or descriptive reasoning, overlooking quantitative reasoning critical to clinical decision-making. Existing datasets like MedCalc-Bench cover few calculation tasks and fail to reflect real-world computational scenarios.
We introduce MedCalc-Eval, the largest benchmark for assessing LLMs' medical calculation abilities, comprising 700+ tasks across two types: equation-based (e.g., Cockcroft-Gault, BMI, BSA) and rule-based scoring systems (e.g., Apgar, Glasgow Coma Scale). These tasks span diverse specialties including internal medicine, surgery, pediatrics, and cardiology, offering a broader and more challenging evaluation setting.
To improve performance, we further develop MedCalc-Env, a reinforcement learning environment built on the InternBootcamp framework, enabling multi-step clinical reasoning and planning. Fine-tuning a Qwen2.5-32B model within this environment achieves state-of-the-art results on MedCalc-Eval, with notable gains in numerical sensitivity, formula selection, and reasoning robustness. Remaining challenges include unit conversion, multi-condition logic, and contextual understanding.
Code and datasets are available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型进入医疗领域，大多数基准测试主要评估其在问答或描述性推理方面的性能，忽视了临床决策中至关重要的定量推理。现有的数据集如MedCalc-Bench覆盖的计算任务较少，未能反映真实的计算场景。
我们介绍了MedCalc-Eval，这是评估LLMs医疗计算能力的最大规模基准，包含700多个任务，分为基于方程的任务（如Cockcroft-Gault、BMI、BSA）和基于规则的评分系统任务（如Apgar、Glasgow昏迷量表）。这些任务涵盖了内科、外科、儿科和心脏病学等多个专科，提供了更为广泛和更具挑战性的评估环境。
为了提高性能，我们进一步开发了MedCalc-Env，该环境基于InternBootcamp框架构建，支持多步临床推理和计划。在该环境中 fine-tuning Qwen2.5-32B 模型达到了MedCalc-Eval上的最佳性能，在数值敏感性、公式选择和推理稳健性方面取得了显著提升。仍存在的挑战包括单位转换、多条件逻辑和上下文理解。
代码和数据集可在以下链接获取。 

---
# Higher-order Linear Attention 

**Title (ZH)**: 高阶线性注意力 

**Authors**: Yifan Zhang, Zhen Qin, Quanquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27258)  

**Abstract**: The quadratic cost of scaled dot-product attention is a central obstacle to scaling autoregressive language models to long contexts. Linear-time attention and State Space Models (SSMs) provide scalable alternatives but are typically restricted to first-order or kernel-based approximations, which can limit expressivity. We introduce Higher-order Linear Attention (HLA), a causal, streaming mechanism that realizes higher interactions via compact prefix sufficient statistics. In the second-order case, HLA maintains a constant-size state and computes per-token outputs in linear time without materializing any $n \times n$ matrices. We give closed-form streaming identities, a strictly causal masked variant using two additional summaries, and a chunk-parallel training scheme based on associative scans that reproduces the activations of a serial recurrence exactly. We further outline extensions to third and higher orders. Collectively, these results position HLA as a principled, scalable building block that combines attention-like, data-dependent mixing with the efficiency of modern recurrent architectures. Project Page: this https URL. 

**Abstract (ZH)**: 缩放点积注意力的二次成本是将自回归语言模型扩展到长上下文的主要障碍。高阶线性注意力（HLA）是一种因果 Streaming 机制，通过紧凑的前缀充分统计量实现更高阶交互。在二阶情况下，HLA 维持恒定大小的状态并在线性时间内计算每词输出，无需构造任何 \(n \times n\) 矩阵。我们给出了闭合形式的 Streaming 标识，一个严格因果的屏蔽变体使用两个额外的摘要，以及基于关联扫描的分块并行训练方案，该方案能够准确再现串行递归的激活。我们还概述了高阶（三阶及以上）的扩展。这些结果将HLA定位为一个兼具注意力样式的数据依赖混合与现代递归架构效率的有原则且可扩展的基本构建块。项目页面：这个 https URL。 

---
# Languages are Modalities: Cross-Lingual Alignment via Encoder Injection 

**Title (ZH)**: 语言是模态性：通过编码器插入选项实现跨语言对齐 

**Authors**: Rajan Agarwal, Aarush Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.27254)  

**Abstract**: Instruction-tuned Large Language Models (LLMs) underperform on low resource, non-Latin scripts due to tokenizer fragmentation and weak cross-lingual coupling. We present LLINK (Latent Language Injection for Non-English Knowledge), a compute efficient language-as-modality method that conditions an instruction-tuned decoder without changing the tokenizer or retraining the decoder. First, we align sentence embeddings from a frozen multilingual encoder to the decoder's latent embedding space at a reserved position via a lightweight contrastive projector. Second, the vector is expanded into K soft slots and trained with minimal adapters so the frozen decoder consumes the signal. LLINK substantially improves bilingual retrieval and achieves 81.3% preference over the base model and 63.6% over direct fine-tuning in LLM-judged Q&A evaluations. We further find that improvements can be attributed to reduced tokenization inflation and a stronger cross lingual alignment, despite the model having residual weaknesses in numeric fidelity. Treating low resource languages as a modality offers a practical path to stronger cross-lingual alignment in lightweight LLMs. 

**Abstract (ZH)**: 非拉丁小字稿语料稀缺条件下指令调优大型语言模型性能欠佳，主要由于分片的分词器和薄弱的跨语言耦合。我们提出LLINK（潜在语言注入非英文知识），这是一种计算高效的语言作为模态的方法，能够在不更改分词器或重新训练解码器的情况下，条件化指令调优的解码器。首先，通过一个轻量级对比投影器将冻结的多语言编码器的句子嵌入对齐到解码器的潜在嵌入空间中的一个预留位置。其次，该向量被扩展为K个软槽并使用最小适配器进行训练，使冻结的解码器消费信号。LLINK显著提高了双语检索性能，在LLM评判的问答评估中，基模型的偏好度提高至81.3%，直接调优提高63.6%。进一步分析表明，性能提升归因于减少了分词膨胀和更强的跨语言对齐，尽管该模型在数值保真度上依然存在残余缺陷。将语料稀缺的语言视作一种模态为轻量级大型语言模型中的更强跨语言对齐提供了一条实用途径。 

---
# Not All Instances Are Equally Valuable: Towards Influence-Weighted Dataset Distillation 

**Title (ZH)**: 并非所有实例的价值都相同：面向影响加权数据集蒸馏 

**Authors**: Qiyan Deng, Changqian Zheng, Lianpeng Qiao, Yuping Wang, Chengliang Chai, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2510.27253)  

**Abstract**: Dataset distillation condenses large datasets into synthetic subsets, achieving performance comparable to training on the full dataset while substantially reducing storage and computation costs. Most existing dataset distillation methods assume that all real instances contribute equally to the process. In practice, real-world datasets contain both informative and redundant or even harmful instances, and directly distilling the full dataset without considering data quality can degrade model performance. In this work, we present Influence-Weighted Distillation IWD, a principled framework that leverages influence functions to explicitly account for data quality in the distillation process. IWD assigns adaptive weights to each instance based on its estimated impact on the distillation objective, prioritizing beneficial data while downweighting less useful or harmful ones. Owing to its modular design, IWD can be seamlessly integrated into diverse dataset distillation frameworks. Our empirical results suggest that integrating IWD tends to improve the quality of distilled datasets and enhance model performance, with accuracy gains of up to 7.8%. 

**Abstract (ZH)**: 数据集蒸馏将大规模数据集凝练为合成子集，在显著减少存储和计算成本的同时，实现与使用完整数据集训练相当的性能。现有大多数数据集蒸馏方法假设所有真实实例对过程的贡献均等。实际上，真实世界数据集包含既有信息性又有冗余或甚至有害的实例，如果不考虑数据质量直接蒸馏完整数据集可能会降低模型性能。在本文中，我们提出了影响加权蒸馏（IWD），这是一种原理性的框架，利用影响函数明确考虑数据质量。IWD 根据每个实例对蒸馏目标的估计影响赋予自适应权重，优先选择有益的数据，同时降低无用或有害数据的权重。由于其模块化设计，IWD 可以无缝集成到各种数据集蒸馏框架中。我们的实验证据表明，集成 IWD 通常可以提高蒸馏数据集的质量并增强模型性能，准确率提升高达 7.8%。 

---
# Reconstructing Unseen Sentences from Speech-related Biosignals for Open-vocabulary Neural Communication 

**Title (ZH)**: 从语音相关生物信号重构未见句子的开放词汇神经通信 

**Authors**: Deok-Seon Kim, Seo-Hyun Lee, Kang Yin, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27247)  

**Abstract**: Brain-to-speech (BTS) systems represent a groundbreaking approach to human communication by enabling the direct transformation of neural activity into linguistic expressions. While recent non-invasive BTS studies have largely focused on decoding predefined words or sentences, achieving open-vocabulary neural communication comparable to natural human interaction requires decoding unconstrained speech. Additionally, effectively integrating diverse signals derived from speech is crucial for developing personalized and adaptive neural communication and rehabilitation solutions for patients. This study investigates the potential of speech synthesis for previously unseen sentences across various speech modes by leveraging phoneme-level information extracted from high-density electroencephalography (EEG) signals, both independently and in conjunction with electromyography (EMG) signals. Furthermore, we examine the properties affecting phoneme decoding accuracy during sentence reconstruction and offer neurophysiological insights to further enhance EEG decoding for more effective neural communication solutions. Our findings underscore the feasibility of biosignal-based sentence-level speech synthesis for reconstructing unseen sentences, highlighting a significant step toward developing open-vocabulary neural communication systems adapted to diverse patient needs and conditions. Additionally, this study provides meaningful insights into the development of communication and rehabilitation solutions utilizing EEG-based decoding technologies. 

**Abstract (ZH)**: 基于脑电的无约束语音合成及其在神经交流系统中的应用研究 

---
# Beyond a Million Tokens: Benchmarking and Enhancing Long-Term Memory in LLMs 

**Title (ZH)**: 超越百万Tokens：大语言模型长期记忆的基准测试与增强 

**Authors**: Mohammad Tavakoli, Alireza Salemi, Carrie Ye, Mohamed Abdalla, Hamed Zamani, J Ross Mitchell  

**Link**: [PDF](https://arxiv.org/pdf/2510.27246)  

**Abstract**: Evaluating the abilities of large language models (LLMs) for tasks that require long-term memory and thus long-context reasoning, for example in conversational settings, is hampered by the existing benchmarks, which often lack narrative coherence, cover narrow domains, and only test simple recall-oriented tasks. This paper introduces a comprehensive solution to these challenges. First, we present a novel framework for automatically generating long (up to 10M tokens), coherent, and topically diverse conversations, accompanied by probing questions targeting a wide range of memory abilities. From this, we construct BEAM, a new benchmark comprising 100 conversations and 2,000 validated questions. Second, to enhance model performance, we propose LIGHT-a framework inspired by human cognition that equips LLMs with three complementary memory systems: a long-term episodic memory, a short-term working memory, and a scratchpad for accumulating salient facts. Our experiments on BEAM reveal that even LLMs with 1M token context windows (with and without retrieval-augmentation) struggle as dialogues lengthen. In contrast, LIGHT consistently improves performance across various models, achieving an average improvement of 3.5%-12.69% over the strongest baselines, depending on the backbone LLM. An ablation study further confirms the contribution of each memory component. 

**Abstract (ZH)**: 评估大型语言模型在需要长期记忆和长上下文推理的任务中的能力，例如在对话情境中的能力受限于现有基准的不足，这些基准通常缺乏叙事连贯性、覆盖范围狭窄且仅测试简单的回忆任务。本文提出了一个综合解决方案来应对这些挑战。首先，我们提出了一种新的框架，用于自动生成长（多达10M词元）、连贯且主题多样的对话，并附带针对各种记忆能力的探测性问题。在此基础上，我们构建了BEAM这一包含100个对话和2000个验证问题的新基准。其次，为了提升模型性能，我们提出了LIGHT框架，该框架借鉴了人类认知，为大型语言模型配备了三种互补的记忆系统：长期 episodic 记忆、短期工作记忆和便笺区以积累重要事实。我们在BEAM上的实验表明，即使具备1M词元上下文窗口（带有和不带检索增强）的大型语言模型，在对话变长时也会遇到困难。相比之下，LIGHT在多种模型上持续提升了性能，平均改进幅度为3.5%-12.69%，这取决于所使用的底层大型语言模型。进一步的消融研究还证实了每种记忆模块的贡献。 

---
# Vintage Code, Modern Judges: Meta-Validation in Low Data Regimes 

**Title (ZH)**: 陈旧代码，现代法官：低数据情况下的元验证 

**Authors**: Ora Nova Fandina, Gal Amram, Eitan Farchi, Shmulik Froimovich, Raviv Gal, Wesam Ibraheem, Rami Katan, Alice Podolsky, Orna Raz  

**Link**: [PDF](https://arxiv.org/pdf/2510.27244)  

**Abstract**: Application modernization in legacy languages such as COBOL, PL/I, and REXX faces an acute shortage of resources, both in expert availability and in high-quality human evaluation data. While Large Language Models as a Judge (LaaJ) offer a scalable alternative to expert review, their reliability must be validated before being trusted in high-stakes workflows. Without principled validation, organizations risk a circular evaluation loop, where unverified LaaJs are used to assess model outputs, potentially reinforcing unreliable judgments and compromising downstream deployment decisions. Although various automated approaches to validating LaaJs have been proposed, alignment with human judgment remains a widely used and conceptually grounded validation strategy. In many real-world domains, the availability of human-labeled evaluation data is severely limited, making it difficult to assess how well a LaaJ aligns with human judgment. We introduce SparseAlign, a formal framework for assessing LaaJ alignment with sparse human-labeled data. SparseAlign combines a novel pairwise-confidence concept with a score-sensitive alignment metric that jointly capture ranking consistency and score proximity, enabling reliable evaluator selection even when traditional statistical methods are ineffective due to limited annotated examples. SparseAlign was applied internally to select LaaJs for COBOL code explanation. The top-aligned evaluators were integrated into assessment workflows, guiding model release decisions. We present a case study of four LaaJs to demonstrate SparseAlign's utility in real-world evaluation scenarios. 

**Abstract (ZH)**: 基于少量人工标注数据评估大型语言模型作为评判者的稀疏对齐框架 

---
# DRAMA: Unifying Data Retrieval and Analysis for Open-Domain Analytic Queries 

**Title (ZH)**: DRAMA: 统一开放域分析查询中的数据检索与分析 

**Authors**: Chuxuan Hu, Maxwell Yang, James Weiland, Yeji Lim, Suhas Palawala, Daniel Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27238)  

**Abstract**: Manually conducting real-world data analyses is labor-intensive and inefficient. Despite numerous attempts to automate data science workflows, none of the existing paradigms or systems fully demonstrate all three key capabilities required to support them effectively: (1) open-domain data collection, (2) structured data transformation, and (3) analytic reasoning.
To overcome these limitations, we propose DRAMA, an end-to-end paradigm that answers users' analytic queries in natural language on large-scale open-domain data. DRAMA unifies data collection, transformation, and analysis as a single pipeline. To quantitatively evaluate system performance on tasks representative of DRAMA, we construct a benchmark, DRAMA-Bench, consisting of two categories of tasks: claim verification and question answering, each comprising 100 instances. These tasks are derived from real-world applications that have gained significant public attention and require the retrieval and analysis of open-domain data. We develop DRAMA-Bot, a multi-agent system designed following DRAMA. It comprises a data retriever that collects and transforms data by coordinating the execution of sub-agents, and a data analyzer that performs structured reasoning over the retrieved data. We evaluate DRAMA-Bot on DRAMA-Bench together with five state-of-the-art baseline agents. DRAMA-Bot achieves 86.5% task accuracy at a cost of $0.05, outperforming all baselines with up to 6.9 times the accuracy and less than 1/6 of the cost. DRAMA is publicly available at this https URL. 

**Abstract (ZH)**: 手动进行现实世界数据的分析劳动密集且效率低下。尽管已经尝试通过自动化数据科学工作流来解决这一问题，但现有的范式或系统无法全面展示支持这些工作流所需的三项关键能力：(1) 开放领域数据收集，(2) 结构化数据转换，以及 (3) 分析推理。为了克服这些限制，我们提出DRAMA，一种端到端范式，能够使用自然语言在大规模开放领域数据上回答用户的分析查询。DRAMA将数据收集、转换和分析统一为一个单一的工作流。为了定量评估系统性能，我们构建了DRAMA-Bench基准测试，包含两类任务：声明验证和问答，每类各包含100个实例。这些任务来自具有显著公众关注的实际应用，需要检索和分析开放领域数据。我们开发了遵循DRAMA设计的多智能体系统DRAMA-Bot。它包括一个数据检索器，通过协调子智能体的执行来收集和转换数据，并且包括一个数据分析师，对检索到的数据进行结构化推理。我们使用DRAMA-Bench与五种最先进的基线智能体一起评估了DRAMA-Bot。DRAMA-Bot在任务准确率方面达到86.5%，成本为0.05，比所有基线的准确率高出多达6.9倍，且成本低于基线的1/6。DRAMA可以在以下链接公开访问：this https URL。DRAMA端到端范式在大规模开放领域数据上的自然语言分析查询 

---
# Soft Task-Aware Routing of Experts for Equivariant Representation Learning 

**Title (ZH)**: 专家软任务感知路由的不变表示学习 

**Authors**: Jaebyeong Jeon, Hyeonseo Jang, Jy-yong Sohn, Kibok Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.27222)  

**Abstract**: Equivariant representation learning aims to capture variations induced by input transformations in the representation space, whereas invariant representation learning encodes semantic information by disregarding such transformations. Recent studies have shown that jointly learning both types of representations is often beneficial for downstream tasks, typically by employing separate projection heads. However, this design overlooks information shared between invariant and equivariant learning, which leads to redundant feature learning and inefficient use of model capacity. To address this, we introduce Soft Task-Aware Routing (STAR), a routing strategy for projection heads that models them as experts. STAR induces the experts to specialize in capturing either shared or task-specific information, thereby reducing redundant feature learning. We validate this effect by observing lower canonical correlations between invariant and equivariant embeddings. Experimental results show consistent improvements across diverse transfer learning tasks. The code is available at this https URL. 

**Abstract (ZH)**: 变换成 invariant 和 equivariant 表征的学习方法旨在分别捕获输入变换在表示空间中引起的变化和不变的信息，而 recent 研究表明，同时学习这两种类型的表征通常对下游任务是有益的，通常通过使用分开的投影头来实现。但这种设计忽略了不变学习和变换成学习之间共享的信息，导致冗余特征学习和模型容量的低效利用。为了解决这一问题，我们引入了 Soft Task-Aware Routing (STAR) —— 一种将投影头建模为专家的路由策略，STAR 促使专家专门捕获共享或任务特定的信息，从而减少冗余特征学习。我们通过观察不变和变换成嵌入的统计相关性的降低验证了这一效果。实验结果表明，STAR 在多种迁移学习任务上一致性地提高了性能。代码可在该链接获取。 

---
# Privacy-Aware Continual Self-Supervised Learning on Multi-Window Chest Computed Tomography for Domain-Shift Robustness 

**Title (ZH)**: 面向隐私保护的多窗格胸部 computed tomography 持续自监督学习及其在领域迁移稳健性中的应用 

**Authors**: Ren Tasai, Guang Li, Ren Togo, Takahiro Ogawa, Kenji Hirata, Minghui Tang, Takaaki Yoshimura, Hiroyuki Sugimori, Noriko Nishioka, Yukie Shimizu, Kohsuke Kudo, Miki Haseyama  

**Link**: [PDF](https://arxiv.org/pdf/2510.27213)  

**Abstract**: We propose a novel continual self-supervised learning (CSSL) framework for simultaneously learning diverse features from multi-window-obtained chest computed tomography (CT) images and ensuring data privacy. Achieving a robust and highly generalizable model in medical image diagnosis is challenging, mainly because of issues, such as the scarcity of large-scale, accurately annotated datasets and domain shifts inherent to dynamic healthcare environments. Specifically, in chest CT, these domain shifts often arise from differences in window settings, which are optimized for distinct clinical purposes. Previous CSSL frameworks often mitigated domain shift by reusing past data, a typically impractical approach owing to privacy constraints. Our approach addresses these challenges by effectively capturing the relationship between previously learned knowledge and new information across different training stages through continual pretraining on unlabeled images. Specifically, by incorporating a latent replay-based mechanism into CSSL, our method mitigates catastrophic forgetting due to domain shifts during continual pretraining while ensuring data privacy. Additionally, we introduce a feature distillation technique that integrates Wasserstein distance-based knowledge distillation (WKD) and batch-knowledge ensemble (BKE), enhancing the ability of the model to learn meaningful, domain-shift-robust representations. Finally, we validate our approach using chest CT images obtained across two different window settings, demonstrating superior performance compared with other approaches. 

**Abstract (ZH)**: 我们提出了一种新颖的持续自监督学习（CSSL）框架，用于同时从多窗口获取的胸部计算机断层扫描（CT）图像中学习多种特征，并确保数据隐私。 

---
# Multi-Modal Feature Fusion for Spatial Morphology Analysis of Traditional Villages via Hierarchical Graph Neural Networks 

**Title (ZH)**: 基于层次图神经网络的多模态特征融合传统 villages 空间形态分析 

**Authors**: Jiaxin Zhang, Zehong Zhu, Junye Deng, Yunqin Li, and Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27208)  

**Abstract**: Villages areas hold significant importance in the study of human-land relationships. However, with the advancement of urbanization, the gradual disappearance of spatial characteristics and the homogenization of landscapes have emerged as prominent issues. Existing studies primarily adopt a single-disciplinary perspective to analyze villages spatial morphology and its influencing factors, relying heavily on qualitative analysis methods. These efforts are often constrained by the lack of digital infrastructure and insufficient data. To address the current research limitations, this paper proposes a Hierarchical Graph Neural Network (HGNN) model that integrates multi-source data to conduct an in-depth analysis of villages spatial morphology. The framework includes two types of nodes-input nodes and communication nodes-and two types of edges-static input edges and dynamic communication edges. By combining Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT), the proposed model efficiently integrates multimodal features under a two-stage feature update mechanism. Additionally, based on existing principles for classifying villages spatial morphology, the paper introduces a relational pooling mechanism and implements a joint training strategy across 17 subtypes. Experimental results demonstrate that this method achieves significant performance improvements over existing approaches in multimodal fusion and classification tasks. Additionally, the proposed joint optimization of all sub-types lifts mean accuracy/F1 from 0.71/0.83 (independent models) to 0.82/0.90, driven by a 6% gain for parcel tasks. Our method provides scientific evidence for exploring villages spatial patterns and generative logic. 

**Abstract (ZH)**: 农村地区在人类-土地关系研究中具有重要意义。然而，随着城市化进程的推进，空间特征的逐渐消失和景观的同质化成为了突出问题。现有研究主要从单一学科视角分析村庄的空间形态及其影响因素，依赖于定性分析方法。这些努力往往受限于缺乏数字基础设施和数据不足的问题。为了解决现有研究的局限性，本文提出了一种层次图神经网络（Hierarchical Graph Neural Network, HGNN）模型，该模型整合多源数据以深入分析村庄的空间形态。该框架包括输入节点和通信节点两种类型，以及静态输入边和动态通信边两种类型。通过结合图卷积网络（Graph Convolutional Networks, GCN）和图注意网络（Graph Attention Networks, GAT），所提出的模型在两阶段特征更新机制下高效地整合了多模态特征。此外，基于现有的村庄空间形态分类原则，本文引入了关系聚池化机制，并通过综合培训策略实施了17种亚型间的联合训练。实验结果表明，该方法在多模态融合和分类任务中相比现有方法取得了显著性能提升。此外，对所有亚型的联合优化进一步将平均准确度/F1分数从独立模型的0.71/0.83提升至0.82/0.90， parcel任务提高了6%。该方法为探索村庄空间模式和生成逻辑提供了科学依据。 

---
# Feature-Function Curvature Analysis: A Geometric Framework for Explaining Differentiable Models 

**Title (ZH)**: 特征-函数曲率分析：可微模型解释的几何框架 

**Authors**: Hamed Najafi, Dongsheng Luo, Jason Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27207)  

**Abstract**: Explainable AI (XAI) is critical for building trust in complex machine learning models, yet mainstream attribution methods often provide an incomplete, static picture of a model's final state. By collapsing a feature's role into a single score, they are confounded by non-linearity and interactions. To address this, we introduce Feature-Function Curvature Analysis (FFCA), a novel framework that analyzes the geometry of a model's learned function. FFCA produces a 4-dimensional signature for each feature, quantifying its: (1) Impact, (2) Volatility, (3) Non-linearity, and (4) Interaction. Crucially, we extend this framework into Dynamic Archetype Analysis, which tracks the evolution of these signatures throughout the training process. This temporal view moves beyond explaining what a model learned to revealing how it learns. We provide the first direct, empirical evidence of hierarchical learning, showing that models consistently learn simple linear effects before complex interactions. Furthermore, this dynamic analysis provides novel, practical diagnostics for identifying insufficient model capacity and predicting the onset of overfitting. Our comprehensive experiments demonstrate that FFCA, through its static and dynamic components, provides the essential geometric context that transforms model explanation from simple quantification to a nuanced, trustworthy analysis of the entire learning process. 

**Abstract (ZH)**: 可解释人工智能（XAI）对于建立对复杂机器学习模型的信任至关重要，然而主流的归因方法往往提供的是模型最终状态的一种不完整、静态的视角。通过将特征的作用归结为单一评分，它们会受到非线性和交互作用的影响。为了解决这一问题，我们引入了特征函数曲率分析（FFCA）这一新颖框架，该框架分析了模型学习函数的几何结构。FFCA为每个特征生成了一个4维签名，量化了其：（1）影响，（2）波动性，（3）非线性，以及（4）交互作用。关键的是，我们进一步将这一框架扩展到动态典型模式分析，该分析追踪这些签名在整个训练过程中的演变。这种时间维度的视角不仅解释了模型学到了什么，还揭示了它是如何学习的。提供了首个直接的实证证据，展示了分层次学习的现象，表明模型在学习复杂交互作用之前一直学习简单的线性效应。此外，这一动态分析还提供了新的实用诊断方法，用于识别模型容量不足并预测过度拟合的出现。我们的全面实验表明，FFCA通过其静态和动态部分，提供了模型解释所需的必要几何上下文，将模型解释从简单的量化转变为对整个学习过程的细微、可靠的分析。 

---
# MemeArena: Automating Context-Aware Unbiased Evaluation of Harmfulness Understanding for Multimodal Large Language Models 

**Title (ZH)**: MemeArena：自动化多模态大型语言模型危害性理解的上下文感知公平评估 

**Authors**: Zixin Chen, Hongzhan Lin, Kaixin Li, Ziyang Luo, Yayue Deng, Jing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.27196)  

**Abstract**: The proliferation of memes on social media necessitates the capabilities of multimodal Large Language Models (mLLMs) to effectively understand multimodal harmfulness. Existing evaluation approaches predominantly focus on mLLMs' detection accuracy for binary classification tasks, which often fail to reflect the in-depth interpretive nuance of harmfulness across diverse contexts. In this paper, we propose MemeArena, an agent-based arena-style evaluation framework that provides a context-aware and unbiased assessment for mLLMs' understanding of multimodal harmfulness. Specifically, MemeArena simulates diverse interpretive contexts to formulate evaluation tasks that elicit perspective-specific analyses from mLLMs. By integrating varied viewpoints and reaching consensus among evaluators, it enables fair and unbiased comparisons of mLLMs' abilities to interpret multimodal harmfulness. Extensive experiments demonstrate that our framework effectively reduces the evaluation biases of judge agents, with judgment results closely aligning with human preferences, offering valuable insights into reliable and comprehensive mLLM evaluations in multimodal harmfulness understanding. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 社交媒体上 meme 的泛滥 necessitates 多模态大型语言模型 (mLLMs) 能够有效理解多模态有害性。现有的评估方法主要集中在 mLLMs 对二元分类任务的检测准确性上，这往往无法反映不同情境下有害性的深入诠释细微差别。在本文中，我们提出了一种基于代理的竞技场式评估框架 MemeArena，提供一种具有情境意识和无偏见的评估方法，用于评估 mLLMs 对多模态有害性的理解。具体而言，MemeArena 通过模拟不同的诠释情境来制定评估任务，促使 mLLMs 进行视角特定的分析。通过对多种观点进行整合并达成评价者的共识，它使 mLLMs 解释多模态有害性的能力的公平和无偏见比较成为可能。大量实验表明，我们的框架有效地减少了评价者偏见，评价结果与人类偏好高度一致，为多模态有害性理解中的可靠和全面的 mLLM 评估提供了有价值的见解。我们的代码和数据可在该网址公开访问：this https URL。 

---
# Vectorized Online POMDP Planning 

**Title (ZH)**: 向量化在线POMDP规划 

**Authors**: Marcus Hoerger, Muhammad Sudrajat, Hanna Kurniawati  

**Link**: [PDF](https://arxiv.org/pdf/2510.27191)  

**Abstract**: Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization of today's hardware, but parallelizing POMDP solvers has been challenging. They rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can quickly offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation that analytically solves part of the optimization component, leaving only the estimation of expectations for numerical computation. VOPP represents all data structures related to planning as a collection of tensors and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel solver with no dependencies and synchronization bottlenecks between parallel computations. Experimental results indicate that VOPP is at least 20X more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver. 

**Abstract (ZH)**: 部分可观测性的规划是自主机器人的一项基本能力。部分可观测马尔可夫决策过程（POMDP）为部分可观测性问题提供了一个强大的框架，捕捉行动的随机效应以及通过噪声观测获得的有限信息。POMDP的求解可以从当今硬件的并行化中受益匪浅，但并行化POMDP求解器颇具挑战性。它们依赖于行动的数值优化和其价值的估计交织进行，这在并行过程中产生了依赖性和同步瓶颈，这些瓶颈会迅速抵消并行化的益处。在本文中，我们提出了一种新颖的并行在线求解器——向量在线POMDP规划器（VOPP），该求解器利用了最近的POMDP形式化方法，该方法对优化组件的一部分进行了分析解算，仅留期望估计用于数值计算。VOPP将所有与规划相关的数据结构表示为张量集合，并实现所有规划步骤为对该表示的完全向量化计算。结果是一种无依赖性和同步瓶颈的并行求解器。实验结果表明，与现有最先进的并行在线求解器相比，VOPP在计算近最优解时至少快20倍。 

---
# Unvalidated Trust: Cross-Stage Vulnerabilities in Large Language Model Architectures 

**Title (ZH)**: 未经验证的信任：大型语言模型架构中的跨阶段漏洞 

**Authors**: Dominik Schwarz  

**Link**: [PDF](https://arxiv.org/pdf/2510.27190)  

**Abstract**: As Large Language Models (LLMs) are increasingly integrated into automated, multi-stage pipelines, risk patterns that arise from unvalidated trust between processing stages become a practical concern. This paper presents a mechanism-centered taxonomy of 41 recurring risk patterns in commercial LLMs. The analysis shows that inputs are often interpreted non-neutrally and can trigger implementation-shaped responses or unintended state changes even without explicit commands. We argue that these behaviors constitute architectural failure modes and that string-level filtering alone is insufficient. To mitigate such cross-stage vulnerabilities, we recommend zero-trust architectural principles, including provenance enforcement, context sealing, and plan revalidation, and we introduce "Countermind" as a conceptual blueprint for implementing these defenses. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益嵌入自动化多阶段管道中，由此产生的未验证信任所引发的风险模式成为一个实际关切。本文提出了一种以机制为中心的分类法，概述了41种在商用LLMs中反复出现的风险模式。分析表明，输入通常被非中立地解释，并可能在没有明确命令的情况下触发实现导向的响应或无意状态变化。我们认为这些行为构成架构性故障模式，而仅依赖于字符串级过滤是不足的。为减轻此类跨阶段漏洞，我们建议采用零信任架构原则，包括溯源强制、上下文密封和计划再验证，并引入“Countermind”作为实施这些防御的概念蓝图。 

---
# Sparse Model Inversion: Efficient Inversion of Vision Transformers for Data-Free Applications 

**Title (ZH)**: 稀疏模型反转：视觉变换器的高效无数据应用反演 

**Authors**: Zixuan Hu, Yongxian Wei, Li Shen, Zhenyi Wang, Lei Li, Chun Yuan, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.27186)  

**Abstract**: Model inversion, which aims to reconstruct the original training data from pre-trained discriminative models, is especially useful when the original training data is unavailable due to privacy, usage rights, or size constraints. However, existing dense inversion methods attempt to reconstruct the entire image area, making them extremely inefficient when inverting high-resolution images from large-scale Vision Transformers (ViTs). We further identify two underlying causes of this inefficiency: the redundant inversion of noisy backgrounds and the unintended inversion of spurious correlations--a phenomenon we term "hallucination" in model inversion. To address these limitations, we propose a novel sparse model inversion strategy, as a plug-and-play extension to speed up existing dense inversion methods with no need for modifying their original loss functions. Specifically, we selectively invert semantic foregrounds while stopping the inversion of noisy backgrounds and potential spurious correlations. Through both theoretical and empirical studies, we validate the efficacy of our approach in achieving significant inversion acceleration (up to 3.79 faster) while maintaining comparable or even enhanced downstream performance in data-free model quantization and data-free knowledge transfer. Code is available at this https URL. 

**Abstract (ZH)**: 从预训练辨别模型重建原始训练数据的模型反演，特别是在原始训练数据因隐私、使用权限或大小限制等原因不可用时，尤其有用。然而，现有的密集反演方法试图重构整幅图像区域，使得它们在从大规模视觉变换器（ViTs）反演高分辨率图像时极其低效。我们进一步识别出这一低效性的两个根本原因：噪声背景的冗余反演和未预期的伪相关性的反演——我们将这种现象在模型反演中称为“幻觉”。为解决这些局限，我们提出了一种新的稀疏模型反演策略，作为即插即用扩展，无需修改原始损失函数即可加速现有的密集反演方法。具体来说，我们选择性地反演语义前景，而停止噪声背景和潜在伪相关性的反演。通过理论和实验研究，我们验证了该方法在实现显著反演加速（最高3.79倍）的同时，保持或甚至提升了数据免费模型量化和数据免费知识传递的下游性能。代码可在以下链接获取。 

---
# Dual-level Progressive Hardness-Aware Reweighting for Cross-View Geo-Localization 

**Title (ZH)**: 双层面渐进式难度感知加权用于跨视图地理定位 

**Authors**: Guozheng Zheng, Jian Guan, Mingjie Xie, Xuanjia Zhao, Congyi Fan, Shiheng Zhang, Pengming Feng  

**Link**: [PDF](https://arxiv.org/pdf/2510.27181)  

**Abstract**: Cross-view geo-localization (CVGL) between drone and satellite imagery remains challenging due to severe viewpoint gaps and the presence of hard negatives, which are visually similar but geographically mismatched samples. Existing mining or reweighting strategies often use static weighting, which is sensitive to distribution shifts and prone to overemphasizing difficult samples too early, leading to noisy gradients and unstable convergence. In this paper, we present a Dual-level Progressive Hardness-aware Reweighting (DPHR) strategy. At the sample level, a Ratio-based Difficulty-Aware (RDA) module evaluates relative difficulty and assigns fine-grained weights to negatives. At the batch level, a Progressive Adaptive Loss Weighting (PALW) mechanism exploits a training-progress signal to attenuate noisy gradients during early optimization and progressively enhance hard-negative mining as training matures. Experiments on the University-1652 and SUES-200 benchmarks demonstrate the effectiveness and robustness of the proposed DPHR, achieving consistent improvements over state-of-the-art methods. 

**Abstract (ZH)**: 无人机和卫星影像之间的跨视角地理定位（CVGL）由于严重的视角差距和难以处理的负样本（地理上不匹配但视觉上相似的样本）存在挑战。现有的挖掘或重权重策略通常使用静态权重，这种权重对分布偏移敏感，容易过早强调困难样本，导致梯度噪声和不稳定收敛。在本文中，我们提出了一种双层渐进难度感知重权重（DPHR）策略。在样本层面，一种基于比例的难度感知（RDA）模块评估相对难度并为负样本分配细粒度权重。在批次层面，一种渐进自适应损失权重（PALW）机制利用训练进度信号在早期优化过程中减弱噪声梯度，并随着训练成熟逐步增强难以处理的负样本挖掘。在University-1652和SUES-200基准上的实验展示了所提出的DPHR的有效性和鲁棒性，实现了对现有最佳方法的一致改进。 

---
# FMint-SDE: A Multimodal Foundation Model for Accelerating Numerical Simulation of SDEs via Error Correction 

**Title (ZH)**: FMint-SDE：一种通过误差校正加速SDE数值模拟的多模态基础模型 

**Authors**: Jiaxin Yuan, Haizhao Yang, Maria Cameron  

**Link**: [PDF](https://arxiv.org/pdf/2510.27173)  

**Abstract**: Fast and accurate simulation of dynamical systems is a fundamental challenge across scientific and engineering domains. Traditional numerical integrators often face a trade-off between accuracy and computational efficiency, while existing neural network-based approaches typically require training a separate model for each case. To overcome these limitations, we introduce a novel multi-modal foundation model for large-scale simulations of differential equations: FMint-SDE (Foundation Model based on Initialization for stochastic differential equations). Based on a decoder-only transformer with in-context learning, FMint-SDE leverages numerical and textual modalities to learn a universal error-correction scheme. It is trained using prompted sequences of coarse solutions generated by conventional solvers, enabling broad generalization across diverse systems. We evaluate our models on a suite of challenging SDE benchmarks spanning applications in molecular dynamics, mechanical systems, finance, and biology. Experimental results show that our approach achieves a superior accuracy-efficiency tradeoff compared to classical solvers, underscoring the potential of FMint-SDE as a general-purpose simulation tool for dynamical systems. 

**Abstract (ZH)**: 快速且准确地模拟动态系统是跨各个科学和工程领域的一项基本挑战。传统数值积分器往往在精度和计算效率之间存在权衡，而现有的基于神经网络的方法通常需要为每种情况训练一个单独的模型。为克服这些限制，我们提出了一种基于初始化的新型多模态基础模型，用于大规模微分方程模拟：FMint-SDE（基于初始化的面向随机微分方程的基础模型）。基于只有解码器的变压器并利用上下文学习，FMint-SDE 利用数值和文本模态学习一种通用的误差校正方案。它通过使用由传统求解器生成的粗糙解序列进行提示学习，从而在不同系统之间实现广泛的泛化能力。我们在涵盖分子动力学、机械系统、金融和生物学等多个应用领域的随机微分方程基准测试上评估了我们的模型。实验结果表明，与经典求解器相比，我们的方法在准确性和效率之间取得了更优的权衡，突显了FMint-SDE 作为动态系统通用模拟工具的潜力。 

---
# Adaptive Defense against Harmful Fine-Tuning for Large Language Models via Bayesian Data Scheduler 

**Title (ZH)**: 面向大型语言模型的贝叶斯数据调度自适应防御有害微调 

**Authors**: Zixuan Hu, Li Shen, Zhenyi Wang, Yongxian Wei, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.27172)  

**Abstract**: Harmful fine-tuning poses critical safety risks to fine-tuning-as-a-service for large language models. Existing defense strategies preemptively build robustness via attack simulation but suffer from fundamental limitations: (i) the infeasibility of extending attack simulations beyond bounded threat models due to the inherent difficulty of anticipating unknown attacks, and (ii) limited adaptability to varying attack settings, as simulation fails to capture their variability and complexity. To address these challenges, we propose Bayesian Data Scheduler (BDS), an adaptive tuning-stage defense strategy with no need for attack simulation. BDS formulates harmful fine-tuning defense as a Bayesian inference problem, learning the posterior distribution of each data point's safety attribute, conditioned on the fine-tuning and alignment datasets. The fine-tuning process is then constrained by weighting data with their safety attributes sampled from the posterior, thus mitigating the influence of harmful data. By leveraging the post hoc nature of Bayesian inference, the posterior is conditioned on the fine-tuning dataset, enabling BDS to tailor its defense to the specific dataset, thereby achieving adaptive defense. Furthermore, we introduce a neural scheduler based on amortized Bayesian learning, enabling efficient transfer to new data without retraining. Comprehensive results across diverse attack and defense settings demonstrate the state-of-the-art performance of our approach. Code is available at this https URL. 

**Abstract (ZH)**: 有害微调对大型语言模型微调即服务造成严重安全风险。现有防御策略通过攻击模拟预先构建鲁棒性，但受到根本性的限制：（i）由于难以预料未知攻击，扩展攻击模拟至超越限定威胁模型的不可行性；（ii）适应不同攻击环境的能力有限，因为模拟无法捕捉其多样性和复杂性。为应对这些挑战，我们提出了贝叶斯数据调度器（BDS），这是一种无需攻击模拟的自适应微调阶段防御策略。BDS 将有害微调防御形式化为贝叶斯推理问题，在微调和对齐数据集的条件下，学习每个数据点的安全属性后验分布。然后通过按权重对数据进行加权（其安全属性来自后验分布），约束微调过程，从而减轻有害数据的影响。通过利用贝叶斯推理的后验性质，后验分布可以条件于微调数据集，使BDS能够针对具体数据集进行定制，从而实现自适应防御。此外，我们引入了一种基于近似贝叶斯学习的神经调度器，能够在无需重新训练的情况下高效地转移到新数据。在多种攻击和防御设置下的综合结果显示了我们方法的先进性能。代码可在以下链接获得：this https URL。 

---
# H2-Cache: A Novel Hierarchical Dual-Stage Cache for High-Performance Acceleration of Generative Diffusion Models 

**Title (ZH)**: H2-Cache: 一种用于生成性扩散模型高性能加速的新型分层两级缓存结构 

**Authors**: Mingyu Sung, Il-Min Kim, Sangseok Yun, Jae-Mo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27171)  

**Abstract**: Diffusion models have emerged as state-of-the-art in image generation, but their practical deployment is hindered by the significant computational cost of their iterative denoising process. While existing caching techniques can accelerate inference, they often create a challenging trade-off between speed and fidelity, suffering from quality degradation and high computational overhead. To address these limitations, we introduce H2-Cache, a novel hierarchical caching mechanism designed for modern generative diffusion model architectures. Our method is founded on the key insight that the denoising process can be functionally separated into a structure-defining stage and a detail-refining stage. H2-cache leverages this by employing a dual-threshold system, using independent thresholds to selectively cache each stage. To ensure the efficiency of our dual-check approach, we introduce pooled feature summarization (PFS), a lightweight technique for robust and fast similarity estimation. Extensive experiments on the Flux architecture demonstrate that H2-cache achieves significant acceleration (up to 5.08x) while maintaining image quality nearly identical to the baseline, quantitatively and qualitatively outperforming existing caching methods. Our work presents a robust and practical solution that effectively resolves the speed-quality dilemma, significantly lowering the barrier for the real-world application of high-fidelity diffusion models. Source code is available at this https URL. 

**Abstract (ZH)**: H2-Cache：一种用于现代生成扩散模型架构的新型分层缓存机制 

---
# Generating Accurate and Detailed Captions for High-Resolution Images 

**Title (ZH)**: 生成准确详细的高分辨率图像描述词 

**Authors**: Hankyeol Lee, Gawon Seo, Kyounggyu Lee, Dogun Kim, Kyungwoo Song, Jiyoung Jung  

**Link**: [PDF](https://arxiv.org/pdf/2510.27164)  

**Abstract**: Vision-language models (VLMs) often struggle to generate accurate and detailed captions for high-resolution images since they are typically pre-trained on low-resolution inputs (e.g., 224x224 or 336x336 pixels). Downscaling high-resolution images to these dimensions may result in the loss of visual details and the omission of important objects. To address this limitation, we propose a novel pipeline that integrates vision-language models, large language models (LLMs), and object detection systems to enhance caption quality. Our proposed pipeline refines captions through a novel, multi-stage process. Given a high-resolution image, an initial caption is first generated using a VLM, and key objects in the image are then identified by an LLM. The LLM predicts additional objects likely to co-occur with the identified key objects, and these predictions are verified by object detection systems. Newly detected objects not mentioned in the initial caption undergo focused, region-specific captioning to ensure they are incorporated. This process enriches caption detail while reducing hallucinations by removing references to undetected objects. We evaluate the enhanced captions using pairwise comparison and quantitative scoring from large multimodal models, along with a benchmark for hallucination detection. Experiments on a curated dataset of high-resolution images demonstrate that our pipeline produces more detailed and reliable image captions while effectively minimizing hallucinations. 

**Abstract (ZH)**: Vision-language模型（VLMs）通常难以生成高分辨率图像的准确和详细描述，因为它们通常在低分辨率输入（如224x224或336x336像素）上进行预训练。将高分辨率图像缩小到这些尺寸可能会导致视觉细节损失以及重要对象的遗漏。为了解决这一限制，我们提出了一种新颖的流程，该流程通过集成视觉-语言模型、大型语言模型（LLMs）和物体检测系统来提高描述质量。我们提出的方法通过一个新颖的多阶段过程来细化描述。给定高分辨率图像，首先使用VLM生成初始描述，然后由LLM识别图像中的关键物体。LLM预测与识别的关键物体共生的可能性较大的其他物体，并由物体检测系统验证这些预测。未在初始描述中提及的新检测到的物体将进行聚焦于特定区域的描述，以确保它们被纳入。这一过程通过去除未检测到物体的引用来丰富描述细节并减少幻觉。我们使用两两比较和大型多模态模型的定量得分，以及幻觉检测基准来评估增强的描述。针对高分辨率图像精心策划的数据集的实验表明，我们的流程可以生成更详细和可靠的图像描述，同时有效减少幻觉。 

---
# MARIA: A Framework for Marginal Risk Assessment without Ground Truth in AI Systems 

**Title (ZH)**: MARIA：AI系统中无地面真实数据的边际风险评估框架 

**Authors**: Jieshan Chen, Suyu Ma, Qinghua Lu, Sung Une Lee, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27163)  

**Abstract**: Before deploying an AI system to replace an existing process, it must be compared with the incumbent to ensure improvement without added risk. Traditional evaluation relies on ground truth for both systems, but this is often unavailable due to delayed or unknowable outcomes, high costs, or incomplete data, especially for long-standing systems deemed safe by convention. The more practical solution is not to compute absolute risk but the difference between systems. We therefore propose a marginal risk assessment framework, that avoids dependence on ground truth or absolute risk. It emphasizes three kinds of relative evaluation methodology, including predictability, capability and interaction dominance. By shifting focus from absolute to relative evaluation, our approach equips software teams with actionable guidance: identifying where AI enhances outcomes, where it introduces new risks, and how to adopt such systems responsibly. 

**Abstract (ZH)**: 在部署AI系统替代现有流程之前，必须对比两者以确保无额外风险的改进。传统的评估依赖于两者的ground truth，但在许多情况下由于结果延迟或不可知、高昂的成本或数据不完整，尤其是对于被认为安全的长期系统，ground truth往往不可用。更实用的解决方案是从计算绝对风险转向评估系统之间的差异。因此，我们提出了一种边际风险评估框架，该框架避免了对ground truth或绝对风险的依赖。它强调三种相对评估方法，包括可预测性、能力和相互主导性。通过从绝对评估转向相对评估，我们的方法为软件团队提供了可操作的指导：识别AI如何提升结果、引入新风险的领域，以及如何负责任地采用此类系统。 

---
# Exploring Landscapes for Better Minima along Valleys 

**Title (ZH)**: 探索景观以在山谷中获得更好的极小值 

**Authors**: Tong Zhao, Jiacheng Li, Yuanchang Zhou, Guangming Tan, Weile Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.27153)  

**Abstract**: Finding lower and better-generalizing minima is crucial for deep learning. However, most existing optimizers stop searching the parameter space once they reach a local minimum. Given the complex geometric properties of the loss landscape, it is difficult to guarantee that such a point is the lowest or provides the best generalization. To address this, we propose an adaptor "E" for gradient-based optimizers. The adapted optimizer tends to continue exploring along landscape valleys (areas with low and nearly identical losses) in order to search for potentially better local minima even after reaching a local minimum. This approach increases the likelihood of finding a lower and flatter local minimum, which is often associated with better generalization. We also provide a proof of convergence for the adapted optimizers in both convex and non-convex scenarios for completeness. Finally, we demonstrate their effectiveness in an important but notoriously difficult training scenario, large-batch training, where Lamb is the benchmark optimizer. Our testing results show that the adapted Lamb, ALTO, increases the test accuracy (generalization) of the current state-of-the-art optimizer by an average of 2.5% across a variety of large-batch training tasks. This work potentially opens a new research direction in the design of optimization algorithms. 

**Abstract (ZH)**: 寻找更低且泛化性能更好的局部极小值对于深度学习至关重要。然而，现有的优化器在达到局部极小值后通常会停止搜索参数空间。鉴于损失景观具有复杂的几何特性，难以保证这样的点是全局最低点或提供最佳泛化性能。为此，我们提出了一个用于梯度基优化器的“E”适配器。适配后的优化器倾向于继续在景观谷地（低且几乎相同的损失区域）中探索，即使在达到局部极小值后也在寻找潜在更好的局部极小值。这种方法增加了找到更低且更平坦的局部极小值的可能性，这通常与更好的泛化性能相关。我们还为适配优化器提供了凸和非凸场景下的收敛性证明以保持完整性。最后，我们在一个关键但历来难以处理的训练场景——大规模训练——中展示了它们的有效性，其中Lamb是基准优化器。我们的测试结果表明，适配后的Lamb（ALTO）在各种大规模训练任务中平均提高了当前最佳优化器的测试准确率（泛化性能）2.5%。这项工作可能为优化算法的设计开辟一个新的研究方向。 

---
# ZEBRA: Towards Zero-Shot Cross-Subject Generalization for Universal Brain Visual Decoding 

**Title (ZH)**: ZEBRA: 向零样本跨受试者泛化的通用脑视觉解码 

**Authors**: Haonan Wang, Jingyu Lu, Hongrui Li, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.27128)  

**Abstract**: Recent advances in neural decoding have enabled the reconstruction of visual experiences from brain activity, positioning fMRI-to-image reconstruction as a promising bridge between neuroscience and computer vision. However, current methods predominantly rely on subject-specific models or require subject-specific fine-tuning, limiting their scalability and real-world applicability. In this work, we introduce ZEBRA, the first zero-shot brain visual decoding framework that eliminates the need for subject-specific adaptation. ZEBRA is built on the key insight that fMRI representations can be decomposed into subject-related and semantic-related components. By leveraging adversarial training, our method explicitly disentangles these components to isolate subject-invariant, semantic-specific representations. This disentanglement allows ZEBRA to generalize to unseen subjects without any additional fMRI data or retraining. Extensive experiments show that ZEBRA significantly outperforms zero-shot baselines and achieves performance comparable to fully finetuned models on several metrics. Our work represents a scalable and practical step toward universal neural decoding. Code and model weights are available at: this https URL. 

**Abstract (ZH)**: 近期神经解码的进展使得从大脑活动重建视觉体验成为可能，定位fMRI到图像重建为神经科学与计算机视觉之间的桥梁开辟了前景。然而，当前方法主要依赖于个体化的模型或需要进行个体化的微调，这限制了它们的可扩展性和实际应用性。在本文中，我们介绍了ZEBRA，这是一个首个零样本脑视觉解码框架，消除了个体化调整的需要。ZEBRA基于一个关键见解，即fMRI表示可以分解为个体相关的和语义相关的部分。通过利用对抗训练，我们的方法显式地分离这些部分，以隔离个体不变、语义特定的表示。这种分离允许ZEBRA在没有任何额外的fMRI数据或重新训练的情况下泛化到未见过的个体。广泛的经验表明，ZEBRA在多个指标上显著优于零样本基线，并且实现了与完全微调模型相当的性能。我们的工作代表了大规模和实用的通用神经解码步骤。代码和模型权重可在以下链接获取：this https URL。 

---
# AURA: A Reinforcement Learning Framework for AI-Driven Adaptive Conversational Surveys 

**Title (ZH)**: AURA：一种基于强化学习的AI驱动自适应对话调查框架 

**Authors**: Jinwen Tang, Yi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27126)  

**Abstract**: Conventional online surveys provide limited personalization, often resulting in low engagement and superficial responses. Although AI survey chatbots improve convenience, most are still reactive: they rely on fixed dialogue trees or static prompt templates and therefore cannot adapt within a session to fit individual users, which leads to generic follow-ups and weak response quality. We address these limitations with AURA (Adaptive Understanding through Reinforcement Learning for Assessment), a reinforcement learning framework for AI-driven adaptive conversational surveys. AURA quantifies response quality using a four-dimensional LSDE metric (Length, Self-disclosure, Emotion, and Specificity) and selects follow-up question types via an epsilon-greedy policy that updates the expected quality gain within each session. Initialized with priors extracted from 96 prior campus-climate conversations (467 total chatbot-user exchanges), the system balances exploration and exploitation across 10-15 dialogue exchanges, dynamically adapting to individual participants in real time. In controlled evaluations, AURA achieved a +0.12 mean gain in response quality and a statistically significant improvement over non-adaptive baselines (p=0.044, d=0.66), driven by a 63% reduction in specification prompts and a 10x increase in validation behavior. These results demonstrate that reinforcement learning can give survey chatbots improved adaptivity, transforming static questionnaires into interactive, self-improving assessment systems. 

**Abstract (ZH)**: 基于强化学习的自适应评估对话调研框架AURA：提高调研个性化与响应质量 

---
# Expressive Range Characterization of Open Text-to-Audio Models 

**Title (ZH)**: 开放文本到语音模型的表达范围characterization 

**Authors**: Jonathan Morse, Azadeh Naderi, Swen Gaudl, Mark Cartwright, Amy K. Hoover, Mark J. Nelson  

**Link**: [PDF](https://arxiv.org/pdf/2510.27102)  

**Abstract**: Text-to-audio models are a type of generative model that produces audio output in response to a given textual prompt. Although level generators and the properties of the functional content that they create (e.g., playability) dominate most discourse in procedurally generated content (PCG), games that emotionally resonate with players tend to weave together a range of creative and multimodal content (e.g., music, sounds, visuals, narrative tone), and multimodal models have begun seeing at least experimental use for this purpose. However, it remains unclear what exactly such models generate, and with what degree of variability and fidelity: audio is an extremely broad class of output for a generative system to target.
Within the PCG community, expressive range analysis (ERA) has been used as a quantitative way to characterize generators' output space, especially for level generators. This paper adapts ERA to text-to-audio models, making the analysis tractable by looking at the expressive range of outputs for specific, fixed prompts. Experiments are conducted by prompting the models with several standardized prompts derived from the Environmental Sound Classification (ESC-50) dataset. The resulting audio is analyzed along key acoustic dimensions (e.g., pitch, loudness, and timbre). More broadly, this paper offers a framework for ERA-based exploratory evaluation of generative audio models. 

**Abstract (ZH)**: 基于文本的音频模型是一类生成模型，能够根据给定的文本提示生成音频输出。尽管层级生成器及其生成的功能内容属性（如可玩性）主导了程序化生成内容（PCG）的大部分讨论，但能够与玩家产生情感共鸣的游戏往往会综合各种创意和多模态内容（如音乐、声音、视觉、叙述语气），并且多模态模型已经开始为此目的进行实验性使用。然而，尚不清楚此类模型生成的具体内容及其多变性和保真度：音频是生成系统需要为目标的极其广泛的一类输出。在PCG领域，表达范围分析（ERA）已被用作一种定量方法来表征生成器的输出空间，尤其是对于层级生成器。本文将ERA应用于基于文本的音频模型，通过分析特定固定提示的输出表达范围使分析变得可行。实验通过使用源自Environmental Sound Classification (ESC-50)数据集的多个标准化提示来提示模型，分析生成的音频在关键声学维度上的表现（如音高、响度和音色）。更广泛地说，本文提供了一种基于ERA的探索性评估生成音频模型的框架。 

---
# QiNN-QJ: A Quantum-inspired Neural Network with Quantum Jump for Multimodal Sentiment Analysis 

**Title (ZH)**: QiNN-QJ：一种基于量子跃迁的多模态情感分析量子启发神经网络 

**Authors**: Yiwei Chen, Kehuan Yan, Yu Pan, Daoyi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2510.27091)  

**Abstract**: Quantum theory provides non-classical principles, such as superposition and entanglement, that inspires promising paradigms in machine learning. However, most existing quantum-inspired fusion models rely solely on unitary or unitary-like transformations to generate quantum entanglement. While theoretically expressive, such approaches often suffer from training instability and limited generalizability. In this work, we propose a Quantum-inspired Neural Network with Quantum Jump (QiNN-QJ) for multimodal entanglement modelling. Each modality is firstly encoded as a quantum pure state, after which a differentiable module simulating the QJ operator transforms the separable product state into the entangled representation. By jointly learning Hamiltonian and Lindblad operators, QiNN-QJ generates controllable cross-modal entanglement among modalities with dissipative dynamics, where structured stochasticity and steady-state attractor properties serve to stabilize training and constrain entanglement shaping. The resulting entangled states are projected onto trainable measurement vectors to produce predictions. In addition to achieving superior performance over the state-of-the-art models on benchmark datasets, including CMU-MOSI, CMU-MOSEI, and CH-SIMS, QiNN-QJ facilitates enhanced post-hoc interpretability through von-Neumann entanglement entropy. This work establishes a principled framework for entangled multimodal fusion and paves the way for quantum-inspired approaches in modelling complex cross-modal correlations. 

**Abstract (ZH)**: 基于量子跃迁的量子启发式神经网络多模态纠缠建模（QiNN-QJ） 

---
# Adapting Large Language Models to Emerging Cybersecurity using Retrieval Augmented Generation 

**Title (ZH)**: 适应新兴网络安全的大型语言模型检索增强生成方法 

**Authors**: Arnabh Borah, Md Tanvirul Alam, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27080)  

**Abstract**: Security applications are increasingly relying on large language models (LLMs) for cyber threat detection; however, their opaque reasoning often limits trust, particularly in decisions that require domain-specific cybersecurity knowledge. Because security threats evolve rapidly, LLMs must not only recall historical incidents but also adapt to emerging vulnerabilities and attack patterns. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in general LLM applications, but its potential for cybersecurity remains underexplored. In this work, we introduce a RAG-based framework designed to contextualize cybersecurity data and enhance LLM accuracy in knowledge retention and temporal reasoning. Using external datasets and the Llama-3-8B-Instruct model, we evaluate baseline RAG, an optimized hybrid retrieval approach, and conduct a comparative analysis across multiple performance metrics. Our findings highlight the promise of hybrid retrieval in strengthening the adaptability and reliability of LLMs for cybersecurity tasks. 

**Abstract (ZH)**: 基于检索增强生成的框架在提升 cybersecurity 数据上下文理解及提升 LLM 知识保留与时间推理准确度中的应用研究 

---
# Towards a Measure of Algorithm Similarity 

**Title (ZH)**: 向着算法相似性度量的研究 

**Authors**: Shairoz Sohail, Taher Ali  

**Link**: [PDF](https://arxiv.org/pdf/2510.27063)  

**Abstract**: Given two algorithms for the same problem, can we determine whether they are meaningfully different? In full generality, the question is uncomputable, and empirically it is muddied by competing notions of similarity. Yet, in many applications (such as clone detection or program synthesis) a pragmatic and consistent similarity metric is necessary. We review existing equivalence and similarity notions and introduce EMOC: An Evaluation-Memory-Operations-Complexity framework that embeds algorithm implementations into a feature space suitable for downstream tasks. We compile PACD, a curated dataset of verified Python implementations across three problems, and show that EMOC features support clustering and classification of algorithm types, detection of near-duplicates, and quantification of diversity in LLM-generated programs. Code, data, and utilities for computing EMOC embeddings are released to facilitate reproducibility and future work on algorithm similarity. 

**Abstract (ZH)**: 给定同一问题的两个算法，我们能否确定它们是否有意义的差异？在一般情况下，这个问题是不可计算的，而且由于相似性概念的竞争性，从经验上来看会变得模糊不清。然而，在许多应用中（如克隆检测或程序合成），需要一种实用且一致的相似性度量标准。我们回顾了现有的等价性和相似性概念，并引入了EMOC：一种评价-内存-操作复杂性框架，将算法实现嵌入到适合下游任务的特征空间中。我们编译了PACD，这是一个包含三个问题的验证Python实现的精编数据集，并展示了EMOC特征支持算法类型聚类和分类、接近重复检测以及LLM生成程序多样性的量化。提供了用于计算EMOC嵌入的代码、数据和工具，以促进算法相似性研究的可重复性和未来工作。 

---
# Consistency Training Helps Stop Sycophancy and Jailbreaks 

**Title (ZH)**: 一致性训练有助于防止逢迎和闯狱攻击 

**Authors**: Alex Irpan, Alexander Matt Turner, Mark Kurzeja, David K. Elson, Rohin Shah  

**Link**: [PDF](https://arxiv.org/pdf/2510.27062)  

**Abstract**: An LLM's factuality and refusal training can be compromised by simple changes to a prompt. Models often adopt user beliefs (sycophancy) or satisfy inappropriate requests which are wrapped within special text (jailbreaking). We explore \emph{consistency training}, a self-supervised paradigm that teaches a model to be invariant to certain irrelevant cues in the prompt. Instead of teaching the model what exact response to give on a particular prompt, we aim to teach the model to behave identically across prompt data augmentations (like adding leading questions or jailbreak text). We try enforcing this invariance in two ways: over the model's external outputs (\emph{Bias-augmented Consistency Training} (BCT) from Chua et al. [2025]) and over its internal activations (\emph{Activation Consistency Training} (ACT), a method we introduce). Both methods reduce Gemini 2.5 Flash's susceptibility to irrelevant cues. Because consistency training uses responses from the model itself as training data, it avoids issues that arise from stale training data, such as degrading model capabilities or enforcing outdated response guidelines. While BCT and ACT reduce sycophancy equally well, BCT does better at jailbreak reduction. We think that BCT can simplify training pipelines by removing reliance on static datasets. We argue that some alignment problems are better viewed not in terms of optimal responses, but rather as consistency issues. 

**Abstract (ZH)**: 一种提示简单变化就可能破坏LLM事实性和拒绝训练的现象：一致性训练探究 

---
# Detecting Data Contamination in LLMs via In-Context Learning 

**Title (ZH)**: 通过内省式学习检测LLMs中的数据污染 

**Authors**: Michał Zawalski, Meriem Boubdir, Klaudia Bałazy, Besmira Nushi, Pablo Ribalta  

**Link**: [PDF](https://arxiv.org/pdf/2510.27055)  

**Abstract**: We present Contamination Detection via Context (CoDeC), a practical and accurate method to detect and quantify training data contamination in large language models. CoDeC distinguishes between data memorized during training and data outside the training distribution by measuring how in-context learning affects model performance. We find that in-context examples typically boost confidence for unseen datasets but may reduce it when the dataset was part of training, due to disrupted memorization patterns. Experiments show that CoDeC produces interpretable contamination scores that clearly separate seen and unseen datasets, and reveals strong evidence of memorization in open-weight models with undisclosed training corpora. The method is simple, automated, and both model- and dataset-agnostic, making it easy to integrate with benchmark evaluations. 

**Abstract (ZH)**: Contamination Detection via Context (CoDeC):一种实用且准确的大型语言模型训练数据污染检测方法 

---
# Dataset Creation and Baseline Models for Sexism Detection in Hausa 

**Title (ZH)**: Hausa性別主義检测的数据集创建与基线模型构建 

**Authors**: Fatima Adam Muhammad, Shamsuddeen Muhammad Hassan, Isa Inuwa-Dutse  

**Link**: [PDF](https://arxiv.org/pdf/2510.27038)  

**Abstract**: Sexism reinforces gender inequality and social exclusion by perpetuating stereotypes, bias, and discriminatory norms. Noting how online platforms enable various forms of sexism to thrive, there is a growing need for effective sexism detection and mitigation strategies. While computational approaches to sexism detection are widespread in high-resource languages, progress remains limited in low-resource languages where limited linguistic resources and cultural differences affect how sexism is expressed and perceived. This study introduces the first Hausa sexism detection dataset, developed through community engagement, qualitative coding, and data augmentation. For cultural nuances and linguistic representation, we conducted a two-stage user study (n=66) involving native speakers to explore how sexism is defined and articulated in everyday discourse. We further experiment with both traditional machine learning classifiers and pre-trained multilingual language models and evaluating the effectiveness few-shot learning in detecting sexism in Hausa. Our findings highlight challenges in capturing cultural nuance, particularly with clarification-seeking and idiomatic expressions, and reveal a tendency for many false positives in such cases. 

**Abstract (ZH)**: 性别歧视通过传播刻板印象、偏见和歧视性规范固化性别不平等和社会排斥。鉴于在线平台使得各种形式的性别歧视得以滋生，有效检测和减轻性别歧视的策略需求日益增加。尽管在高资源语言中存在广泛的性别歧视检测计算方法，但在低资源语言中，由于有限的语言资源和文化差异影响性别歧视的表达和感知，进展仍然有限。本研究通过社区参与、定性编码和数据增广，引入了首个豪萨语性别歧视检测数据集。为探究文化细微差别和语言表达，我们进行了两阶段用户研究（n=66），邀请母语者探索性别歧视在日常 discourse 中的定义和表达方式。我们进一步尝试了传统机器学习分类器和预训练多语言语言模型，并评估了少样本学习在检测豪萨语性别歧视中的有效性。我们的研究结果强调了在捕捉文化细微差别方面面临的挑战，特别是在含糊不清和成语表达的情况下，发现了许多假阳性的情况。 

---
# Elastic Architecture Search for Efficient Language Models 

**Title (ZH)**: 高效的语言模型弹性架构搜索 

**Authors**: Shang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27037)  

**Abstract**: As large pre-trained language models become increasingly critical to natural language understanding (NLU) tasks, their substantial computational and memory requirements have raised significant economic and environmental concerns. Addressing these challenges, this paper introduces the Elastic Language Model (ELM), a novel neural architecture search (NAS) method optimized for compact language models. ELM extends existing NAS approaches by introducing a flexible search space with efficient transformer blocks and dynamic modules for dimension and head number adjustment. These innovations enhance the efficiency and flexibility of the search process, which facilitates more thorough and effective exploration of model architectures. We also introduce novel knowledge distillation losses that preserve the unique characteristics of each block, in order to improve the discrimination between architectural choices during the search process. Experiments on masked language modeling and causal language modeling tasks demonstrate that models discovered by ELM significantly outperform existing methods. 

**Abstract (ZH)**: 随着大型预训练语言模型在自然语言理解任务中的作用日益重要，它们对计算和内存的大量需求引起了显著的经济和环境关注。为应对这些挑战，本文提出了弹性语言模型（ELM），一种优化紧凑型语言模型的新型神经架构搜索（NAS）方法。ELM 通过引入高效的变压器块和动态模块以灵活调整维度和头数，扩展了现有的 NAS 方法。这些创新增强了搜索过程的效率和灵活性，从而促进了更深入和有效的模型架构探索。我们还引入了新的知识蒸馏损失，以保留每个模块的独特特性，在搜索过程中改善架构选择之间的区分能力。实验结果表明，由 ELM 发现的模型显著优于现有方法。 

---
# A Multi-Modal Neuro-Symbolic Approach for Spatial Reasoning-Based Visual Grounding in Robotics 

**Title (ZH)**: 基于空间推理的多模态神经符号方法在机器人视觉定位中的应用 

**Authors**: Simindokht Jahangard, Mehrzad Mohammadi, Abhinav Dhall, Hamid Rezatofighi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27033)  

**Abstract**: Visual reasoning, particularly spatial reasoning, is a challenging cognitive task that requires understanding object relationships and their interactions within complex environments, especially in robotics domain. Existing vision_language models (VLMs) excel at perception tasks but struggle with fine-grained spatial reasoning due to their implicit, correlation-driven reasoning and reliance solely on images. We propose a novel neuro_symbolic framework that integrates both panoramic-image and 3D point cloud information, combining neural perception with symbolic reasoning to explicitly model spatial and logical relationships. Our framework consists of a perception module for detecting entities and extracting attributes, and a reasoning module that constructs a structured scene graph to support precise, interpretable queries. Evaluated on the JRDB-Reasoning dataset, our approach demonstrates superior performance and reliability in crowded, human_built environments while maintaining a lightweight design suitable for robotics and embodied AI applications. 

**Abstract (ZH)**: 视觉推理，特别是空间推理，是一项具有挑战性的认知任务，要求理解对象关系及其在复杂环境中的相互作用，尤其是在机器人领域。现有的视觉语言模型（VLMs）在感知任务上表现出色，但在细粒度的空间推理方面存在困难，这主要是由于它们依赖于图像的隐式、关联驱动的推理。我们提出了一种新颖的神经-符号框架，该框架结合全景图像和3D点云信息，将神经感知与符号推理结合，明确建模空间和逻辑关系。该框架包括一个感知模块用于检测实体并提取属性，以及一个推理模块构建结构化的场景图以支持精确、可解释的查询。在JRDB-Reasoning数据集上的评估表明，我们的方法在拥挤的人造环境中展现了优越的性能和可靠性，并保持了适合机器人和具身AI应用的轻量级设计。 

---
# Jasmine: A Simple, Performant and Scalable JAX-based World Modeling Codebase 

**Title (ZH)**: Jasmine: 一个基于JAX的简单、高性能且可扩展的世界建模代码库 

**Authors**: Mihir Mahajan, Alfred Nguyen, Franz Srambical, Stefan Bauer  

**Link**: [PDF](https://arxiv.org/pdf/2510.27002)  

**Abstract**: While world models are increasingly positioned as a pathway to overcoming data scarcity in domains such as robotics, open training infrastructure for world modeling remains nascent. We introduce Jasmine, a performant JAX-based world modeling codebase that scales from single hosts to hundreds of accelerators with minimal code changes. Jasmine achieves an order-of-magnitude faster reproduction of the CoinRun case study compared to prior open implementations, enabled by performance optimizations across data loading, training and checkpointing. The codebase guarantees fully reproducible training and supports diverse sharding configurations. By pairing Jasmine with curated large-scale datasets, we establish infrastructure for rigorous benchmarking pipelines across model families and architectural ablations. 

**Abstract (ZH)**: 尽管世界模型在克服机器人等领域数据稀缺性方面越来越受到重视，但开放训练基础设施的世界建模仍处于初级阶段。我们引入了Jasmine，这是一个基于JAX的世界模型代码库，可以从单个主机扩展到数百个加速器，并且只需进行最少的代码更改即可实现规模扩展。Jasmine通过在数据加载、训练和检查点方面进行性能优化，相比于之前的开放实现实现了数量级的速度提升。该代码库保证了训练的完全可重复性，并支持多种切分配置。通过将Jasmine与精心挑选的大规模数据集相结合，我们建立了跨模型家族和架构删减的严格基准测试管道基础设施。 

---
# A Framework for Fair Evaluation of Variance-Aware Bandit Algorithms 

**Title (ZH)**: 面向方差感知bandit算法公平评估的框架 

**Authors**: Elise Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2510.27001)  

**Abstract**: Multi-armed bandit (MAB) problems serve as a fundamental building block for more complex reinforcement learning algorithms. However, evaluating and comparing MAB algorithms remains challenging due to the lack of standardized conditions and replicability. This is particularly problematic for variance-aware extensions of classical methods like UCB, whose performance can heavily depend on the underlying environment. In this study, we address how performance differences between bandit algorithms can be reliably observed, and under what conditions variance-aware algorithms outperform classical ones. We present a reproducible evaluation designed to systematically compare eight classical and variance-aware MAB algorithms. The evaluation framework, implemented in our Bandit Playground codebase, features clearly defined experimental setups, multiple performance metrics (reward, regret, reward distribution, value-at-risk, and action optimality), and an interactive evaluation interface that supports consistent and transparent analysis. We show that variance-aware algorithms can offer advantages in settings with high uncertainty where the difficulty arises from subtle differences between arm rewards. In contrast, classical algorithms often perform equally well or better in more separable scenarios or if fine-tuned extensively. Our contributions are twofold: (1) a framework for systematic evaluation of MAB algorithms, and (2) insights into the conditions under which variance-aware approaches outperform their classical counterparts. 

**Abstract (ZH)**: 多臂-bandit (MAB) 问题作为更复杂强化学习算法的基本构建块至关重要。然而，由于缺乏标准化条件和可重复性，评估和比较MAB算法仍然具有挑战性。这特别对经典方法如UCB的方差感知扩展提出了问题，其性能可能强烈依赖于底层环境。在本研究中，我们探讨了如何可靠地观察不同bandit算法之间的性能差异，并在什么条件下方差感知算法优于经典算法。我们呈现了一个可重复的评估框架，旨在系统比较八种经典和方差感知MAB算法。该评估框架在我们的Bandit Playground代码库中实现，包含明确定义的实验设置、多种性能指标（奖励、遗憾、奖励分布、值-at-风险和动作优化性），以及支持一致和透明分析的交互式评估界面。我们展示了方差感知算法在高不确定性环境下具有优势，这些环境因手臂奖励之间的微妙差异而具有挑战性。相比之下，在更分离的场景或经过广泛微调的情况下，经典算法通常表现得同样好或更好。我们的贡献有两个方面：(1) 一种MAB算法系统的评估框架，(2) 方差感知方法在什么条件下优于其经典对手的见解。 

---
# AIOT based Smart Education System: A Dual Layer Authentication and Context-Aware Tutoring Framework for Learning Environments 

**Title (ZH)**: 基于AIOT的智能教育系统：学习环境中基于双层认证和情景感知的辅导框架 

**Authors**: Adithya Neelakantan, Pratik Satpute, Prerna Shinde, Tejas Manjunatha Devang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26999)  

**Abstract**: The AIoT-Based Smart Education System integrates Artificial Intelligence and IoT to address persistent challenges in contemporary classrooms: attendance fraud, lack of personalization, student disengagement, and inefficient resource use. The unified platform combines four core modules: (1) a dual-factor authentication system leveraging RFID-based ID scans and WiFi verification for secure, fraud-resistant attendance; (2) an AI-powered assistant that provides real-time, context-aware support and dynamic quiz generation based on instructor-supplied materials; (3) automated test generators to streamline adaptive assessment and reduce administrative overhead; and (4) the EcoSmart Campus module, which autonomously regulates classroom lighting, air quality, and temperature using IoT sensors and actuators. Simulated evaluations demonstrate the system's effectiveness in delivering robust real-time monitoring, fostering inclusive engagement, preventing fraudulent practices, and supporting operational scalability. Collectively, the AIoT-Based Smart Education System offers a secure, adaptive, and efficient learning environment, providing a scalable blueprint for future educational innovation and improved student outcomes through the synergistic application of artificial intelligence and IoT technologies. 

**Abstract (ZH)**: 基于AIoT的智能教育系统整合人工智能和物联网以应对当代教室中长期存在的挑战：考勤作弊、个性化不足、学生活动参与度低和资源使用效率低下。该统一平台结合了四个核心模块：（1）基于RFID身份扫描和WiFi验证的双因素认证系统，用于安全、防作弊的考勤；（2）基于人工智能的助手，提供实时、情境感知的支持和依据教师提供的材料生成的动态测验；（3）自动化测试生成器，以简化适应性评估并减少行政负担；以及（4）EcoSmart校园模块，利用物联网传感器和执行器自主调节教室照明、空气质量及温度。模拟评估验证了该系统的有效性，能够在实时监控、促进包容性参与、防止欺诈行为以及支持操作扩展方面发挥重要作用。总体而言，基于AIoT的智能教育系统提供了一个安全、适应性强且高效的的学习环境，为未来教育创新和通过人工智能与物联网技术的协同应用提高学生成果提供了可扩展的蓝图。 

---
# LLMs are Overconfident: Evaluating Confidence Interval Calibration with FermiEval 

**Title (ZH)**: LLMs过于自信：基于FermiEval评估置信区间校准 

**Authors**: Elliot L. Epstein, John Winnicki, Thanawat Sornwanee, Rajat Dwaraknath  

**Link**: [PDF](https://arxiv.org/pdf/2510.26995)  

**Abstract**: Large language models (LLMs) excel at numerical estimation but struggle to correctly quantify uncertainty. We study how well LLMs construct confidence intervals around their own answers and find that they are systematically overconfident. To evaluate this behavior, we introduce FermiEval, a benchmark of Fermi-style estimation questions with a rigorous scoring rule for confidence interval coverage and sharpness. Across several modern models, nominal 99\% intervals cover the true answer only 65\% of the time on average. With a conformal prediction based approach that adjusts the intervals, we obtain accurate 99\% observed coverage, and the Winkler interval score decreases by 54\%. We also propose direct log-probability elicitation and quantile adjustment methods, which further reduce overconfidence at high confidence levels. Finally, we develop a perception-tunnel theory explaining why LLMs exhibit overconfidence: when reasoning under uncertainty, they act as if sampling from a truncated region of their inferred distribution, neglecting its tails. 

**Abstract (ZH)**: 大规模语言模型在数值估计方面表现出色，但在正确量化不确定性方面存在困难。我们研究了大规模语言模型在其自身答案周围构建置信区间的能力，并发现它们系统地过于自信。为了评估这种行为，我们引入了FermiEval，这是一个基于费米风格估计问题的基准，其中包含严格的置信区间覆盖率和锋利度评分规则。在多个现代模型中，名义上的99%区间平均只有65%的时间包含真实答案。通过基于一致性预测的方法调整区间，我们获得了准确的99%实际覆盖率，Winkler区间分数降低了54%。我们还提出了直接概率对数 elicitation 和分位数调整方法，这些方法进一步减少了高置信水平下的过于自信现象。最后，我们发展了一种感知隧道理论来解释为什么大型语言模型表现出过于自信：在不确定性的推理过程中，它们似乎是从其推断分布的截断区域中进行采样，忽视了其尾部。 

---
# Fine-Grained Iterative Adversarial Attacks with Limited Computation Budget 

**Title (ZH)**: 细粒度迭代对抗攻击在有限计算预算下 

**Authors**: Zhichao Hou, Weizhi Gao, Xiaorui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26981)  

**Abstract**: This work tackles a critical challenge in AI safety research under limited compute: given a fixed computation budget, how can one maximize the strength of iterative adversarial attacks? Coarsely reducing the number of attack iterations lowers cost but substantially weakens effectiveness. To fulfill the attainable attack efficacy within a constrained budget, we propose a fine-grained control mechanism that selectively recomputes layer activations across both iteration-wise and layer-wise levels. Extensive experiments show that our method consistently outperforms existing baselines at equal cost. Moreover, when integrated into adversarial training, it attains comparable performance with only 30% of the original budget. 

**Abstract (ZH)**: 在有限计算资源下最大化迭代对抗攻击效果的研究：一种细粒度控制机制 

---
# Overview of the MEDIQA-OE 2025 Shared Task on Medical Order Extraction from Doctor-Patient Consultations 

**Title (ZH)**: 2025年MEDIQA-OE共享任务：从医生-患者咨询中提取医疗 orders 概览 

**Authors**: Jean-Philippe Corbeil, Asma Ben Abacha, Jerome Tremblay, Phillip Swazinna, Akila Jeeson Daniel, Miguel Del-Agua, Francois Beaulieu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26974)  

**Abstract**: Clinical documentation increasingly uses automatic speech recognition and summarization, yet converting conversations into actionable medical orders for Electronic Health Records remains unexplored. A solution to this problem can significantly reduce the documentation burden of clinicians and directly impact downstream patient care. We introduce the MEDIQA-OE 2025 shared task, the first challenge on extracting medical orders from doctor-patient conversations. Six teams participated in the shared task and experimented with a broad range of approaches, and both closed- and open-weight large language models (LLMs). In this paper, we describe the MEDIQA-OE task, dataset, final leaderboard ranking, and participants' solutions. 

**Abstract (ZH)**: 临床文档越来越多地使用自动语音识别和总结技术，然而将对话转换为电子健康记录中的可操作医疗订单仍然未被探索。解决这一问题可以显著减轻临床人员的文档负担，并直接影响下游患者的护理。我们介绍了MEDIQA-OE 2025共享任务，这是首次针对从医生-患者对话中抽取医疗订单的挑战。六支队伍参与了共享任务，并尝试了广泛的方法，包括闭合权重和开放权重的大语言模型。在本文中，我们描述了MEDIQA-OE任务、数据集、最终排行榜以及参赛队伍的解决方案。 

---
# Frame Semantic Patterns for Identifying Underreporting of Notifiable Events in Healthcare: The Case of Gender-Based Violence 

**Title (ZH)**: 基于框架语义模式识别医疗保健中可报告事件上报不足的情况：以基于性别的暴力为例 

**Authors**: Lívia Dutra, Arthur Lorenzi, Laís Berno, Franciany Campos, Karoline Biscardi, Kenneth Brown, Marcelo Viridiano, Frederico Belcavello, Ely Matos, Olívia Guaranha, Erik Santos, Sofia Reinach, Tiago Timponi Torrent  

**Link**: [PDF](https://arxiv.org/pdf/2510.26969)  

**Abstract**: We introduce a methodology for the identification of notifiable events in the domain of healthcare. The methodology harnesses semantic frames to define fine-grained patterns and search them in unstructured data, namely, open-text fields in e-medical records. We apply the methodology to the problem of underreporting of gender-based violence (GBV) in e-medical records produced during patients' visits to primary care units. A total of eight patterns are defined and searched on a corpus of 21 million sentences in Brazilian Portuguese extracted from e-SUS APS. The results are manually evaluated by linguists and the precision of each pattern measured. Our findings reveal that the methodology effectively identifies reports of violence with a precision of 0.726, confirming its robustness. Designed as a transparent, efficient, low-carbon, and language-agnostic pipeline, the approach can be easily adapted to other health surveillance contexts, contributing to the broader, ethical, and explainable use of NLP in public health systems. 

**Abstract (ZH)**: 一种在医疗领域识别可报告事件的方法及其在电子医疗记录中基于性别的暴力报告识别中的应用 

---
# Using Salient Object Detection to Identify Manipulative Cookie Banners that Circumvent GDPR 

**Title (ZH)**: 使用显著目标检测识别绕过GDPR的操纵性Cookie弹窗 

**Authors**: Riley Grossman, Michael Smith, Cristian Borcea, Yi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26967)  

**Abstract**: The main goal of this paper is to study how often cookie banners that comply with the General Data Protection Regulation (GDPR) contain aesthetic manipulation, a design tactic to draw users' attention to the button that permits personal data sharing. As a byproduct of this goal, we also evaluate how frequently the banners comply with GDPR and the recommendations of national data protection authorities regarding banner designs. We visited 2,579 websites and identified the type of cookie banner implemented. Although 45% of the relevant websites have fully compliant banners, we found aesthetic manipulation on 38% of the compliant banners. Unlike prior studies of aesthetic manipulation, we use a computer vision model for salient object detection to measure how salient (i.e., attention-drawing) each banner element is. This enables the discovery of new types of aesthetic manipulation (e.g., button placement), and leads us to conclude that aesthetic manipulation is more common than previously reported (38% vs 27% of banners). To study the effects of user and/or website location on cookie banner design, we include websites within the European Union (EU), where privacy regulation enforcement is more stringent, and websites outside the EU. We visited websites from IP addresses in the EU and from IP addresses in the United States (US). We find that 13.9% of EU websites change their banner design when the user is from the US, and EU websites are roughly 48.3% more likely to use aesthetic manipulation than non-EU websites, highlighting their innovative responses to privacy regulation. 

**Abstract (ZH)**: 本文的主要目标是研究哪些符合《通用数据保护条例》（GDPR）的cookie弹窗中包含多少美学操控，这是一种设计手法，旨在吸引用户注意允许分享个人数据的按钮。作为这一目标的副产品，我们还评估了合规的cookie弹窗及其设计是否符合GDPR和国家数据保护机构的建议。我们访问了2,579个网站并确定了实施的cookie弹窗类型。尽管45%的相关网站拥有完全合规的弹窗，但我们发现38%的合规弹窗包含美学操控。不同于先前对美学操控的研究，我们使用计算机视觉模型进行显著对象检测，衡量每个弹窗元素的显著性（即，注意吸引程度），这使得我们能够发现新的美学操控类型（例如，按钮位置），并得出美学操控比之前报道的更为常见的结论（38% vs 27%的弹窗）。为了研究用户和/或网站位置对cookie弹窗设计的影响，我们包含了位于欧盟（EU）内的网站，这些网站的隐私法规执行更为严格，以及位于欧盟之外的网站。我们访问了来自欧盟IP地址和来自美国IP地址的网站。我们发现13.9%的欧盟网站会在用户来自美国时更改其弹窗设计，并且欧盟网站比非欧盟网站约高出48.3%使用美学操控的可能性，这突显了其对隐私法规的创新回应。 

---
# Can machines think efficiently? 

**Title (ZH)**: 机器能高效思考吗？ 

**Authors**: Adam Winchell  

**Link**: [PDF](https://arxiv.org/pdf/2510.26954)  

**Abstract**: The Turing Test is no longer adequate for distinguishing human and machine intelligence. With advanced artificial intelligence systems already passing the original Turing Test and contributing to serious ethical and environmental concerns, we urgently need to update the test. This work expands upon the original imitation game by accounting for an additional factor: the energy spent answering the questions. By adding the constraint of energy, the new test forces us to evaluate intelligence through the lens of efficiency, connecting the abstract problem of thinking to the concrete reality of finite resources. Further, this proposed new test ensures the evaluation of intelligence has a measurable, practical finish line that the original test lacks. This additional constraint compels society to weigh the time savings of using artificial intelligence against its total resource cost. 

**Abstract (ZH)**: 图灵测试已不足以区分人类和机器智能。鉴于先进人工智能系统已通过原始图灵测试并引发严重伦理和环境问题，我们迫切需要更新这一测试。本研究在原始模仿游戏中引入了额外的因素：回答问题所消耗的能源。通过增加能源约束，新模式试要求我们从效率的角度评估智能，将抽象的思考问题与有限资源的现实联系起来。此外，这一提出的新型测试确保了智能评估有一个可度量且实用的终点，这是原始测试所缺乏的。这一额外的约束促使社会权衡使用人工智能节省的时间与总体资源成本之间的关系。 

---
# LLM-based Multi-class Attack Analysis and Mitigation Framework in IoT/IIoT Networks 

**Title (ZH)**: 基于LLM的物联网/工业物联网网络多类攻击分析与缓解框架 

**Authors**: Seif Ikbarieh, Maanak Gupta, Elmahedi Mahalal  

**Link**: [PDF](https://arxiv.org/pdf/2510.26941)  

**Abstract**: The Internet of Things has expanded rapidly, transforming communication and operations across industries but also increasing the attack surface and security breaches. Artificial Intelligence plays a key role in securing IoT, enabling attack detection, attack behavior analysis, and mitigation suggestion. Despite advancements, evaluations remain purely qualitative, and the lack of a standardized, objective benchmark for quantitatively measuring AI-based attack analysis and mitigation hinders consistent assessment of model effectiveness. In this work, we propose a hybrid framework combining Machine Learning (ML) for multi-class attack detection with Large Language Models (LLMs) for attack behavior analysis and mitigation suggestion. After benchmarking several ML and Deep Learning (DL) classifiers on the Edge-IIoTset and CICIoT2023 datasets, we applied structured role-play prompt engineering with Retrieval-Augmented Generation (RAG) to guide ChatGPT-o3 and DeepSeek-R1 in producing detailed, context-aware responses. We introduce novel evaluation metrics for quantitative assessment to guide us and an ensemble of judge LLMs, namely ChatGPT-4o, DeepSeek-V3, Mixtral 8x7B Instruct, Gemini 2.5 Flash, Meta Llama 4, TII Falcon H1 34B Instruct, xAI Grok 3, and Claude 4 Sonnet, to independently evaluate the responses. Results show that Random Forest has the best detection model, and ChatGPT-o3 outperformed DeepSeek-R1 in attack analysis and mitigation. 

**Abstract (ZH)**: 物联网的迅猛扩展正在改变各行业的通信和操作，同时也增加了攻击面和安全漏洞。人工智能在保障物联网安全中发挥关键作用，能够实现攻击检测、攻击行为分析和缓解建议。尽管取得了进展，但评估仍然主要依赖定性评价，缺乏标准化和客观的基准来定量测量基于AI的攻击分析和缓解效果，这阻碍了对模型有效性的一致评估。在本研究中，我们提出了一种结合机器学习（ML）进行多类攻击检测和大规模语言模型（LLMs）进行攻击行为分析和缓解建议的混合框架。在对Edge-IIoTset和CICIoT2023数据集中的多种机器学习（ML）和深度学习（DL）分类器进行基准测试后，我们利用检索增强生成（RAG）结构化角色扮演提示工程指导ChatGPT-o3和DeepSeek-R1生成详细且情境相关的回应。我们引入了新的评估指标进行定量评估，并由ChatGPT-4o、DeepSeek-V3、Mixtral 8x7B Instruct、Gemini 2.5 Flash、Meta Llama 4、TII Falcon H1 34B Instruct、xAI Grok 3和Claude 4 Sonnet等多个评判大规模语言模型独立评估这些回应。结果表明，随机森林具有最佳的检测模型，而ChatGPT-o3在攻击分析和缓解方面优于DeepSeek-R1。 

---
# Mind the Gaps: Auditing and Reducing Group Inequity in Large-Scale Mobility Prediction 

**Title (ZH)**: 注意差距：审计和减少大规模移动预测中的群体不公正性 

**Authors**: Ashwin Kumar, Hanyu Zhang, David A. Schweidel, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2510.26940)  

**Abstract**: Next location prediction underpins a growing number of mobility, retail, and public-health applications, yet its societal impacts remain largely unexplored. In this paper, we audit state-of-the-art mobility prediction models trained on a large-scale dataset, highlighting hidden disparities based on user demographics. Drawing from aggregate census data, we compute the difference in predictive performance on racial and ethnic user groups and show a systematic disparity resulting from the underlying dataset, resulting in large differences in accuracy based on location and user groups. To address this, we propose Fairness-Guided Incremental Sampling (FGIS), a group-aware sampling strategy designed for incremental data collection settings. Because individual-level demographic labels are unavailable, we introduce Size-Aware K-Means (SAKM), a clustering method that partitions users in latent mobility space while enforcing census-derived group proportions. This yields proxy racial labels for the four largest groups in the state: Asian, Black, Hispanic, and White. Built on these labels, our sampling algorithm prioritizes users based on expected performance gains and current group representation. This method incrementally constructs training datasets that reduce demographic performance gaps while preserving overall accuracy. Our method reduces total disparity between groups by up to 40\% with minimal accuracy trade-offs, as evaluated on a state-of-art MetaPath2Vec model and a transformer-encoder model. Improvements are most significant in early sampling stages, highlighting the potential for fairness-aware strategies to deliver meaningful gains even in low-resource settings. Our findings expose structural inequities in mobility prediction pipelines and demonstrate how lightweight, data-centric interventions can improve fairness with little added complexity, especially for low-data applications. 

**Abstract (ZH)**: 基于大规模数据训练的最新移动预测模型在不同用户群体间隐含差异未被充分探索。本文审查了这些模型，揭示了基于用户人口统计学的隐藏不平等。通过汇总的人口普查数据，我们计算了不同种族和 Ethnic 用户群体的预测性能差异，并展示了由于底层数据集导致的系统性差异，从而引起了基于位置和用户群体的准确性差异。为解决这一问题，我们提出了一种公平指导的增量采样（FGIS）方法，这是一种针对增量数据收集场景设计的群体意识采样策略。由于无法获取个体级别的人口统计标签，我们引入了一种大小感知的 K-均值聚类方法（SAKM），该方法在潜在移动空间中分区用户，同时强制执行人口普查推断的群体比例。这为州内四个最大的群体（亚裔、非裔、拉丁裔和白人）提供了代理种族标签。基于这些标签，我们的采样算法根据预期性能提升和当前群体代表性来优先选择用户。这种方法逐步构建训练数据集，以减少人口统计学性能差距，同时保持总体准确性。在评估 MetaPath2Vec 模型和 transformer-编码模型时，我们的方法将组间的总不平等性降低了最多40%，且对准确性影响微乎其微。早期采样阶段的改进最为显著，强调了公平意识策略即使在资源有限的情况下也能带来显著的改进潜力。我们的研究揭示了移动预测管道中的结构性不平等，并展示了轻量级的数据导向干预措施如何在增加复杂性极小的情况下改善公平性，特别是在数据稀缺的应用场景中。 

---
# RepV: Safety-Separable Latent Spaces for Scalable Neurosymbolic Plan Verification 

**Title (ZH)**: RepV: 安全分离的潜在空间以实现可扩展的神经符号计划验证 

**Authors**: Yunhao Yang, Neel P. Bhatt, Pranay Samineni, Rohan Siva, Zhanyang Wang, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26935)  

**Abstract**: As AI systems migrate to safety-critical domains, verifying that their actions comply with well-defined rules remains a challenge. Formal methods provide provable guarantees but demand hand-crafted temporal-logic specifications, offering limited expressiveness and accessibility. Deep learning approaches enable evaluation of plans against natural-language constraints, yet their opaque decision process invites misclassifications with potentially severe consequences. We introduce RepV, a neurosymbolic verifier that unifies both views by learning a latent space where safe and unsafe plans are linearly separable. Starting from a modest seed set of plans labeled by an off-the-shelf model checker, RepV trains a lightweight projector that embeds each plan, together with a language model-generated rationale, into a low-dimensional space; a frozen linear boundary then verifies compliance for unseen natural-language rules in a single forward pass.
Beyond binary classification, RepV provides a probabilistic guarantee on the likelihood of correct verification based on its position in the latent space. This guarantee enables a guarantee-driven refinement of the planner, improving rule compliance without human annotations. Empirical evaluations show that RepV improves compliance prediction accuracy by up to 15% compared to baseline methods while adding fewer than 0.2M parameters. Furthermore, our refinement framework outperforms ordinary fine-tuning baselines across various planning domains. These results show that safety-separable latent spaces offer a scalable, plug-and-play primitive for reliable neurosymbolic plan verification. Code and data are available at: this https URL. 

**Abstract (ZH)**: 随着AI系统迁移到安全关键领域，验证其行为是否符合预定义规则仍是一项挑战。形式化方法可以提供可证明的保证，但需要手工构建时间逻辑规范，这在表达能力和可访问性方面都有限。深度学习方法允许用自然语言约束评估计划，但其不透明的决策过程可能导致潜在严重后果的误分类。我们提出了RepV，这是一种神经符号验证器，通过学习一个潜在空间统一了这两种视角，在该潜在空间中，安全和不安全的计划是线性可分的。从一个现成模型检查器标注的初始计划种子集开始，RepV 训练一个轻量级的投影器，将每个计划与其语言模型生成的解释嵌入到低维空间中；然后，一个冻结的线性边界在单次前向传播中验证未见过的自然语言规则的合规性。 

---
# Scale-Aware Curriculum Learning for Ddata-Efficient Lung Nodule Detection with YOLOv11 

**Title (ZH)**: 面向数据高效肺结节检测的尺度aware课程学习方法-YOLOv11 

**Authors**: Yi Luo, Yike Guo, Hamed Hooshangnejad, Kai Ding  

**Link**: [PDF](https://arxiv.org/pdf/2510.26923)  

**Abstract**: Lung nodule detection in chest CT is crucial for early lung cancer diagnosis, yet existing deep learning approaches face challenges when deployed in clinical settings with limited annotated data. While curriculum learning has shown promise in improving model training, traditional static curriculum strategies fail in data-scarce scenarios. We propose Scale Adaptive Curriculum Learning (SACL), a novel training strategy that dynamically adjusts curriculum design based on available data scale. SACL introduces three key mechanisms:(1) adaptive epoch scheduling, (2) hard sample injection, and (3) scale-aware optimization. We evaluate SACL on the LUNA25 dataset using YOLOv11 as the base detector. Experimental results demonstrate that while SACL achieves comparable performance to static curriculum learning on the full dataset in mAP50, it shows significant advantages under data-limited conditions with 4.6%, 3.5%, and 2.0% improvements over baseline at 10%, 20%, and 50% of training data respectively. By enabling robust training across varying data scales without architectural modifications, SACL provides a practical solution for healthcare institutions to develop effective lung nodule detection systems despite limited annotation resources. 

**Abstract (ZH)**: 胸部CT中肺结节检测对于早期肺癌诊断至关重要，然而现有深度学习方法在临床应用中受限于标注数据匮乏。尽管 curriculum 学习在提高模型训练性能方面显示出潜力，传统静态 curriculum 战略在数据稀缺场景中表现不佳。我们提出了一种新的训练策略——适应性缩放 curriculum 学习 (SACL)，它能够根据可用数据量动态调整 curriculum 设计。SACL 引入了三种关键机制：(1) 自适应epoch调度，(2) 困难样本注入，(3) 缩放感知优化。我们使用YOLOv11作为基检测器在LUNA25数据集上评估SACL。实验结果表明，尽管在mAP50度量上SACL与静态curriculum学习的性能相当，但在数据受限条件下（分别在训练数据的10%、20%和50%时），SACL分别显示出4.6%、3.5%和2.0%的性能提升。通过在不修改架构的情况下实现跨不同数据规模的稳健训练，SACL为医疗保健机构提供了在标注资源有限的情况下开发有效肺结节检测系统的可行解决方案。 

---
# Heterogeneous Robot Collaboration in Unstructured Environments with Grounded Generative Intelligence 

**Title (ZH)**: 具有地基生成智能的异构机器人在非结构化环境下的协同作业 

**Authors**: Zachary Ravichandran, Fernando Cladera, Ankit Prabhu, Jason Hughes, Varun Murali, Camillo Taylor, George J. Pappas, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.26915)  

**Abstract**: Heterogeneous robot teams operating in realistic settings often must accomplish complex missions requiring collaboration and adaptation to information acquired online. Because robot teams frequently operate in unstructured environments -- uncertain, open-world settings without prior maps -- subtasks must be grounded in robot capabilities and the physical world. While heterogeneous teams have typically been designed for fixed specifications, generative intelligence opens the possibility of teams that can accomplish a wide range of missions described in natural language. However, current large language model (LLM)-enabled teaming methods typically assume well-structured and known environments, limiting deployment in unstructured environments. We present SPINE-HT, a framework that addresses these limitations by grounding the reasoning abilities of LLMs in the context of a heterogeneous robot team through a three-stage process. Given language specifications describing mission goals and team capabilities, an LLM generates grounded subtasks which are validated for feasibility. Subtasks are then assigned to robots based on capabilities such as traversability or perception and refined given feedback collected during online operation. In simulation experiments with closed-loop perception and control, our framework achieves nearly twice the success rate compared to prior LLM-enabled heterogeneous teaming approaches. In real-world experiments with a Clearpath Jackal, a Clearpath Husky, a Boston Dynamics Spot, and a high-altitude UAV, our method achieves an 87\% success rate in missions requiring reasoning about robot capabilities and refining subtasks with online feedback. More information is provided at this https URL. 

**Abstract (ZH)**: 异构机器人团队在现实场景中 often 必须执行需要协作和适应在线获取信息的复杂任务。由于机器人团队通常在未结构化的环境中运作——即不确定的、开放的世界环境，没有先验的地图——子任务必须基于机器人的能力和物理世界。虽然异构团队通常被设计为固定规格，但生成性智能开启了能够通过自然语言描述执行广泛任务的团队的可能性。然而，当前的大规模语言模型（LLM）启用的团队方法通常假设结构化的和已知的环境，限制了其在未结构化环境中的部署。我们提出了 SPINE-HT 框架，该框架通过一个三阶段过程将 LLM 的推理能力锚定在一个异构机器人团队的上下文中，从而解决这些限制。给定描述任务目标和团队能力的语言规范，LLM 生成可验证可行性的地面化子任务。然后根据诸如可通行性或感知之类的机器人能力将子任务分配给机器人，并在收到在线操作期间收集的反馈后进一步优化。在具有闭环感知和控制的仿真实验中，我们框架的成功率几乎是之前 LLM 启用的异构团队方法的两倍。在涉及 Clearpath Jackal、Clearpath Husky、Boston Dynamics Spot 和高空无人机的实际实验中，我们的方法在需要推理机器人能力和在线优化子任务的任务中取得了 87% 的成功率。更多详细信息请访问 <https://www.example.com>。 

---
# How Similar Are Grokipedia and Wikipedia? A Multi-Dimensional Textual and Structural Comparison 

**Title (ZH)**: Grokipedia和Wikipedia相似度研究：多维度文本与结构比较 

**Authors**: Taha Yasseri  

**Link**: [PDF](https://arxiv.org/pdf/2510.26899)  

**Abstract**: The launch of Grokipedia, an AI-generated encyclopedia developed by Elon Musk's xAI, was presented as a response to perceived ideological and structural biases in Wikipedia, aiming to produce "truthful" entries via the large language model Grok. Yet whether an AI-driven alternative can escape the biases and limitations of human-edited platforms remains unclear. This study undertakes a large-scale computational comparison of 382 matched article pairs between Grokipedia and Wikipedia. Using metrics across lexical richness, readability, structural organization, reference density, and semantic similarity, we assess how closely the two platforms align in form and substance. The results show that while Grokipedia exhibits strong semantic and stylistic alignment with Wikipedia, it typically produces longer but less lexically diverse articles, with fewer references per word and more variable structural depth. These findings suggest that AI-generated encyclopedic content currently mirrors Wikipedia's informational scope but diverges in editorial norms, favoring narrative expansion over citation-based verification. The implications highlight new tensions around transparency, provenance, and the governance of knowledge in an era of automated text generation. 

**Abstract (ZH)**: Grokipedia：一项由xAI开发的AI生成百科的推出及其与Wikipedia的对比研究 

---
# BI-DCGAN: A Theoretically Grounded Bayesian Framework for Efficient and Diverse GANs 

**Title (ZH)**: BI-DCGAN：一个理论支持的贝叶斯框架，用于高效的多样化生成对抗网络 

**Authors**: Mahsa Valizadeh, Rui Tuo, James Caverlee  

**Link**: [PDF](https://arxiv.org/pdf/2510.26892)  

**Abstract**: Generative Adversarial Networks (GANs) are proficient at generating synthetic data but continue to suffer from mode collapse, where the generator produces a narrow range of outputs that fool the discriminator but fail to capture the full data distribution. This limitation is particularly problematic, as generative models are increasingly deployed in real-world applications that demand both diversity and uncertainty awareness. In response, we introduce BI-DCGAN, a Bayesian extension of DCGAN that incorporates model uncertainty into the generative process while maintaining computational efficiency. BI-DCGAN integrates Bayes by Backprop to learn a distribution over network weights and employs mean-field variational inference to efficiently approximate the posterior distribution during GAN training. We establishes the first theoretical proof, based on covariance matrix analysis, that Bayesian modeling enhances sample diversity in GANs. We validate this theoretical result through extensive experiments on standard generative benchmarks, demonstrating that BI-DCGAN produces more diverse and robust outputs than conventional DCGANs, while maintaining training efficiency. These findings position BI-DCGAN as a scalable and timely solution for applications where both diversity and uncertainty are critical, and where modern alternatives like diffusion models remain too resource-intensive. 

**Abstract (ZH)**: 生成对抗网络（GANs）在生成合成数据方面表现出色，但仍然遭受模式崩溃的问题，即生成器产生的输出范围狭窄，能够欺骗判别器，但却未能捕捉到完整的数据分布。这一局限性尤其成问题，因为生成模型在现实世界应用中越来越受到多样化和不确定性意识的需求挑战。为应对这一挑战，我们介绍了BI-DCGAN，这是一种基于DCGAN的贝叶斯扩展，它在保持计算效率的同时将模型不确定性融入生成过程。BI-DCGAN结合了Bayes by Backprop来学习网络权重的分布，并采用均场变分推断来在GAN训练过程中高效地近似后验分布。我们首次基于协方差矩阵分析建立了理论证明，表明贝叶斯模型可以增强GAN中的样本多样性。通过在标准生成基准上的广泛实验验证了这一理论结果，表明BI-DCGAN能够生成比传统DCGAN更具多样性和鲁棒性的输出，同时保持训练效率。这些发现使BI-DCGAN成为在需要多样化和不确定性的应用中可扩展且及时的解决方案，而现代替代方案如扩散模型仍然过于资源密集。 

---
# Do Vision-Language Models Measure Up? Benchmarking Visual Measurement Reading with MeasureBench 

**Title (ZH)**: 视觉-语言模型达标了吗？使用MeasureBench评估视觉测量阅读能力 

**Authors**: Fenfen Lin, Yesheng Liu, Haiyu Xu, Chen Yue, Zheqi He, Mingxuan Zhao, Miguel Hu Chen, Jiakang Liu, JG Yao, Xi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26865)  

**Abstract**: Reading measurement instruments is effortless for humans and requires relatively little domain expertise, yet it remains surprisingly challenging for current vision-language models (VLMs) as we find in preliminary evaluation. In this work, we introduce MeasureBench, a benchmark on visual measurement reading covering both real-world and synthesized images of various types of measurements, along with an extensible pipeline for data synthesis. Our pipeline procedurally generates a specified type of gauge with controllable visual appearance, enabling scalable variation in key details such as pointers, scales, fonts, lighting, and clutter. Evaluation on popular proprietary and open-weight VLMs shows that even the strongest frontier VLMs struggle measurement reading in general. A consistent failure mode is indicator localization: models can read digits or labels but misidentify the key positions of pointers or alignments, leading to big numeric errors despite plausible textual reasoning. We have also conducted preliminary experiments with reinforcement learning over synthetic data, and find encouraging results on in-domain synthetic subset but less promising for real-world images. Our analysis highlights a fundamental limitation of current VLMs in fine-grained spatial grounding. We hope this resource can help future advances on visually grounded numeracy and precise spatial perception of VLMs, bridging the gap between recognizing numbers and measuring the world. 

**Abstract (ZH)**: 视觉测量阅读对于人类来说是容易的，只需要少量的领域专业知识，然而当前的视觉-语言模型（VLMs）在这一任务上仍然面临挑战。在本工作中，我们介绍了MeasureBench，一个涵盖各类测量的真实和合成图像的基准测试，以及一个扩展的数据合成流水线。我们的流水线程序化地生成指定类型的测量表盘，并控制其视觉外观，从而实现关键细节，如指针、刻度、字体、照明和杂乱环境的可扩展变化。对流行且开放的视觉-语言模型的评估显示，即使是最先进的视觉-语言模型在视觉测量阅读上也表现不佳。常见的失败模式是指示器定位：模型能够读取数字或标签，但错误识别指针的关键位置或对齐方式，即使进行合理的文本推理，也会导致较大的数值错误。我们还进行了初步的强化学习实验，使用合成数据，发现对领域内合成子集结果令人鼓舞，但在真实世界图像上表现不佳。我们的分析突显了当前视觉-语言模型在精细空间定位上的基本局限性。我们希望这一资源能帮助未来在视觉引导下的数字能力和精确的空间感知方面取得进展，弥合识别数字和测量世界之间的差距。 

---
# Leveraging Foundation Models for Enhancing Robot Perception and Action 

**Title (ZH)**: 利用基础模型增强机器人感知与行动能力 

**Authors**: Reihaneh Mirjalili  

**Link**: [PDF](https://arxiv.org/pdf/2510.26855)  

**Abstract**: This thesis investigates how foundation models can be systematically leveraged to enhance robotic capabilities, enabling more effective localization, interaction, and manipulation in unstructured environments. The work is structured around four core lines of inquiry, each addressing a fundamental challenge in robotics while collectively contributing to a cohesive framework for semantics-aware robotic intelligence. 

**Abstract (ZH)**: 本论文探究基础模型如何系统性地被利用以增强机器人能力，从而在非结构化环境中实现更加有效的定位、互动和操作。工作围绕四个核心研究方向展开，每个方向都解决机器人领域的一项基本挑战，共同构建起一种语义意识机器人智能的综合框架。 

---
# Broken-Token: Filtering Obfuscated Prompts by Counting Characters-Per-Token 

**Title (ZH)**: 断裂的标记：通过按标记计数字符过滤混淆的提示 

**Authors**: Shaked Zychlinski, Yuval Kainan  

**Link**: [PDF](https://arxiv.org/pdf/2510.26847)  

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks where malicious prompts are disguised using ciphers and character-level encodings to bypass safety guardrails. While these guardrails often fail to interpret the encoded content, the underlying models can still process the harmful instructions. We introduce CPT-Filtering, a novel, model-agnostic with negligible-costs and near-perfect accuracy guardrail technique that aims to mitigate these attacks by leveraging the intrinsic behavior of Byte-Pair Encoding (BPE) tokenizers. Our method is based on the principle that tokenizers, trained on natural language, represent out-of-distribution text, such as ciphers, using a significantly higher number of shorter tokens. Our technique uses a simple yet powerful artifact of using language models: the average number of Characters Per Token (CPT) in the text. This approach is motivated by the high compute cost of modern methods - relying on added modules such as dedicated LLMs or perplexity models. We validate our approach across a large dataset of over 100,000 prompts, testing numerous encoding schemes with several popular tokenizers. Our experiments demonstrate that a simple CPT threshold robustly identifies encoded text with high accuracy, even for very short inputs. CPT-Filtering provides a practical defense layer that can be immediately deployed for real-time text filtering and offline data curation. 

**Abstract (ZH)**: 大型语言模型（LLMs）易遭受جا伊布攻击，恶意提示可能通过加密和字符级编码方式进行伪装以绕过安全防护。虽然这些防护措施往往无法解读编码内容，但底层模型仍能处理这些有害指令。我们提出了一种名为CPT-Filtering的新颖、模型无关且低成本、高准确性边栏技术，该技术利用字节对编码（BPE）分词器的内在行为来缓解这些攻击。该方法基于这样的原理：分词器在自然语言上进行训练，会使用显著更多的较短分词来表示分布外文本，如加密文本。我们的方法利用语言模型的一个简单而强大的特性：文本中字符每分词平均数（CPT）。这种方法受到现代方法高昂计算成本的启发——这些方法依赖于专用LLM或困惑度模型等附加模块。我们在超过100,000个提示的大规模数据集上验证了该方法，测试了多种编码方案和多个流行分词器。实验表明，即使对于非常短的输入，简单的CPT阈值也能以高准确性可靠地识别编码文本。CPT-Filtering提供了一种实用的防御层，可以立即部署用于实时文本过滤和离线数据整理。 

---
# CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs 

**Title (ZH)**: CAS-Spec：级联自适应自投机解码以实现大模型即刻无损推理加速 

**Authors**: Zhiyuan Ning, Jiawei Shao, Ruge Xu, Xinfei Guo, Jun Zhang, Chi Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.26843)  

**Abstract**: Speculative decoding has become a widely adopted as an effective technique for lossless inference acceleration when deploying large language models (LLMs). While on-the-fly self-speculative methods offer seamless integration and broad utility, they often fall short of the speed gains achieved by methods relying on specialized training. Cascading a hierarchy of draft models promises further acceleration and flexibility, but the high cost of training multiple models has limited its practical application. In this paper, we propose a novel Cascade Adaptive Self-Speculative Decoding (CAS-Spec) method which constructs speculative draft models by leveraging dynamically switchable inference acceleration (DSIA) strategies, including layer sparsity and activation quantization. Furthermore, traditional vertical and horizontal cascade algorithms are inefficient when applied to self-speculative decoding methods. We introduce a Dynamic Tree Cascade (DyTC) algorithm that adaptively routes the multi-level draft models and assigns the draft lengths, based on the heuristics of acceptance rates and latency prediction. Our CAS-Spec method achieves state-of-the-art acceleration compared to existing on-the-fly speculative decoding methods, with an average speedup from $1.1\times$ to $2.3\times$ over autoregressive decoding across various LLMs and datasets. DyTC improves the average speedup by $47$\% and $48$\% over cascade-based baseline and tree-based baseline algorithms, respectively. CAS-Spec can be easily integrated into most existing LLMs and holds promising potential for further acceleration as self-speculative decoding techniques continue to evolve. 

**Abstract (ZH)**: 基于动态可切换推理加速的级联自推测解码（CAS-Spec）方法 

---
# Accurate Target Privacy Preserving Federated Learning Balancing Fairness and Utility 

**Title (ZH)**: 准确的目标隐私保护联邦学习：平衡公平性和效用 

**Authors**: Kangkang Sun, Jun Wu, Minyi Guo, Jianhua Li, Jianwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.26841)  

**Abstract**: Federated Learning (FL) enables collaborative model training without data sharing, yet participants face a fundamental challenge, e.g., simultaneously ensuring fairness across demographic groups while protecting sensitive client data. We introduce a differentially private fair FL algorithm (\textit{FedPF}) that transforms this multi-objective optimization into a zero-sum game where fairness and privacy constraints compete against model utility. Our theoretical analysis reveals a surprising inverse relationship, i.e., stricter privacy protection fundamentally limits the system's ability to detect and correct demographic biases, creating an inherent tension between privacy and fairness. Counterintuitively, we prove that moderate fairness constraints initially improve model generalization before causing performance degradation, where a non-monotonic relationship that challenges conventional wisdom about fairness-utility tradeoffs. Experimental validation demonstrates up to 42.9 % discrimination reduction across three datasets while maintaining competitive accuracy, but more importantly, reveals that the privacy-fairness tension is unavoidable, i.e., achieving both objectives simultaneously requires carefully balanced compromises rather than optimization of either in isolation. The source code for our proposed algorithm is publicly accessible at this https URL. 

**Abstract (ZH)**: 联邦学习（FL）允许在不共享数据的情况下进行协作模型训练，但参与者面临着一个基本挑战，即在保护敏感客户端数据的同时，确保跨不同人口组的公平性。我们提出了一种基于差分隐私的公平联邦学习算法（FedPF），将多目标优化问题转化为公平性和隐私约束与模型性能之间的零和博弈。我们的理论分析揭示了一个令人意外的反相关关系，即更为严格的隐私保护从根本上限制了系统检测和纠正人口偏差的能力，从而在隐私与公平之间产生了固有的矛盾。出乎意料的是，我们证明了适度的公平性约束最初会改善模型泛化能力，但在后续过程中导致性能下降，形成了一个挑战传统公平性-性能权衡观念的非单调关系。实验验证结果显示，在保持竞争力的同时，该算法在三个数据集上实现了高达42.9%的歧视减少，更重要的是，揭示了隐私-公平性权衡不可避免，即同时实现双重目标需要精心平衡的妥协，而非单独优化任一方。我们所提出的算法的源代码可在以下网址公开访问：this https URL。 

---
# SpotIt: Evaluating Text-to-SQL Evaluation with Formal Verification 

**Title (ZH)**: SpotIt: 使用形式验证评估文本到SQL转换 

**Authors**: Rocky Klopfenstein, Yang He, Andrew Tremante, Yuepeng Wang, Nina Narodytska, Haoze Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26840)  

**Abstract**: Community-driven Text-to-SQL evaluation platforms play a pivotal role in tracking the state of the art of Text-to-SQL performance. The reliability of the evaluation process is critical for driving progress in the field. Current evaluation methods are largely test-based, which involves comparing the execution results of a generated SQL query and a human-labeled ground-truth on a static test database. Such an evaluation is optimistic, as two queries can coincidentally produce the same output on the test database while actually being different. In this work, we propose a new alternative evaluation pipeline, called SpotIt, where a formal bounded equivalence verification engine actively searches for a database that differentiates the generated and ground-truth SQL queries. We develop techniques to extend existing verifiers to support a richer SQL subset relevant to Text-to-SQL. A performance evaluation of ten Text-to-SQL methods on the high-profile BIRD dataset suggests that test-based methods can often overlook differences between the generated query and the ground-truth. Further analysis of the verification results reveals a more complex picture of the current Text-to-SQL evaluation. 

**Abstract (ZH)**: 社区驱动的文本到SQL评估平台在追踪文本到SQL性能的最新状态中发挥着关键作用。评估过程的可靠性对于推动该领域的发展至关重要。当前的评估方法主要是基于测试的方法，涉及将生成的SQL查询的执行结果与人工标注的_ground-truth_在静态测试数据库上的结果进行比较。此类评估可能过于乐观，因为两个查询可能在测试数据库上巧合地产生相同的结果，但实际上却是不同的。在此工作中，我们提出了一种新的替代评估管道，称为SpotIt，其中正式边界等价性验证引擎主动寻找一个能够区分生成的和地标的SQL查询的数据库。我们开发了技术，将现有的验证器扩展以支持与文本到SQL相关的更丰富的SQL子集。对包括高度关注的BIRD数据集在内的十个文本到SQL方法的性能评估表明，基于测试的方法经常忽略生成查询与地标的差异。进一步分析验证结果揭示了当前文本到SQL评估更为复杂的情况。 

---
# Category-Aware Semantic Caching for Heterogeneous LLM Workloads 

**Title (ZH)**: 面向类别感知语义缓存的异构LLM工作负载 

**Authors**: Chen Wang, Xunzhuo Liu, Yue Zhu, Alaa Youssef, Priya Nagpurkar, Huamin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.26835)  

**Abstract**: LLM serving systems process heterogeneous query workloads where different categories exhibit different characteristics. Code queries cluster densely in embedding space while conversational queries distribute sparsely. Content staleness varies from minutes (stock data) to months (code patterns). Query repetition patterns range from power-law (code) to uniform (conversation), producing long tail cache hit rate distributions: high-repetition categories achieve 40-60% hit rates while low-repetition or volatile categories achieve 5-15% hit rates. Vector databases must exclude the long tail because remote search costs (30ms) require 15--20% hit rates to break even, leaving 20-30% of production traffic uncached. Uniform cache policies compound this problem: fixed thresholds cause false positives in dense spaces and miss valid paraphrases in sparse spaces; fixed TTLs waste memory or serve stale data. This paper presents category-aware semantic caching where similarity thresholds, TTLs, and quotas vary by query category. We present a hybrid architecture separating in-memory HNSW search from external document storage, reducing miss cost from 30ms to 2ms. This reduction makes low-hit-rate categories economically viable (break-even at 3-5% versus 15-20%), enabling cache coverage across the entire workload distribution. Adaptive load-based policies extend this framework to respond to downstream model load, dynamically adjusting thresholds and TTLs to reduce traffic to overloaded models by 9-17% in theoretical projections. 

**Abstract (ZH)**: LLM服务系统处理不同类别具有不同特征的异构查询工作负载：代码查询在嵌入空间中紧密聚类，而对话查询则分布稀疏。内容陈旧性从分钟（股票数据）到数月（代码模式）不等。查询重复模式从幂律分布（代码）到均匀分布（对话），产生长尾缓存命中率分布：高重复类别可实现40-60%的命中率，而低重复或易变类别则可实现5-15%的命中率。向量数据库必须排除长尾部分，因为远程搜索成本（30毫秒）要求至少15-20%的命中率才能达到盈亏平衡点，从而留下20-30%的生产流量未被缓存。固定缓存策略加剧了这一问题：固定的阈值在密集空间中导致误报，并在稀疏空间中遗漏有效的同义表达；固定的TTLs要么浪费内存，要么提供过时的数据。本文提出了一种基于类别的语义缓存方法，其中相似性阈值、TTLs和配额根据查询类别而异。我们提出了一种混合架构，将内存中的HNSW搜索与外部文档存储分离，将缺失成本从30毫秒降低到2毫秒。这一减少使低命中率类别在经济上变得可行（盈亏平衡点从15-20%降低到3-5%），从而在整个负载分布范围内提供缓存覆盖。基于负载的自适应策略将这一框架扩展到可动态调整阈值和TTLs以减少对过载模型的流量，理论预测中可减少9-17%的流量。 

---
# Diffusion-Driven Generation of Minimally Preprocessed Brain MRI 

**Title (ZH)**: 最小预处理脑MRI的扩散驱动生成 

**Authors**: Samuel W. Remedios, Aaron Carass, Jerry L. Prince, Blake E. Dewey  

**Link**: [PDF](https://arxiv.org/pdf/2510.26834)  

**Abstract**: The purpose of this study is to present and compare three denoising diffusion probabilistic models (DDPMs) that generate 3D $T_1$-weighted MRI human brain images. Three DDPMs were trained using 80,675 image volumes from 42,406 subjects spanning 38 publicly available brain MRI datasets. These images had approximately 1 mm isotropic resolution and were manually inspected by three human experts to exclude those with poor quality, field-of-view issues, and excessive pathology. The images were minimally preprocessed to preserve the visual variability of the data. Furthermore, to enable the DDPMs to produce images with natural orientation variations and inhomogeneity, the images were neither registered to a common coordinate system nor bias field corrected. Evaluations included segmentation, Frechet Inception Distance (FID), and qualitative inspection. Regarding results, all three DDPMs generated coherent MR brain volumes. The velocity and flow prediction models achieved lower FIDs than the sample prediction model. However, all three models had higher FIDs compared to real images across multiple cohorts. In a permutation experiment, the generated brain regional volume distributions differed statistically from real data. However, the velocity and flow prediction models had fewer statistically different volume distributions in the thalamus and putamen. In conclusion this work presents and releases the first 3D non-latent diffusion model for brain data without skullstripping or registration. Despite the negative results in statistical testing, the presented DDPMs are capable of generating high-resolution 3D $T_1$-weighted brain images. All model weights and corresponding inference code are publicly available at this https URL . 

**Abstract (ZH)**: 本研究的目的是呈现并比较三种生成3D $T_1$加权MRI人类大脑图像的去噪扩散概率模型（DDPMs）。三种DDPMs使用来自42,406个受试者的80,675个图像体素进行了训练，这些图像体素来自38个公开的大脑MRI数据集。这些图像具有约1 mm等向分辨率，并由三位专家手工检查，排除了质量差、视野问题和病理情况过多的图像。图像进行了最少的预处理，以保留数据的视觉变异性。此外，为了使DDPMs能够生成具有自然方向变化和不均匀性的图像，这些图像既未注册到共同坐标系统，也没有进行偏差场校正。评估包括分割、弗雷谢特入渗距离（FID）和定性检查。关于结果，三种DDPMs均生成了连贯的MR大脑体积。速度和流预测模型的FID低于样本预测模型。然而，与真实图像相比，所有三种模型在多个队列中具有较高的FID。在替换实验中，生成的大脑区域体积分布与真实数据存在统计学差异。然而，速度和流预测模型在丘脑和壳核中具有较少的统计学差异的体积分布。总之，本工作呈现并发布了首个不进行颅骨剥离或注册的3D非潜藏扩散模型用于大脑数据。尽管统计测试结果为负，但所呈现的DDPMs能够生成高分辨率的3D $T_1$加权大脑图像。所有模型权重和相应的推理代码均可在该网址公开获取。 

---
# VISAT: Benchmarking Adversarial and Distribution Shift Robustness in Traffic Sign Recognition with Visual Attributes 

**Title (ZH)**: VISAT: 依据视觉属性在交通标志识别中评估对抗性和分布偏移 robustness 的基准测试 

**Authors**: Simon Yu, Peilin Yu, Hongbo Zheng, Huajie Shao, Han Zhao, Lui Sha  

**Link**: [PDF](https://arxiv.org/pdf/2510.26833)  

**Abstract**: We present VISAT, a novel open dataset and benchmarking suite for evaluating model robustness in the task of traffic sign recognition with the presence of visual attributes. Built upon the Mapillary Traffic Sign Dataset (MTSD), our dataset introduces two benchmarks that respectively emphasize robustness against adversarial attacks and distribution shifts. For our adversarial attack benchmark, we employ the state-of-the-art Projected Gradient Descent (PGD) method to generate adversarial inputs and evaluate their impact on popular models. Additionally, we investigate the effect of adversarial attacks on attribute-specific multi-task learning (MTL) networks, revealing spurious correlations among MTL tasks. The MTL networks leverage visual attributes (color, shape, symbol, and text) that we have created for each traffic sign in our dataset. For our distribution shift benchmark, we utilize ImageNet-C's realistic data corruption and natural variation techniques to perform evaluations on the robustness of both base and MTL models. Moreover, we further explore spurious correlations among MTL tasks through synthetic alterations of traffic sign colors using color quantization techniques. Our experiments focus on two major backbones, ResNet-152 and ViT-B/32, and compare the performance between base and MTL models. The VISAT dataset and benchmarking framework contribute to the understanding of model robustness for traffic sign recognition, shedding light on the challenges posed by adversarial attacks and distribution shifts. We believe this work will facilitate advancements in developing more robust models for real-world applications in autonomous driving and cyber-physical systems. 

**Abstract (ZH)**: VISAT：一种面向具有视觉属性交通标志识别任务的新型开源数据集和基准测试套件 

---
# R3GAN-based Optimal Strategy for Augmenting Small Medical Dataset 

**Title (ZH)**: 基于R3GAN的优化策略以扩充小型医疗数据集 

**Authors**: Tsung-Wei Pan, Chang-Hong Wu, Jung-Hua Wang, Ming-Jer Chen, Yu-Chiao Yi, Tsung-Hsien Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.26828)  

**Abstract**: Medical image analysis often suffers from data scarcity and class imbalance, limiting the effectiveness of deep learning models in clinical applications. Using human embryo time-lapse imaging (TLI) as a case study, this work investigates how generative adversarial networks (GANs) can be optimized for small datasets to generate realistic and diagnostically meaningful images. Based on systematic experiments with R3GAN, we established effective training strategies and designed an optimized configuration for 256x256-resolution datasets, featuring a full burn-in phase and a low, gradually increasing gamma range (5 -> 40). The generated samples were used to balance an imbalanced embryo dataset, leading to substantial improvement in classification performance. The recall and F1-score of t3 increased from 0.06 to 0.69 and 0.11 to 0.60, respectively, without compromising other classes. These results demonstrate that tailored R3GAN training strategies can effectively alleviate data scarcity and improve model robustness in small-scale medical imaging tasks. 

**Abstract (ZH)**: 医学图像分析往往受到数据稀缺性和类别不平衡的限制，这限制了深度学习模型在临床应用中的有效性。以人类胚胎时间 lapse 成像（TLI）为例，本工作探讨了如何通过生成对抗网络（GANs）优化小数据集以生成具有诊断意义的现实图像。基于R3GAN的系统实验，我们制定了有效的训练策略，并为256x256分辨率的数据集设计了优化配置，包括完整的预热阶段和低、逐渐增加的伽马范围（5 -> 40）。生成的样本用于平衡不平衡的胚胎数据集，显著提高了分类性能。t3的召回率和F1分数分别从0.06提高到0.69和0.11提高到0.60，而其他类别的性能保持不变。这些结果表明，定制的R3GAN训练策略可以有效地缓解数据稀缺性并提高小型医学成像任务中模型的鲁棒性。 

---
# LeMat-Synth: a multi-modal toolbox to curate broad synthesis procedure databases from scientific literature 

**Title (ZH)**: LeMat-Synth：多模态工具箱，用于从科学文献中整理广泛的合成程序数据库 

**Authors**: Magdalena Lederbauer, Siddharth Betala, Xiyao Li, Ayush Jain, Amine Sehaba, Georgia Channing, Grégoire Germain, Anamaria Leonescu, Faris Flaifil, Alfonso Amayuelas, Alexandre Nozadze, Stefan P. Schmid, Mohd Zaki, Sudheesh Kumar Ethirajan, Elton Pan, Mathilde Franckel, Alexandre Duval, N. M. Anoop Krishnan, Samuel P. Gleason  

**Link**: [PDF](https://arxiv.org/pdf/2510.26824)  

**Abstract**: The development of synthesis procedures remains a fundamental challenge in materials discovery, with procedural knowledge scattered across decades of scientific literature in unstructured formats that are challenging for systematic analysis. In this paper, we propose a multi-modal toolbox that employs large language models (LLMs) and vision language models (VLMs) to automatically extract and organize synthesis procedures and performance data from materials science publications, covering text and figures. We curated 81k open-access papers, yielding LeMat-Synth (v 1.0): a dataset containing synthesis procedures spanning 35 synthesis methods and 16 material classes, structured according to an ontology specific to materials science. The extraction quality is rigorously evaluated on a subset of 2.5k synthesis procedures through a combination of expert annotations and a scalable LLM-as-a-judge framework. Beyond the dataset, we release a modular, open-source software library designed to support community-driven extension to new corpora and synthesis domains. Altogether, this work provides an extensible infrastructure to transform unstructured literature into machine-readable information. This lays the groundwork for predictive modeling of synthesis procedures as well as modeling synthesis--structure--property relationships. 

**Abstract (ZH)**: 材料科学合成程序的开发仍然是材料发现中的一个基本挑战，相关知识散落于数十年的科学文献中，以无结构格式存在，难以系统分析。本文提出一个多模态工具箱，利用大规模语言模型（LLMs）和视觉语言模型（VLMs）自动从材料科学出版物中提取和组织合成程序和性能数据，涵盖文本和图表。我们整理了81,000篇开放获取论文，生成了LeMat-Synth（v 1.0）：一个包含35种合成方法和16种材料类别合成程序的数据集，按照特定于材料科学的本体进行结构化。通过专家注解和可扩展的LLM作为法官框架，对2,500条合成程序的子集进行了提取质量的严格评估。除了数据集，我们还发布了模块化开源软件库，旨在支持社区驱动的新语料库和合成领域的扩展。整体而言，这项工作提供了一个可扩展的基础设施，将无结构文献转化为可机器读取的信息。这为合成程序的预测建模以及合成-结构-性能关系的建模奠定了基础。 

---
# Cross-Corpus Validation of Speech Emotion Recognition in Urdu using Domain-Knowledge Acoustic Features 

**Title (ZH)**: 使用领域知识声学特征的乌尔都语语音情感识别跨语料库验证 

**Authors**: Unzela Talpur, Zafi Sherhan Syed, Muhammad Shehram Shah Syed, Abbas Shah Syed  

**Link**: [PDF](https://arxiv.org/pdf/2510.26823)  

**Abstract**: Speech Emotion Recognition (SER) is a key affective computing technology that enables emotionally intelligent artificial intelligence. While SER is challenging in general, it is particularly difficult for low-resource languages such as Urdu. This study investigates Urdu SER in a cross-corpus setting, an area that has remained largely unexplored. We employ a cross-corpus evaluation framework across three different Urdu emotional speech datasets to test model generalization. Two standard domain-knowledge based acoustic feature sets, eGeMAPS and ComParE, are used to represent speech signals as feature vectors which are then passed to Logistic Regression and Multilayer Perceptron classifiers. Classification performance is assessed using unweighted average recall (UAR) whilst considering class-label imbalance. Results show that Self-corpus validation often overestimates performance, with UAR exceeding cross-corpus evaluation by up to 13%, underscoring that cross-corpus evaluation offers a more realistic measure of model robustness. Overall, this work emphasizes the importance of cross-corpus validation for Urdu SER and its implications contribute to advancing affective computing research for underrepresented language communities. 

**Abstract (ZH)**: 乌尔都语语音情感识别中的跨语料库研究 

---
# See the Speaker: Crafting High-Resolution Talking Faces from Speech with Prior Guidance and Region Refinement 

**Title (ZH)**: 基于先验引导和区域精炼的高分辨率说话人脸生成 

**Authors**: Jinting Wang, Jun Wang, Hei Victor Cheng, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26819)  

**Abstract**: Unlike existing methods that rely on source images as appearance references and use source speech to generate motion, this work proposes a novel approach that directly extracts information from the speech, addressing key challenges in speech-to-talking face. Specifically, we first employ a speech-to-face portrait generation stage, utilizing a speech-conditioned diffusion model combined with statistical facial prior and a sample-adaptive weighting module to achieve high-quality portrait generation. In the subsequent speech-driven talking face generation stage, we embed expressive dynamics such as lip movement, facial expressions, and eye movements into the latent space of the diffusion model and further optimize lip synchronization using a region-enhancement module. To generate high-resolution outputs, we integrate a pre-trained Transformer-based discrete codebook with an image rendering network, enhancing video frame details in an end-to-end manner. Experimental results demonstrate that our method outperforms existing approaches on the HDTF, VoxCeleb, and AVSpeech datasets. Notably, this is the first method capable of generating high-resolution, high-quality talking face videos exclusively from a single speech input. 

**Abstract (ZH)**: 不同于现有的方法依赖源图像作为外观参考并使用源语音生成运动，本工作提出了一种新颖的方法，直接从语音中提取信息，以解决语音到 talking face 的关键挑战。具体而言，我们首先采用语音到人脸肖像生成阶段，利用语音条件下的扩散模型结合统计面部先验和样本自适应加权模块，实现高质量肖像生成。在后续的语音驱动 talking face 生成阶段，我们将表达性动态（如唇动、面部表情和眼动）嵌入到扩散模型的潜在空间中，并进一步使用区域增强模块优化唇部同步。为了生成高分辨率输出，我们结合了预训练的基于 Transformer 的离散码本和图像渲染网络，以端到端的方式增强视频帧细节。实验结果表明，与 HDTF、VoxCeleb 和 AVSpeech 数据集上的现有方法相比，本方法表现更优。特别地，这是首个仅从单个语音输入生成高分辨率、高质量 talking face 视频的方法。 

---
# GACA-DiT: Diffusion-based Dance-to-Music Generation with Genre-Adaptive Rhythm and Context-Aware Alignment 

**Title (ZH)**: 基于扩散的跨风格节奏适应与上下文感知对齐的舞蹈音乐生成方法：GACA-DiT 

**Authors**: Jinting Wang, Chenxing Li, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.26818)  

**Abstract**: Dance-to-music (D2M) generation aims to automatically compose music that is rhythmically and temporally aligned with dance movements. Existing methods typically rely on coarse rhythm embeddings, such as global motion features or binarized joint-based rhythm values, which discard fine-grained motion cues and result in weak rhythmic alignment. Moreover, temporal mismatches introduced by feature downsampling further hinder precise synchronization between dance and music. To address these problems, we propose \textbf{GACA-DiT}, a diffusion transformer-based framework with two novel modules for rhythmically consistent and temporally aligned music generation. First, a \textbf{genre-adaptive rhythm extraction} module combines multi-scale temporal wavelet analysis and spatial phase histograms with adaptive joint weighting to capture fine-grained, genre-specific rhythm patterns. Second, a \textbf{context-aware temporal alignment} module resolves temporal mismatches using learnable context queries to align music latents with relevant dance rhythm features. Extensive experiments on the AIST++ and TikTok datasets demonstrate that GACA-DiT outperforms state-of-the-art methods in both objective metrics and human evaluation. Project page: this https URL. 

**Abstract (ZH)**: 舞蹈伴随音乐（D2M）生成旨在自动生成与舞蹈动作在节奏和时间上匹配的音乐。现有的方法通常依赖粗粒度的节奏嵌入，如全局运动特征或基于关节的二值化节奏值，这些方法忽略了细粒度的运动线索并导致节奏对齐较弱。此外，特征下采样引入的时间不匹配进一步阻碍了舞蹈和音乐的精确同步。为了解决这些问题，我们提出了基于扩散变换器的GACA-DiT框架，该框架包含两个新型模块以实现节奏一致性和时间对齐的音乐生成。首先，一个自适应节奏提取模块结合多尺度时域小波分析和自适应关节加权以及空间相位直方图，以捕获细粒度、特定于曲风的节奏模式。其次，一个上下文感知的时间对齐模块使用可学习的上下文查询来解决时间不匹配问题，将音乐潜在特征与相关的舞蹈节奏特征对齐。在AIST++和TikTok数据集上的广泛实验表明，GACA-DiT在客观指标和人工评估方面均优于现有方法。项目页面: this https URL。 

---
# Systematic Absence of Low-Confidence Nighttime Fire Detections in VIIRS Active Fire Product: Evidence of Undocumented Algorithmic Filtering 

**Title (ZH)**: VIIRS活性火产品中低置信度夜间火灾检测系统性缺失的证据：未记录的算法过滤现象 

**Authors**: Rohit Rajendra Dhage  

**Link**: [PDF](https://arxiv.org/pdf/2510.26816)  

**Abstract**: The Visible Infrared Imaging Radiometer Suite (VIIRS) active fire product is widely used for global fire monitoring, yet its confidence classification scheme exhibits an undocumented systematic pattern. Through analysis of 21,540,921 fire detections spanning one year (January 2023 - January 2024), I demonstrate a complete absence of low-confidence classifications during nighttime observations. Of 6,007,831 nighttime fires, zero were classified as low confidence, compared to an expected 696,908 under statistical independence (chi-squared = 1,474,795, p < 10^-15, Z = -833). This pattern persists globally across all months, latitude bands, and both NOAA-20 and Suomi-NPP satellites. Machine learning reverse-engineering (88.9% accuracy), bootstrap simulation (1,000 iterations), and spatial-temporal analysis confirm this is an algorithmic constraint rather than a geophysical phenomenon. Brightness temperature analysis reveals nighttime fires below approximately 295K are likely excluded entirely rather than flagged as low-confidence, while daytime fires show normal confidence distributions. This undocumented behavior affects 27.9% of all VIIRS fire detections and has significant implications for fire risk assessment, day-night detection comparisons, confidence-weighted analyses, and any research treating confidence levels as uncertainty metrics. I recommend explicit documentation of this algorithmic constraint in VIIRS user guides and reprocessing strategies for affected analyses. 

**Abstract (ZH)**: VIIRS可见红外成像辐射计套件主动火产品存在未记录的系统性模式：夜间观测中低置信度分类完全缺失及其影响 

---
# Impact of clinical decision support systems (cdss) on clinical outcomes and healthcare delivery in low- and middle-income countries: protocol for a systematic review and meta-analysis 

**Title (ZH)**: 临床决策支持系统（CDSS）对低收入和中等收入国家临床结果和医疗保健交付的影响：系统评价和meta分析的方案 

**Authors**: Garima Jain, Anand Bodade, Sanghamitra Pati  

**Link**: [PDF](https://arxiv.org/pdf/2510.26812)  

**Abstract**: Clinical decision support systems (CDSS) are used to improve clinical and service outcomes, yet evidence from low- and middle-income countries (LMICs) is dispersed. This protocol outlines methods to quantify the impact of CDSS on patient and healthcare delivery outcomes in LMICs. We will include comparative quantitative designs (randomized trials, controlled before-after, interrupted time series, comparative cohorts) evaluating CDSS in World Bank-defined LMICs. Standalone qualitative studies are excluded; mixed-methods studies are eligible only if they report comparative quantitative outcomes, for which we will extract the quantitative component. Searches (from inception to 30 September 2024) will cover MEDLINE, Embase, CINAHL, CENTRAL, Web of Science, Global Health, Scopus, IEEE Xplore, LILACS, African Index Medicus, and IndMED, plus grey sources. Screening and extraction will be performed in duplicate. Risk of bias will be assessed with RoB 2 (randomized trials) and ROBINS-I (non-randomized). Random-effects meta-analysis will be performed where outcomes are conceptually or statistically comparable; otherwise, a structured narrative synthesis will be presented. Heterogeneity will be explored using relative and absolute metrics and a priori subgroups or meta-regression (condition area, care level, CDSS type, readiness proxies, study design). 

**Abstract (ZH)**: 临床决策支持系统（CDSS）在改善临床和服务结果方面得到应用，然而低收入和中等收入国家（LMICs）的相关证据分散。本研究方案概述了量化CDSS对LMICs患者和医疗服务结果影响的方法。我们将包括世界银行定义的LMICs中评估CDSS的比较定量设计（随机试验、控制前后设计、中断时间序列、比较队列研究）。独立定性研究将被排除；只有报告比较定量结果的混合方法研究才符合条件，我们将提取其中的定量部分。文献检索（从创刊号到2024年9月30日）将覆盖MEDLINE、Embase、CINAHL、CENTRAL、Web of Science、Global Health、Scopus、IEEE Xplore、LILACS、African Index Medicus和IndMED，以及灰色文献。筛查和数据提取将进行双重操作。偏倚风险将使用RoB 2（随机试验）和ROBINS-I（非随机试验）评估。当结果在概念或统计上可比时，将进行随机效应元分析；否则，将呈现结构化的综述合成。异质性将通过相对和绝对指标以及先验子组或meta回归（条件领域、护理级别、CDSS类型、准备度指标、研究设计）进行探索。 

---
# Reinforcement Learning for Accelerator Beamline Control: a simulation-based approach 

**Title (ZH)**: 基于仿真的一种加速器束线控制的强化学习方法 

**Authors**: Anwar Ibrahim, Alexey Petrenko, Maxim Kaledin, Ehab Suleiman, Fedor Ratnikov, Denis Derkach  

**Link**: [PDF](https://arxiv.org/pdf/2510.26805)  

**Abstract**: Particle accelerators play a pivotal role in advancing scientific research, yet optimizing beamline configurations to maximize particle transmission remains a labor-intensive task requiring expert intervention. In this work, we introduce RLABC (Reinforcement Learning for Accelerator Beamline Control), a Python-based library that reframes beamline optimization as a reinforcement learning (RL) problem. Leveraging the Elegant simulation framework, RLABC automates the creation of an RL environment from standard lattice and element input files, enabling sequential tuning of magnets to minimize particle losses. We define a comprehensive state representation capturing beam statistics, actions for adjusting magnet parameters, and a reward function focused on transmission efficiency. Employing the Deep Deterministic Policy Gradient (DDPG) algorithm, we demonstrate RLABC's efficacy on two beamlines, achieving transmission rates of 94% and 91%, comparable to expert manual optimizations. This approach bridges accelerator physics and machine learning, offering a versatile tool for physicists and RL researchers alike to streamline beamline tuning. 

**Abstract (ZH)**: 粒子加速器在推动科学研究中发挥着关键作用，然而，优化束线配置以最大化粒子传输效率仍是一项劳-intensive的任务，需要专家干预。本文介绍了一种基于Python的RLABC（Reinforcement Learning for Accelerator Beamline Control）库，将束线优化重新定义为一个强化学习（RL）问题。借助Elegant仿真框架，RLABC自动从标准束线和元件输入文件创建RL环境，实现对磁铁的顺序调谐以最小化粒子损失。我们定义了一个全面的状态表示，包括束流统计、调整磁铁参数的动作以及专注于传输效率的奖励函数。采用深度确定性策略梯度（DDPG）算法，我们在两个束线上展示了RLABC的有效性，分别实现了94%和91%的传输率，与专家手动优化相当。该方法将加速器物理与机器学习相结合，为物理学家和RL研究人员提供了一个灵活的工具，用于简化束线调谐。 

---
# EARS-UDE: Evaluating Auditory Response in Sensory Overload with Universal Differential Equations 

**Title (ZH)**: EARS-UDE: 评估感官超载中听觉反应的通用微分方程方法 

**Authors**: Miheer Salunke, Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2510.26804)  

**Abstract**: Auditory sensory overload affects 50-70% of individuals with Autism Spectrum Disorder (ASD), yet existing approaches, such as mechanistic models (Hodgkin Huxley type, Wilson Cowan, excitation inhibition balance), clinical tools (EEG/MEG, Sensory Profile scales), and ML methods (Neural ODEs, predictive coding), either assume fixed parameters or lack interpretability, missing autism heterogeneity. We present a Scientific Machine Learning approach using Universal Differential Equations (UDEs) to model sensory adaptation dynamics in autism. Our framework combines ordinary differential equations grounded in biophysics with neural networks to capture both mechanistic understanding and individual variability. We demonstrate that UDEs achieve a 90.8% improvement over pure Neural ODEs while using 73.5% fewer parameters. The model successfully recovers physiological parameters within the 2% error and provides a quantitative risk assessment for sensory overload, predicting 17.2% risk for pulse stimuli with specific temporal patterns. This framework establishes foundations for personalized, evidence-based interventions in autism, with direct applications to wearable technology and clinical practice. 

**Abstract (ZH)**: 听觉感觉过载影响自闭症谱系障碍（ASD）个体的50-70%，现有方法如机制模型（Hodgkin Huxley类型、Wilson Cowan模型、兴奋与抑制平衡）、临床工具（EEG/MEG、感觉量表）及机器学习方法（神经ODEs、预测编码）要么假设固定参数，要么缺乏解释性，未能捕捉自闭症异质性。我们提出了一种使用通用微分方程（UDEs）的科学机器学习方法，以建模自闭症的感觉适应动力学。该框架结合了源自生物物理的常微分方程和神经网络，以捕捉机制理解与个体变异性。研究表明，UDEs在参数减少73.5%的情况下，相比纯神经ODEs实现了90.8%的性能提升。该模型成功地在2%误差范围内恢复了生理参数，并提供了感觉过载的定量风险评估，预测特定时间模式的脉冲刺激有17.2%的风险。该框架为自闭症个性化、证据为基础的干预奠定了基础，并直接应用于可穿戴技术及临床实践。 

---
# VeriStruct: AI-assisted Automated Verification of Data-Structure Modules in Verus 

**Title (ZH)**: VeriStruct: AI辅助的数据结构模块自动验证在Verus中的应用 

**Authors**: Chuyue Sun, Yican Sun, Daneshvar Amrollahi, Ethan Zhang, Shuvendu Lahiri, Shan Lu, David Dill, Clark Barrett  

**Link**: [PDF](https://arxiv.org/pdf/2510.25015)  

**Abstract**: We introduce VeriStruct, a novel framework that extends AI-assisted automated verification from single functions to more complex data structure modules in Verus. VeriStruct employs a planner module to orchestrate the systematic generation of abstractions, type invariants, specifications, and proof code. To address the challenge that LLMs often misunderstand Verus' annotation syntax and verification-specific semantics, VeriStruct embeds syntax guidance within prompts and includes a repair stage to automatically correct annotation errors. In an evaluation on eleven Rust data structure modules, VeriStruct succeeds on ten of the eleven, successfully verifying 128 out of 129 functions (99.2%) in total. These results represent an important step toward the goal of automatic AI-assisted formal verification. 

**Abstract (ZH)**: 我们介绍了VeriStruct，这是一种新型框架，它将AI辅助自动化验证从单个函数扩展到Verus中更复杂的数据结构模块。VeriStruct采用规划器模块来协调抽象、类型不变式、规范和证明代码的系统生成。为了应对LLMs常误解Verus注解语法和验证特定语义的挑战，VeriStruct在提示中嵌入了语法指导，并包含一个修复阶段以自动纠正注解错误。在对 eleven 个 Rust 数据结构模块的评估中，VeriStruct 在 eleven 个模块中有十个成功，总共成功验证了 129 个函数中的 128 个（99.2%）。这些结果代表了朝着自动AI辅助形式化验证目标迈出的重要一步。 

---
# Detecting Prefix Bias in LLM-based Reward Models 

**Title (ZH)**: 基于LLM的奖励模型中的前缀偏见检测 

**Authors**: Ashwin Kumar, Yuzi He, Aram H. Markosyan, Bobbie Chern, Imanol Arrieta-Ibarra  

**Link**: [PDF](https://arxiv.org/pdf/2505.13487)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has emerged as a key paradigm for task-specific fine-tuning of language models using human preference data. While numerous publicly available preference datasets provide pairwise comparisons of responses, the potential for biases in the resulting reward models remains underexplored. In this work, we introduce novel methods to detect and evaluate prefix bias -- a systematic shift in model preferences triggered by minor variations in query prefixes -- in LLM-based reward models trained on such datasets. We leverage these metrics to reveal significant biases in preference models across racial and gender dimensions. Our comprehensive evaluation spans diverse open-source preference datasets and reward model architectures, demonstrating susceptibility to this kind of bias regardless of the underlying model architecture. Furthermore, we propose a data augmentation strategy to mitigate these biases, showing its effectiveness in reducing the impact of prefix bias. Our findings highlight the critical need for bias-aware dataset design and evaluation in developing fair and reliable reward models, contributing to the broader discourse on fairness in AI. 

**Abstract (ZH)**: 基于人类反馈的强化学习（RLHF）已成为使用人类偏好数据对语言模型进行任务特定微调的关键范式。虽然众多公开可用的偏好数据集提供了响应的成对比较，但结果奖励模型中潜在的偏差仍需进一步探索。在本工作中，我们介绍了用于检测和评估前缀偏差（由于查询前缀的微小变化而引起模型偏好系统性偏移）的新方法。我们利用这些指标揭示了偏好模型在种族和性别维度上的显著偏差。我们的全面评估涵盖了多种开源偏好数据集和奖励模型架构，证明了无论基础模型架构如何，这种偏差的普适性。此外，我们提出了一种数据增强策略来减轻这些偏差，并展示了其在减少前缀偏差影响方面的有效性。我们的发现强调了在开发公平可靠的奖励模型时关注偏差的必要性，为AI公平性更广泛的讨论做出贡献。 

---
# A Transformer-based Neural Architecture Search Method 

**Title (ZH)**: 基于Transformer的神经架构搜索方法 

**Authors**: Shang Wang, Huanrong Tang, Jianquan Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01314)  

**Abstract**: This paper presents a neural architecture search method based on Transformer architecture, searching cross multihead attention computation ways for different number of encoder and decoder combinations. In order to search for neural network structures with better translation results, we considered perplexity as an auxiliary evaluation metric for the algorithm in addition to BLEU scores and iteratively improved each individual neural network within the population by a multi-objective genetic algorithm. Experimental results show that the neural network structures searched by the algorithm outperform all the baseline models, and that the introduction of the auxiliary evaluation metric can find better models than considering only the BLEU score as an evaluation metric. 

**Abstract (ZH)**: 基于Transformer架构的神经架构搜索方法：跨多头注意力机制的编码器-解码器组合搜索 

---
# A Neural Architecture Search Method using Auxiliary Evaluation Metric based on ResNet Architecture 

**Title (ZH)**: 基于ResNet架构的辅助评价指标神经架构搜索方法 

**Authors**: Shang Wang, Huanrong Tang, Jianquan Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2505.01313)  

**Abstract**: This paper proposes a neural architecture search space using ResNet as a framework, with search objectives including parameters for convolution, pooling, fully connected layers, and connectivity of the residual network. In addition to recognition accuracy, this paper uses the loss value on the validation set as a secondary objective for optimization. The experimental results demonstrate that the search space of this paper together with the optimisation approach can find competitive network architectures on the MNIST, Fashion-MNIST and CIFAR100 datasets. 

**Abstract (ZH)**: 本文提出了一种以ResNet为准架构的神经架构搜索空间，搜索目标包括卷积参数、池化参数、全连接层参数以及残差网络的连接性。除了识别精度外，本文还将验证集的损失值作为优化的次要目标。实验结果表明，本文提出的搜索空间与优化方法可以在MNIST、Fashion-MNIST和CIFAR100数据集上找到具有竞争力的网络架构。 

---
