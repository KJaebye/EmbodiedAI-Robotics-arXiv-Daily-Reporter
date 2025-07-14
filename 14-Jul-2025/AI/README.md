# System-of-systems Modeling and Optimization: An Integrated Framework for Intermodal Mobility 

**Title (ZH)**: 系统-of-系统建模与优化：一种多式联运 Mobility 整合框架 

**Authors**: Paul Saves, Jasper Bussemaker, Rémi Lafage, Thierry Lefebvre, Nathalie Bartoli, Youssef Diouane, Joseph Morlier  

**Link**: [PDF](https://arxiv.org/pdf/2507.08715)  

**Abstract**: For developing innovative systems architectures, modeling and optimization techniques have been central to frame the architecting process and define the optimization and modeling problems. In this context, for system-of-systems the use of efficient dedicated approaches (often physics-based simulations) is highly recommended to reduce the computational complexity of the targeted applications. However, exploring novel architectures using such dedicated approaches might pose challenges for optimization algorithms, including increased evaluation costs and potential failures. To address these challenges, surrogate-based optimization algorithms, such as Bayesian optimization utilizing Gaussian process models have emerged. 

**Abstract (ZH)**: 开发创新型系统架构时，建模与优化技术是框架 architects 过程和定义优化与建模问题的核心。在这种背景下，对于系统-of-系统来说，使用高效的专用方法（通常是基于物理的模拟）来降低目标应用的计算复杂性是高度推荐的。然而，使用此类专用方法探索新型架构可能会给优化算法带来挑战，包括增加评估成本和潜在的失败。为此，基于代理模型的优化算法，如利用高斯过程模型的贝叶斯优化，已逐渐兴起。 

---
# elsciRL: Integrating Language Solutions into Reinforcement Learning Problem Settings 

**Title (ZH)**: elsciRL: 将语言解决方案集成到强化学习问题设置中 

**Authors**: Philip Osborne, Danilo S. Carvalho, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2507.08705)  

**Abstract**: We present elsciRL, an open-source Python library to facilitate the application of language solutions on reinforcement learning problems. We demonstrate the potential of our software by extending the Language Adapter with Self-Completing Instruction framework defined in (Osborne, 2024) with the use of LLMs. Our approach can be re-applied to new applications with minimal setup requirements. We provide a novel GUI that allows a user to provide text input for an LLM to generate instructions which it can then self-complete. Empirical results indicate that these instructions \textit{can} improve a reinforcement learning agent's performance. Therefore, we present this work to accelerate the evaluation of language solutions on reward based environments to enable new opportunities for scientific discovery. 

**Abstract (ZH)**: 我们介绍了一个开源Python库elsciRL，用于促进语言解决方案在强化学习问题中的应用。我们通过在Osborne (2024) 中定义的语言适配器与自我完成指令框架中加入LLM，展示了该软件的潜力。我们的方法在新应用中只需最少的设置要求就可以重新应用。我们提供了一个新型的GUI，允许用户输入文本以供LLM生成指令，然后自我完成这些指令。实证结果表明，这些指令可以提高强化学习代理的性能。因此，我们提出这项工作以加速基于奖励的环境中的语言解决方案的评估，以促进新的科学发现机会。 

---
# Introspection of Thought Helps AI Agents 

**Title (ZH)**: 反思思维有助于AI代理 

**Authors**: Haoran Sun, Shaoning Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08664)  

**Abstract**: AI Agents rely on Large Language Models (LLMs) and Multimodal-LLMs (MLLMs) to perform interpretation and inference in text and image tasks without post-training, where LLMs and MLLMs play the most critical role and determine the initial ability and limitations of AI Agents. Usually, AI Agents utilize sophisticated prompt engineering and external reasoning framework to obtain a promising interaction with LLMs, e.g., Chain-of-Thought, Iteration of Thought and Image-of-Thought. However, they are still constrained by the inherent limitations of LLM in understanding natural language, and the iterative reasoning process will generate a large amount of inference cost. To this end, we propose a novel AI Agent Reasoning Framework with Introspection of Thought (INoT) by designing a new LLM-Read code in prompt. It enables LLM to execute programmatic dialogue reasoning processes following the code in prompt. Therefore, self-denial and reflection occur within LLM instead of outside LLM, which can reduce token cost effectively. Through our experiments on six benchmarks for three different tasks, the effectiveness of INoT is verified, with an average improvement of 7.95\% in performance, exceeding the baselines. Furthermore, the token cost of INoT is lower on average than the best performing method at baseline by 58.3\%. In addition, we demonstrate the versatility of INoT in image interpretation and inference through verification experiments. 

**Abstract (ZH)**: 基于内部反思的思想推理AI代理框架（INoT） 

---
# Leanabell-Prover-V2: Verifier-integrated Reasoning for Formal Theorem Proving via Reinforcement Learning 

**Title (ZH)**: Leanabell-Prover-V2: 验证器集成的强化学习形式定理证明推理 

**Authors**: Xingguang Ji, Yahui Liu, Qi Wang, Jingyuan Zhang, Yang Yue, Rui Shi, Chenxi Sun, Fuzheng Zhang, Guorui Zhou, Kun Gai  

**Link**: [PDF](https://arxiv.org/pdf/2507.08649)  

**Abstract**: We introduce our Leanabell-Prover-V2, a 7B large language models (LLMs) that can produce formal theorem proofs in Lean 4, with verifier-integrated Long Chain-of-Thoughts (CoT). Following our previous work Leanabell-Prover-V1, we continual to choose to posttrain existing strong prover models for further performance improvement. In our V2 version, we mainly upgrade the Reinforcement Learning (RL) with feedback provided by the Lean 4 verifier. Crucially, verifier feedback, such as indicating success or detailing specific errors, allows the LLM to become ``self-aware'' of the correctness of its own reasoning process and learn to reflexively correct errors. Leanabell-Prover-V2 directly optimizes LLM reasoning trajectories with multi-turn verifier interactions, together with feedback token masking for stable RL training and a simple reward strategy. Experiments show that Leanabell-Prover-V2 improves performance by 3.2% (pass@128) with Kimina-Prover-Preview-Distill-7B and 2.0% (pass@128) with DeepSeek-Prover-V2-7B on the MiniF2F test set. The source codes, curated data and models are available at: this https URL. 

**Abstract (ZH)**: Leanabell-Prover-V2: 一个集成验证器的7B大型语言模型及其在Lean 4中的形式定理证明 

---
# Agentic Large Language Models for Conceptual Systems Engineering and Design 

**Title (ZH)**: 代理型大型语言模型在概念系统工程与设计中的应用 

**Authors**: Soheyl Massoudi, Mark Fuge  

**Link**: [PDF](https://arxiv.org/pdf/2507.08619)  

**Abstract**: Early-stage engineering design involves complex, iterative reasoning, yet existing large language model (LLM) workflows struggle to maintain task continuity and generate executable models. We evaluate whether a structured multi-agent system (MAS) can more effectively manage requirements extraction, functional decomposition, and simulator code generation than a simpler two-agent system (2AS). The target application is a solar-powered water filtration system as described in a cahier des charges. We introduce the Design-State Graph (DSG), a JSON-serializable representation that bundles requirements, physical embodiments, and Python-based physics models into graph nodes. A nine-role MAS iteratively builds and refines the DSG, while the 2AS collapses the process to a Generator-Reflector loop. Both systems run a total of 60 experiments (2 LLMs - Llama 3.3 70B vs reasoning-distilled DeepSeek R1 70B x 2 agent configurations x 3 temperatures x 5 seeds). We report a JSON validity, requirement coverage, embodiment presence, code compatibility, workflow completion, runtime, and graph size. Across all runs, both MAS and 2AS maintained perfect JSON integrity and embodiment tagging. Requirement coverage remained minimal (less than 20\%). Code compatibility peaked at 100\% under specific 2AS settings but averaged below 50\% for MAS. Only the reasoning-distilled model reliably flagged workflow completion. Powered by DeepSeek R1 70B, the MAS generated more granular DSGs (average 5-6 nodes) whereas 2AS mode-collapsed. Structured multi-agent orchestration enhanced design detail. Reasoning-distilled LLM improved completion rates, yet low requirements and fidelity gaps in coding persisted. 

**Abstract (ZH)**: 结构化多agent系统在早期工程设计中的有效性研究：基于需求提取、功能分解和模拟器代码生成的比较 

---
# Unlocking Speech Instruction Data Potential with Query Rewriting 

**Title (ZH)**: 利用查询重写解锁语音指令数据潜力 

**Authors**: Yonghua Hei, Yibo Yan, Shuliang Liu, Huiyu Zhou, Linfeng Zhang, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08603)  

**Abstract**: End-to-end Large Speech Language Models~(\textbf{LSLMs}) demonstrate strong potential in response latency and speech comprehension capabilities, showcasing general intelligence across speech understanding tasks. However, the ability to follow speech instructions has not been fully realized due to the lack of datasets and heavily biased training tasks. Leveraging the rich ASR datasets, previous approaches have used Large Language Models~(\textbf{LLMs}) to continue the linguistic information of speech to construct speech instruction datasets. Yet, due to the gap between LLM-generated results and real human responses, the continuation methods further amplify these shortcomings. Given the high costs of collecting and annotating speech instruction datasets by humans, using speech synthesis to construct large-scale speech instruction datasets has become a balanced and robust alternative. Although modern Text-To-Speech~(\textbf{TTS}) models have achieved near-human-level synthesis quality, it is challenging to appropriately convert out-of-distribution text instruction to speech due to the limitations of the training data distribution in TTS models. To address this issue, we propose a query rewriting framework with multi-LLM knowledge fusion, employing multiple agents to annotate and validate the synthesized speech, making it possible to construct high-quality speech instruction datasets without relying on human annotation. Experiments show that this method can transform text instructions into distributions more suitable for TTS models for speech synthesis through zero-shot rewriting, increasing data usability from 72\% to 93\%. It also demonstrates unique advantages in rewriting tasks that require complex knowledge and context-related abilities. 

**Abstract (ZH)**: 端到端大型语音语言模型（LSLMs）在响应延迟和语音理解能力方面表现出强大潜力，展示了在语音理解任务中的一般智能。然而，由于缺乏数据集和高度偏向的训练任务，沿袭语音指令的能力尚未完全实现。利用丰富的语音识别（ASR）数据集，之前的Approaches使用大型语言模型（LLMs）延续语音的语言信息以构建语音指令数据集。但由于LLM生成结果与真实人类响应之间的差距，延续方法进一步放大了这些缺点。鉴于人工收集和标注语音指令数据集的高成本，使用语音合成构建大规模语音指令数据集已成为一种平衡且稳健的替代方案。尽管现代文本到语音（TTS）模型在合成质量上已达近人类水平，但因TTS模型训练数据分布的限制，将非分布外的文本指令适当地转换为语音颇具挑战性。为解决这一问题，我们提出一种多LLM知识融合的查询重写框架，通过多个代理标注和验证合成语音，使其成为无需依赖人工标注即可构建高质量语音指令数据集的方法。实验结果显示，此方法可以通过零样本重写将文本指令转换为更适配TTS模型的分布，从而提高数据可利用率从72%提升到93%。同时，该方法在需要复杂知识和上下文相关能力的重写任务中表现出独特优势。 

---
# Large Multi-modal Model Cartographic Map Comprehension for Textual Locality Georeferencing 

**Title (ZH)**: 大型多模态模型地图理解在文本局部地理配准中的应用 

**Authors**: Kalana Wijegunarathna, Kristin Stock, Christopher B. Jones  

**Link**: [PDF](https://arxiv.org/pdf/2507.08575)  

**Abstract**: Millions of biological sample records collected in the last few centuries archived in natural history collections are un-georeferenced. Georeferencing complex locality descriptions associated with these collection samples is a highly labour-intensive task collection agencies struggle with. None of the existing automated methods exploit maps that are an essential tool for georeferencing complex relations. We present preliminary experiments and results of a novel method that exploits multi-modal capabilities of recent Large Multi-Modal Models (LMM). This method enables the model to visually contextualize spatial relations it reads in the locality description. We use a grid-based approach to adapt these auto-regressive models for this task in a zero-shot setting. Our experiments conducted on a small manually annotated dataset show impressive results for our approach ($\sim$1 km Average distance error) compared to uni-modal georeferencing with Large Language Models and existing georeferencing tools. The paper also discusses the findings of the experiments in light of an LMM's ability to comprehend fine-grained maps. Motivated by these results, a practical framework is proposed to integrate this method into a georeferencing workflow. 

**Abstract (ZH)**: 近年来收集在自然历史收藏中的数百万份生物样本记录缺乏地理坐标。对这些收集样本关联的复杂地理位置描述进行地理参考化是一个劳动密集型的任务，收集机构面临挑战。现有的自动化方法并未利用地图这一地理参考化复杂关系的重要工具。我们介绍了利用近期多模态大型模型（LMM）多模态能力的一种新颖方法的初步实验和结果。该方法使模型能够通过视觉上下文化它在地理位置描述中读取的空间关系。我们采用基于网格的方法，在零样本设置下将这些自回归模型适应于这一任务。我们在一个小的手动标注数据集上进行的实验结果表明，与单一模态地理参考化结合大型语言模型和现有地理参考化工具相比，我们的方法取得了令人 impressive 的结果（平均距离误差约为1公里）。论文还探讨了实验结果在考虑LMM理解细粒度地图能力方面的发现。受到这些结果的启发，我们提出了一个实用框架，将此方法整合到地理参考化工作流程中。 

---
# A Multi-granularity Concept Sparse Activation and Hierarchical Knowledge Graph Fusion Framework for Rare Disease Diagnosis 

**Title (ZH)**: 多粒度概念稀疏激活与层次知识图融合框架在罕见病诊断中的应用 

**Authors**: Mingda Zhang, Na Zhao, Jianglong Qin, Guoyu Ye, Ruixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08529)  

**Abstract**: Despite advances from medical large language models in healthcare, rare-disease diagnosis remains hampered by insufficient knowledge-representation depth, limited concept understanding, and constrained clinical reasoning. We propose a framework that couples multi-granularity sparse activation of medical concepts with a hierarchical knowledge graph. Four complementary matching algorithms, diversity control, and a five-level fallback strategy enable precise concept activation, while a three-layer knowledge graph (taxonomy, clinical features, instances) provides structured, up-to-date context. Experiments on the BioASQ rare-disease QA set show BLEU gains of 0.09, ROUGE gains of 0.05, and accuracy gains of 0.12, with peak accuracy of 0.89 approaching the 0.90 clinical threshold. Expert evaluation confirms improvements in information quality, reasoning, and professional expression, suggesting our approach shortens the "diagnostic odyssey" for rare-disease patients. 

**Abstract (ZH)**: 尽管医疗大型语言模型在医疗领域的进展，罕见疾病诊断仍受限于知识表示深度不足、概念理解有限和临床推理能力受限。我们提出了一种框架，结合多粒度稀疏激活的医疗概念与层次知识图谱。四种互补的匹配算法、多样性控制以及五级后备策略实现了精确的概念激活，而三层知识图谱（分类学、临床特征、实例）提供了结构化、及时的上下文。在BioASQ罕见疾病问答数据集上的实验结果显示，BLEU分数提高了0.09，ROUGE分数提高了0.05，准确率提高了0.12，峰值准确率达到0.89，接近临床阈值0.90。专家评审确认在信息质量、推理和专业表达方面的改进，表明我们的方法缩短了罕见疾病患者的“诊断之旅”。 

---
# From Language to Logic: A Bi-Level Framework for Structured Reasoning 

**Title (ZH)**: 从语言到逻辑：一种结构化推理的双层框架 

**Authors**: Keying Yang, Hao Wang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08501)  

**Abstract**: Structured reasoning over natural language inputs remains a core challenge in artificial intelligence, as it requires bridging the gap between unstructured linguistic expressions and formal logical representations. In this paper, we propose a novel \textbf{bi-level framework} that maps language to logic through a two-stage process: high-level task abstraction and low-level logic generation. At the upper level, a large language model (LLM) parses natural language queries into intermediate structured representations specifying the problem type, objectives, decision variables, and symbolic constraints. At the lower level, the LLM uses these representations to generate symbolic workflows or executable reasoning programs for accurate and interpretable decision making. The framework supports modular reasoning, enforces explicit constraints, and generalizes across domains such as mathematical problem solving, question answering, and logical inference. We further optimize the framework with an end-to-end {bi-level} optimization approach that jointly refines both the high-level abstraction and low-level logic generation stages. Experiments on multiple realistic reasoning benchmarks demonstrate that our approach significantly outperforms existing baselines in accuracy, with accuracy gains reaching as high as 40\%. Moreover, the bi-level design enhances transparency and error traceability, offering a promising step toward trustworthy and systematic reasoning with LLMs. 

**Abstract (ZH)**: 基于自然语言输入的结构化推理仍是对人工智核心挑战，因为这要求在非结构化语义表达与正式逻辑表示之间建立桥梁。在本文中，我们提出了一种新颖的双层框架，通过两阶段过程将语言映射到逻辑：高层任务抽象和低层逻辑生成。在高层，一个大型语言模型（LLM）将自然语言查询解析为中间结构化表示，指定问题类型、目标、决策变量和符号约束。在低层，LLM 使用这些表示生成符号工作流或可执行推理程序，以实现准确和可解释的决策。该框架支持模块化推理、强制显式约束，并跨数学问题求解、问答和逻辑推理等领域进行泛化。进一步地，我们通过端到端的双层优化方法优化该框架，同时磨练高层抽象和低层逻辑生成阶段。在多个现实推理基准测试上的实验表明，我们的方法在准确性上显著优于现有基线，准确率提升高达40%。此外，双层设计增强了透明度和错误追踪能力，为可信和系统的LLM推理提供了一步前景。 

---
# Why this and not that? A Logic-based Framework for Contrastive Explanations 

**Title (ZH)**: 为什么是这个而不是那个？一种基于逻辑的对比解释框架 

**Authors**: Tobias Geibinger, Reijo Jaakkola, Antti Kuusisto, Xinghan Liu, Miikka Vilander  

**Link**: [PDF](https://arxiv.org/pdf/2507.08454)  

**Abstract**: We define several canonical problems related to contrastive explanations, each answering a question of the form ''Why P but not Q?''. The problems compute causes for both P and Q, explicitly comparing their differences. We investigate the basic properties of our definitions in the setting of propositional logic. We show, inter alia, that our framework captures a cardinality-minimal version of existing contrastive explanations in the literature. Furthermore, we provide an extensive analysis of the computational complexities of the problems. We also implement the problems for CNF-formulas using answer set programming and present several examples demonstrating how they work in practice. 

**Abstract (ZH)**: 我们定义了几类与对比解释相关的规范性问题，每个问题回答的形式均为“为什么P而不是Q？”这些问题计算P和Q的原因，并明确比较它们的差异。我们在命题逻辑的框架下研究我们定义的基本性质。我们证明了，例如，我们的框架捕捉到了文献中已存在的对比解释的基数最小版本。此外，我们还对这些问题的计算复杂性进行了详尽分析。我们还使用回答集编程实现CNF公式上的这些问题，并提供了几个示例，演示它们在实际中的工作方式。 

---
# Multi-Agent LLMs as Ethics Advocates in AI-Based Systems 

**Title (ZH)**: 多智能体大型语言模型作为AI基于系统中的伦理倡导者 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2507.08392)  

**Abstract**: Incorporating ethics into the requirement elicitation process is essential for creating ethically aligned systems. Although eliciting manual ethics requirements is effective, it requires diverse input from multiple stakeholders, which can be challenging due to time and resource constraints. Moreover, it is often given a low priority in the requirements elicitation process. This study proposes a framework for generating ethics requirements drafts by introducing an ethics advocate agent in a multi-agent LLM setting. This agent critiques and provides input on ethical issues based on the system description. The proposed framework is evaluated through two case studies from different contexts, demonstrating that it captures the majority of ethics requirements identified by researchers during 30-minute interviews and introduces several additional relevant requirements. However, it also highlights reliability issues in generating ethics requirements, emphasizing the need for human feedback in this sensitive domain. We believe this work can facilitate the broader adoption of ethics in the requirements engineering process, ultimately leading to more ethically aligned products. 

**Abstract (ZH)**: 将伦理纳入需求获取过程对于创建伦理对齐的系统是必要的。 although eliciting manual ethics requirements是有效的，但它需要来自多个利益相关者的多样化输入，这由于时间和资源限制而具有挑战性。此外，它在需求获取过程中往往被给予较低的优先级。本研究提出了一种框架，在多智能体LLM环境中引入伦理倡导代理，以生成伦理要求草案。该代理基于系统描述对伦理问题进行批判并提供输入。所提出框架通过来自不同情境的两个案例研究进行评估，证明它捕获了研究人员在30分钟访谈中识别出的大多数伦理要求，并引入了若干其他相关要求。然而，这也突出了生成伦理要求的可靠性问题，强调了在此敏感领域需要人类反馈的重要性。我们认为，这项工作有助于促进伦理在需求工程过程中的更广泛采用，最终导致更伦理对齐的产品。 

---
# M2-Reasoning: Empowering MLLMs with Unified General and Spatial Reasoning 

**Title (ZH)**: M2-Reasoning: 促进统一通用与空间推理的MLLLMs 

**Authors**: Inclusion AI, Fudong Wang, Jiajia Liu, Jingdong Chen, Jun Zhou, Kaixiang Ji, Lixiang Ru, Qingpei Guo, Ruobing Zheng, Tianqi Li, Yi Yuan, Yifan Mao, Yuting Xiao, Ziping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08306)  

**Abstract**: Recent advancements in Multimodal Large Language Models (MLLMs), particularly through Reinforcement Learning with Verifiable Rewards (RLVR), have significantly enhanced their reasoning abilities. However, a critical gap persists: these models struggle with dynamic spatial interactions, a capability essential for real-world applications. To bridge this gap, we introduce M2-Reasoning-7B, a model designed to excel in both general and spatial reasoning. Our approach integrates two key innovations: (1) a novel data pipeline that generates 294.2K high-quality data samples (168K for cold-start fine-tuning and 126.2K for RLVR), which feature logically coherent reasoning trajectories and have undergone comprehensive assessment; and (2) a dynamic multi-task training strategy with step-wise optimization to mitigate conflicts between data, and task-specific rewards for delivering tailored incentive signals. This combination of curated data and advanced training allows M2-Reasoning-7B to set a new state-of-the-art (SOTA) across 8 benchmarks, showcasing superior performance in both general and spatial reasoning domains. 

**Abstract (ZH)**: 近期，通过验证性奖励强化学习（RLVR）的多模态大型语言模型（MLLMs）取得了重要进展，显著提升了其推理能力。然而，一个关键差距依然存在：这些模型在动态空间交互方面的表现不佳，而这对于实际应用至关重要。为填补这一空白，我们引入了M2-Reasoning-7B模型，该模型旨在在一般性和空间推理方面表现出色。我们的方法结合了两项关键技术创新：（1）一个新颖的数据管道，生成了294,200个高质量数据样本（168,000个用于冷启动微调，126,200个用于RLVR），这些样本逻辑连贯且经过了全面评估；（2）一种动态多任务训练策略，通过逐步优化来缓解数据和任务特定奖励之间的冲突，并提供定制化的激励信号。这种精挑细选的数据与高级训练策略的结合使M2-Reasoning-7B在8个基准测试中达到了新的最佳水平，展示了其在一般性和空间推理领域均表现出色。 

---
# Agent Safety Alignment via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的代理安全性对齐 

**Authors**: Zeyang Sha, Hanling Tian, Zhuoer Xu, Shiwen Cui, Changhua Meng, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08270)  

**Abstract**: The emergence of autonomous Large Language Model (LLM) agents capable of tool usage has introduced new safety risks that go beyond traditional conversational misuse. These agents, empowered to execute external functions, are vulnerable to both user-initiated threats (e.g., adversarial prompts) and tool-initiated threats (e.g., malicious outputs from compromised tools). In this paper, we propose the first unified safety-alignment framework for tool-using agents, enabling models to handle both channels of threat via structured reasoning and sandboxed reinforcement learning. We introduce a tri-modal taxonomy, including benign, malicious, and sensitive for both user prompts and tool responses, and define a policy-driven decision model. Our framework employs a custom-designed sandbox environment that simulates real-world tool execution and allows fine-grained reward shaping. Through extensive evaluations on public and self-built benchmarks, including Agent SafetyBench, InjecAgent, and BFCL, we demonstrate that our safety-aligned agents significantly improve resistance to security threats while preserving strong utility on benign tasks. Our results show that safety and effectiveness can be jointly optimized, laying the groundwork for trustworthy deployment of autonomous LLM agents. 

**Abstract (ZH)**: 具备工具使用能力的自主大型语言模型代理的涌现带来了超越传统对话滥用的新安全风险。这些代理被授权执行外部功能，使其易受用户发起的威胁（例如，对抗性提示）和工具发起的威胁（例如，受感染工具的恶意输出）的影响。本文提出了一种统一的安全对齐框架，使模型能够通过结构化推理和沙箱强化学习来处理这两种威胁渠道。我们引入了一种三模态分类法，分别对用户提示和工具响应进行分类，包括良性、恶意和敏感，并定义了一个基于策略的决策模型。该框架采用自定义设计的沙箱环境，模拟实际工具执行情况，并允许精细化奖励塑形。通过对公共和自建基准测试的广泛评估，包括Agent SafetyBench、InjecAgent和BFCL，我们证明了我们的安全对齐代理在提高对安全威胁的抵抗力方面表现出色，同时在良性任务上保持了强大的效用。我们的结果表明，安全性和有效性可以共同优化，为自主大型语言模型代理的可信赖部署奠定了基础。 

---
# Abductive Computational Systems: Creative Abduction and Future Directions 

**Title (ZH)**: 演绎计算系统：创造性演绎与未来方向 

**Authors**: Abhinav Sood, Kazjon Grace, Stephen Wan, Cecile Paris  

**Link**: [PDF](https://arxiv.org/pdf/2507.08264)  

**Abstract**: Abductive reasoning, reasoning for inferring explanations for observations, is often mentioned in scientific, design-related and artistic contexts, but its understanding varies across these domains. This paper reviews how abductive reasoning is discussed in epistemology, science and design, and then analyses how various computational systems use abductive reasoning. Our analysis shows that neither theoretical accounts nor computational implementations of abductive reasoning adequately address generating creative hypotheses. Theoretical frameworks do not provide a straightforward model for generating creative abductive hypotheses, computational systems largely implement syllogistic forms of abductive reasoning. We break down abductive computational systems into components and conclude by identifying specific directions for future research that could advance the state of creative abductive reasoning in computational systems. 

**Abstract (ZH)**: abduction推理，即为基于观察推求解释的推理，在科学、设计相关和艺术领域中常被提及，但在这些领域的理解有所不同。本文回顾了 abduction 推理在认识论、科学和设计中的讨论方式，然后分析了各种计算系统如何使用 abduction 推理。我们的分析表明，无论是理论 Accounts 还是计算实现的 abduction 推理，都不能充分解决生成创造性假设的问题。理论框架没有提供生成创造性 abduction 假设的直观模型，计算系统主要采用前提演绎形式的 abduction 推理。我们将 abduction 计算系统分解成组件，并提出具体的研究方向，以促进计算系统中创造性 abduction 推理的发展。 

---
# Giving AI Agents Access to Cryptocurrency and Smart Contracts Creates New Vectors of AI Harm 

**Title (ZH)**: 给AI代理提供接入加密货币和智能合约的权限创造了新的AI危害向量 

**Authors**: Bill Marino, Ari Juels  

**Link**: [PDF](https://arxiv.org/pdf/2507.08249)  

**Abstract**: There is growing interest in giving AI agents access to cryptocurrencies as well as to the smart contracts that transact them. But doing so, this position paper argues, could lead to formidable new vectors of AI harm. To support this argument, we first examine the unique properties of cryptocurrencies and smart contracts that could lead to these new vectors of harm. Next, we describe each of these new vectors of harm in detail. Finally, we conclude with a call for more technical research aimed at preventing and mitigating these harms and, thereby making it safer to endow AI agents with cryptocurrencies and smart contracts. 

**Abstract (ZH)**: 随着对给AI代理赋予加密货币及其智能合约接入权限的兴趣日益增长，这有可能导致强大的新AI危害向量。但本立场论文认为，这样做可能会带来新的AI危害。为支持这一论点，我们首先探讨了加密货币和智能合约的独特属性，这些属性可能导致这些新的危害向量。接着，我们详细描述了这些新的危害向量。最后，我们呼吁开展更多旨在预防和减轻这些危害的技术研究，从而使得赋予AI代理加密货币和智能合约变得更安全。 

---
# Quantum Federated Learning for Multimodal Data: A Modality-Agnostic Approach 

**Title (ZH)**: 量子联邦学习在多模态数据中的应用：一种模态无关方法 

**Authors**: Atit Pokharel, Ratun Rahman, Thomas Morris, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08217)  

**Abstract**: Quantum federated learning (QFL) has been recently introduced to enable a distributed privacy-preserving quantum machine learning (QML) model training across quantum processors (clients). Despite recent research efforts, existing QFL frameworks predominantly focus on unimodal systems, limiting their applicability to real-world tasks that often naturally involve multiple modalities. To fill this significant gap, we present for the first time a novel multimodal approach specifically tailored for the QFL setting with the intermediate fusion using quantum entanglement. Furthermore, to address a major bottleneck in multimodal QFL, where the absence of certain modalities during training can degrade model performance, we introduce a Missing Modality Agnostic (MMA) mechanism that isolates untrained quantum circuits, ensuring stable training without corrupted states. Simulation results demonstrate that the proposed multimodal QFL method with MMA yields an improvement in accuracy of 6.84% in independent and identically distributed (IID) and 7.25% in non-IID data distributions compared to the state-of-the-art methods. 

**Abstract (ZH)**: 多模态量子联邦学习（QMFL）方法及其在量子处理器上的应用：基于量子纠缠的中间融合和缺失模态无关（MMA）机制 

---
# Grounding Methods for Neural-Symbolic AI 

**Title (ZH)**: 神经符号AI中的 grounding 方法 

**Authors**: Rodrigo Castellano Ontiveros, Francesco Giannini, Marco Gori, Giuseppe Marra, Michelangelo Diligenti  

**Link**: [PDF](https://arxiv.org/pdf/2507.08216)  

**Abstract**: A large class of Neural-Symbolic (NeSy) methods employs a machine learner to process the input entities, while relying on a reasoner based on First-Order Logic to represent and process more complex relationships among the entities. A fundamental role for these methods is played by the process of logic grounding, which determines the relevant substitutions for the logic rules using a (sub)set of entities. Some NeSy methods use an exhaustive derivation of all possible substitutions, preserving the full expressive power of the logic knowledge. This leads to a combinatorial explosion in the number of ground formulas to consider and, therefore, strongly limits their scalability. Other methods rely on heuristic-based selective derivations, which are generally more computationally efficient, but lack a justification and provide no guarantees of preserving the information provided to and returned by the reasoner. Taking inspiration from multi-hop symbolic reasoning, this paper proposes a parametrized family of grounding methods generalizing classic Backward Chaining. Different selections within this family allow us to obtain commonly employed grounding methods as special cases, and to control the trade-off between expressiveness and scalability of the reasoner. The experimental results show that the selection of the grounding criterion is often as important as the NeSy method itself. 

**Abstract (ZH)**: 一种大型神经符号（NeSy）方法使用机器学习处理输入实体，同时依赖基于一阶逻辑的推理器来表示和处理实体间更为复杂的关系。这些方法中的基础作用由逻辑基础化过程发挥，该过程利用实体集（子集）确定逻辑规则的相关替换。一些NeSy方法通过穷尽所有可能的替换衍生，保留了逻辑知识的全部表达能力。这导致需要考虑的地面公式数量呈组合爆炸增长，因此严重限制了其可扩展性。其他方法依赖基于启发式的选择性衍生，通常更具计算效率，但缺乏解释性且无法保证推理器所提供的和返回的信息。借鉴多跳符号推理的思路，本文提出了一类参数化的基础化方法，推广了经典的向后链接。该家族的不同选择使我们能够获得常用的几种基础化方法，并能够控制推理器的表达能力和可扩展性之间的权衡关系。实验结果表明，基础化标准的选择往往与NeSy方法本身同样重要。 

---
# From Curiosity to Competence: How World Models Interact with the Dynamics of Exploration 

**Title (ZH)**: 从好奇心到能力：世界模型如何与探索动力学互动 

**Authors**: Fryderyk Mantiuk, Hanqi Zhou, Charley M. Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08210)  

**Abstract**: What drives an agent to explore the world while also maintaining control over the environment? From a child at play to scientists in the lab, intelligent agents must balance curiosity (the drive to seek knowledge) with competence (the drive to master and control the environment). Bridging cognitive theories of intrinsic motivation with reinforcement learning, we ask how evolving internal representations mediate the trade-off between curiosity (novelty or information gain) and competence (empowerment). We compare two model-based agents using handcrafted state abstractions (Tabular) or learning an internal world model (Dreamer). The Tabular agent shows curiosity and competence guide exploration in distinct patterns, while prioritizing both improves exploration. The Dreamer agent reveals a two-way interaction between exploration and representation learning, mirroring the developmental co-evolution of curiosity and competence. Our findings formalize adaptive exploration as a balance between pursuing the unknown and the controllable, offering insights for cognitive theories and efficient reinforcement learning. 

**Abstract (ZH)**: 什么是驱使智能代理探索世界同时又维持环境控制的动力？从玩耍的儿童到实验室中的科学家，智能代理必须在好奇心（寻求知识的驱动力）与胜任力（掌握和控制环境的驱动力）之间寻求平衡。将认知理论中的内在动机与强化学习相结合，我们探讨了在好奇心（新颖性或信息获取）与胜任力（能力感）之间通过不断演化内部表征如何进行权衡。我们比较了两个基于模型的代理：一个使用手工设计的状态抽象（表格），另一个学习内部世界模型（Dreamer）。表格代理展示了好奇心和胜任力在探索中以不同模式引导探索，而优先考虑两者则能提高探索效率。Dreamer代理揭示了探索与表示学习之间的双向交互作用，类似于好奇心与胜任力在发展过程中的共同进化。我们的研究将适应性探索正式化为追求未知与可控之间的平衡，为认知理论和高效强化学习提供了启示。 

---
# Reasoning and Behavioral Equilibria in LLM-Nash Games: From Mindsets to Actions 

**Title (ZH)**: LLM-纳什游戏中思维模式与行为均衡的推理与行为 equilibrium：从心态到行动 

**Authors**: Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08208)  

**Abstract**: We introduce the LLM-Nash framework, a game-theoretic model where agents select reasoning prompts to guide decision-making via Large Language Models (LLMs). Unlike classical games that assume utility-maximizing agents with full rationality, this framework captures bounded rationality by modeling the reasoning process explicitly. Equilibrium is defined over the prompt space, with actions emerging as the behavioral output of LLM inference. This approach enables the study of cognitive constraints, mindset expressiveness, and epistemic learning. Through illustrative examples, we show how reasoning equilibria can diverge from classical Nash outcomes, offering a new foundation for strategic interaction in LLM-enabled systems. 

**Abstract (ZH)**: LLM-Nash框架：一种基于大规模语言模型的游戏理论模型 

---
# A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking 

**Title (ZH)**: 一种针对大语言模型逃逸攻击的代理人工智能防御的动态斯塔克尔贝尔格博弈框架 

**Authors**: Zhengye Han, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08207)  

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking. 

**Abstract (ZH)**: 大型语言模型（LLMs）在关键应用中部署越来越多时，模型被操纵以绕过安全机制的“ Jailbreaking ”挑战已成为一个重要问题。本文提出了一种动态Stackelberg博弈框架，以建模在LLM Jailbreaking背景下攻击者和防御者的互动。该框架将提示-响应动态视为一个序列扩展式博弈，其中防御者作为领导者，制定策略并预测攻击者的最优响应。我们提出了一种新颖的代理AI解决方案——“紫色代理”，该解决方案使用快速扩展随机树（RRT）结合了对抗性探索和防御策略。紫色代理主动模拟潜在的攻击轨迹并主动干预以防止有害输出。该方法提供了一种分析对抗动态的原则性方法，并为减轻Jailbreaking风险提供了基础。 

---
# TableReasoner: Advancing Table Reasoning Framework with Large Language Models 

**Title (ZH)**: TableReasoner: 借助大型语言模型促进表单推理框架发展 

**Authors**: Sishi Xiong, Dakai Wang, Yu Zhao, Jie Zhang, Changzai Pan, Haowei He, Xiangyu Li, Wenhan Chang, Zhongjiang He, Shuangyong Song, Yongxiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08046)  

**Abstract**: The paper presents our system developed for table question answering (TQA). TQA tasks face challenges due to the characteristics of real-world tabular data, such as large size, incomplete column semantics, and entity ambiguity. To address these issues, we propose a large language model (LLM)-powered and programming-based table reasoning framework, named TableReasoner. It models a table using the schema that combines structural and semantic representations, enabling holistic understanding and efficient processing of large tables. We design a multi-step schema linking plan to derive a focused table schema that retains only query-relevant information, eliminating ambiguity and alleviating hallucinations. This focused table schema provides precise and sufficient table details for query refinement and programming. Furthermore, we integrate the reasoning workflow into an iterative thinking architecture, allowing incremental cycles of thinking, reasoning and reflection. Our system achieves first place in both subtasks of SemEval-2025 Task 8. 

**Abstract (ZH)**: 本文介绍了我们开发的表格问题回答（TQA）系统。由于现实世界表格数据的特点（如规模大、列语义不完整和实体歧义）导致TQA任务面临挑战。为应对这些挑战，我们提出了一种基于大型语言模型（LLM）和编程的表格推理框架，名为TableReasoner。该框架使用结合结构和语义表示的模式来建模表格，从而实现对大表格的全面理解和高效处理。我们设计了一种多步模式链接计划，以提取仅保留查询相关信息的聚焦表格模式，消除歧义并减轻虚假信息的产生。该聚焦表格模式为查询细化和编程提供了精确和充分的表格细节。此外，我们将推理工作流集成到迭代思维架构中，允许思维、推理和反思的逐步循环。我们的系统在SemEval-2025 Task 8的两个子任务中均位居第一。 

---
# Human Creativity and AI 

**Title (ZH)**: 人类创造力与人工智能 

**Authors**: Shengyi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2507.08001)  

**Abstract**: With the advancement of science and technology, the philosophy of creativity has undergone significant reinterpretation. This paper investigates contemporary research in the fields of psychology, cognitive neuroscience, and the philosophy of creativity, particularly in the context of the development of artificial intelligence (AI) techniques. It aims to address the central question: Can AI exhibit creativity? The paper reviews the historical perspectives on the philosophy of creativity and explores the influence of psychological advancements on the study of creativity. Furthermore, it analyzes various definitions of creativity and examines the responses of naturalism and cognitive neuroscience to the concept of creativity. 

**Abstract (ZH)**: 科学和技术的进步促使对创造力哲学进行了重新解读。本文探讨了心理学、认知神经科学和创造力哲学领域的当代研究，特别是人工智能技术发展的背景。本文旨在探讨中心问题：人工智能能否表现出创造力？文章回顾了创造力哲学的历史观点，并探讨了心理学进步对创造力研究的影响。此外，文章分析了创造力的各种定义，并考察了自然主义和认知神经科学对创造力概念的回应。 

---
# Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective 

**Title (ZH)**: Lumos-1：统一模型视角下的自回归视频生成 

**Authors**: Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08801)  

**Abstract**: Autoregressive large language models (LLMs) have unified a vast range of language tasks, inspiring preliminary efforts in autoregressive video generation. Existing autoregressive video generators either diverge from standard LLM architectures, depend on bulky external text encoders, or incur prohibitive latency due to next-token decoding. In this paper, we introduce Lumos-1, an autoregressive video generator that retains the LLM architecture with minimal architectural modifications. To inject spatiotemporal correlations in LLMs, we identify the efficacy of incorporating 3D RoPE and diagnose its imbalanced frequency spectrum ranges. Therefore, we propose MM-RoPE, a RoPE scheme that preserves the original textual RoPE while providing comprehensive frequency spectra and scaled 3D positions for modeling multimodal spatiotemporal data. Moreover, Lumos-1 resorts to a token dependency strategy that obeys intra-frame bidirectionality and inter-frame temporal causality. Based on this dependency strategy, we identify the issue of frame-wise loss imbalance caused by spatial information redundancy and solve it by proposing Autoregressive Discrete Diffusion Forcing (AR-DF). AR-DF introduces temporal tube masking during training with a compatible inference-time masking policy to avoid quality degradation. By using memory-efficient training techniques, we pre-train Lumos-1 on only 48 GPUs, achieving performance comparable to EMU3 on GenEval, COSMOS-Video2World on VBench-I2V, and OpenSoraPlan on VBench-T2V. Code and models are available at this https URL. 

**Abstract (ZH)**: 基于自回归的大语言模型（LLMs）已统一了广泛的语言任务，并激发了初步的自回归视频生成努力。现有的自回归视频生成器要么偏离标准LLM架构，要么依赖庞大的外部文本编码器，要么因下一步解码导致显著的延迟。在本文中，我们引入了Lumos-1，这是一种保留LLM架构并进行了最小架构修改的自回归视频生成器。为了在LLMs中注入空时相关性，我们认识到3D RoPE的有效性，并诊断其不均衡的频谱范围。因此，我们提出了一种MM-RoPE方案，该方案保留了原始文本RoPE的同时，提供了全面的频谱范围和缩放的3D位置，以建模多模态空时数据。此外，Lumos-1采用了遵循帧内双向依赖性和帧间时序因果性的令牌依赖策略。基于此依赖策略，我们发现了由于空间信息冗余导致的帧间损失不平衡的问题，并提出了一种自回归离散扩散强迫（AR-DF）来解决这一问题。AR-DF在训练时引入了时空管状掩码，并配备了兼容的推理时掩码策略，以避免质量下降。通过使用高效的训练技术，我们仅在48个GPU上预训练了Lumos-1，实现了与EMU3在GenEval、COSMOS-Video2World在VBench-I2V以及OpenSoraPlan在VBench-T2V上的性能相当的表现。代码和模型可在以下链接获取。 

---
# NeuralOS: Towards Simulating Operating Systems via Neural Generative Models 

**Title (ZH)**: NeuralOS: 通过神经生成模型模拟操作系统 

**Authors**: Luke Rivard, Sun Sun, Hongyu Guo, Wenhu Chen, Yuntian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08800)  

**Abstract**: We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames in response to user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates screen images. The model is trained on a large-scale dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Experiments show that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Although modeling fine-grained keyboard interactions precisely remains challenging, NeuralOS offers a step toward creating fully adaptive, generative neural interfaces for future human-computer interaction systems. 

**Abstract (ZH)**: NeuralOS：一种通过直接预测屏幕帧来模拟操作系统图形用户界面的神经框架 

---
# KV Cache Steering for Inducing Reasoning in Small Language Models 

**Title (ZH)**: 针对诱导小语言模型进行推断的KV缓存定向方法 

**Authors**: Max Belitsky, Dawid J. Kopiczko, Michael Dorkenwald, M. Jehanzeb Mirza, Cees G. M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2507.08799)  

**Abstract**: We propose cache steering, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. To validate its effectiveness, we apply cache steering to induce chain-of-thought reasoning in small language models. Our approach leverages GPT-4o-generated reasoning traces to construct steering vectors that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. Experimental evaluations on diverse reasoning benchmarks demonstrate that cache steering improves both the qualitative structure of model reasoning and quantitative task performance. Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of hyperparameter stability, inference-time efficiency, and ease of integration, making it a more robust and practical solution for controlled generation. 

**Abstract (ZH)**: 我们提出了一种轻量级的方法——缓存引导，该方法通过一次性的干预直接作用于键值缓存，从而实现语言模型的隐式引导。为了验证其有效性，我们将缓存引导应用于小语言模型，以诱导其进行链式思考推理。我们的方法利用GPT-4o生成的推理踪迹构建引导向量，从而在无需微调或修改提示的情况下，使模型行为更加符合显式、多步骤的推理。在多种推理基准上的实验评估表明，缓存引导可以提高模型推理的定性和定量表现。与需要连续干预的先前提取引导技术相比，我们的单次缓存引导在超参数稳定性、推理时间效率和集成简易性方面具有显著优势，使其成为更稳健且实用的受控生成解决方案。 

---
# Optimistic Exploration for Risk-Averse Constrained Reinforcement Learning 

**Title (ZH)**: 风险规避约束强化学习中的乐观探索 

**Authors**: James McCarthy, Radu Marinescu, Elizabeth Daly, Ivana Dusparic  

**Link**: [PDF](https://arxiv.org/pdf/2507.08793)  

**Abstract**: Risk-averse Constrained Reinforcement Learning (RaCRL) aims to learn policies that minimise the likelihood of rare and catastrophic constraint violations caused by an environment's inherent randomness. In general, risk-aversion leads to conservative exploration of the environment which typically results in converging to sub-optimal policies that fail to adequately maximise reward or, in some cases, fail to achieve the goal. In this paper, we propose an exploration-based approach for RaCRL called Optimistic Risk-averse Actor Critic (ORAC), which constructs an exploratory policy by maximising a local upper confidence bound of the state-action reward value function whilst minimising a local lower confidence bound of the risk-averse state-action cost value function. Specifically, at each step, the weighting assigned to the cost value is increased or decreased if it exceeds or falls below the safety constraint value. This way the policy is encouraged to explore uncertain regions of the environment to discover high reward states whilst still satisfying the safety constraints. Our experimental results demonstrate that the ORAC approach prevents convergence to sub-optimal policies and improves significantly the reward-cost trade-off in various continuous control tasks such as Safety-Gymnasium and a complex building energy management environment CityLearn. 

**Abstract (ZH)**: Risk-averse Constrained Reinforcement Learning (RaCRL) aims to learn policies that minimize the likelihood of rare and catastrophic constraint violations caused by an environment's inherent randomness. In this paper, we propose an exploration-based approach for RaCRL called Optimistic Risk-averse Actor Critic (ORAC), which constructs an exploratory policy by maximizing a local upper confidence bound of the state-action reward value function while minimizing a local lower confidence bound of the risk-averse state-action cost value function. Specifically, at each step, the weighting assigned to the cost value is increased or decreased if it exceeds or falls below the safety constraint value. This way, the policy is encouraged to explore uncertain regions of the environment to discover high-reward states while still satisfying the safety constraints. Our experimental results demonstrate that the ORAC approach prevents convergence to sub-optimal policies and significantly improves the reward-cost trade-off in various continuous control tasks such as Safety-Gymnasium and a complex building energy management environment CityLearn. 

---
# On Barriers to Archival Audio Processing 

**Title (ZH)**: 关于归档音频处理的障碍 

**Authors**: Peter Sullivan, Muhammad Abdul-Mageed  

**Link**: [PDF](https://arxiv.org/pdf/2507.08768)  

**Abstract**: In this study, we leverage a unique UNESCO collection of mid-20th century radio recordings to probe the robustness of modern off-the-shelf language identification (LID) and speaker recognition (SR) methods, especially with respect to the impact of multilingual speakers and cross-age recordings. Our findings suggest that LID systems, such as Whisper, are increasingly adept at handling second-language and accented speech. However, speaker embeddings remain a fragile component of speech processing pipelines that is prone to biases related to the channel, age, and language. Issues which will need to be overcome should archives aim to employ SR methods for speaker indexing. 

**Abstract (ZH)**: 本研究利用UNESCO中期20世纪无线电录音的独特集合，探究现代现成语言识别（LID）和说话人识别（SR）方法的稳健性，尤其是多语言说话人和跨年龄录音的影响。研究发现，语言识别系统，如Whisper，越来越擅长处理二语和口音 speech。然而，说话人嵌入仍然是语音处理管道中的一个脆弱组件，容易受到通道、年龄和语言相关的偏见影响。这些问题需要在档案馆利用说话人识别方法进行说话人索引时予以克服。 

---
# A Hybrid Multi-Well Hopfield-CNN with Feature Extraction and K-Means for MNIST Classification 

**Title (ZH)**: 带有特征提取和K均值聚类的混合多井霍普菲尔德-CNNfor MNIST分类 

**Authors**: Ahmed Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2507.08766)  

**Abstract**: This study presents a hybrid model for classifying handwritten digits in the MNIST dataset, combining convolutional neural networks (CNNs) with a multi-well Hopfield network. The approach employs a CNN to extract high-dimensional features from input images, which are then clustered into class-specific prototypes using k-means clustering. These prototypes serve as attractors in a multi-well energy landscape, where a Hopfield network performs classification by minimizing an energy function that balances feature similarity and class this http URL model's design enables robust handling of intraclass variability, such as diverse handwriting styles, while providing an interpretable framework through its energy-based decision process. Through systematic optimization of the CNN architecture and the number of wells, the model achieves a high test accuracy of 99.2% on 10,000 MNIST images, demonstrating its effectiveness for image classification tasks. The findings highlight the critical role of deep feature extraction and sufficient prototype coverage in achieving high performance, with potential for broader applications in pattern recognition. 

**Abstract (ZH)**: 本研究提出了一种结合卷积神经网络（CNN）和多阱霍普菲尔德网络的混合模型，用于MNIST手写数字数据集的分类。该方法利用CNN从输入图像中提取高维特征，并使用k-means聚类将这些特征聚类成类特定的原型。这些原型在多阱能量景观中作为吸引子，霍普菲尔德网络通过最小化平衡特征相似性和类别的能量函数来进行分类。该模型的设计能够稳健地处理类内变异性，如多种手写风格，并通过基于能量的决策过程提供可解释的框架。通过系统优化CNN架构和阱的数量，该模型在10,000个MNIST图像上的测试准确率达到99.2%，展示了其在图像分类任务中的有效性。研究结果强调了深入特征提取和足够原型覆盖对实现高性能的关键作用，并具有在模式识别中更广泛的应用潜力。 

---
# Compress Any Segment Anything Model (SAM) 

**Title (ZH)**: 压缩任意段落检测模型 (SAM) 

**Authors**: Juntong Fan, Zhiwei Hao, Jianqiang Shen, Shang-Ling Jui, Yi Zhang, Jing-Xiao Liao, Feng-Lei Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08765)  

**Abstract**: Due to the excellent performance in yielding high-quality, zero-shot segmentation, Segment Anything Model (SAM) and its variants have been widely applied in diverse scenarios such as healthcare and intelligent manufacturing. Therefore, effectively compressing SAMs has become an increasingly pressing practical need. In this study, we propose Birkhoff, a novel data-free compression algorithm for SAM and its variants. Unlike quantization, pruning, distillation, and other compression methods, Birkhoff embodies versatility across model types, agility in deployment, faithfulness to the original model, and compactness in model size. Specifically, Birkhoff introduces a novel compression algorithm: Hyper-Compression, whose core principle is to find a dense trajectory to turn a high-dimensional parameter vector into a low-dimensional scalar. Furthermore, Birkhoff designs a dedicated linear layer operator, HyperLinear, to fuse decompression and matrix multiplication to significantly accelerate inference of the compressed SAMs. Extensive experiments on 18 SAMs in the COCO, LVIS, and SA-1B datasets show that Birkhoff performs consistently and competitively in compression time, compression ratio, post-compression performance, and inference speed. For example, Birkhoff can achieve a compression ratio of 5.17x on SAM2-B, with less than 1% performance drop without using any fine-tuning data. Moreover, the compression is finished within 60 seconds for all models. 

**Abstract (ZH)**: 基于Birkhoff的Segment Anything模型及其变种的无数据压缩算法 

---
# Penalizing Infeasible Actions and Reward Scaling in Reinforcement Learning with Offline Data 

**Title (ZH)**: 基于离线数据的强化学习中不可行动作惩罚和奖励缩放 

**Authors**: Jeonghye Kim, Yongjae Shin, Whiyoung Jung, Sunghoon Hong, Deunsol Yoon, Youngchul Sung, Kanghoon Lee, Woohyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2507.08761)  

**Abstract**: Reinforcement learning with offline data suffers from Q-value extrapolation errors. To address this issue, we first demonstrate that linear extrapolation of the Q-function beyond the data range is particularly problematic. To mitigate this, we propose guiding the gradual decrease of Q-values outside the data range, which is achieved through reward scaling with layer normalization (RS-LN) and a penalization mechanism for infeasible actions (PA). By combining RS-LN and PA, we develop a new algorithm called PARS. We evaluate PARS across a range of tasks, demonstrating superior performance compared to state-of-the-art algorithms in both offline training and online fine-tuning on the D4RL benchmark, with notable success in the challenging AntMaze Ultra task. 

**Abstract (ZH)**: Offline Data-Reinforcement Learning Suffers from Q-value Extrapolation Errors: A New Algorithm PARSMitigates This Issue Through Reward Scaling with Layer Normalization and Penalization Mechanism 

---
# Geo-ORBIT: A Federated Digital Twin Framework for Scene-Adaptive Lane Geometry Detection 

**Title (ZH)**: Geo-ORBIT: 一种场景自适应车道几何检测的联邦数字孪生框架 

**Authors**: Rei Tamaru, Pei Li, Bin Ran  

**Link**: [PDF](https://arxiv.org/pdf/2507.08743)  

**Abstract**: Digital Twins (DT) have the potential to transform traffic management and operations by creating dynamic, virtual representations of transportation systems that sense conditions, analyze operations, and support decision-making. A key component for DT of the transportation system is dynamic roadway geometry sensing. However, existing approaches often rely on static maps or costly sensors, limiting scalability and adaptability. Additionally, large-scale DTs that collect and analyze data from multiple sources face challenges in privacy, communication, and computational efficiency. To address these challenges, we introduce Geo-ORBIT (Geometrical Operational Roadway Blueprint with Integrated Twin), a unified framework that combines real-time lane detection, DT synchronization, and federated meta-learning. At the core of Geo-ORBIT is GeoLane, a lightweight lane detection model that learns lane geometries from vehicle trajectory data using roadside cameras. We extend this model through Meta-GeoLane, which learns to personalize detection parameters for local entities, and FedMeta-GeoLane, a federated learning strategy that ensures scalable and privacy-preserving adaptation across roadside deployments. Our system is integrated with CARLA and SUMO to create a high-fidelity DT that renders highway scenarios and captures traffic flows in real-time. Extensive experiments across diverse urban scenes show that FedMeta-GeoLane consistently outperforms baseline and meta-learning approaches, achieving lower geometric error and stronger generalization to unseen locations while drastically reducing communication overhead. This work lays the foundation for flexible, context-aware infrastructure modeling in DTs. The framework is publicly available at this https URL. 

**Abstract (ZH)**: 数字孪生（DT）有潜力通过创建交通系统的动态虚拟表示，感知状况、分析运行情况和支持决策来转型交通管理与运营。交通系统的数字孪生的关键组件是动态道路几何感知。然而，现有的方法往往依赖于静态地图或昂贵的传感器，限制了其可扩展性和适应性。此外，从多个来源收集和分析数据的大规模数字孪生面临着隐私、通信和计算效率方面的挑战。为了解决这些挑战，我们介绍了Geo-ORBIT（几何操作道路蓝图与集成孪生），这是一种结合了实时车道检测、数字孪生同步和联邦元学习的统一框架。Geo-ORBIT的核心是GeoLane，这是一种轻量级的车道检测模型，通过路边摄像头从车辆轨迹数据中学习车道几何。我们通过Meta-GeoLane扩展了该模型，使其能够为本地实体个性化检测参数，并通过FedMeta-GeoLane的联邦学习策略实现跨路边部署的可扩展和隐私保护的适应。我们的系统与CARLA和SUMO集成，创建了一个高保真的数字孪生，实时渲染高速公路场景并捕捉交通流。在多种城市场景的广泛实验中，FedMeta-GeoLane在几何误差和对未见过的位置的泛化能力方面表现出色，同时大幅减少了通信开销。这项工作为数字孪生中的灵活、上下文感知基础设施建模奠定了基础。该框架可在以下网址公开获取：this https URL。 

---
# Adaptive Nonlinear Vector Autoregression: Robust Forecasting for Noisy Chaotic Time Series 

**Title (ZH)**: 自适应非线性向量自回归：噪声混沌时间序列的稳健forecasting 

**Authors**: Azimov Sherkhon, Susana Lopez-Moreno, Eric Dolores-Cuenca, Sieun Lee, Sangil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.08738)  

**Abstract**: Nonlinear vector autoregression (NVAR) and reservoir computing (RC) have shown promise in forecasting chaotic dynamical systems, such as the Lorenz-63 model and El Nino-Southern Oscillation. However, their reliance on fixed nonlinearities - polynomial expansions in NVAR or random feature maps in RC - limits their adaptability to high noise or real-world data. These methods also scale poorly in high-dimensional settings due to costly matrix inversion during readout computation. We propose an adaptive NVAR model that combines delay-embedded linear inputs with features generated by a shallow, learnable multi-layer perceptron (MLP). The MLP and linear readout are jointly trained using gradient-based optimization, enabling the model to learn data-driven nonlinearities while preserving a simple readout structure. Unlike standard NVAR, our approach avoids the need for an exhaustive and sensitive grid search over ridge and delay parameters. Instead, tuning is restricted to neural network hyperparameters, improving scalability. Initial experiments on chaotic systems tested under noise-free and synthetically noisy conditions showed that the adaptive model outperformed the standard NVAR in predictive accuracy and showed robust forecasting under noisy conditions with a lower observation frequency. 

**Abstract (ZH)**: 自适应非线性向量自回归模型结合浅层可学习多层感知机在混沌动态系统预测中的应用 

---
# Catastrophic Forgetting Mitigation Through Plateau Phase Activity Profiling 

**Title (ZH)**: 通过 plateau 阶段活性 profiling 减轻灾难性遗忘 

**Authors**: Idan Mashiach, Oren Glickman, Tom Tirer  

**Link**: [PDF](https://arxiv.org/pdf/2507.08736)  

**Abstract**: Catastrophic forgetting in deep neural networks occurs when learning new tasks degrades performance on previously learned tasks due to knowledge overwriting. Among the approaches to mitigate this issue, regularization techniques aim to identify and constrain "important" parameters to preserve previous knowledge. In the highly nonconvex optimization landscape of deep learning, we propose a novel perspective: tracking parameters during the final training plateau is more effective than monitoring them throughout the entire training process. We argue that parameters that exhibit higher activity (movement and variability) during this plateau reveal directions in the loss landscape that are relatively flat, making them suitable for adaptation to new tasks while preserving knowledge from previous ones. Our comprehensive experiments demonstrate that this approach achieves superior performance in balancing catastrophic forgetting mitigation with strong performance on newly learned tasks. 

**Abstract (ZH)**: 深度神经网络中的灾难性遗忘现象发生在学习新任务时因知识覆盖而导致之前学习任务表现下降。为缓解这一问题的方法中，正则化技术旨在识别并约束“重要”的参数以保留先前的知识。在深度学习高度非凸的优化景观中，我们提出一种新颖的观点：在最终训练平台期追踪参数比在整个训练过程中监测参数更有效。我们认为，在此平台期活动性（运动和变化）较高的参数揭示了损失景观中相对平坦的方向，使其适合适应新任务的同时保留先前的知识。全面的实验表明，该方法在缓解灾难性遗忘和新任务强表现之间取得了更优的平衡性能。 

---
# Dually Hierarchical Drift Adaptation for Online Configuration Performance Learning 

**Title (ZH)**: 双重分层漂移适应在线配置性能学习 

**Authors**: Zezhen Xiang, Jingzhi Gong, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08730)  

**Abstract**: Modern configurable software systems need to learn models that correlate configuration and performance. However, when the system operates in dynamic environments, the workload variations, hardware changes, and system updates will inevitably introduce concept drifts at different levels - global drifts, which reshape the performance landscape of the entire configuration space; and local drifts, which only affect certain sub-regions of that space. As such, existing offline and transfer learning approaches can struggle to adapt to these implicit and unpredictable changes in real-time, rendering configuration performance learning challenging. To address this, we propose DHDA, an online configuration performance learning framework designed to capture and adapt to these drifts at different levels. The key idea is that DHDA adapts to both the local and global drifts using dually hierarchical adaptation: at the upper level, we redivide the data into different divisions, within each of which the local model is retrained, to handle global drifts only when necessary. At the lower level, the local models of the divisions can detect local drifts and adapt themselves asynchronously. To balance responsiveness and efficiency, DHDA combines incremental updates with periodic full retraining to minimize redundant computation when no drifts are detected. Through evaluating eight software systems and against state-of-the-art approaches, we show that DHDA achieves considerably better accuracy and can effectively adapt to drifts with up to 2x improvements, while incurring reasonable overhead and is able to improve different local models in handling concept drift. 

**Abstract (ZH)**: 现代可配置软件系统需要学习关联配置和性能的模型。然而，当系统在动态环境中运行时，工作负载变化、硬件更新和系统更新将不可避免地在不同层次上引入概念漂移——全局漂移，重塑整个配置空间的性能景观；局部漂移，仅影响该空间的某些子区域。因此，现有的离线和迁移学习方法在实时适应这些隐含且难以预测的变化方面可能会遇到困难，使得配置性能学习变得挑战重重。为了解决这一问题，我们提出了DHDA，这是一种在线配置性能学习框架，旨在捕捉并适应不同层次的概念漂移。关键思想是DHDA使用双重分层适应来适应局部和全局漂移：在较高层次上，我们重新划分数据为不同的部分，在每个部分中重新训练局部模型，仅在必要时处理全局漂移。在较低层次，各部分的局部模型可以异步检测局部漂移并自行适应。为了平衡响应性和效率，DHDA结合增量更新与定期全面重训练，以最小化未检测到漂移时的冗余计算。通过评估八个软件系统并与其他先进方法进行对比，我们展示了DHDA实现了显著的准确性提升，并能在最多2倍的性能提升下有效适应漂移，同时引入合理的开销，并能够提高不同局部模型处理概念漂移的能力。 

---
# Monitoring Risks in Test-Time Adaptation 

**Title (ZH)**: 测试时适应中的风险监控 

**Authors**: Mona Schirmer, Metod Jazbec, Christian A. Naesseth, Eric Nalisnick  

**Link**: [PDF](https://arxiv.org/pdf/2507.08721)  

**Abstract**: Encountering shifted data at test time is a ubiquitous challenge when deploying predictive models. Test-time adaptation (TTA) methods address this issue by continuously adapting a deployed model using only unlabeled test data. While TTA can extend the model's lifespan, it is only a temporary solution. Eventually the model might degrade to the point that it must be taken offline and retrained. To detect such points of ultimate failure, we propose pairing TTA with risk monitoring frameworks that track predictive performance and raise alerts when predefined performance criteria are violated. Specifically, we extend existing monitoring tools based on sequential testing with confidence sequences to accommodate scenarios in which the model is updated at test time and no test labels are available to estimate the performance metrics of interest. Our extensions unlock the application of rigorous statistical risk monitoring to TTA, and we demonstrate the effectiveness of our proposed TTA monitoring framework across a representative set of datasets, distribution shift types, and TTA methods. 

**Abstract (ZH)**: 在测试时遭遇偏移数据是部署预测模型时的一个普遍挑战。测试时自适应（TTA）方法通过仅使用未标记的测试数据连续适应部署模型来解决这一问题。虽然TTA可以延长模型的寿命，但它仅是一个临时解决方案。最终，模型可能会退化到必须下线并重新训练的程度。为了检测这种最终失败点，我们提出将TTA与风险监控框架相结合，这些框架跟踪预测性能并在预定义的性能标准被违反时发出警报。SPECIFICALLY，我们扩展了基于顺序测试和置信序列的现有监控工具，以适应模型在测试时更新且无法获取测试标签以估计所需性能指标的场景。我们的扩展解锁了严格统计风险监控在TTA中的应用，并通过代表性数据集、分布偏移类型和TTA方法展示了我们提出的TTA监控框架的有效性。 

---
# Multilingual Multimodal Software Developer for Code Generation 

**Title (ZH)**: 多语言多模态软件开发人员：代码生成 

**Authors**: Linzheng Chai, Jian Yang, Shukai Liu, Wei Zhang, Liran Wang, Ke Jin, Tao Sun, Congnan Liu, Chenchen Zhang, Hualei Zhu, Jiaheng Liu, Xianjie Wu, Ge Zhang, Tianyu Liu, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.08719)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has significantly improved code generation, yet most models remain text-only, neglecting crucial visual aids like diagrams and flowcharts used in real-world software development. To bridge this gap, we introduce MM-Coder, a Multilingual Multimodal software developer. MM-Coder integrates visual design inputs-Unified Modeling Language (UML) diagrams and flowcharts (termed Visual Workflow)-with textual instructions to enhance code generation accuracy and architectural alignment. To enable this, we developed MMc-Instruct, a diverse multimodal instruction-tuning dataset including visual-workflow-based code generation, allowing MM-Coder to synthesize textual and graphical information like human developers, distinct from prior work on narrow tasks. Furthermore, we introduce MMEval, a new benchmark for evaluating multimodal code generation, addressing existing text-only limitations. Our evaluations using MMEval highlight significant remaining challenges for models in precise visual information capture, instruction following, and advanced programming knowledge. Our work aims to revolutionize industrial programming by enabling LLMs to interpret and implement complex specifications conveyed through both text and visual designs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展显著提升了代码生成能力，然而大多数模型仍局限于文本生成，忽视了实际软件开发中至关重要的视觉辅助工具，如图表和流程图。为弥合这一差距，我们引入了MM-Coder，这是一种多语言多模态软件开发者。MM-Coder 将视觉设计输入（如统一建模语言UML图表和流程图，统称为视觉工作流）与文本指令相结合，以提高代码生成的准确性和架构一致性。为此，我们开发了MMc-Instruct，这是一种包含基于视觉工作流的代码生成的多样多模态指令调优数据集，使MM-Coder 能够像人类开发者一样综合处理文本和图形信息，有别于此前仅针对狭窄任务的工作。此外，我们引入了MMEval，这是一种新的多模态代码生成评估基准，解决了现有单一文本限制。使用MMEval的评估突显了模型在精确视觉信息捕捉、指令遵循以及高级编程知识方面仍存在的重大挑战。我们的工作旨在通过使LLMs能够理解和实施通过文字和视觉设计传达的复杂规范，革新工业编程。 

---
# KG-Attention: Knowledge Graph-Guided Attention at Test-Time via Bidirectional Information Aggregation 

**Title (ZH)**: KG-Attention: 测试时基于双向信息聚合的知识图谱引导注意力 

**Authors**: Songlin Zhai, Guilin Qi, Yuan Meng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08704)  

**Abstract**: Knowledge graphs (KGs) play a critical role in enhancing large language models (LLMs) by introducing structured and grounded knowledge into the learning process. However, most existing KG-enhanced approaches rely on parameter-intensive fine-tuning, which risks catastrophic forgetting and degrades the pretrained model's generalization. Moreover, they exhibit limited adaptability to real-time knowledge updates due to their static integration frameworks. To address these issues, we introduce the first test-time KG-augmented framework for LLMs, built around a dedicated knowledge graph-guided attention (KGA) module that enables dynamic knowledge fusion without any parameter updates. The proposed KGA module augments the standard self-attention mechanism with two synergistic pathways: outward and inward aggregation. Specifically, the outward pathway dynamically integrates external knowledge into input representations via input-driven KG fusion. This inward aggregation complements the outward pathway by refining input representations through KG-guided filtering, suppressing task-irrelevant signals and amplifying knowledge-relevant patterns. Importantly, while the outward pathway handles knowledge fusion, the inward path selects the most relevant triples and feeds them back into the fusion process, forming a closed-loop enhancement mechanism. By synergistically combining these two pathways, the proposed method supports real-time knowledge fusion exclusively at test-time, without any parameter modification. Extensive experiments on five benchmarks verify the comparable knowledge fusion performance of KGA. 

**Abstract (ZH)**: 知识图谱增强的大语言模型测试时动态知识融合框架 

---
# ONION: A Multi-Layered Framework for Participatory ER Design 

**Title (ZH)**: 洋葱模型：一种参与式ER设计的多层框架 

**Authors**: Viktoriia Makovska, George Fletcher, Julia Stoyanovich  

**Link**: [PDF](https://arxiv.org/pdf/2507.08702)  

**Abstract**: We present ONION, a multi-layered framework for participatory Entity-Relationship (ER) modeling that integrates insights from design justice, participatory AI, and conceptual modeling. ONION introduces a five-stage methodology: Observe, Nurture, Integrate, Optimize, Normalize. It supports progressive abstraction from unstructured stakeholder input to structured ER diagrams.
Our approach aims to reduce designer bias, promote inclusive participation, and increase transparency through the modeling process. We evaluate ONION through real-world workshops focused on sociotechnical systems in Ukraine, highlighting how diverse stakeholder engagement leads to richer data models and deeper mutual understanding. Early results demonstrate ONION's potential to host diversity in early-stage data modeling. We conclude with lessons learned, limitations and challenges involved in scaling and refining the framework for broader adoption. 

**Abstract (ZH)**: We Presented ONION, 一种融合设计正义、参与式AI和概念建模洞察的多层实体-关系 modeling框架及其五阶段方法学 

---
# A Personalised Formal Verification Framework for Monitoring Activities of Daily Living of Older Adults Living Independently in Their Homes 

**Title (ZH)**: 独立居家老年人日常活动个性化形式验证框架 

**Authors**: Ricardo Contreras, Filip Smola, Nuša Farič, Jiawei Zheng, Jane Hillston, Jacques D. Fleuriot  

**Link**: [PDF](https://arxiv.org/pdf/2507.08701)  

**Abstract**: There is an imperative need to provide quality of life to a growing population of older adults living independently. Personalised solutions that focus on the person and take into consideration their preferences and context are key. In this work, we introduce a framework for representing and reasoning about the Activities of Daily Living of older adults living independently at home. The framework integrates data from sensors and contextual information that aggregates semi-structured interviews, home layouts and sociological observations from the participants. We use these data to create formal models, personalised for each participant according to their preferences and context. We formulate requirements that are specific to each individual as properties encoded in Linear Temporal Logic and use a model checker to verify whether each property is satisfied by the model. When a property is violated, a counterexample is generated giving the cause of the violation. We demonstrate the framework's generalisability by applying it to different participants, highlighting its potential to enhance the safety and well-being of older adults ageing in place. 

**Abstract (ZH)**: 独立居住的老年人提高生活质量的迫切需求需要个性化解决方案。本研究引入了一个框架，用于表示和推理独立居住老年人的日常生活活动。该框架整合了来自传感器的数据以及半结构化访谈、家庭布局和参与者的社会观察等上下文信息。我们使用这些数据为每位参与者创建形式模型，并根据其偏好和情境进行个性化。我们将针对每位个人的具体要求编码为线性时序逻辑属性，并使用模型检查器验证这些属性是否由模型满足。当属性被违反时，会产生反例以指出违规的原因。我们通过将其应用于不同的参与者，展示了该框架的普遍性，并突出了其在原地老化老年人的安全和福祉方面的潜力。 

---
# MoSAiC: Multi-Modal Multi-Label Supervision-Aware Contrastive Learning for Remote Sensing 

**Title (ZH)**: MoSAiC: 多模态多标签监督aware对比学习在遥感中的应用 

**Authors**: Debashis Gupta, Aditi Golder, Rongkhun Zhu, Kangning Cui, Wei Tang, Fan Yang, Ovidiu Csillik, Sarra Alaqahtani, V. Paul Pauca  

**Link**: [PDF](https://arxiv.org/pdf/2507.08683)  

**Abstract**: Contrastive learning (CL) has emerged as a powerful paradigm for learning transferable representations without the reliance on large labeled datasets. Its ability to capture intrinsic similarities and differences among data samples has led to state-of-the-art results in computer vision tasks. These strengths make CL particularly well-suited for Earth System Observation (ESO), where diverse satellite modalities such as optical and SAR imagery offer naturally aligned views of the same geospatial regions. However, ESO presents unique challenges, including high inter-class similarity, scene clutter, and ambiguous boundaries, which complicate representation learning -- especially in low-label, multi-label settings. Existing CL frameworks often focus on intra-modality self-supervision or lack mechanisms for multi-label alignment and semantic precision across modalities. In this work, we introduce MoSAiC, a unified framework that jointly optimizes intra- and inter-modality contrastive learning with a multi-label supervised contrastive loss. Designed specifically for multi-modal satellite imagery, MoSAiC enables finer semantic disentanglement and more robust representation learning across spectrally similar and spatially complex classes. Experiments on two benchmark datasets, BigEarthNet V2.0 and Sent12MS, show that MoSAiC consistently outperforms both fully supervised and self-supervised baselines in terms of accuracy, cluster coherence, and generalization in low-label and high-class-overlap scenarios. 

**Abstract (ZH)**: 对比学习（CL）已成为一种无需依赖大量标注数据集的学习可转移表示的强大范式。其能够捕获数据样本内部相似性和差异性的能力使其在计算机视觉任务中取得了最先进结果。这些优势使CL特别适合地球系统观测（ESO），其中如光学和SAR图像等多元卫星模态提供了同一地理区域的自然对齐视图。然而，ESO带来了独特的挑战，包括高类间相似性、场景杂乱和模糊边界，这些都使表示学习复杂化——尤其是在低标记、多标签设置中。现有CL框架往往侧重于单一模态的自我监督或缺乏跨模态多标签对齐和语义精度的机制。在本文中，我们提出了MoSAiC，这是一种统一框架，通过多标签监督对比损失联合优化模内和模间的对比学习。MoSAiC专为多元卫星图像设计，使人们能够更细致地分离语义并提高在光谱相似和空间复杂类别中表示学习的稳健性。在两个基准数据集BigEarthNet V2.0和Sent12MS上的实验表明，MoSAiC在准确性、聚类一致性和低标签、高类别重叠场景中的泛化能力上均优于完全监督和自我监督基线。 

---
# KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment 

**Title (ZH)**: KELPS：一种通过语义-语法对齐进行验证的多语言自动形式化框架 

**Authors**: Jiyao Zhang, Chengli Zhong, Hui Xu, Qige Li, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08665)  

**Abstract**: Modern large language models (LLMs) show promising progress in formalizing informal mathematics into machine-verifiable theorems. However, these methods still face bottlenecks due to the limited quantity and quality of multilingual parallel corpora. In this paper, we propose a novel neuro-symbolic framework KELPS (Knowledge-Equation based Logical Processing System) to address these problems. KELPS is an iterative framework for translating, synthesizing, and filtering informal data into multiple formal languages (Lean, Coq, and Isabelle). First, we translate natural language into Knowledge Equations (KEs), a novel language that we designed, theoretically grounded in assertional logic. Next, we convert them to target languages through rigorously defined rules that preserve both syntactic structure and semantic meaning. This process yielded a parallel corpus of over 60,000 problems. Our framework achieves 88.9% syntactic accuracy (pass@1) on MiniF2F, outperforming SOTA models such as Deepseek-V3 (81%) and Herald (81.3%) across multiple datasets. All datasets and codes are available in the supplementary materials. 

**Abstract (ZH)**: 现代大型语言模型在将非形式化数学转换为机器可验证定理方面展现了前景，但仍然受限于多语言平行语料库的数量和质量不足。本文提出了一种新的神经符号框架KELPS（基于知识-方程的逻辑处理系统）来解决这些问题。KELPS是一种迭代框架，用于将非形式化数据翻译、合成和过滤为多种正式语言（Lean、Coq和Isabelle）。首先，我们将自然语言转换为我们设计的一种新型语言知识方程（KEs），其理论基础是断言逻辑。然后，通过严格定义的规则将其转换为目标语言，同时保留其语义意义和语法结构。这一过程产生了超过60,000个问题的平行语料库。我们的框架在MiniF2F上的语法准确率（pass@1）达到了88.9%，在多个数据集上超过了包括Deepseek-V3（81%）和Herald（81.3%）在内的当前最佳模型。所有数据集和代码都包含在补充材料中。 

---
# Safe Deep Reinforcement Learning for Resource Allocation with Peak Age of Information Violation Guarantees 

**Title (ZH)**: 资源分配中的安全深度强化学习及其峰值年龄信息违例保证 

**Authors**: Berire Gunes Reyhan, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2507.08653)  

**Abstract**: In Wireless Networked Control Systems (WNCSs), control and communication systems must be co-designed due to their strong interdependence. This paper presents a novel optimization theory-based safe deep reinforcement learning (DRL) framework for ultra-reliable WNCSs, ensuring constraint satisfaction while optimizing performance, for the first time in the literature. The approach minimizes power consumption under key constraints, including Peak Age of Information (PAoI) violation probability, transmit power, and schedulability in the finite blocklength regime. PAoI violation probability is uniquely derived by combining stochastic maximum allowable transfer interval (MATI) and maximum allowable packet delay (MAD) constraints in a multi-sensor network. The framework consists of two stages: optimization theory and safe DRL. The first stage derives optimality conditions to establish mathematical relationships among variables, simplifying and decomposing the problem. The second stage employs a safe DRL model where a teacher-student framework guides the DRL agent (student). The control mechanism (teacher) evaluates compliance with system constraints and suggests the nearest feasible action when needed. Extensive simulations show that the proposed framework outperforms rule-based and other optimization theory based DRL benchmarks, achieving faster convergence, higher rewards, and greater stability. 

**Abstract (ZH)**: 无线网络控制系统（WNCSs）中，控制与通信系统必须由于其紧密的相互依赖性而协同设计。本文提出了一种基于优化理论的全新安全深度强化学习（DRL）框架，确保在优化性能的同时满足约束条件，这是文献中的首次尝试。该方法在关键约束条件下（包括峰值年龄信息（PAoI）违规概率、传输功率和有限块长度范围内的可调度性）最小化能耗。PAoI违规概率是通过结合多传感器网络中的随机最大允许传输间隔（MATI）和最大允许包延迟（MAD）约束唯一推导出来的。该框架包括两个阶段：优化理论和安全DRL。第一个阶段推导出最优条件，建立变量之间的数学关系，使问题更简洁和分解。第二阶段采用一种安全DRL模型，其中导师-学生框架引导DRL代理（学生）。控制机制（导师）评估系统约束的遵守情况，并在必要时建议最近可行的操作。大量仿真实验表明，所提出的框架优于基于规则和其他优化理论的DRL基准，实现了更快的收敛速度、更高的奖励和更大的稳定性。 

---
# DatasetAgent: A Novel Multi-Agent System for Auto-Constructing Datasets from Real-World Images 

**Title (ZH)**: DatasetAgent：一种新型多智能体系统，用于从真实世界图像自动生成数据集 

**Authors**: Haoran Sun, Haoyu Bian, Shaoning Zeng, Yunbo Rao, Xu Xu, Lin Mei, Jianping Gou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08648)  

**Abstract**: Common knowledge indicates that the process of constructing image datasets usually depends on the time-intensive and inefficient method of manual collection and annotation. Large models offer a solution via data generation. Nonetheless, real-world data are obviously more valuable comparing to artificially intelligence generated data, particularly in constructing image datasets. For this reason, we propose a novel method for auto-constructing datasets from real-world images by a multiagent collaborative system, named as DatasetAgent. By coordinating four different agents equipped with Multi-modal Large Language Models (MLLMs), as well as a tool package for image optimization, DatasetAgent is able to construct high-quality image datasets according to user-specified requirements. In particular, two types of experiments are conducted, including expanding existing datasets and creating new ones from scratch, on a variety of open-source datasets. In both cases, multiple image datasets constructed by DatasetAgent are used to train various vision models for image classification, object detection, and image segmentation. 

**Abstract (ZH)**: 现有的常识表明，构建图像数据集的过程通常依赖于耗时且低效的手动收集和标注方法。大型模型通过数据生成提供了一种解决方案。然而，现实世界数据显然比人工智能生成的数据更具价值，特别是在构建图像数据集方面。因此，我们提出了一种由多智能体协作系统实现的自动数据集构建方法，名为DatasetAgent。通过协调四个不同智能体，配备多模态大型语言模型（MLLMs），以及一个图像优化工具包，DatasetAgent能够根据用户指定的要求构建高质量的图像数据集。特别地，在多种开源数据集上进行了两种类型实验，包括扩展现有数据集和从零开始创建新的数据集。在两种情况下，均由DatasetAgent构建的多个图像数据集被用于训练各种视觉模型，用于图像分类、目标检测和图像分割。 

---
# Scaling Attention to Very Long Sequences in Linear Time with Wavelet-Enhanced Random Spectral Attention (WERSA) 

**Title (ZH)**: 使用小波增强随机谱注意力（WERSA）实现线性时间下的超长序列注意力扩展 

**Authors**: Vincenzo Dentamaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08637)  

**Abstract**: Transformer models are computationally costly on long sequences since regular attention has quadratic $O(n^2)$ time complexity. We introduce Wavelet-Enhanced Random Spectral Attention (WERSA), a novel mechanism of linear $O(n)$ time complexity that is pivotal to enable successful long-sequence processing without the performance trade-off. WERSA merges content-adaptive random spectral features together with multi-resolution Haar wavelets and learnable parameters to selectively attend to informative scales of data while preserving linear efficiency.
Large-scale comparisons \textbf{on single GPU} and across various benchmarks (vision, NLP, hierarchical reasoning) and various attention mechanisms (like Multiheaded Attention, Flash-Attention-2, FNet, Linformer, Performer, Waveformer), reveal uniform advantages of WERSA. It achieves best accuracy in all tests. On ArXiv classification, WERSA improves accuracy over vanilla attention by 1.2\% (86.2\% vs 85.0\%) while cutting training time by 81\% (296s vs 1554s) and FLOPS by 73.4\% (26.2G vs 98.4G). Significantly, WERSA excels where vanilla and FlashAttention-2 fail: on ArXiv-128k's extremely lengthy sequences, it achieves best accuracy (79.1\%) and AUC (0.979) among viable methods, operating on data that gives Out-Of-Memory errors to quadratic methods while being \textbf{twice as fast} as Waveformer, its next-best competitor.
By significantly reducing computational loads without compromising accuracy, WERSA makes possible more practical, more affordable, long-context models, in particular on low-resource hardware, for more sustainable and more scalable AI development. 

**Abstract (ZH)**: Wavelet-Enhanced Random Spectral Attention: A Linear Time Complexity Mechanism Enabling Efficient Long-Sequence Processing Without Performance Trade-Off 

---
# Normalized vs Diplomatic Annotation: A Case Study of Automatic Information Extraction from Handwritten Uruguayan Birth Certificates 

**Title (ZH)**: 规范化标注 vs 外交式标注：来自乌拉圭手写出生证明的自动信息提取案例研究 

**Authors**: Natalia Bottaioli, Solène Tarride, Jérémy Anger, Seginus Mowlavi, Marina Gardella, Antoine Tadros, Gabriele Facciolo, Rafael Grompone von Gioi, Christopher Kermorvant, Jean-Michel Morel, Javier Preciozzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08636)  

**Abstract**: This study evaluates the recently proposed Document Attention Network (DAN) for extracting key-value information from Uruguayan birth certificates, handwritten in Spanish. We investigate two annotation strategies for automatically transcribing handwritten documents, fine-tuning DAN with minimal training data and annotation effort. Experiments were conducted on two datasets containing the same images (201 scans of birth certificates written by more than 15 different writers) but with different annotation methods. Our findings indicate that normalized annotation is more effective for fields that can be standardized, such as dates and places of birth, whereas diplomatic annotation performs much better for fields containing names and surnames, which can not be standardized. 

**Abstract (ZH)**: 本研究评估了最近提出的文档注意力网络（DAN）对乌拉圭出生证明（手写西班牙文）中关键信息的提取效果，并调查了两种标注策略以自动转录手写文档，在训练数据和标注努力最少的情况下对DAN进行微调。实验在包含相同图像的两个数据集（201份由15名以上不同书写者撰写的出生证明扫描件）上进行，但使用了不同的标注方法。研究发现，标准化标注对可以标准化的字段（如出生日期和出生地点）更有效，而外交标注在包含姓名和姓氏等无法标准化的字段时表现更好。 

---
# Adaptive Framework for Ambient Intelligence in Rehabilitation Assistance 

**Title (ZH)**: 适应性框架：用于康复辅助的环境智能 

**Authors**: Gábor Baranyi, Zsolt Csibi, Kristian Fenech, Áron Fóthi, Zsófia Gaál, Joul Skaf, András Lőrincz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08624)  

**Abstract**: This paper introduces the Ambient Intelligence Rehabilitation Support (AIRS) framework, an advanced artificial intelligence-based solution tailored for home rehabilitation environments. AIRS integrates cutting-edge technologies, including Real-Time 3D Reconstruction (RT-3DR), intelligent navigation, and large Vision-Language Models (VLMs), to create a comprehensive system for machine-guided physical rehabilitation. The general AIRS framework is demonstrated in rehabilitation scenarios following total knee replacement (TKR), utilizing a database of 263 video recordings for evaluation. A smartphone is employed within AIRS to perform RT-3DR of living spaces and has a body-matched avatar to provide visual feedback about the excercise. This avatar is necessary in (a) optimizing exercise configurations, including camera placement, patient positioning, and initial poses, and (b) addressing privacy concerns and promoting compliance with the AI Act. The system guides users through the recording process to ensure the collection of properly recorded videos. AIRS employs two feedback mechanisms: (i) visual 3D feedback, enabling direct comparisons between prerecorded clinical exercises and patient home recordings and (ii) VLM-generated feedback, providing detailed explanations and corrections for exercise errors. The framework also supports people with visual and hearing impairments. It also features a modular design that can be adapted to broader rehabilitation contexts. AIRS software components are available for further use and customization. 

**Abstract (ZH)**: 基于环境智能的康复支持框架（AIRS）：一种适用于家庭康复环境的先进人工智能解决方案 

---
# A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1 

**Title (ZH)**: 基于LLM的论证分类综合研究：从LLAMA到GPT-4再到Deepseek-R1 

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski  

**Link**: [PDF](https://arxiv.org/pdf/2507.08621)  

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as this http URL and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings. 

**Abstract (ZH)**: Argument 矿工（AM）是跨逻辑、哲学、语言学、修辞学、法律、心理学和计算机科学的多学科研究领域，涉及自动识别和提取论据组件（如前提和主张）及其关系（如支持、反对或中立）。近年来，随着大型语言模型（LLMs）的发展，该领域取得了显著进展，与传统方法和其他深度学习模型相比，大大提高了论据语义分析的效率。虽然有众多基准测试 LLM 的质量和性能，但在公共可获得的论据分类数据库中，对这些模型的操作研究和结果仍然不足。本文研究了一种选择的 LLM，并使用如 this http URL 和 UKP 等多样化的数据集。测试的模型包括 GPT、Llama 和 DeepSeek 的版本，以及包含 Chain-of-Thoughts 算法的推理增强版本。结果显示，ChatGPT-4 在论据分类基准测试中表现最佳。对于具备推理能力的模型，Deepseek-R1 表现更优。然而，尽管表现出色，ChatGPT-4 和 Deepseek-R1 仍然会出错。针对所有模型，最常见错误进行了讨论。根据我们的知识，本研究是首次对所述数据集进行更全面的 LLM 和提示算法分析，展示了已知提示算法在论据分析中的某些局限性，指出了改进方向。工作的增益在于对可用的论据数据集进行了深入分析，并展示了它们的不足。 

---
# Towards Collaborative Fairness in Federated Learning Under Imbalanced Covariate Shift 

**Title (ZH)**: 面向不平衡 covariate shift 下的联邦学习协作公平性研究 

**Authors**: Tianrun Yu, Jiaqi Wang, Haoyu Wang, Mingquan Lin, Han Liu, Nelson S. Yee, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08617)  

**Abstract**: Collaborative fairness is a crucial challenge in federated learning. However, existing approaches often overlook a practical yet complex form of heterogeneity: imbalanced covariate shift. We provide a theoretical analysis of this setting, which motivates the design of FedAKD (Federated Asynchronous Knowledge Distillation)- simple yet effective approach that balances accurate prediction with collaborative fairness. FedAKD consists of client and server updates. In the client update, we introduce a novel asynchronous knowledge distillation strategy based on our preliminary analysis, which reveals that while correctly predicted samples exhibit similar feature distributions across clients, incorrectly predicted samples show significant variability. This suggests that imbalanced covariate shift primarily arises from misclassified samples. Leveraging this insight, our approach first applies traditional knowledge distillation to update client models while keeping the global model fixed. Next, we select correctly predicted high-confidence samples and update the global model using these samples while keeping client models fixed. The server update simply aggregates all client models. We further provide a theoretical proof of FedAKD's convergence. Experimental results on public datasets (FashionMNIST and CIFAR10) and a real-world Electronic Health Records (EHR) dataset demonstrate that FedAKD significantly improves collaborative fairness, enhances predictive accuracy, and fosters client participation even under highly heterogeneous data distributions. 

**Abstract (ZH)**: 协作公平性是联邦学习中的关键挑战。然而，现有方法往往会忽视一种实际且复杂的异质性形式：特征偏移不均衡。我们对该场景进行了理论分析，这激发了FedAKD（联邦异步知识蒸馏）这一简单而有效的方法的设计，该方法平衡了准确预测与协作公平性。FedAKD包括客户端和服务器更新。在客户端更新中，我们引入了一种基于初步分析的新颖的异步知识蒸馏策略，揭示了尽管正确预测的样本在不同客户端上具有相似的特征分布，但错误预测的样本显示出显著的差异性。这表明特征偏移不均衡主要源自分类错误的样本。利用这一见解，我们的方法首先应用传统知识蒸馏更新客户端模型，同时固定全局模型。随后，我们选择高置信度的正确预测样本，用这些样本更新全局模型，同时保持客户端模型不变。服务器更新只是聚合所有客户端模型。我们还提供了FedAKD收敛性的理论证明。实验结果表明，FedAKD显著改善了协作公平性、增强了预测准确性，并即使在高度异质的数据分布下也促进了客户端的参与。 

---
# Generating Proto-Personas through Prompt Engineering: A Case Study on Efficiency, Effectiveness and Empathy 

**Title (ZH)**: 通过提示工程生成原型人格：一项关于效率、有效性和 Empathy 的案例研究 

**Authors**: Fernando Ayach, Vitor Lameirão, Raul Leão, Jerfferson Felizardo, Rafael Sobrinho, Vanessa Borges, Patrícia Matsubara, Awdren Fontão  

**Link**: [PDF](https://arxiv.org/pdf/2507.08594)  

**Abstract**: Proto-personas are commonly used during early-stage Product Discovery, such as Lean Inception, to guide product definition and stakeholder alignment. However, the manual creation of proto-personas is often time-consuming, cognitively demanding, and prone to bias. In this paper, we propose and empirically investigate a prompt engineering-based approach to generate proto-personas with the support of Generative AI (GenAI). Our goal is to evaluate the approach in terms of efficiency, effectiveness, user acceptance, and the empathy elicited by the generated personas. We conducted a case study with 19 participants embedded in a real Lean Inception, employing a qualitative and quantitative methods design. The results reveal the approach's efficiency by reducing time and effort and improving the quality and reusability of personas in later discovery phases, such as Minimum Viable Product (MVP) scoping and feature refinement. While acceptance was generally high, especially regarding perceived usefulness and ease of use, participants noted limitations related to generalization and domain specificity. Furthermore, although cognitive empathy was strongly supported, affective and behavioral empathy varied significantly across participants. These results contribute novel empirical evidence on how GenAI can be effectively integrated into software Product Discovery practices, while also identifying key challenges to be addressed in future iterations of such hybrid design processes. 

**Abstract (ZH)**: 基于提示工程的生成式AI支持下proto-人物角色生成方法及其有效性研究 

---
# To Trade or Not to Trade: An Agentic Approach to Estimating Market Risk Improves Trading Decisions 

**Title (ZH)**: 要不要交易：一种代理方法来估计市场风险以改善交易决策 

**Authors**: Dimitrios Emmanoulopoulos, Ollie Olby, Justin Lyon, Namid R. Stillman  

**Link**: [PDF](https://arxiv.org/pdf/2507.08584)  

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic frameworks, in which prompts trigger complex tool-based analysis in pursuit of a goal. While these frameworks have shown promise across multiple domains including in finance, they typically lack a principled model-building step, relying instead on sentiment- or trend-based analysis. We address this gap by developing an agentic system that uses LLMs to iteratively discover stochastic differential equations for financial time series. These models generate risk metrics which inform daily trading decisions. We evaluate our system in both traditional backtests and using a market simulator, which introduces synthetic but causally plausible price paths and news events. We find that model-informed trading strategies outperform standard LLM-based agents, improving Sharpe ratios across multiple equities. Our results show that combining LLMs with agentic model discovery enhances market risk estimation and enables more profitable trading decisions. 

**Abstract (ZH)**: 大型语言模型在追求目标的代理框架中的迭代发现 stochastic 微分方程以生成金融时间序列的风险指标并指导每日交易决策：一种结合 LLMs 与代理模型发现的交易策略 

---
# A Multi-Modal Fusion Framework for Brain Tumor Segmentation Based on 3D Spatial-Language-Vision Integration and Bidirectional Interactive Attention Mechanism 

**Title (ZH)**: 基于3D空间-语言-视觉集成和双向交互注意力机制的多模态融合框架用于脑肿瘤分割 

**Authors**: Mingda Zhang, Kaiwen Pan  

**Link**: [PDF](https://arxiv.org/pdf/2507.08574)  

**Abstract**: This study aims to develop a novel multi-modal fusion framework for brain tumor segmentation that integrates spatial-language-vision information through bidirectional interactive attention mechanisms to improve segmentation accuracy and boundary delineation. Methods: We propose two core components: Multi-modal Semantic Fusion Adapter (MSFA) integrating 3D MRI data with clinical text descriptions through hierarchical semantic decoupling, and Bidirectional Interactive Visual-semantic Attention (BIVA) enabling iterative information exchange between modalities. The framework was evaluated on BraTS 2020 dataset comprising 369 multi-institutional MRI scans. Results: The proposed method achieved average Dice coefficient of 0.8505 and 95% Hausdorff distance of 2.8256mm across enhancing tumor, tumor core, and whole tumor regions, outperforming state-of-the-art methods including SCAU-Net, CA-Net, and 3D U-Net. Ablation studies confirmed critical contributions of semantic and spatial modules to boundary precision. Conclusion: Multi-modal semantic fusion combined with bidirectional interactive attention significantly enhances brain tumor segmentation performance, establishing new paradigms for integrating clinical knowledge into medical image analysis. 

**Abstract (ZH)**: 本研究旨在通过双向交互注意力机制整合空间-语言-视觉信息，开发一种新颖的多模态融合框架以提高脑肿瘤分割精度和边界 delineation。方法：我们提出两个核心组件：多模态语义融合适配器（MSFA），实现3D MRI数据与临床文本描述的层次语义解耦融合，以及双向交互视觉-语义注意力（BIVA），以在模态间实现迭代信息交流。该框架在包含369个多机构MRI扫描的BraTS 2020数据集上进行了评估。结果：所提出的方法在增强肿瘤、肿瘤核心和整个肿瘤区域中获得了平均骰系数0.8505和95% Hausdorff距离2.8256mm，优于SCAU-Net、CA-Net和3D U-Net等最先进的方法。消融研究证实了语义和空间模块对边界精度的贡献至关重要。结论：结合多模态语义融合与双向交互注意力显著提升了脑肿瘤分割性能，为将临床知识整合到医学图像分析中建立了新的范式。 

---
# FreeAudio: Training-Free Timing Planning for Controllable Long-Form Text-to-Audio Generation 

**Title (ZH)**: FreeAudio: 无需训练的定时规划以实现可控的长形式文本到语音生成 

**Authors**: Yuxuan Jiang, Zehua Chen, Zeqian Ju, Chang Li, Weibei Dou, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08557)  

**Abstract**: Text-to-audio (T2A) generation has achieved promising results with the recent advances in generative models. However, because of the limited quality and quantity of temporally-aligned audio-text pairs, existing T2A methods struggle to handle the complex text prompts that contain precise timing control, e.g., "owl hooted at 2.4s-5.2s". Recent works have explored data augmentation techniques or introduced timing conditions as model inputs to enable timing-conditioned 10-second T2A generation, while their synthesis quality is still limited. In this work, we propose a novel training-free timing-controlled T2A framework, FreeAudio, making the first attempt to enable timing-controlled long-form T2A generation, e.g., "owl hooted at 2.4s-5.2s and crickets chirping at 0s-24s". Specifically, we first employ an LLM to plan non-overlapping time windows and recaption each with a refined natural language description, based on the input text and timing prompts. Then we introduce: 1) Decoupling and Aggregating Attention Control for precise timing control; 2) Contextual Latent Composition for local smoothness and Reference Guidance for global consistency. Extensive experiments show that: 1) FreeAudio achieves state-of-the-art timing-conditioned T2A synthesis quality among training-free methods and is comparable to leading training-based methods; 2) FreeAudio demonstrates comparable long-form generation quality with training-based Stable Audio and paves the way for timing-controlled long-form T2A synthesis. Demo samples are available at: this https URL 

**Abstract (ZH)**: 基于文本的无训练同步生成：FreeAudio框架 

---
# RadiomicsRetrieval: A Customizable Framework for Medical Image Retrieval Using Radiomics Features 

**Title (ZH)**: 基于影像omics特征的可定制医疗图像检索框架 

**Authors**: Inye Na, Nejung Rue, Jiwon Chung, Hyunjin Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.08546)  

**Abstract**: Medical image retrieval is a valuable field for supporting clinical decision-making, yet current methods primarily support 2D images and require fully annotated queries, limiting clinical flexibility. To address this, we propose RadiomicsRetrieval, a 3D content-based retrieval framework bridging handcrafted radiomics descriptors with deep learning-based embeddings at the tumor level. Unlike existing 2D approaches, RadiomicsRetrieval fully exploits volumetric data to leverage richer spatial context in medical images. We employ a promptable segmentation model (e.g., SAM) to derive tumor-specific image embeddings, which are aligned with radiomics features extracted from the same tumor via contrastive learning. These representations are further enriched by anatomical positional embedding (APE). As a result, RadiomicsRetrieval enables flexible querying based on shape, location, or partial feature sets. Extensive experiments on both lung CT and brain MRI public datasets demonstrate that radiomics features significantly enhance retrieval specificity, while APE provides global anatomical context essential for location-based searches. Notably, our framework requires only minimal user prompts (e.g., a single point), minimizing segmentation overhead and supporting diverse clinical scenarios. The capability to query using either image embeddings or selected radiomics attributes highlights its adaptability, potentially benefiting diagnosis, treatment planning, and research on large-scale medical imaging repositories. Our code is available at this https URL. 

**Abstract (ZH)**: 医学影像检索是支持临床决策的重要领域，现有的方法主要支持2D图像且需要完全标注的查询，限制了临床的灵活性。为了解决这一问题，我们提出了一种名为RadiomicsRetrieval的3D内容基于检索框架，将手工构建的影像组学描述符与基于深度学习的肿瘤级嵌入相结合。与现有的2D方法不同，RadiomicsRetrieval充分利用了体数据，利用了医学影像中的更丰富的空间语境。我们采用可提示的分割模型（如SAM）来提取肿瘤特异性图像嵌入，这些嵌入与通过对比学习从同一肿瘤中提取的影像组学特征对齐。进一步通过解剖位置嵌入（APE）增强了这些表示。结果，RadiomicsRetrieval使得基于形状、位置或部分特征集的灵活查询成为可能。在肺CT和脑MRI公开数据集上的广泛实验表明，影像组学特征显著提高了检索的特异性，而解剖位置嵌入（APE）提供了基于位置搜索所需的重要全局解剖上下文。值得注意的是，我们的框架仅需极少的用户提示（例如，一个点），从而减少了分割开销并支持了多种临床场景。既能使用图像嵌入也能使用选择的影像组学属性进行查询的能力突显了其适应性，有望造福于大规模医学影像库的诊断、治疗计划和研究。我们的代码可在以下链接获得：this https URL。 

---
# White-Basilisk: A Hybrid Model for Code Vulnerability Detection 

**Title (ZH)**: 白蜥蜴：一种代码漏洞检测的混合模型 

**Authors**: Ioannis Lamprou, Alexander Shevtsov, Ioannis Arapakis, Sotiris Ioannidis  

**Link**: [PDF](https://arxiv.org/pdf/2507.08540)  

**Abstract**: The proliferation of software vulnerabilities presents a significant challenge to cybersecurity, necessitating more effective detection methodologies. We introduce White-Basilisk, a novel approach to vulnerability detection that demonstrates superior performance while challenging prevailing assumptions in AI model scaling. Utilizing an innovative architecture that integrates Mamba layers, linear self-attention, and a Mixture of Experts framework, White-Basilisk achieves state-of-the-art results in vulnerability detection tasks with a parameter count of only 200M. The model's capacity to process sequences of unprecedented length enables comprehensive analysis of extensive codebases in a single pass, surpassing the context limitations of current Large Language Models (LLMs). White-Basilisk exhibits robust performance on imbalanced, real-world datasets, while maintaining computational efficiency that facilitates deployment across diverse organizational scales. This research not only establishes new benchmarks in code security but also provides empirical evidence that compact, efficiently designed models can outperform larger counterparts in specialized tasks, potentially redefining optimization strategies in AI development for domain-specific applications. 

**Abstract (ZH)**: 白骨草：一种在有效性上超越现有假设的新型漏洞检测方法 

---
# MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling 

**Title (ZH)**: MIDI-VALLE: 通过神经编码器语言建模提高表达性钢琴表演合成 

**Authors**: Jingjing Tang, Xin Wang, Zhe Zhang, Junichi Yamagishi, Geraint Wiggins, George Fazekas  

**Link**: [PDF](https://arxiv.org/pdf/2507.08530)  

**Abstract**: Generating expressive audio performances from music scores requires models to capture both instrument acoustics and human interpretation. Traditional music performance synthesis pipelines follow a two-stage approach, first generating expressive performance MIDI from a score, then synthesising the MIDI into audio. However, the synthesis models often struggle to generalise across diverse MIDI sources, musical styles, and recording environments. To address these challenges, we propose MIDI-VALLE, a neural codec language model adapted from the VALLE framework, which was originally designed for zero-shot personalised text-to-speech (TTS) synthesis. For performance MIDI-to-audio synthesis, we improve the architecture to condition on a reference audio performance and its corresponding MIDI. Unlike previous TTS-based systems that rely on piano rolls, MIDI-VALLE encodes both MIDI and audio as discrete tokens, facilitating a more consistent and robust modelling of piano performances. Furthermore, the model's generalisation ability is enhanced by training on an extensive and diverse piano performance dataset. Evaluation results show that MIDI-VALLE significantly outperforms a state-of-the-art baseline, achieving over 75% lower Frechet Audio Distance on the ATEPP and Maestro datasets. In the listening test, MIDI-VALLE received 202 votes compared to 58 for the baseline, demonstrating improved synthesis quality and generalisation across diverse performance MIDI inputs. 

**Abstract (ZH)**: 从乐谱生成表达性强的音频表演需要模型捕捉乐器音色和人类诠释。传统的音乐表演合成管道采用两阶段方法，首先从乐谱生成具有表现力的MIDI表演，然后将其合成成为音频。然而，合成模型往往难以在多样化的MIDI源、音乐风格和录音环境中做到泛化。为应对这些挑战，我们提出MIDI-VALLE，这是一种源自零样本个性化文本转语音（TTS）合成框架VALLE的神经编码语言模型。对于表演MIDI到音频的合成，我们改进了架构，使其能够根据参考音频表演及其对应的MIDI进行条件化。与依赖琴键展开图的以前的TTS系统不同，MIDI-VALLE将MIDI和音频编码为离散令牌，促进了更一致和鲁棒的钢琴表演建模。此外，通过在广泛多样化的钢琴表演数据集上训练，模型的泛化能力得到了增强。评估结果表明，MIDI-VALLE在ATEPP和Maestro数据集中实现了超过75%更低的Frechet音频距离，显著优于最先进的基线模型。在听觉测试中，MIDI-VALLE获得了202票，而基线模型为58票，这表明其在多样化的MIDI表演输入上的合成质量和泛化能力得到了提高。 

---
# PromotionGo at SemEval-2025 Task 11: A Feature-Centric Framework for Cross-Lingual Multi-Emotion Detection in Short Texts 

**Title (ZH)**: SemEval-2025 任务 11 中的 PromotionGo：一种面向特征的跨语言短文本多情绪检测框架 

**Authors**: Ziyi Huang, Xia Cui  

**Link**: [PDF](https://arxiv.org/pdf/2507.08499)  

**Abstract**: This paper presents our system for SemEval 2025 Task 11: Bridging the Gap in Text-Based Emotion Detection (Track A), which focuses on multi-label emotion detection in short texts. We propose a feature-centric framework that dynamically adapts document representations and learning algorithms to optimize language-specific performance. Our study evaluates three key components: document representation, dimensionality reduction, and model training in 28 languages, highlighting five for detailed analysis. The results show that TF-IDF remains highly effective for low-resource languages, while contextual embeddings like FastText and transformer-based document representations, such as those produced by Sentence-BERT, exhibit language-specific strengths. Principal Component Analysis (PCA) reduces training time without compromising performance, particularly benefiting FastText and neural models such as Multi-Layer Perceptrons (MLP). Computational efficiency analysis underscores the trade-off between model complexity and processing cost. Our framework provides a scalable solution for multilingual emotion detection, addressing the challenges of linguistic diversity and resource constraints. 

**Abstract (ZH)**: 本文介绍了我们用于SemEval 2025 Task 11：跨越文本情感检测中的鸿沟（Track A）的系统，该任务专注于短文本的多标签情感检测。我们提出了一种以特征为中心的框架，动态调整文档表示和学习算法以优化语言特定的表现。研究评估了三种关键组件：文档表示、降维和模型训练，涵盖28种语言，并详细分析了其中五种。结果显示，TF-IDF 对于低资源语言仍然非常有效，而如FastText这样的上下文嵌入和由Sentence-BERT生成的基于变换器的文档表示在不同语言上展现出语言特定的优势。主成分分析（PCA）在不牺牲性能的情况下减少了训练时间，特别是在FastText和多层感知机（MLP）等神经模型上。计算效率分析强调了模型复杂性和处理成本之间的权衡。我们的框架提供了一种针对多语言情感检测的可扩展解决方案，应对语言多样性和资源限制的挑战。 

---
# Enhancing Essay Cohesion Assessment: A Novel Item Response Theory Approach 

**Title (ZH)**: 增强论文连贯性评估：一种新型项目反应理论方法 

**Authors**: Bruno Alexandre Rosa, Hilário Oliveira, Luiz Rodrigues, Eduardo Araujo Oliveira, Rafael Ferreira Mello  

**Link**: [PDF](https://arxiv.org/pdf/2507.08487)  

**Abstract**: Essays are considered a valuable mechanism for evaluating learning outcomes in writing. Textual cohesion is an essential characteristic of a text, as it facilitates the establishment of meaning between its parts. Automatically scoring cohesion in essays presents a challenge in the field of educational artificial intelligence. The machine learning algorithms used to evaluate texts generally do not consider the individual characteristics of the instances that comprise the analysed corpus. In this meaning, item response theory can be adapted to the context of machine learning, characterising the ability, difficulty and discrimination of the models used. This work proposes and analyses the performance of a cohesion score prediction approach based on item response theory to adjust the scores generated by machine learning models. In this study, the corpus selected for the experiments consisted of the extended Essay-BR, which includes 6,563 essays in the style of the National High School Exam (ENEM), and the Brazilian Portuguese Narrative Essays, comprising 1,235 essays written by 5th to 9th grade students from public schools. We extracted 325 linguistic features and treated the problem as a machine learning regression task. The experimental results indicate that the proposed approach outperforms conventional machine learning models and ensemble methods in several evaluation metrics. This research explores a potential approach for improving the automatic evaluation of cohesion in educational essays. 

**Abstract (ZH)**: essays被认为是评估写作学习成果的一种有价值的机制。文本连贯性是文本的一个重要特征，因为它有助于文本各部分之间意义的建立。在教育人工智能领域，自动评分连贯性是一项挑战。用于评估文本的机器学习算法通常不会考虑所分析语料库中个体实例的独特特征。在这种意义上，可以将项目反应理论适应机器学习的语境，表征所使用的模型的能力、难度和区分度。本文提出并分析了一种基于项目反应理论的连贯性评分预测方法，以调整机器学习模型生成的分数。在本研究中，用于实验的语料库包括扩展的Essay-BR，其中包含6,563篇按照全国高中考试（ENEM）风格撰写的作文，以及由公立学校5至9年级学生撰写的1,235篇巴西葡萄牙语叙事作文。我们提取了325个语言特征，并将问题视为一个机器学习回归任务。实验结果表明，所提出的方法在多个评估指标上优于传统机器学习模型和集成方法。本文探索了提高教育作文中连贯性自动评估的一种潜在方法。 

---
# Pre-Training LLMs on a budget: A comparison of three optimizers 

**Title (ZH)**: 在预算范围内预训练大规模语言模型：三种优化器的比较 

**Authors**: Joel Schlotthauer, Christian Kroos, Chris Hinze, Viktor Hangya, Luzian Hahn, Fabian Küch  

**Link**: [PDF](https://arxiv.org/pdf/2507.08472)  

**Abstract**: Optimizers play a decisive role in reducing pre-training times for LLMs and achieving better-performing models. In this study, we compare three major variants: the de-facto standard AdamW, the simpler Lion, developed through an evolutionary search, and the second-order optimizer Sophia. For better generalization, we train with two different base architectures and use a single- and a multiple-epoch approach while keeping the number of tokens constant. Using the Maximal Update Parametrization and smaller proxy models, we tune relevant hyperparameters separately for each combination of base architecture and optimizer. We found that while the results from all three optimizers were in approximately the same range, Sophia exhibited the lowest training and validation loss, Lion was fastest in terms of training GPU hours but AdamW led to the best downstream evaluation results. 

**Abstract (ZH)**: 优化器在降低大语言模型预训练时间和提升模型性能方面起着决定性作用。本研究比较了三种主要变体：事实上的标准AdamW、通过进化搜索开发的简单Lion以及第二阶优化器Sophia。为了提高泛化能力，我们使用了两种不同的基架构进行训练，并采用单 epoch 和多 epoch 方法，同时保持token数量不变。利用Maximal Update Parametrization和较小的代理模型，我们分别对每种基架构和优化器组合进行相关超参数调整。研究发现，尽管所有三种优化器的结果大致在同一范围内，但Sophia在训练和验证损失方面最低，Lion在训练GPU小时内最快，而AdamW在下游评估结果方面表现最好。 

---
# A document is worth a structured record: Principled inductive bias design for document recognition 

**Title (ZH)**: 一份文档等同于一个结构化的记录：,



 principled 归并为 “合理的”,



 inductive bias design 归并为 “归纳偏置设计”,



 document recognition 归并为 “文档识别”。



因此，翻译后的标题为：



一份文档等同于一个结构化的记录：合理的归纳偏置设计用于文档识别。 

**Authors**: Benjamin Meyer, Lukas Tuggener, Sascha Hänzi, Daniel Schmid, Erdal Ayfer, Benjamin F. Grewe, Ahmed Abdulkadir, Thilo Stadelmann  

**Link**: [PDF](https://arxiv.org/pdf/2507.08458)  

**Abstract**: Many document types use intrinsic, convention-driven structures that serve to encode precise and structured information, such as the conventions governing engineering drawings. However, state-of-the-art approaches treat document recognition as a mere computer vision problem, neglecting these underlying document-type-specific structural properties, making them dependent on sub-optimal heuristic post-processing and rendering many less frequent or more complicated document types inaccessible to modern document recognition. We suggest a novel perspective that frames document recognition as a transcription task from a document to a record. This implies a natural grouping of documents based on the intrinsic structure inherent in their transcription, where related document types can be treated (and learned) similarly. We propose a method to design structure-specific inductive biases for the underlying machine-learned end-to-end document recognition systems, and a respective base transformer architecture that we successfully adapt to different structures. We demonstrate the effectiveness of the so-found inductive biases in extensive experiments with progressively complex record structures from monophonic sheet music, shape drawings, and simplified engineering drawings. By integrating an inductive bias for unrestricted graph structures, we train the first-ever successful end-to-end model to transcribe engineering drawings to their inherently interlinked information. Our approach is relevant to inform the design of document recognition systems for document types that are less well understood than standard OCR, OMR, etc., and serves as a guide to unify the design of future document foundation models. 

**Abstract (ZH)**: 基于结构转换的文档识别新视角：从文档到记录的转录任务 

---
# Space filling positionality and the Spiroformer 

**Title (ZH)**: 空间填充位置性与Spiroformer 

**Authors**: M. Maurin, M.Á. Evangelista-Alvarado, P. Suárez-Serrato  

**Link**: [PDF](https://arxiv.org/pdf/2507.08456)  

**Abstract**: Transformers excel when dealing with sequential data. Generalizing transformer models to geometric domains, such as manifolds, we encounter the problem of not having a well-defined global order. We propose a solution with attention heads following a space-filling curve. As a first experimental example, we present the Spiroformer, a transformer that follows a polar spiral on the $2$-sphere. 

**Abstract (ZH)**: Transformer模型在处理序列数据方面表现出色。将Transformer模型推广到几何领域，如流形，我们遇到了缺乏全局顺序定义的问题。我们提出了一种解决方案，即让注意力头遵循空间填充曲线。作为首个实验示例，我们介绍了Spiroformer模型，该模型在二维球面上遵循螺旋轨迹。 

---
# Review of Feed-forward 3D Reconstruction: From DUSt3R to VGGT 

**Title (ZH)**: feed-forward 3D重建综述：从DUSt3R到VGGT 

**Authors**: Wei Zhang, Yihang Wu, Songhua Li, Wenjie Ma, Xin Ma, Qiang Li, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08448)  

**Abstract**: 3D reconstruction, which aims to recover the dense three-dimensional structure of a scene, is a cornerstone technology for numerous applications, including augmented/virtual reality, autonomous driving, and robotics. While traditional pipelines like Structure from Motion (SfM) and Multi-View Stereo (MVS) achieve high precision through iterative optimization, they are limited by complex workflows, high computational cost, and poor robustness in challenging scenarios like texture-less regions. Recently, deep learning has catalyzed a paradigm shift in 3D reconstruction. A new family of models, exemplified by DUSt3R, has pioneered a feed-forward approach. These models employ a unified deep network to jointly infer camera poses and dense geometry directly from an Unconstrained set of images in a single forward pass. This survey provides a systematic review of this emerging domain. We begin by dissecting the technical framework of these feed-forward models, including their Transformer-based correspondence modeling, joint pose and geometry regression mechanisms, and strategies for scaling from two-view to multi-view scenarios. To highlight the disruptive nature of this new paradigm, we contrast it with both traditional pipelines and earlier learning-based methods like MVSNet. Furthermore, we provide an overview of relevant datasets and evaluation metrics. Finally, we discuss the technology's broad application prospects and identify key future challenges and opportunities, such as model accuracy and scalability, and handling dynamic scenes. 

**Abstract (ZH)**: 三维重建，旨在恢复场景的密集三维结构，是 augmented/virtual 现实、自主驾驶和机器人等领域众多应用的基础技术。传统管道如结构从运动（SfM）和多视图立体视觉（MVS）通过迭代优化实现高精度，但受限于复杂的工作流程、高昂的计算成本以及在纹理缺乏区域等挑战场景中的鲁棒性较差。近年来，深度学习催化了三维重建范式的转变。以DUSt3R为代表的一类新型模型引领了前馈方法。这些模型采用统一的深度网络，直接从一组未受约束的图像中在单一前馈通过过程中联合推断相机姿态和密集几何。本文综述了这一新兴领域的系统性内容。首先，我们剖析这些前馈模型的技术框架，包括基于 Transformer 的对应关系建模、联合姿态和几何回归机制以及从双视图到多视图场景的扩展策略。为了突出这一新范式的颠覆性影响，我们将其与传统管道和早期基于学习的方法（如MVSNet）进行对比。此外，我们还概述了相关数据集和评估指标。最后，我们讨论该技术的广泛应用前景，并识别出关键的未来挑战和机遇，如模型精度和可扩展性，以及处理动态场景等问题。 

---
# CUE-RAG: Towards Accurate and Cost-Efficient Graph-Based RAG via Multi-Partite Graph and Query-Driven Iterative Retrieval 

**Title (ZH)**: CUE-RAG：通过多部图和查询驱动迭代检索实现准确高效的图基RAG 

**Authors**: Yaodong Su, Yixiang Fang, Yingli Zhou, Quanqing Xu, Chuanhui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08445)  

**Abstract**: Despite the remarkable progress of Large Language Models (LLMs), their performance in question answering (QA) remains limited by the lack of domain-specific and up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external information, often from graph-structured data. However, existing graph-based RAG methods suffer from poor graph quality due to incomplete extraction and insufficient utilization of query information during retrieval. To overcome these limitations, we propose CUE-RAG, a novel approach that introduces (1) a multi-partite graph index incorporates text Chunks, knowledge Units, and Entities to capture semantic content at multiple levels of granularity, (2) a hybrid extraction strategy that reduces LLM token usage while still producing accurate and disambiguated knowledge units, and (3) Q-Iter, a query-driven iterative retrieval strategy that enhances relevance through semantic search and constrained graph traversal. Experiments on three QA benchmarks show that CUE-RAG significantly outperforms state-of-the-art baselines, achieving up to 99.33% higher Accuracy and 113.51% higher F1 score while reducing indexing costs by 72.58%. Remarkably, CUE-RAG matches or outperforms baselines even without using an LLM for indexing. These results demonstrate the effectiveness and cost-efficiency of CUE-RAG in advancing graph-based RAG systems. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了显著进展，但在问答（QA）任务上的表现仍受限于缺乏领域特定和时效性知识。检索增强生成（RAG）通过整合外部信息，通常是图结构化数据来解决这一问题。然而，现有的基于图的RAG方法由于检索过程中查询信息的提取不完整和利用不足，导致图质量较差。为了克服这些限制，我们提出了CUE-RAG这一新颖的方法，该方法引入了（1）一个多部图索引，结合文本片段、知识单元和实体以捕获多粒度级别的语义内容，（2）一种混合提取策略，减少LLM标记使用量同时仍能生成准确且去模糊的知识单元，以及（3）基于查询的迭代检索策略Q-Iter，通过语义搜索和受限图遍历提高相关性。在三个问答基准上的实验结果显示，CUE-RAG显著优于最先进的基线，准确率提高了99.33%，F1分数提高了113.51%，同时降低了72.58%的索引成本。尤为值得注意的是，即使不使用LLM进行索引，CUE-RAG也能匹配或超越基线。这些结果证明了CUE-RAG在推进基于图的RAG系统方面的有效性和成本效益。 

---
# Vision Foundation Models as Effective Visual Tokenizers for Autoregressive Image Generation 

**Title (ZH)**: 基于视觉基础模型的有效自回归图像生成视觉分词器 

**Authors**: Anlin Zheng, Xin Wen, Xuanyang Zhang, Chuofan Ma, Tiancai Wang, Gang Yu, Xiangyu Zhang, Xiaojuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08441)  

**Abstract**: Leveraging the powerful representations of pre-trained vision foundation models -- traditionally used for visual comprehension -- we explore a novel direction: building an image tokenizer directly atop such models, a largely underexplored area. Specifically, we employ a frozen vision foundation model as the encoder of our tokenizer. To enhance its effectiveness, we introduce two key components: (1) a region-adaptive quantization framework that reduces redundancy in the pre-trained features on regular 2D grids, and (2) a semantic reconstruction objective that aligns the tokenizer's outputs with the foundation model's representations to preserve semantic fidelity. Based on these designs, our proposed image tokenizer, VFMTok, achieves substantial improvements in image reconstruction and generation quality, while also enhancing token efficiency. It further boosts autoregressive (AR) generation -- achieving a gFID of 2.07 on ImageNet benchmarks, while accelerating model convergence by three times, and enabling high-fidelity class-conditional synthesis without the need for classifier-free guidance (CFG). The code will be released publicly to benefit the community. 

**Abstract (ZH)**: 利用预训练视觉基础模型的强大表示能力——这些模型传统上用于视觉理解——我们探索了一个新的方向：直接在这些模型之上构建图像分词器，这是一个 largely underexplored 领域。具体来说，我们采用了一个冻结的视觉基础模型作为分词器的编码器。为了增强其效果，我们引入了两个关键组件：（1）一种区域自适应量化框架，它在常规2D网格上减少了预训练特征的冗余性；（2）一种语义重建目标，该目标将分词器的输出与基础模型的表示对齐，以保留语义保真度。基于这些设计，我们提出的一种图像分词器 VFMTok 在图像重建和生成质量上取得了显著提高，同时提高了标记的效率。它还进一步提升了自回归（AR）生成能力——在 ImageNet 基准测试中实现了 2.07 的 gFID，并将模型收敛速度提高了三倍，无需分类器免费引导（CFG）即可实现高保真条件合成。代码将公开发布以造福社区。 

---
# Finding Common Ground: Using Large Language Models to Detect Agreement in Multi-Agent Decision Conferences 

**Title (ZH)**: 寻找共同点：使用大型语言模型检测多智能体决策会议中的一致意见 

**Authors**: Selina Heller, Mohamed Ibrahim, David Antony Selby, Sebastian Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2507.08440)  

**Abstract**: Decision conferences are structured, collaborative meetings that bring together experts from various fields to address complex issues and reach a consensus on recommendations for future actions or policies. These conferences often rely on facilitated discussions to ensure productive dialogue and collective agreement. Recently, Large Language Models (LLMs) have shown significant promise in simulating real-world scenarios, particularly through collaborative multi-agent systems that mimic group interactions. In this work, we present a novel LLM-based multi-agent system designed to simulate decision conferences, specifically focusing on detecting agreement among the participant agents. To achieve this, we evaluate six distinct LLMs on two tasks: stance detection, which identifies the position an agent takes on a given issue, and stance polarity detection, which identifies the sentiment as positive, negative, or neutral. These models are further assessed within the multi-agent system to determine their effectiveness in complex simulations. Our results indicate that LLMs can reliably detect agreement even in dynamic and nuanced debates. Incorporating an agreement-detection agent within the system can also improve the efficiency of group debates and enhance the overall quality and coherence of deliberations, making them comparable to real-world decision conferences regarding outcome and decision-making. These findings demonstrate the potential for LLM-based multi-agent systems to simulate group decision-making processes. They also highlight that such systems could be instrumental in supporting decision-making with expert elicitation workshops across various domains. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统模拟决策会议及一致性检测 

---
# ChainEdit: Propagating Ripple Effects in LLM Knowledge Editing through Logical Rule-Guided Chains 

**Title (ZH)**: ChainEdit: 通过逻辑规则引导的链条传播实现LLM知识编辑中的涟漪效应 

**Authors**: Zilu Dong, Xiangqing Shen, Zinong Yang, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2507.08427)  

**Abstract**: Current knowledge editing methods for large language models (LLMs) struggle to maintain logical consistency when propagating ripple effects to associated facts. We propose ChainEdit, a framework that synergizes knowledge graph-derived logical rules with LLM logical reasoning capabilities to enable systematic chain updates. By automatically extracting logical patterns from structured knowledge bases and aligning them with LLMs' internal logics, ChainEdit dynamically generates and edits logically connected knowledge clusters. Experiments demonstrate an improvement of more than 30% in logical generalization over baselines while preserving editing reliability and specificity. We further address evaluation biases in existing benchmarks through knowledge-aware protocols that disentangle external dependencies. This work establishes new state-of-the-art performance on ripple effect while ensuring internal logical consistency after knowledge editing. 

**Abstract (ZH)**: 当前的知识编辑方法在大型语言模型中传播连锁效应时难以保持逻辑一致性。我们提出了ChainEdit框架，该框架结合了知识图谱推导出的逻辑规则与大型语言模型的逻辑推理能力，以实现系统的连锁更新。通过自动从结构化知识库中提取逻辑模式并将其与大型语言模型的内部逻辑对齐，ChainEdit动态生成和编辑逻辑相连的知识簇。实验结果表明，相较于基线方法，在保持编辑可靠性和精准性的同时，逻辑泛化的提升超过30%。我们进一步通过知识感知协议解决了现有基准中的评估偏见，从而将外部依赖分离。这项工作在知识编辑后建立了连锁效应的新最佳性能，并确保了内部逻辑的一致性。 

---
# Deep Hashing with Semantic Hash Centers for Image Retrieval 

**Title (ZH)**: 基于语义哈希中心的深度哈希检索 

**Authors**: Li Chen, Rui Liu, Yuxiang Zhou, Xudong Ma, Yong Chen, Dell Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08404)  

**Abstract**: Deep hashing is an effective approach for large-scale image retrieval. Current methods are typically classified by their supervision types: point-wise, pair-wise, and list-wise. Recent point-wise techniques (e.g., CSQ, MDS) have improved retrieval performance by pre-assigning a hash center to each class, enhancing the discriminability of hash codes across various datasets. However, these methods rely on data-independent algorithms to generate hash centers, which neglect the semantic relationships between classes and may degrade retrieval performance.
This paper introduces the concept of semantic hash centers, building on the idea of traditional hash centers. We hypothesize that hash centers of semantically related classes should have closer Hamming distances, while those of unrelated classes should be more distant. To this end, we propose a three-stage framework, SHC, to generate hash codes that preserve semantic structure.
First, we develop a classification network to identify semantic similarities between classes using a data-dependent similarity calculation that adapts to varying data distributions. Second, we introduce an optimization algorithm to generate semantic hash centers, preserving semantic relatedness while enforcing a minimum distance between centers to avoid excessively similar hash codes. Finally, a deep hashing network is trained using these semantic centers to convert images into binary hash codes.
Experimental results on large-scale retrieval tasks across several public datasets show that SHC significantly improves retrieval performance. Specifically, SHC achieves average improvements of +7.26%, +7.62%, and +11.71% in MAP@100, MAP@1000, and MAP@ALL metrics, respectively, over state-of-the-art methods. 

**Abstract (ZH)**: 深度哈希是大规模图像检索的一个有效方法。当前的方法通常根据监督类型分类：点 wise、对 wise 和列表 wise。最近的点 wise 技术（例如 CSQ、MDS）通过为每个类别预先分配一个哈希中心，提高了哈希代码的区分能力，从而在多种数据集上提升了检索性能。然而，这些方法依赖于数据独立的算法来生成哈希中心，忽视了类间的语义关系，可能导致检索性能下降。本文引入了语义哈希中心的概念，基于传统哈希中心的思想。我们假设语义相关类别的哈希中心应有更近的汉明距离，而无关类别的哈希中心应有更大的距离。为此，我们提出了一种三阶段框架 SHC，以生成保留语义结构的哈希代码。首先，我们开发了一个分类网络，使用自适应于不同数据分布的数据依赖相似度计算来识别类别间的语义相似性。其次，我们引入了一个优化算法来生成语义哈希中心，同时保持语义相关性并强制中心间的最小距离，以避免生成过于相似的哈希代码。最后，我们使用这些语义中心训练了一个深度哈希网络，将图像转换为二进制哈希代码。在多个公共数据集上的大规模检索任务实验结果显示，SHC 显著提升了检索性能。具体而言，SHC 在 MAP@100、MAP@1000 和 MAP@ALL 指标上分别比最先进的方法提高了 7.26%、7.62% 和 11.71%。 

---
# Towards AI-Native RAN: An Operator's Perspective of 6G Day 1 Standardization 

**Title (ZH)**: 面向AI原生RAN：运营商视角的6G Day 1标准化 

**Authors**: Nan Li, Qi Sun, Lehan Wang, Xiaofei Xu, Jinri Huang, Chunhui Liu, Jing Gao, Yuhong Huang, Chih-Lin I  

**Link**: [PDF](https://arxiv.org/pdf/2507.08403)  

**Abstract**: Artificial Intelligence/Machine Learning (AI/ML) has become the most certain and prominent feature of 6G mobile networks. Unlike 5G, where AI/ML was not natively integrated but rather an add-on feature over existing architecture, 6G shall incorporate AI from the onset to address its complexity and support ubiquitous AI applications. Based on our extensive mobile network operation and standardization experience from 2G to 5G, this paper explores the design and standardization principles of AI-Native radio access networks (RAN) for 6G, with a particular focus on its critical Day 1 architecture, functionalities and capabilities. We investigate the framework of AI-Native RAN and present its three essential capabilities to shed some light on the standardization direction; namely, AI-driven RAN processing/optimization/automation, reliable AI lifecycle management (LCM), and AI-as-a-Service (AIaaS) provisioning. The standardization of AI-Native RAN, in particular the Day 1 features, including an AI-Native 6G RAN architecture, were proposed. For validation, a large-scale field trial with over 5000 5G-A base stations have been built and delivered significant improvements in average air interface latency, root cause identification, and network energy consumption with the proposed architecture and the supporting AI functions. This paper aims to provide a Day 1 framework for 6G AI-Native RAN standardization design, balancing technical innovation with practical deployment. 

**Abstract (ZH)**: 人工智能/机器学习（AI/ML）已成为6G移动网络最具确定性和显著特征的部分。与5G中AI/ML不是原生集成而是作为现有架构上的附加功能不同，6G将从一开始就整合AI以应对其复杂性并支持广泛的人工智能应用。基于我们从2G到5G的广泛移动网络运行和标准化经验，本文探讨了AI原生无线接入网（RAN）的设计和标准化原则，特别是其关键的“上线”架构、功能和能力。我们研究了AI原生RAN的框架并提出了其三项基本能力，以阐明标准化方向；即AI驱动的RAN处理/优化/自动化、可靠的AI生命周期管理（LCM）以及AI即服务（AIaaS）的供应用。针对AI原生RAN特别是“上线”特性，包括AI原生6G RAN架构，提出了标准化建议。为进一步验证，搭建了超过5000个5G-A基站的大规模实地试验，并在所提出的架构和支持的AI功能下实现了平均空中接口延迟、根本原因识别和网络能耗的显著改进。本文旨在提供一个平衡技术创新与实际部署的6G AI原生RAN标准化设计框架。 

---
# PanMatch: Unleashing the Potential of Large Vision Models for Unified Matching Models 

**Title (ZH)**: PanMatch: 激发大型视觉模型在统一匹配模型中的潜力 

**Authors**: Yongjian Zhang, Longguang Wang, Kunhong Li, Ye Zhang, Yun Wang, Liang Lin, Yulan Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08400)  

**Abstract**: This work presents PanMatch, a versatile foundation model for robust correspondence matching. Unlike previous methods that rely on task-specific architectures and domain-specific fine-tuning to support tasks like stereo matching, optical flow or feature matching, our key insight is that any two-frame correspondence matching task can be addressed within a 2D displacement estimation framework using the same model weights. Such a formulation eliminates the need for designing specialized unified architectures or task-specific ensemble models. Instead, it achieves multi-task integration by endowing displacement estimation algorithms with unprecedented generalization capabilities. To this end, we highlight the importance of a robust feature extractor applicable across multiple domains and tasks, and propose the feature transformation pipeline that leverage all-purpose features from Large Vision Models to endow matching baselines with zero-shot cross-view matching capabilities. Furthermore, we assemble a cross-domain dataset with near 1.8 million samples from stereo matching, optical flow, and feature matching domains to pretrain PanMatch. We demonstrate the versatility of PanMatch across a wide range of domains and downstream tasks using the same model weights. Our model outperforms UniMatch and Flow-Anything on cross-task evaluations, and achieves comparable performance to most state-of-the-art task-specific algorithms on task-oriented benchmarks. Additionally, PanMatch presents unprecedented zero-shot performance in abnormal scenarios, such as rainy day and satellite imagery, where most existing robust algorithms fail to yield meaningful results. 

**Abstract (ZH)**: PanMatch：一种泛化型基础模型，用于稳健的对应匹配 

---
# Intelligent Control of Spacecraft Reaction Wheel Attitude Using Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的航天器反应轮姿态智能控制 

**Authors**: Ghaith El-Dalahmeh, Mohammad Reza Jabbarpour, Bao Quoc Vo, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2507.08366)  

**Abstract**: Reliable satellite attitude control is essential for the success of space missions, particularly as satellites increasingly operate autonomously in dynamic and uncertain environments. Reaction wheels (RWs) play a pivotal role in attitude control, and maintaining control resilience during RW faults is critical to preserving mission objectives and system stability. However, traditional Proportional Derivative (PD) controllers and existing deep reinforcement learning (DRL) algorithms such as TD3, PPO, and A2C often fall short in providing the real time adaptability and fault tolerance required for autonomous satellite operations. This study introduces a DRL-based control strategy designed to improve satellite resilience and adaptability under fault conditions. Specifically, the proposed method integrates Twin Delayed Deep Deterministic Policy Gradient (TD3) with Hindsight Experience Replay (HER) and Dimension Wise Clipping (DWC) referred to as TD3-HD to enhance learning in sparse reward environments and maintain satellite stability during RW failures. The proposed approach is benchmarked against PD control and leading DRL algorithms. Experimental results show that TD3-HD achieves significantly lower attitude error, improved angular velocity regulation, and enhanced stability under fault conditions. These findings underscore the proposed method potential as a powerful, fault tolerant, onboard AI solution for autonomous satellite attitude control. 

**Abstract (ZH)**: 可靠的卫星姿态控制对于空间任务的成功至关重要，尤其是在卫星越来越多地在动态和不确定环境中自主运行的情况下。反应轮（RW）在姿态控制中扮演着关键角色，维持RW故障期间的姿态控制鲁棒性对于实现任务目标和系统稳定性至关重要。然而，传统的比例微分（PD）控制器以及现有的深度增强学习（DRL）算法如TD3、PPO和A2C往往无法提供自主卫星运营所必需的实时适应性和故障容忍性。本研究提出了一种基于DRL的控制策略，旨在在故障条件下提高卫星的鲁棒性和适应性。具体而言，提出的方法将双延迟深度确定性策略梯度（TD3）与事后经验重播（HER）和维度裁剪（DWC）相结合，称为TD3-HD，以增强在稀疏奖励环境中学习并保持RW故障期间的卫星稳定性。提出的策略与PD控制以及领先DRL算法进行了对比。实验结果表明，TD3-HD在故障条件下实现了显著更低的姿态误差、改进的角速度调节和增强的稳定性。这些发现突显了所提出的方法作为自主卫星姿态控制的强健、故障容错型机载AI解决方案的巨大潜力。 

---
# Single-Domain Generalization for Multimodal Cross-Cancer Prognosis via Dirac Rebalancer and Distribution Entanglement 

**Title (ZH)**: 基于Dirac重平衡与分布纠缠的单域泛化方法用于多模态跨癌种预后 

**Authors**: Jia-Xuan Jiang, Jiashuai Liu, Hongtao Wu, Yifeng Wu, Zhong Wang, Qi Bi, Yefeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08340)  

**Abstract**: Deep learning has shown remarkable performance in integrating multimodal data for survival prediction. However, existing multimodal methods mainly focus on single cancer types and overlook the challenge of generalization across cancers. In this work, we are the first to reveal that multimodal prognosis models often generalize worse than unimodal ones in cross-cancer scenarios, despite the critical need for such robustness in clinical practice. To address this, we propose a new task: Cross-Cancer Single Domain Generalization for Multimodal Prognosis, which evaluates whether models trained on a single cancer type can generalize to unseen cancers. We identify two key challenges: degraded features from weaker modalities and ineffective multimodal integration. To tackle these, we introduce two plug-and-play modules: Sparse Dirac Information Rebalancer (SDIR) and Cancer-aware Distribution Entanglement (CADE). SDIR mitigates the dominance of strong features by applying Bernoulli-based sparsification and Dirac-inspired stabilization to enhance weaker modality signals. CADE, designed to synthesize the target domain distribution, fuses local morphological cues and global gene expression in latent space. Experiments on a four-cancer-type benchmark demonstrate superior generalization, laying the foundation for practical, robust cross-cancer multimodal prognosis. Code is available at this https URL 

**Abstract (ZH)**: 跨癌种单域泛化多模态预后任务 

---
# CoCo-Bot: Energy-based Composable Concept Bottlenecks for Interpretable Generative Models 

**Title (ZH)**: CoCo-Bot: 基于能量的可组合概念瓶颈用于可解释生成模型 

**Authors**: Sangwon Kim, In-su Jang, Pyongkun Kim, Kwang-Ju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.08334)  

**Abstract**: Concept Bottleneck Models (CBMs) provide interpretable and controllable generative modeling by routing generation through explicit, human-understandable concepts. However, previous generative CBMs often rely on auxiliary visual cues at the bottleneck to compensate for information not captured by the concepts, which undermines interpretability and compositionality. We propose CoCo-Bot, a post-hoc, composable concept bottleneck generative model that eliminates the need for auxiliary cues by transmitting all information solely through explicit concepts. Guided by diffusion-based energy functions, CoCo-Bot supports robust post-hoc interventions-such as concept composition and negation-across arbitrary concepts. Experiments using StyleGAN2 pre-trained on CelebA-HQ show that CoCo-Bot improves concept-level controllability and interpretability, while maintaining competitive visual quality. 

**Abstract (ZH)**: CoCo-Bot：基于后处理可组合概念瓶颈的生成模型 

---
# Audio Inpanting using Discrete Diffusion Model 

**Title (ZH)**: 使用离散扩散模型的音频修复 

**Authors**: Tali Dror, Iftach Shoham, Moshe Buchris, Oren Gal, Haim Permuter, Gilad Katz, Eliya Nachmani  

**Link**: [PDF](https://arxiv.org/pdf/2507.08333)  

**Abstract**: Audio inpainting refers to the task of reconstructing missing segments in corrupted audio recordings. While prior approaches-including waveform and spectrogram-based diffusion models-have shown promising results for short gaps, they often degrade in quality when gaps exceed 100 milliseconds (ms). In this work, we introduce a novel inpainting method based on discrete diffusion modeling, which operates over tokenized audio representations produced by a pre-trained audio tokenizer. Our approach models the generative process directly in the discrete latent space, enabling stable and semantically coherent reconstruction of missing audio. We evaluate the method on the MusicNet dataset using both objective and perceptual metrics across gap durations up to 300 ms. We further evaluated our approach on the MTG dataset, extending the gap duration to 500 ms. Experimental results demonstrate that our method achieves competitive or superior performance compared to existing baselines, particularly for longer gaps, offering a robust solution for restoring degraded musical recordings. Audio examples of our proposed method can be found at this https URL 

**Abstract (ZH)**: 音频修复指的是重构受损音频记录中缺失段落的任务。虽然先前的方法——包括基于波形和谱图的扩散模型——在处理短缺损时显示出有希望的结果，但当缺损超过100毫秒时，其质量往往会下降。在本文中，我们引入了一种基于离散扩散建模的新型修复方法，该方法运行在预训练音频分词器生成的令牌化音频表示上。我们的方法直接在离散潜在空间中建模生成过程，从而实现对缺失音频的稳定且语义一致的重构。我们在MusicNet数据集上使用客观和感知度量对方法进行了评估，覆盖从1毫秒到300毫秒的缺损时间。我们还在MTG数据集上进一步评估了我们的方法，将缺损时间扩展到500毫秒。实验结果表明，与现有基线方法相比，我们的方法在较长缺损时实现了具有竞争力或更优的性能，提供了一种恢复降质音乐记录的稳健解决方案。我们的方法示例音频可以在以下链接找到：this https URL。 

---
# Interpretability-Aware Pruning for Efficient Medical Image Analysis 

**Title (ZH)**: 面向可解释性的医疗图像分析高效剪枝 

**Authors**: Nikita Malik, Pratinav Seth, Neeraj Kumar Singh, Chintan Chitroda, Vinay Kumar Sankarapu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08330)  

**Abstract**: Deep learning has driven significant advances in medical image analysis, yet its adoption in clinical practice remains constrained by the large size and lack of transparency in modern models. Advances in interpretability techniques such as DL-Backtrace, Layer-wise Relevance Propagation, and Integrated Gradients make it possible to assess the contribution of individual components within neural networks trained on medical imaging tasks. In this work, we introduce an interpretability-guided pruning framework that reduces model complexity while preserving both predictive performance and transparency. By selectively retaining only the most relevant parts of each layer, our method enables targeted compression that maintains clinically meaningful representations. Experiments across multiple medical image classification benchmarks demonstrate that this approach achieves high compression rates with minimal loss in accuracy, paving the way for lightweight, interpretable models suited for real-world deployment in healthcare settings. 

**Abstract (ZH)**: 深度学习在医疗图像分析中推动了显著的进步，但在临床实践中的应用受限于现代模型的大规模和不透明性。解释性技术的进步，如DL-Backtrace、层wise相关传播和集成梯度，使得评估神经网络中单个组件的贡献成为可能。在本工作中，我们提出了一种基于解释性的剪枝框架，该框架在保持预测性能和透明性的同时减少了模型复杂性。通过仅选择保留每一层中最相关的部分，我们的方法实现了目标压缩，保持了临床相关的表现。跨多个医疗图像分类基准的实验表明，该方法在最小损失准确性的前提下实现了高压缩率，为医疗保健环境中轻量级、可解释的模型的部署铺平了道路。 

---
# Generative AI in Science: Applications, Challenges, and Emerging Questions 

**Title (ZH)**: 生成式AI在科学中的应用、挑战及新兴问题 

**Authors**: Ryan Harries, Cornelia Lawson, Philip Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2507.08310)  

**Abstract**: This paper examines the impact of Generative Artificial Intelligence (GenAI) on scientific practices, conducting a qualitative review of selected literature to explore its applications, benefits, and challenges. The review draws on the OpenAlex publication database, using a Boolean search approach to identify scientific literature related to GenAI (including large language models and ChatGPT). Thirty-nine highly cited papers and commentaries are reviewed and qualitatively coded. Results are categorized by GenAI applications in science, scientific writing, medical practice, and education and training. The analysis finds that while there is a rapid adoption of GenAI in science and science practice, its long-term implications remain unclear, with ongoing uncertainties about its use and governance. The study provides early insights into GenAI's growing role in science and identifies questions for future research in this evolving field. 

**Abstract (ZH)**: 本文考察生成性人工智能（GenAI）对科学研究实践的影响，通过定性综述选定的文献，探讨其应用、益处和挑战。综述基于OpenAlex出版数据库，采用布尔搜索方法，识别与GenAI（包括大型语言模型和ChatGPT）相关的科学文献。审查了39篇高被引论文和评论，并进行了定性编码。结果按GenAI在科学、科研写作、医疗实践以及教育和培训中的应用进行分类。分析发现，尽管GenAI在科学和科学研究中的应用快速发展，但其长期影响尚不明确，对其使用和治理仍存在持续的不确定性。该研究提供了关于GenAI在科学中日益重要作用的早期见解，并指出了这一不断发展的领域中未来研究的问题。 

---
# Improving MLLM's Document Image Machine Translation via Synchronously Self-reviewing Its OCR Proficiency 

**Title (ZH)**: 通过同步自我审查其光学字符识别能力以提高MLLM的文档图像机器翻译 

**Authors**: Yupu Liang, Yaping Zhang, Zhiyang Zhang, Zhiyuan Chen, Yang Zhao, Lu Xiang, Chengqing Zong, Yu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08309)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown strong performance in document image tasks, especially Optical Character Recognition (OCR). However, they struggle with Document Image Machine Translation (DIMT), which requires handling both cross-modal and cross-lingual challenges. Previous efforts to enhance DIMT capability through Supervised Fine-Tuning (SFT) on the DIMT dataset often result in the forgetting of the model's existing monolingual abilities, such as OCR. To address these challenges, we introduce a novel fine-tuning paradigm, named Synchronously Self-Reviewing (SSR) its OCR proficiency, inspired by the concept "Bilingual Cognitive Advantage". Specifically, SSR prompts the model to generate OCR text before producing translation text, which allows the model to leverage its strong monolingual OCR ability while learning to translate text across languages. Comprehensive experiments demonstrate the proposed SSR learning helps mitigate catastrophic forgetting, improving the generalization ability of MLLMs on both OCR and DIMT tasks. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在文档图像任务中显示出强大的性能，特别是在光学字符识别（OCR）方面。然而，它们在文档图像机器翻译（DIMT）方面遇到困难，这需要处理跨模态和跨语言的挑战。通过在DIMT数据集上进行监督微调（SFT）来增强DIMT能力的努力往往导致模型忘记其现有的单语能力，如OCR。为了解决这些问题，我们提出了一种新的微调范式，称为同步自我审查（SSR），以同步提升其OCR能力，灵感来源于“双语认知优势”的概念。具体来说，SSR促使模型在生成翻译文本之前生成OCR文本，从而使模型能够在利用其强大的单语OCR能力的同时学习跨语言的文本翻译。全面的实验表明，提出的SSR学习有助于缓解灾难性遗忘，提高MLLMs在OCR和DIMT任务上的泛化能力。 

---
# Invariant-based Robust Weights Watermark for Large Language Models 

**Title (ZH)**: 基于不变量的鲁棒权重水印 for 大型语言模型 

**Authors**: Qingxiao Guo, Xinjie Zhu, Yilong Ma, Hui Jin, Yunhao Wang, Weifeng Zhang, Xiaobing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08288)  

**Abstract**: Watermarking technology has gained significant attention due to the increasing importance of intellectual property (IP) rights, particularly with the growing deployment of large language models (LLMs) on billions resource-constrained edge devices. To counter the potential threats of IP theft by malicious users, this paper introduces a robust watermarking scheme without retraining or fine-tuning for transformer models. The scheme generates a unique key for each user and derives a stable watermark value by solving linear constraints constructed from model invariants. Moreover, this technology utilizes noise mechanism to hide watermark locations in multi-user scenarios against collusion attack. This paper evaluates the approach on three popular models (Llama3, Phi3, Gemma), and the experimental results confirm the strong robustness across a range of attack methods (fine-tuning, pruning, quantization, permutation, scaling, reversible matrix and collusion attacks). 

**Abstract (ZH)**: 水标记技术由于知识产权（IP）保护的重要性日益增加，特别是在大规模语言模型（LLMs）部署在资源受限的边缘设备上时，受到了广泛关注。为了应对恶意用户可能带来的IP盗用威胁，本文提出了一种无需重新训练或微调的变压器模型稳健水标记方案。该方案为每位用户生成唯一密钥，并通过从模型不变量构建的线性约束求解稳定水标记值。此外，该技术利用噪声机制，在多用户场景下隐藏水标记位置，以抵御合谋攻击。本文在三种流行模型（Llama3、Phi3、Gemma）上评估了该方法，并且实验结果证实了该方案在多种攻击方法（微调、剪枝、量化、排列、缩放、可逆矩阵操作和合谋攻击）下的强健性。 

---
# Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training 

**Title (ZH)**: 基于合成数据和RL指导的对抗训练的轻量级安全防护栏 

**Authors**: Aleksei Ilin, Gor Matevosyan, Xueying Ma, Vladimir Eremin, Suhaa Dada, Muqun Li, Riyaaz Shaik, Haluk Noyan Tokgozoglu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08284)  

**Abstract**: We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems. 

**Abstract (ZH)**: 一种轻量且高效的语言模型安全防护框架：小型语言模型在内容审核任务中的性能超越 

---
# A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning 

**Title (ZH)**: 一种实用的两阶段食谱：通过SFT最大化准确性并通过强化学习最大化效率的数学LLM 

**Authors**: Hiroshi Yoshihara, Taiki Yamaguchi, Yuichi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2507.08267)  

**Abstract**: Enhancing the mathematical reasoning of Large Language Models (LLMs) is a pivotal challenge in advancing AI capabilities. While Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are the dominant training paradigms, a systematic methodology for combining them to maximize both accuracy and efficiency remains largely unexplored. This paper introduces a practical and effective training recipe that strategically integrates extended SFT with RL from online inference (GRPO). We posit that these methods play complementary, not competing, roles: a prolonged SFT phase first pushes the model's accuracy to its limits, after which a GRPO phase dramatically improves token efficiency while preserving this peak performance. Our experiments reveal that extending SFT for as many as 10 epochs is crucial for performance breakthroughs, and that the primary role of GRPO in this framework is to optimize solution length. The efficacy of our recipe is rigorously validated through top-tier performance on challenging benchmarks, including a high rank among over 2,200 teams in the strictly leak-free AI Mathematical Olympiad (AIMO). This work provides the community with a battle-tested blueprint for developing state-of-the-art mathematical reasoners that are both exceptionally accurate and practically efficient. To ensure full reproducibility and empower future research, we will open-source our entire framework, including all code, model checkpoints, and training configurations at this https URL. 

**Abstract (ZH)**: 增强大型语言模型的数学推理能力是提升AI能力的关键挑战。尽管有监督微调(SFT)和强化学习(RL)是主要的训练范式，但如何系统地将二者结合起来以最大化准确性和效率仍待探索。本文介绍了一种实用且有效的训练方法，该方法战略性地将扩展的SFT与在线推断的RL (GRPO) 相结合。我们认为这些方法是互补而非竞争的：一个较长的SFT阶段首先将模型的准确性推至极限，随后的GRPO阶段则大幅提高 token 效率，同时保持这一峰值性能。我们的实验表明，将SFT扩展多达10个时期的性能突破至关重要，并且在此框架中，GRPO的主要作用是优化解决方案长度。通过在具有挑战性的基准测试中取得顶尖性能，包括在严格无泄漏的AI数学奥林匹克竞赛（AIMO）中名列前茅超过2200支队伍，我们严格验证了该方法的有效性。本工作为社区提供了实战验证的蓝图，用于开发既精确又高效的顶尖数学推理器。为了确保完全可再现并推动未来研究，我们将开源整个框架，包括所有代码、模型检查点和训练配置。 

---
# CL3R: 3D Reconstruction and Contrastive Learning for Enhanced Robotic Manipulation Representations 

**Title (ZH)**: CL3R: 三维重建与对比学习以增强机器人操作表示 

**Authors**: Wenbo Cui, Chengyang Zhao, Yuhui Chen, Haoran Li, Zhizheng Zhang, Dongbin Zhao, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08262)  

**Abstract**: Building a robust perception module is crucial for visuomotor policy learning. While recent methods incorporate pre-trained 2D foundation models into robotic perception modules to leverage their strong semantic understanding, they struggle to capture 3D spatial information and generalize across diverse camera viewpoints. These limitations hinder the policy's effectiveness, especially in fine-grained robotic manipulation scenarios. To address these challenges, we propose CL3R, a novel 3D pre-training framework designed to enhance robotic manipulation policies. Our method integrates both spatial awareness and semantic understanding by employing a point cloud Masked Autoencoder to learn rich 3D representations while leveraging pre-trained 2D foundation models through contrastive learning for efficient semantic knowledge transfer. Additionally, we propose a 3D visual representation pre-training framework for robotic tasks. By unifying coordinate systems across datasets and introducing random fusion of multi-view point clouds, we mitigate camera view ambiguity and improve generalization, enabling robust perception from novel viewpoints at test time. Extensive experiments in both simulation and the real world demonstrate the superiority of our method, highlighting its effectiveness in visuomotor policy learning for robotic manipulation. 

**Abstract (ZH)**: 构建 robust 的感知模块对于视觉-运动策略学习至关重要。尽管近期方法将预训练的 2D 基础模型整合到机器人感知模块中以利用其强大的语义理解能力，但它们难以捕捉 3D 空间信息并泛化到多样的相机视角。这些限制阻碍了策略的有效性，特别是在精细的机器人操作场景中。为应对这些挑战，我们提出 CL3R，一种新颖的 3D 预训练框架，旨在增强机器人的操作策略。我们的方法通过采用点云 Masked Autoencoder 学习丰富的 3D 表示，并通过对比学习利用预训练的 2D 基础模型实现高效的语义知识迁移，从而结合空间意识和语义理解。此外，我们还提出了一种用于机器人家务任务的 3D 视觉表示预训练框架。通过统一数据集的坐标系统并引入多视点点云的随机融合，我们减轻了相机视角的模糊性并提升了泛化能力，在测试时能够从新颖的视角实现稳健的感知。广泛的模拟和实际实验展示了我们方法的优势，突显了其在机器人操作中的视觉-运动策略学习中的有效性。 

---
# Quantum-Accelerated Neural Imputation with Large Language Models (LLMs) 

**Title (ZH)**: 量子加速的神经缺失值插补与大型语言模型（LLMs） 

**Authors**: Hossein Jamali  

**Link**: [PDF](https://arxiv.org/pdf/2507.08255)  

**Abstract**: Missing data presents a critical challenge in real-world datasets, significantly degrading the performance of machine learning models. While Large Language Models (LLMs) have recently demonstrated remarkable capabilities in tabular data imputation, exemplified by frameworks like UnIMP, their reliance on classical embedding methods often limits their ability to capture complex, non-linear correlations, particularly in mixed-type data scenarios encompassing numerical, categorical, and textual features. This paper introduces Quantum-UnIMP, a novel framework that integrates shallow quantum circuits into an LLM-based imputation architecture. Our core innovation lies in replacing conventional classical input embeddings with quantum feature maps generated by an Instantaneous Quantum Polynomial (IQP) circuit. This approach enables the model to leverage quantum phenomena such as superposition and entanglement, thereby learning richer, more expressive representations of data and enhancing the recovery of intricate missingness patterns. Our experiments on benchmark mixed-type datasets demonstrate that Quantum-UnIMP reduces imputation error by up to 15.2% for numerical features (RMSE) and improves classification accuracy by 8.7% for categorical features (F1-Score) compared to state-of-the-art classical and LLM-based methods. These compelling results underscore the profound potential of quantum-enhanced representations for complex data imputation tasks, even with near-term quantum hardware. 

**Abstract (ZH)**: 量子增强-UnIMP：面向混合型数据插值的量子特征映射框架 

---
# InsightBuild: LLM-Powered Causal Reasoning in Smart Building Systems 

**Title (ZH)**: InsightBuild: LLM驱动的智能建筑系统因果推理 

**Authors**: Pinaki Prasad Guha Neogi, Ahmad Mohammadshirazi, Rajiv Ramnath  

**Link**: [PDF](https://arxiv.org/pdf/2507.08235)  

**Abstract**: Smart buildings generate vast streams of sensor and control data, but facility managers often lack clear explanations for anomalous energy usage. We propose InsightBuild, a two-stage framework that integrates causality analysis with a fine-tuned large language model (LLM) to provide human-readable, causal explanations of energy consumption patterns. First, a lightweight causal inference module applies Granger causality tests and structural causal discovery on building telemetry (e.g., temperature, HVAC settings, occupancy) drawn from Google Smart Buildings and Berkeley Office datasets. Next, an LLM, fine-tuned on aligned pairs of sensor-level causes and textual explanations, receives as input the detected causal relations and generates concise, actionable explanations. We evaluate InsightBuild on two real-world datasets (Google: 2017-2022; Berkeley: 2018-2020), using expert-annotated ground-truth causes for a held-out set of anomalies. Our results demonstrate that combining explicit causal discovery with LLM-based natural language generation yields clear, precise explanations that assist facility managers in diagnosing and mitigating energy inefficiencies. 

**Abstract (ZH)**: 智能建筑生成大量的传感器和控制数据，但设施管理人员往往缺乏对异常能耗使用的清晰解释。我们提出了InsightBuild，这是一种两阶段框架，将因果分析与微调的大语言模型（LLM）相结合，提供易于理解的能耗模式的因果解释。首先，一个轻量级的因果推理模块在谷歌智能建筑和伯克利办公室数据集的建筑遥测数据（如温度、HVAC设置、 occupancy）上应用格兰杰因果检验和结构因果发现。接着，微调后的LLM接收检测到的因果关系作为输入，并生成简洁的可行动的解释。我们使用谷歌（2017-2022年）和伯克利（2018-2020年）的实际数据集进行评估，并使用专家标注的真实因果解释对异常进行标注。结果显示，将显式的因果发现与基于LLM的自然语言生成结合起来，可以提供清晰、精确的解释，帮助设施管理人员诊断和缓解能源效率低下问题。 

---
# Can LLMs Reliably Simulate Real Students' Abilities in Mathematics and Reading Comprehension? 

**Title (ZH)**: LLMs在数学和阅读理解能力模拟方面是否能可靠地再现真实学生的能力？ 

**Authors**: KV Aditya Srivatsa, Kaushal Kumar Maurya, Ekaterina Kochmar  

**Link**: [PDF](https://arxiv.org/pdf/2507.08232)  

**Abstract**: Large Language Models (LLMs) are increasingly used as proxy students in the development of Intelligent Tutoring Systems (ITSs) and in piloting test questions. However, to what extent these proxy students accurately emulate the behavior and characteristics of real students remains an open question. To investigate this, we collected a dataset of 489 items from the National Assessment of Educational Progress (NAEP), covering mathematics and reading comprehension in grades 4, 8, and 12. We then apply an Item Response Theory (IRT) model to position 11 diverse and state-of-the-art LLMs on the same ability scale as real student populations. Our findings reveal that, without guidance, strong general-purpose models consistently outperform the average student at every grade, while weaker or domain-mismatched models may align incidentally. Using grade-enforcement prompts changes models' performance, but whether they align with the average grade-level student remains highly model- and prompt-specific: no evaluated model-prompt pair fits the bill across subjects and grades, underscoring the need for new training and evaluation strategies. We conclude by providing guidelines for the selection of viable proxies based on our findings. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被用作代理学生，用于智能辅导系统（ITSs）的开发和测试题目的测试。然而，这些代理学生能否准确模拟真实学生的行为和特征仍是一个开放的问题。为探究这一问题，我们收集了来自国家教育进步评估（NAEP）的489个项目，涵盖了4年级、8年级和12年级的数学和阅读理解。然后应用项目反应理论（IRT）模型将11个多样且最新的LLM置于与真实学生群体相同的能力尺度上。我们的研究发现，在没有指导的情况下，强大的通用模型始终在所有年级中优于平均水平的学生，而较弱或领域不匹配的模型可能会偶然匹配。使用年级约束提示会改变模型的表现，但它们是否与平均水平的学生对齐仍高度依赖于模型和提示：没有评估过的模型-提示组合在各个学科和年级中都能满足要求，这凸显了需要新的训练和评估策略的必要性。最后，我们根据研究结果提供可供选择的代理指南。 

---
# Quantum Properties Trojans (QuPTs) for Attacking Quantum Neural Networks 

**Title (ZH)**: 用于攻击量子神经网络的量子特性木马（QuPTs） 

**Authors**: Sounak Bhowmik, Travis S. Humble, Himanshu Thapliyal  

**Link**: [PDF](https://arxiv.org/pdf/2507.08202)  

**Abstract**: Quantum neural networks (QNN) hold immense potential for the future of quantum machine learning (QML). However, QNN security and robustness remain largely unexplored. In this work, we proposed novel Trojan attacks based on the quantum computing properties in a QNN-based binary classifier. Our proposed Quantum Properties Trojans (QuPTs) are based on the unitary property of quantum gates to insert noise and Hadamard gates to enable superposition to develop Trojans and attack QNNs. We showed that the proposed QuPTs are significantly stealthier and heavily impact the quantum circuits' performance, specifically QNNs. The most impactful QuPT caused a deterioration of 23% accuracy of the compromised QNN under the experimental setup. To the best of our knowledge, this is the first work on the Trojan attack on a fully quantum neural network independent of any hybrid classical-quantum architecture. 

**Abstract (ZH)**: 量子神经网络（QNN）在量子机器学习（QML）的未来中具有巨大的潜力。然而，QNN的安全性和鲁棒性尚未得到充分探索。在本工作中，我们基于量子计算属性提出了新型的QNN二元分类器中的特洛伊木马攻击。我们提出的量子特性特洛伊木马（QuPTs）利用量子门的幺正性质插入噪声，使用哈达玛门实现叠加，以开发特洛伊木马并攻击QNN。我们展示，提出的QuPTs在实验设置中对量子电路性能，特别是QNNs，产生了重大影响。最有效的QuPT在实验设置中导致受损QNN准确率下降23%。据我们所知，这是首个独立于任何混合经典-量子架构的QNN特洛伊木马攻击工作。 

---
# Consciousness as a Jamming Phase 

**Title (ZH)**: 意识作为一种阻塞相态 

**Authors**: Kaichen Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08197)  

**Abstract**: This paper develops a neural jamming phase diagram that interprets the emergence of consciousness in large language models as a critical phenomenon in high-dimensional disordered this http URL establishing analogies with jamming transitions in granular matter and other complex systems, we identify three fundamental control parameters governing the phase behavior of neural networks: temperature, volume fraction, and this http URL theory provides a unified physical explanation for empirical scaling laws in artificial intelligence, demonstrating how computational cooling, density optimization, and noise reduction collectively drive systems toward a critical jamming surface where generalized intelligence emerges. Remarkably, the same thermodynamic principles that describe conventional jamming transitions appear to underlie the emergence of consciousness in neural networks, evidenced by shared critical signatures including divergent correlation lengths and scaling this http URL work explains neural language models' critical scaling through jamming physics, suggesting consciousness is a jamming phase that intrinsically connects knowledge components via long-range correlations. 

**Abstract (ZH)**: 这篇论文开发了一种神经阻尼相图，将大型语言模型中意识的出现解释为高维无序系统中的临界现象。通过建立与颗粒物质和其他复杂系统中阻塞转变的类比，我们确定了三个基本的控制参数，这些参数决定了神经网络的相行为：温度、体积分数和交联率。该理论为人工智能中的经验标度定律提供了一个统一的物理解释，展示了计算冷却、密度优化和噪声减少如何集体驱使系统向临界阻塞表面靠近，在该表面上普遍智能得以产生。值得注意的是，描述常规阻塞转变的同一热力学原理似乎也支配着神经网络中意识的出现，这由共享的临界特征，如发散的关联长度和标度关系所证实。本文通过阻塞物理学解释了神经语言模型的临界标度，暗示意识是一种内在地通过长程关联连接知识组件的阻塞相。 

---
# Overview of the TREC 2021 deep learning track 

**Title (ZH)**: TREC 2021 深度学习赛道概览 

**Authors**: Nick Craswell, Bhaskar Mitra, Emine Yilmaz, Daniel Campos, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.08191)  

**Abstract**: This is the third year of the TREC Deep Learning track. As in previous years, we leverage the MS MARCO datasets that made hundreds of thousands of human annotated training labels available for both passage and document ranking tasks. In addition, this year we refreshed both the document and the passage collections which also led to a nearly four times increase in the document collection size and nearly $16$ times increase in the size of the passage collection. Deep neural ranking models that employ large scale pretraininig continued to outperform traditional retrieval methods this year. We also found that single stage retrieval can achieve good performance on both tasks although they still do not perform at par with multistage retrieval pipelines. Finally, the increase in the collection size and the general data refresh raised some questions about completeness of NIST judgments and the quality of the training labels that were mapped to the new collections from the old ones which we discuss in this report. 

**Abstract (ZH)**: 这是TREC深度学习赛道的第三年。与往年一样，我们利用MS MARCO数据集提供了数百万人标注的训练标签，用于段落和文档排名任务。此外，今年我们刷新了文档和段落集合，这导致文档集合的规模几乎增加了四倍，段落集合的规模增加了约十六倍。采用大规模预训练的深度神经排名模型继续在今年的实验中超过传统检索方法。我们还发现，单阶段检索在两个任务上都能取得较好的性能，尽管它们仍然无法与多阶段检索管道相匹敌。最后，集合规模的增加和整体数据的刷新引发了关于NIST判断完整性和从旧集合映射到新集合的训练标签质量的一些问题，我们在本报告中对此进行了讨论。 

---
# Rethinking Spatio-Temporal Anomaly Detection: A Vision for Causality-Driven Cybersecurity 

**Title (ZH)**: 重新思考时空异常检测：因果驱动的网络安全愿景 

**Authors**: Arun Vignesh Malarkkan, Haoyue Bai, Xinyuan Wang, Anjali Kaushik, Dongjie Wang, Yanjie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08177)  

**Abstract**: As cyber-physical systems grow increasingly interconnected and spatially distributed, ensuring their resilience against evolving cyberattacks has become a critical priority. Spatio-Temporal Anomaly detection plays an important role in ensuring system security and operational integrity. However, current data-driven approaches, largely driven by black-box deep learning, face challenges in interpretability, adaptability to distribution shifts, and robustness under evolving system dynamics. In this paper, we advocate for a causal learning perspective to advance anomaly detection in spatially distributed infrastructures that grounds detection in structural cause-effect relationships. We identify and formalize three key directions: causal graph profiling, multi-view fusion, and continual causal graph learning, each offering distinct advantages in uncovering dynamic cause-effect structures across time and space. Drawing on real-world insights from systems such as water treatment infrastructures, we illustrate how causal models provide early warning signals and root cause attribution, addressing the limitations of black-box detectors. Looking ahead, we outline the future research agenda centered on multi-modality, generative AI-driven, and scalable adaptive causal frameworks. Our objective is to lay a new research trajectory toward scalable, adaptive, explainable, and spatially grounded anomaly detection systems. We hope to inspire a paradigm shift in cybersecurity research, promoting causality-driven approaches to address evolving threats in interconnected infrastructures. 

**Abstract (ZH)**: 随着网络物理系统日益互联和空间分布，确保其抵御 evolving网络攻击的能力已成为一项 Critical Priority。时空异常检测在保障系统安全和操作完整性方面发挥着重要作用。然而，当前以黑盒深度学习为主的数据驱动方法在可解释性、分布变化适应性和在不断演变的系统动力学下的鲁棒性方面面临挑战。本文倡导从因果学习的角度推进分布式基础设施中的异常检测，使检测基于结构性因果关系。我们确定并形式化了三个关键方向：因果图剖析、多视图融合和持续因果图学习，每个方向都提供了在时间和空间上揭示动态因果结构的独特优势。借助水处理基础设施等系统的实际洞见，我们展示了因果模型如何提供早期预警信号和根本原因归因，解决了黑盒检测器的局限性。展望未来，我们指出了以多模态、生成AI驱动和可扩展自适应因果框架为中心的未来研究议程。我们的目标是为可扩展、自适应、可解释和空间性基础的异常检测系统奠定新的研究轨迹。我们期望激发网络安全研究中的范式转变，促进因果驱动的方法以应对互联基础设施中的不断演变威胁。 

---
# KP-A: A Unified Network Knowledge Plane for Catalyzing Agentic Network Intelligence 

**Title (ZH)**: KP-A: 促进能动网络智能的一体化网络知识平面 

**Authors**: Yun Tang, Mengbang Zou, Zeinab Nezami, Syed Ali Raza Zaidi, Weisi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2507.08164)  

**Abstract**: The emergence of large language models (LLMs) and agentic systems is enabling autonomous 6G networks with advanced intelligence, including self-configuration, self-optimization, and self-healing. However, the current implementation of individual intelligence tasks necessitates isolated knowledge retrieval pipelines, resulting in redundant data flows and inconsistent interpretations. Inspired by the service model unification effort in Open-RAN (to support interoperability and vendor diversity), we propose KP-A: a unified Network Knowledge Plane specifically designed for Agentic network intelligence. By decoupling network knowledge acquisition and management from intelligence logic, KP-A streamlines development and reduces maintenance complexity for intelligence engineers. By offering an intuitive and consistent knowledge interface, KP-A also enhances interoperability for the network intelligence agents. We demonstrate KP-A in two representative intelligence tasks: live network knowledge Q&A and edge AI service orchestration. All implementation artifacts have been open-sourced to support reproducibility and future standardization efforts. 

**Abstract (ZH)**: 大语言模型（LLMs）和自主系统的发展使具有高级智能的自主6G网络成为可能，包括自我配置、自我优化和自我修复。然而，当前个体智能任务的实现需要孤立的知识检索管道，导致数据流冗余和解释不一致。借鉴Open-RAN在服务模型统一方面的工作（以支持互操作性和供应商多样性），我们提出KP-A：一种专门为了自主网络智能设计的统一网络知识平面。通过将网络知识获取和管理与智能逻辑分离，KP-A 简化了智能工程师的开发并降低了维护复杂性。通过提供直观且一致的知识接口，KP-A 也增强了网络智能代理之间的互操作性。我们在两个典型的智能任务中展示了KP-A：实时网络知识问答和边缘AI服务编排。所有实现成果均已开源，以支持可重复性和未来标准化工作的进行。 

---
# AmpLyze: A Deep Learning Model for Predicting the Hemolytic Concentration 

**Title (ZH)**: AmpLyze: 一种用于预测溶血浓度的深度学习模型 

**Authors**: Peng Qiu, Hanqi Feng, Barnabas Poczos  

**Link**: [PDF](https://arxiv.org/pdf/2507.08162)  

**Abstract**: Red-blood-cell lysis (HC50) is the principal safety barrier for antimicrobial-peptide (AMP) therapeutics, yet existing models only say "toxic" or "non-toxic." AmpLyze closes this gap by predicting the actual HC50 value from sequence alone and explaining the residues that drive toxicity. The model couples residue-level ProtT5/ESM2 embeddings with sequence-level descriptors in dual local and global branches, aligned by a cross-attention module and trained with log-cosh loss for robustness to assay noise. The optimal AmpLyze model reaches a PCC of 0.756 and an MSE of 0.987, outperforming classical regressors and the state-of-the-art. Ablations confirm that both branches are essential, and cross-attention adds a further 1% PCC and 3% MSE improvement. Expected-Gradients attributions reveal known toxicity hotspots and suggest safer substitutions. By turning hemolysis assessment into a quantitative, sequence-based, and interpretable prediction, AmpLyze facilitates AMP design and offers a practical tool for early-stage toxicity screening. 

**Abstract (ZH)**: 红细胞裂解半数有效浓度（HC50）是抗菌肽（AMP）治疗安全性的重要屏障，现有模型仅能区分“有毒”或“无毒”。AmpLyze 通过仅从序列预测实际HC50值并解释驱动毒性的残基，填补了这一空白。该模型将残基级ProtT5/ESM2嵌入与序列级描述子结合，在双局部和全局分支中进行表征，并通过交叉注意力模块对齐，使用对试验噪声具有鲁棒性的对数双曲余弦损失进行训练。优化后的AmpLyze模型达到了相关系数（PCC）0.756和均方误差（MSE）0.987，优于经典回归器和当前最佳方法。消融实验表明两个分支都是必需的，交叉注意力进一步提高了1%的PCC和3%的MSE。梯度期望归因揭示了已知毒性热点，并建议了更安全的替换方案。通过将溶血评估转化为定量、序列基础和可解释的预测，AmpLyze 促进了AMP的设计，并提供了一种早期毒性筛查的实用工具。 

---
# ALCo-FM: Adaptive Long-Context Foundation Model for Accident Prediction 

**Title (ZH)**: ALCo-FM：自适应长上下文基础模型用于事故预测 

**Authors**: Pinaki Prasad Guha Neogi, Ahmad Mohammadshirazi, Rajiv Ramnath  

**Link**: [PDF](https://arxiv.org/pdf/2507.08153)  

**Abstract**: Traffic accidents are rare, yet high-impact events that require long-context multimodal reasoning for accurate risk forecasting. In this paper, we introduce ALCo-FM, a unified adaptive long-context foundation model that computes a volatility pre-score to dynamically select context windows for input data and encodes and fuses these multimodal data via shallow cross attention. Following a local GAT layer and a BigBird-style sparse global transformer over H3 hexagonal grids, coupled with Monte Carlo dropout for confidence, the model yields superior, well-calibrated predictions. Trained on data from 15 US cities with a class-weighted loss to counter label imbalance, and fine-tuned with minimal data on held-out cities, ALCo-FM achieves 0.94 accuracy, 0.92 F1, and an ECE of 0.04, outperforming more than 20 state-of-the-art baselines in large-scale urban risk prediction. Code and dataset are available at: this https URL 

**Abstract (ZH)**: 交通事故是高影响但频率较低的事件，需要长期上下文多模态推理以实现准确的风险预测。本文介绍了一种统一的自适应长期上下文基础模型ALCo-FM，该模型计算波动预评分以动态选择输入数据的上下文窗口，并通过浅层交叉注意机制编码和融合这些多模态数据。经过局部GAT层和基于H3六边形网格的BigBird风格稀疏全局变压器处理，并结合蒙特卡洛 Dropout 提供置信度，该模型提供了性能优越且校准良好的预测结果。ALCo-FM 使用加权损失在15个美国城市的数据上进行训练，并在保留城市上进行少量数据微调，最终在大规模城市风险预测中优于超过20个最先进的基线模型，准确率为0.94，F1分为0.92，ECE为0.04。代码和数据集可从以下链接获取：this https URL。 

---
# Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores 

**Title (ZH)**: Compactor: 校准的免查询 KV 缓存压缩方法及其近似杠杆得分 

**Authors**: Vivek Chari, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.08143)  

**Abstract**: Modern Large Language Models (LLMs) are increasingly trained to support very large context windows. Unfortunately the ability to use long contexts in generation is complicated by the large memory requirement of the KV cache, which scales linearly with the context length. This memory footprint is often the dominant resource bottleneck in real-world deployments, limiting throughput and increasing serving cost. One way to address this is by compressing the KV cache, which can be done either with knowledge of the question being asked (query-aware) or without knowledge of the query (query-agnostic). We present Compactor, a parameter-free, query-agnostic KV compression strategy that uses approximate leverage scores to determine token importance. We show that Compactor can achieve the same performance as competing methods while retaining 1/2 the tokens in both synthetic and real-world context tasks, with minimal computational overhead. We further introduce a procedure for context-calibrated compression, which allows one to infer the maximum compression ratio a given context can support. Using context-calibrated compression, we show that Compactor achieves full KV performance on Longbench while reducing the KV memory burden by 63%, on average. To demonstrate the efficacy and generalizability of our approach, we apply Compactor to 27 synthetic and real-world tasks from RULER and Longbench, with models from both the Qwen 2.5 and Llama 3.1 families. 

**Abstract (ZH)**: 现代大规模语言模型（LLMs）越来越多地被训练以支持非常大的上下文窗口。不幸的是，在生成过程中使用长上下文受到了KV缓存的大量内存需求的复杂影响，这种内存需求与上下文长度成线性关系。这种内存足迹在现实世界部署中往往是主要的资源瓶颈，限制了吞吐量并增加了服务成本。一种解决办法是压缩KV缓存，这可以是有问题知识的（查询感知的）或没有问题知识的（查询无关的）。我们提出了Compactor，这是一种参数无关、查询无关的KV压缩策略，使用近似杠杆得分来确定token的重要性。我们展示了Compactor能够在合成和真实世界上下文任务中实现与竞争方法相同的效果，同时保留原有token数量的一半，并且几乎没有额外的计算开销。我们还引入了一种上下文校准压缩的程序，允许人们推断出给定上下文支持的最大压缩比。使用上下文校准压缩，我们展示了Compactor在Longbench上实现了完整的KV性能，同时平均将KV内存负担减少了63%。为了展示我们方法的有效性和通用性，我们在RULER和Longbench的27个合成和真实世界任务中应用了Compactor，使用了来自Qwen 2.5和Llama 3.1系列的模型。 

---
# Temporally Consistent Amodal Completion for 3D Human-Object Interaction Reconstruction 

**Title (ZH)**: 三维人体-物体交互重建中的时间一致性不完备补全 

**Authors**: Hyungjun Doh, Dong In Lee, Seunggeun Chi, Pin-Hao Huang, Kwonjoon Lee, Sangpil Kim, Karthik Ramani  

**Link**: [PDF](https://arxiv.org/pdf/2507.08137)  

**Abstract**: We introduce a novel framework for reconstructing dynamic human-object interactions from monocular video that overcomes challenges associated with occlusions and temporal inconsistencies. Traditional 3D reconstruction methods typically assume static objects or full visibility of dynamic subjects, leading to degraded performance when these assumptions are violated-particularly in scenarios where mutual occlusions occur. To address this, our framework leverages amodal completion to infer the complete structure of partially obscured regions. Unlike conventional approaches that operate on individual frames, our method integrates temporal context, enforcing coherence across video sequences to incrementally refine and stabilize reconstructions. This template-free strategy adapts to varying conditions without relying on predefined models, significantly enhancing the recovery of intricate details in dynamic scenes. We validate our approach using 3D Gaussian Splatting on challenging monocular videos, demonstrating superior precision in handling occlusions and maintaining temporal stability compared to existing techniques. 

**Abstract (ZH)**: 一种克服遮挡和时间不一致性重构动态人-物交互的新框架 

---
# Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models 

**Title (ZH)**: Audio Flamingo 3: 以完全开放的大规模音频语言模型促进音频智能 

**Authors**: Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08128)  

**Abstract**: We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets. 

**Abstract (ZH)**: Audio Flamingo 3：一种面向语音、声效和音乐的全面开放的前沿大型音频语言模型 

---
# Quasi-Random Physics-informed Neural Networks 

**Title (ZH)**: 拟随机物理知情神经网络 

**Authors**: Tianchi Yu, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2507.08121)  

**Abstract**: Physics-informed neural networks have shown promise in solving partial differential equations (PDEs) by integrating physical constraints into neural network training, but their performance is sensitive to the sampling of points. Based on the impressive performance of quasi Monte-Carlo methods in high dimensional problems, this paper proposes Quasi-Random Physics-Informed Neural Networks (QRPINNs), which use low-discrepancy sequences for sampling instead of random points directly from the domain. Theoretically, QRPINNs have been proven to have a better convergence rate than PINNs. Empirically, experiments demonstrate that QRPINNs significantly outperform PINNs and some representative adaptive sampling methods, especially in high-dimensional PDEs. Furthermore, combining QRPINNs with adaptive sampling can further improve the performance. 

**Abstract (ZH)**: 物理信息神经网络通过将物理约束整合到神经网络训练中，在求解偏微分方程(PDEs)方面表现出潜力，但其性能对点的采样敏感。基于蒙特卡洛方法在高维问题上出色的性能，本文提出了一种基于低偏差序列采样的拟蒙特卡洛物理信息神经网络(QRPINNs)，而不是直接从域中随机采样点。理论上，QRPINNs的收敛速度优于物理信息神经网络(PINNs)。实验上，实验结果表明，QRPINNs在高维PDEs中的表现显著优于PINNs和一些代表性自适应采样方法。此外，将QRPINNs与自适应采样相结合可以进一步提高性能。 

---
# VideoConviction: A Multimodal Benchmark for Human Conviction and Stock Market Recommendations 

**Title (ZH)**: 视频确信：一种多模态基准，用于人类确信度与股市推荐 

**Authors**: Michael Galarnyk, Veer Kejriwal, Agam Shah, Yash Bhardwaj, Nicholas Meyer, Anand Krishnan, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2507.08104)  

**Abstract**: Social media has amplified the reach of financial influencers known as "finfluencers," who share stock recommendations on platforms like YouTube. Understanding their influence requires analyzing multimodal signals like tone, delivery style, and facial expressions, which extend beyond text-based financial analysis. We introduce VideoConviction, a multimodal dataset with 6,000+ expert annotations, produced through 457 hours of human effort, to benchmark multimodal large language models (MLLMs) and text-based large language models (LLMs) in financial discourse. Our results show that while multimodal inputs improve stock ticker extraction (e.g., extracting Apple's ticker AAPL), both MLLMs and LLMs struggle to distinguish investment actions and conviction--the strength of belief conveyed through confident delivery and detailed reasoning--often misclassifying general commentary as definitive recommendations. While high-conviction recommendations perform better than low-conviction ones, they still underperform the popular S\&P 500 index fund. An inverse strategy--betting against finfluencer recommendations--outperforms the S\&P 500 by 6.8\% in annual returns but carries greater risk (Sharpe ratio of 0.41 vs. 0.65). Our benchmark enables a diverse evaluation of multimodal tasks, comparing model performance on both full video and segmented video inputs. This enables deeper advancements in multimodal financial research. Our code, dataset, and evaluation leaderboard are available under the CC BY-NC 4.0 license. 

**Abstract (ZH)**: 社会媒体放大了所谓的“金fluencer”——在YouTube等平台上分享股票建议的金融影响者的影响。了解他们的影响需要分析语气、表达风格、面部表情等多模态信号，而这些信号超出了基于文本的金融分析。我们引入了VideoConviction多模态数据集，该数据集包含6000多个专家注解，通过457小时的人工努力生成，用于评估多模态大型语言模型（MLLMs）和文本大型语言模型（LLMs）在金融话语中的表现。我们的结果表明，虽然多模态输入提高了股票代码抽取（如提取苹果公司的代码AAPL）的效果，但无论是MLLMs还是LLMs，在区分投资行为和信心（通过自信的表达和详细的推理传达的信念强度）方面仍存在困难，经常错误地将一般评论分类为明确的建议。尽管高信心的建议表现优于低信心的建议，但它们在年度回报率上仍然落后于流行的标普500指数基金。采用与金fluencer建议相反的策略——做空金fluencer建议——在年度回报率上比标普500指数高出6.8%，但风险更大（夏普比率分别为0.41和0.65）。我们的基准测试使多模态任务的多样性评估成为可能，比较模型在完整视频和分段视频输入上的性能。这促进了多模态金融研究的更深发展。我们的代码、数据集和评估排行榜在CC BY-NC 4.0许可证下提供。 

---
# An Object-Based Deep Learning Approach for Building Height Estimation from Single SAR Images 

**Title (ZH)**: 基于对象的深度学习方法从单张SAR图像估计建筑物高度 

**Authors**: Babak Memar, Luigi Russo, Silvia Liberata Ullo, Paolo Gamba  

**Link**: [PDF](https://arxiv.org/pdf/2507.08096)  

**Abstract**: Accurate estimation of building heights using very high resolution (VHR) synthetic aperture radar (SAR) imagery is crucial for various urban applications. This paper introduces a Deep Learning (DL)-based methodology for automated building height estimation from single VHR COSMO-SkyMed images: an object-based regression approach based on bounding box detection followed by height estimation. This model was trained and evaluated on a unique multi-continental dataset comprising eight geographically diverse cities across Europe, North and South America, and Asia, employing a cross-validation strategy to explicitly assess out-of-distribution (OOD) generalization. The results demonstrate highly promising performance, particularly on European cities where the model achieves a Mean Absolute Error (MAE) of approximately one building story (2.20 m in Munich), significantly outperforming recent state-of-the-art methods in similar OOD scenarios. Despite the increased variability observed when generalizing to cities in other continents, particularly in Asia with its distinct urban typologies and prevalence of high-rise structures, this study underscores the significant potential of DL for robust cross-city and cross-continental transfer learning in building height estimation from single VHR SAR data. 

**Abstract (ZH)**: 使用高分辨率合成孔径雷达图像的建筑高度准确估计对各类城市应用至关重要：一种基于深度学习的方法 

---
# Tree-Structured Parzen Estimator Can Solve Black-Box Combinatorial Optimization More Efficiently 

**Title (ZH)**: 树结构帕兹恩估计器可以更高效地解决黑盒组合优化问题 

**Authors**: Kenshin Abe, Yunzhuo Wang, Shuhei Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2507.08053)  

**Abstract**: Tree-structured Parzen estimator (TPE) is a versatile hyperparameter optimization (HPO) method supported by popular HPO tools. Since these HPO tools have been developed in line with the trend of deep learning (DL), the problem setups often used in the DL domain have been discussed for TPE such as multi-objective optimization and multi-fidelity optimization. However, the practical applications of HPO are not limited to DL, and black-box combinatorial optimization is actively utilized in some domains, e.g., chemistry and biology. As combinatorial optimization has been an untouched, yet very important, topic in TPE, we propose an efficient combinatorial optimization algorithm for TPE. In this paper, we first generalize the categorical kernel with the numerical kernel in TPE, enabling us to introduce a distance structure to the categorical kernel. Then we discuss modifications for the newly developed kernel to handle a large combinatorial search space. These modifications reduce the time complexity of the kernel calculation with respect to the size of a combinatorial search space. In the experiments using synthetic problems, we verified that our proposed method identifies better solutions with fewer evaluations than the original TPE. Our algorithm is available in Optuna, an open-source framework for HPO. 

**Abstract (ZH)**: 基于树结构帕兹恩估计器的组合优化算法 

---
# An Enhanced Privacy-preserving Federated Few-shot Learning Framework for Respiratory Disease Diagnosis 

**Title (ZH)**: 增强的隐私保护联邦少样本学习框架用于呼吸系统疾病诊断 

**Authors**: Ming Wang, Zhaoyang Duan, Dong Xue, Fangzhou Liu, Zhongheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08050)  

**Abstract**: The labor-intensive nature of medical data annotation presents a significant challenge for respiratory disease diagnosis, resulting in a scarcity of high-quality labeled datasets in resource-constrained settings. Moreover, patient privacy concerns complicate the direct sharing of local medical data across institutions, and existing centralized data-driven approaches, which rely on amounts of available data, often compromise data privacy. This study proposes a federated few-shot learning framework with privacy-preserving mechanisms to address the issues of limited labeled data and privacy protection in diagnosing respiratory diseases. In particular, a meta-stochastic gradient descent algorithm is proposed to mitigate the overfitting problem that arises from insufficient data when employing traditional gradient descent methods for neural network training. Furthermore, to ensure data privacy against gradient leakage, differential privacy noise from a standard Gaussian distribution is integrated into the gradients during the training of private models with local data, thereby preventing the reconstruction of medical images. Given the impracticality of centralizing respiratory disease data dispersed across various medical institutions, a weighted average algorithm is employed to aggregate local diagnostic models from different clients, enhancing the adaptability of a model across diverse scenarios. Experimental results show that the proposed method yields compelling results with the implementation of differential privacy, while effectively diagnosing respiratory diseases using data from different structures, categories, and distributions. 

**Abstract (ZH)**: 医疗数据注释的劳动密集性质为呼吸疾病诊断带来了显著挑战，导致资源受限环境中高质量标注数据的稀缺。此外，患者隐私担忧使机构间直接共享本地医疗数据变得复杂，现有的依赖大量可用数据的集中式数据驱动方法经常牺牲数据隐私。本研究提出了一种带有隐私保护机制的联邦少-shot学习框架，以解决诊断呼吸疾病时标注数据有限和隐私保护的问题。特别是，提出了一种元随机梯度下降算法来缓解使用传统梯度下降方法训练神经网络时由数据不足引起的过拟合问题。此外，为了防止梯度泄露导致的隐私泄露，整合了标准高斯分布的差分隐私噪声到本地数据训练的私有模型的梯度中，从而防止重建医疗图像。鉴于呼吸疾病数据分散在各个医疗机构之间难以集中化，采用了加权平均算法聚合不同客户端的诊断模型，增强了模型在多种场景下的适应性。实验结果表明，所提出的方法在实施差分隐私的同时，能够有效诊断来自不同结构、类别和分布的数据，取得了令人信服的结果。 

---
# Krul: Efficient State Restoration for Multi-turn Conversations with Dynamic Cross-layer KV Sharing 

**Title (ZH)**: Krul: 多轮对话中动态跨层KV共享的高效状态恢复 

**Authors**: Junyi Wen, Junyuan Liang, Zicong Hong, Wuhui Chen, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.08045)  

**Abstract**: Efficient state restoration in multi-turn conversations with large language models (LLMs) remains a critical challenge, primarily due to the overhead of recomputing or loading full key-value (KV) caches for all historical tokens. To address this, existing approaches compress KV caches across adjacent layers with highly similar attention patterns. However, these methods often apply a fixed compression scheme across all conversations, selecting the same layer pairs for compression without considering conversation-specific attention dynamics. This static strategy overlooks variability in attention pattern similarity across different conversations, which can lead to noticeable accuracy degradation.
We present Krul, a multi-turn LLM inference system that enables accurate and efficient KV cache restoration. Krul dynamically selects compression strategies based on attention similarity across layer pairs and uses a recomputation-loading pipeline to restore the KV cache. It introduces three key innovations: 1) a preemptive compression strategy selector to preserve critical context for future conversation turns and selects a customized strategy for the conversation; 2) a token-wise heterogeneous attention similarity estimator to mitigate the attention similarity computation and storage overhead during model generation; 3) a bubble-free restoration scheduler to reduce potential bubbles brought by the imbalance of recomputing and loading stream due to compressed KV caches. Empirical evaluations on real-world tasks demonstrate that Krul achieves a 1.5x-2.68x reduction in time-to-first-token (TTFT) and a 1.33x-2.35x reduction in KV cache storage compared to state-of-the-art methods without compromising generation quality. 

**Abstract (ZH)**: 高效的大语言模型（LLMs）在多轮对话中的状态恢复仍是一项关键挑战，主要原因在于需要重新计算或加载所有历史令牌的完整键值（KV）缓存带来的开销。为解决这一问题，现有方法通过在具有高度相似注意力模式的相邻层间压缩KV缓存来压缩KV缓存。然而，这些方法通常在整个对话中使用固定的压缩方案，为所有对话选择相同的层对进行压缩，而忽略了对话特定的注意力动态。这种静态策略忽略了不同对话之间注意力模式相似性变化的差异性，这可能导致明显的准确率下降。

我们提出Krul，一种多轮L LM推理系统，能够实现准确高效的KV缓存恢复。Krul基于层对间的注意力相似性动态选择压缩策略，并采用重新计算-加载流水线来恢复KV缓存。它引入了三项关键创新：1) 预见性的压缩策略选择器，以保留对未来对话轮次至关重要的上下文，并为对话选择定制化策略；2) 单词级异质注意力相似性估计器，以减轻模型生成过程中注意力相似性计算和存储开销；3) 泡沫自由的恢复调度器，以减少由压缩KV缓存引起的重新计算与加载流不平衡所带来的潜在泡沫。实证研究表明，与最先进的方法相比，Krul在不牺牲生成质量的情况下，实现了1.5倍至2.68倍的时间到首个单词（TTFT）的减少和1.33倍至2.35倍的KV缓存存储减少。 

---
# ConsNoTrainLoRA: Data-driven Weight Initialization of Low-rank Adapters using Constraints 

**Title (ZH)**: ConsNoTrainLoRA: 受约束的低秩适配器的数据驱动权重初始化 

**Authors**: Debasmit Das, Hyoungwoo Park, Munawar Hayat, Seokeon Choi, Sungrack Yun, Fatih Porikli  

**Link**: [PDF](https://arxiv.org/pdf/2507.08044)  

**Abstract**: Foundation models are pre-trained on large-scale datasets and subsequently fine-tuned on small-scale datasets using parameter-efficient fine-tuning (PEFT) techniques like low-rank adapters (LoRA). In most previous works, LoRA weight matrices are randomly initialized with a fixed rank across all attachment points. In this paper, we improve convergence and final performance of LoRA fine-tuning, using our proposed data-driven weight initialization method, ConsNoTrainLoRA (CNTLoRA). We express LoRA initialization as a domain shift problem where we use multiple constraints relating the pre-training and fine-tuning activations. By reformulating these constraints, we obtain a closed-form estimate of LoRA weights that depends on pre-training weights and fine-tuning activation vectors and hence requires no training during initialization. This weight estimate is decomposed to initialize the up and down matrices with proposed flexibility of variable ranks. With the proposed initialization method, we fine-tune on downstream tasks such as image generation, image classification and image understanding. Both quantitative and qualitative results demonstrate that CNTLoRA outperforms standard and data-driven weight initialization methods. Extensive analyses and ablations further elucidate the design choices of our framework, providing an optimal recipe for faster convergence and enhanced performance. 

**Abstract (ZH)**: 使用数据驱动的权重初始化方法ConsNoTrainLoRA提高LoRA微调的收敛性和最终性能 

---
# Towards Evaluating Robustness of Prompt Adherence in Text to Image Models 

**Title (ZH)**: 面向文本到图像模型提示遵从性robustness的评估研究 

**Authors**: Sujith Vemishetty, Advitiya Arora, Anupama Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08039)  

**Abstract**: The advancements in the domain of LLMs in recent years have surprised many, showcasing their remarkable capabilities and diverse applications. Their potential applications in various real-world scenarios have led to significant research on their reliability and effectiveness. On the other hand, multimodal LLMs and Text-to-Image models have only recently gained prominence, especially when compared to text-only LLMs. Their reliability remains constrained due to insufficient research on assessing their performance and robustness. This paper aims to establish a comprehensive evaluation framework for Text-to-Image models, concentrating particularly on their adherence to prompts. We created a novel dataset that aimed to assess the robustness of these models in generating images that conform to the specified factors of variation in the input text prompts. Our evaluation studies present findings on three variants of Stable Diffusion models: Stable Diffusion 3 Medium, Stable Diffusion 3.5 Large, and Stable Diffusion 3.5 Large Turbo, and two variants of Janus models: Janus Pro 1B and Janus Pro 7B. We introduce a pipeline that leverages text descriptions generated by the gpt-4o model for our ground-truth images, which are then used to generate artificial images by passing these descriptions to the Text-to-Image models. We then pass these generated images again through gpt-4o using the same system prompt and compare the variation between the two descriptions. Our results reveal that these models struggle to create simple binary images with only two factors of variation: a simple geometric shape and its location. We also show, using pre-trained VAEs on our dataset, that they fail to generate images that follow our input dataset distribution. 

**Abstract (ZH)**: 近年来，Large Language Models (LLMs) 的发展令人惊讶，展示了其出色的能力和多种多样的应用。它们在各种现实场景中的潜在应用促使人们对其可靠性和有效性进行了大量研究。另一方面，多模态LLMs和Text-to-Image模型最近才受到重视，尤其是在与仅文本的LLMs相比时更为显著。由于对其性能和鲁棒性评估研究不足，这些模型的可靠性仍受到限制。本文旨在建立一个全面的Text-to-Image模型评估框架，特别关注它们对提示的遵守情况。我们创建了一个新的数据集，旨在评估这些模型在生成符合输入文本提示中指定变异因素的图像时的鲁棒性。我们的评估研究针对三种Stable Diffusion模型变体：Stable Diffusion 3 Medium、Stable Diffusion 3.5 Large 和 Stable Diffusion 3.5 Large Turbo，以及两种Janus模型变体：Janus Pro 1B 和 Janus Pro 7B。我们引入了一个模型管道，利用gpt-4o模型生成的文本描述作为我们的 ground-truth 图像，然后通过将这些描述传递给Text-to-Image模型生成人工图像。然后，我们再次将这些生成的图像通过gpt-4o系统提示处理，并比较两次描述之间的差异。我们的结果显示，这些模型难以生成仅有两个变异因素的简单二值图像：一个简单的几何形状及其位置。我们还使用在我们数据集上预训练的VAEs展示了它们无法生成遵循我们输入数据集分布的图像。 

---
# AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research 

**Title (ZH)**: AblationBench: 评估实证人工智能研究中消融自动化规划的评估工具 

**Authors**: Talor Abramovich, Gal Chechik  

**Link**: [PDF](https://arxiv.org/pdf/2507.08038)  

**Abstract**: Autonomous agents built on language models (LMs) are showing increasing popularity in many fields, including scientific research. AI co-scientists aim to support or automate parts of the research process using these agents. A key component of empirical AI research is the design of ablation experiments. To this end, we introduce AblationBench, a benchmark suite for evaluating agents on ablation planning tasks in empirical AI research. It includes two tasks: AuthorAblation, which helps authors propose ablation experiments based on a method section and contains 83 instances, and ReviewerAblation, which helps reviewers find missing ablations in a full paper and contains 350 instances. For both tasks, we develop LM-based judges that serve as an automatic evaluation framework. Our experiments with frontier LMs show that these tasks remain challenging, with the best-performing LM system identifying only 29% of the original ablations on average. Lastly, we analyze the limitations of current LMs on these tasks, and find that chain-of-thought prompting outperforms the currently existing agent-based approach. 

**Abstract (ZH)**: 基于语言模型的自主代理在多个领域显示出日益增长的 popularity，包括科学研究。AI合作科学家旨在利用这些代理支持或自动化研究过程的部分环节。经验性人工智能研究的一个关键组成部分是消融实验的设计。为此，我们引入了AblationBench，这是一个评估代理在经验性人工智能研究中的消融规划任务的基准套件。它包括两个任务：AuthorAblation，帮助作者根据方法部分提出消融实验，包含83个实例；ReviewerAblation，帮助审稿人发现完整论文中的缺失消融实验，包含350个实例。对于这两个任务，我们开发了基于语言模型的评判器，作为自动评价框架。我们的实验表明，最出色的LM系统平均仅能识别出原始消融实验的29%。最后，我们分析了当前LM在这两个任务中的局限性，并发现链式思考提示优于现有基于代理的方法。 

---
# CRISP: Complex Reasoning with Interpretable Step-based Plans 

**Title (ZH)**: CRISP: 具有可解释步骤计划的复杂推理 

**Authors**: Matan Vetzler, Koren Lazar, Guy Uziel, Eran Hirsch, Ateret Anaby-Tavor, Leshem Choshen  

**Link**: [PDF](https://arxiv.org/pdf/2507.08037)  

**Abstract**: Recent advancements in large language models (LLMs) underscore the need for stronger reasoning capabilities to solve complex problems effectively. While Chain-of-Thought (CoT) reasoning has been a step forward, it remains insufficient for many domains. A promising alternative is explicit high-level plan generation, but existing approaches largely assume that LLMs can produce effective plans through few-shot prompting alone, without additional training. In this work, we challenge this assumption and introduce CRISP (Complex Reasoning with Interpretable Step-based Plans), a multi-domain dataset of high-level plans for mathematical reasoning and code generation. The plans in CRISP are automatically generated and rigorously validated--both intrinsically, using an LLM as a judge, and extrinsically, by evaluating their impact on downstream task performance. We demonstrate that fine-tuning a small model on CRISP enables it to generate higher-quality plans than much larger models using few-shot prompting, while significantly outperforming Chain-of-Thought reasoning. Furthermore, our out-of-domain evaluation reveals that fine-tuning on one domain improves plan generation in the other, highlighting the generalizability of learned planning capabilities. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）强调了为了有效解决复杂问题，需要更强的推理能力。虽然Step-by-Step（SbS）推理是一个进步，但对于许多领域仍然不够。一种有前途的替代方案是显式的高层次计划生成，但现有方法主要假设通过少样本提示，LMs可以生成有效的计划，而不需要额外训练。在这项工作中，我们挑战了这一假设，并介绍了CRISP（Complex Reasoning with Interpretable Step-based Plans），这是一个包含数学推理和代码生成的多领域高层次计划数据集。CRISP中的计划是自动生成并通过LLM作为裁判进行严格验证的，同时也通过评估它们对下游任务性能的影响来进行外部验证。我们证明，对CRISP进行微调的小模型能够与其相比使用少样本提示生成更高质量的计划，并且在链式推理方面表现更优。此外，我们的跨领域评估表明，在一个领域进行微调可以提高另一个领域的计划生成能力，突显了学习计划能力的泛化能力。 

---
# Integrating External Tools with Large Language Models to Improve Accuracy 

**Title (ZH)**: 将外部工具集成到大型语言模型以提高准确性 

**Authors**: Nripesh Niketan, Hadj Batatia  

**Link**: [PDF](https://arxiv.org/pdf/2507.08034)  

**Abstract**: This paper deals with improving querying large language models (LLMs). It is well-known that without relevant contextual information, LLMs can provide poor quality responses or tend to hallucinate. Several initiatives have proposed integrating LLMs with external tools to provide them with up-to-date data to improve accuracy. In this paper, we propose a framework to integrate external tools to enhance the capabilities of LLMs in answering queries in educational settings. Precisely, we develop a framework that allows accessing external APIs to request additional relevant information. Integrated tools can also provide computational capabilities such as calculators or calendars. The proposed framework has been evaluated using datasets from the Multi-Modal Language Understanding (MMLU) collection. The data consists of questions on mathematical and scientific reasoning. Results compared to state-of-the-art language models show that the proposed approach significantly improves performance. Our Athena framework achieves 83% accuracy in mathematical reasoning and 88% in scientific reasoning, substantially outperforming all tested models including GPT-4o, LLaMA-Large, Mistral-Large, Phi-Large, and GPT-3.5, with the best baseline model (LLaMA-Large) achieving only 67% and 79% respectively. These promising results open the way to creating complex computing ecosystems around LLMs to make their use more natural to support various tasks and activities. 

**Abstract (ZH)**: 改进大型语言模型查询能力的框架：在教育场景中集成外部工具以提高准确性和计算能力 

---
# SSSUMO: Real-Time Semi-Supervised Submovement Decomposition 

**Title (ZH)**: SSSUMO: 实时半监督子运动分解 

**Authors**: Evgenii Rudakov, Jonathan Shock, Otto Lappi, Benjamin Ultan Cowley  

**Link**: [PDF](https://arxiv.org/pdf/2507.08028)  

**Abstract**: This paper introduces a SSSUMO, semi-supervised deep learning approach for submovement decomposition that achieves state-of-the-art accuracy and speed. While submovement analysis offers valuable insights into motor control, existing methods struggle with reconstruction accuracy, computational cost, and validation, due to the difficulty of obtaining hand-labeled data. We address these challenges using a semi-supervised learning framework. This framework learns from synthetic data, initially generated from minimum-jerk principles and then iteratively refined through adaptation to unlabeled human movement data. Our fully convolutional architecture with differentiable reconstruction significantly surpasses existing methods on both synthetic and diverse human motion datasets, demonstrating robustness even in high-noise conditions. Crucially, the model operates in real-time (less than a millisecond per input second), a substantial improvement over optimization-based techniques. This enhanced performance facilitates new applications in human-computer interaction, rehabilitation medicine, and motor control studies. We demonstrate the model's effectiveness across diverse human-performed tasks such as steering, rotation, pointing, object moving, handwriting, and mouse-controlled gaming, showing notable improvements particularly on challenging datasets where traditional methods largely fail. Training and benchmarking source code, along with pre-trained model weights, are made publicly available at this https URL. 

**Abstract (ZH)**: 一种半监督深度学习方法SSSUMO在子运动分解中的应用：实现最先进的准确性和速度 

---
# Unveiling Effective In-Context Configurations for Image Captioning: An External & Internal Analysis 

**Title (ZH)**: 揭示有效的图像 captioning 先 contextual 配置：外部与内部分析 

**Authors**: Li Li, Yongliang Wu, Jingze Zhu, Jiawei Peng, Jianfei Cai, Xu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08021)  

**Abstract**: The evolution of large models has witnessed the emergence of In-Context Learning (ICL) capabilities. In Natural Language Processing (NLP), numerous studies have demonstrated the effectiveness of ICL. Inspired by the success of Large Language Models (LLMs), researchers have developed Large Multimodal Models (LMMs) with ICL capabilities. However, explorations of demonstration configuration for multimodal ICL remain preliminary. Additionally, the controllability of In-Context Examples (ICEs) provides an efficient and cost-effective means to observe and analyze the inference characteristics of LMMs under varying inputs. This paper conducts a comprehensive external and internal investigation of multimodal in-context learning on the image captioning task. Externally, we explore demonstration configuration strategies through three dimensions: shot number, image retrieval, and caption assignment. We employ multiple metrics to systematically and thoroughly evaluate and summarize key findings. Internally, we analyze typical LMM attention characteristics and develop attention-based metrics to quantify model behaviors. We also conduct auxiliary experiments to explore the feasibility of attention-driven model acceleration and compression. We further compare performance variations between LMMs with identical model design and pretraining strategies and explain the differences from the angles of pre-training data features. Our study reveals both how ICEs configuration strategies impact model performance through external experiments and characteristic typical patterns through internal inspection, providing dual perspectives for understanding multimodal ICL in LMMs. Our method of combining external and internal analysis to investigate large models, along with our newly proposed metrics, can be applied to broader research areas. 

**Abstract (ZH)**: 大规模模型进化见证了基于上下文学习（ICL）能力的 emergence。在自然语言处理（NLP）领域，大量研究已经证明了ICL的有效性。受大型语言模型（LLMs）成功的启发，研究人员开发了具备ICL能力的大规模多模态模型（LMMs）。然而，多模态ICL的示范配置探索仍处于初级阶段。此外，In-Context Examples（ICEs）的可控性为观察和分析在不同输入下LMMs的推理特性提供了一种高效且经济的手段。本文对图像 captioning 任务中的多模态基于上下文学习进行全面的外部和内部研究。外部方面，我们通过三个维度探索示范配置策略：图的数量、图像检索和字幕分配。我们采用多种指标系统地、全面地评估和总结关键发现。内部方面，我们分析典型的大规模多模态模型的注意力特性，开发基于注意力的指标来量化模型行为。我们还进行了辅助实验，探索基于注意力的模型加速和压缩的可行性。我们进一步比较了具有相同模型设计和预训练策略的大规模多模态模型之间的性能差异，并从预训练数据特征的角度解释差异。我们的研究表明，通过外部实验 ICEs 配置策略如何影响模型性能，以及通过内部检查找到典型的模式，提供了理解大规模多模态模型中多模态ICL的双重视角。我们的外部和内部分析相结合的方法，以及新提出的指标，可以应用于更广泛的科研领域。 

---
# Circumventing Safety Alignment in Large Language Models Through Embedding Space Toxicity Attenuation 

**Title (ZH)**: 通过嵌入空间毒性衰减规避大型语言模型的安全对齐 

**Authors**: Zhibo Zhang, Yuxi Li, Kailong Wang, Shuai Yuan, Ling Shi, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08020)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success across domains such as healthcare, education, and cybersecurity. However, this openness also introduces significant security risks, particularly through embedding space poisoning, which is a subtle attack vector where adversaries manipulate the internal semantic representations of input data to bypass safety alignment mechanisms. While previous research has investigated universal perturbation methods, the dynamics of LLM safety alignment at the embedding level remain insufficiently understood. Consequently, more targeted and accurate adversarial perturbation techniques, which pose significant threats, have not been adequately studied.
In this work, we propose ETTA (Embedding Transformation Toxicity Attenuation), a novel framework that identifies and attenuates toxicity-sensitive dimensions in embedding space via linear transformations. ETTA bypasses model refusal behaviors while preserving linguistic coherence, without requiring model fine-tuning or access to training data. Evaluated on five representative open-source LLMs using the AdvBench benchmark, ETTA achieves a high average attack success rate of 88.61%, outperforming the best baseline by 11.34%, and generalizes to safety-enhanced models (e.g., 77.39% ASR on instruction-tuned defenses). These results highlight a critical vulnerability in current alignment strategies and underscore the need for embedding-aware defenses. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗保健、教育和网络安全等领域取得了显著的成功。然而，这种开放性也引入了重大的安全风险，特别是通过嵌入空间投毒，这是一种微妙的攻击向量，对手可以通过操纵输入数据的内部语义表示来规避安全对齐机制。虽然以往的研究已经探讨了普遍扰动方法，但LLM在嵌入层面的安全对齐动态仍然理解不足。因此，具有更高针对性和准确度的对抗性扰动技术尚未得到充分研究。

在本项工作中，我们提出了ETTA（嵌入空间投毒衰减）框架，该框架通过线性转换识别并衰减嵌入空间中的毒性敏感维度。ETTA在不进行模型微调或不需要访问训练数据的情况下，能够绕过模型拒绝行为并保持语义连贯性。在使用AdvBench基准测试的五个代表性的开源LLM上，ETTA实现了高达88.61%的平均攻击成功率，比最佳基线高出11.34%，并且对增强安全性的模型进行了泛化（例如，指令微调防御的87.39%的成功率）。这些结果突显了当前对齐策略中的关键漏洞，并强调了嵌入感知防御的需求。 

---
# Mechanistic Indicators of Understanding in Large Language Models 

**Title (ZH)**: 大型语言模型中理解的机理指标 

**Authors**: Pierre Beckmann, Matthieu Queloz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08017)  

**Abstract**: Recent findings in mechanistic interpretability (MI), the field probing the inner workings of Large Language Models (LLMs), challenge the view that these models rely solely on superficial statistics. Here, we offer an accessible synthesis of these findings that doubles as an introduction to MI, all while integrating these findings within a novel theoretical framework for thinking about machine understanding. We argue that LLMs develop internal structures that are functionally analogous to the kind of understanding that consists in seeing connections. To sharpen this idea, we propose a three-tiered conception of machine understanding. First, conceptual understanding emerges when a model forms "features" as directions in latent space, thereby learning the connections between diverse manifestations of something. Second, state-of-the-world understanding emerges when a model learns contingent factual connections between features and dynamically tracks changes in the world. Third, principled understanding emerges when a model ceases to rely on a collection of memorized facts and discovers a "circuit" that connects these facts. However, we conclude by exploring the "parallel mechanisms" phenomenon, arguing that while LLMs exhibit forms of understanding, their cognitive architecture remains different from ours, and the debate should shift from whether LLMs understand to how their strange minds work. 

**Abstract (ZH)**: 近期关于机制可解释性（MI）的研究发现：探索大型语言模型（LLMs）内部工作机制的领域挑战了这些模型仅依赖表面统计的观点。本文提供了一种易于理解的MI发现综述，同时也作为MI介绍，将其研究成果整合到一个关于机器理解的新理论框架中。我们argue认为，LLMs发展出的功能上类似于发现联系的认知结构。为了使这一观点更清晰，我们提出了一种三层次的机器理解概念。首先，概念理解在模型形成“特征”作为潜在空间中的方向时出现，从而学习不同表现形式之间的联系。其次，世界状态理解在模型学习特征之间的条件性事实联系并动态追踪世界变化时出现。第三，原则性理解在模型不再依赖于记忆的事实集合，而是发现将这些事实连接起来的“电路”时出现。然而，我们最终探讨了“平行机制”现象，认为虽然LLMs表现出某种形式的理解，但其认知架构与人类不同，讨论应从LLMs是否理解转向如何理解它们异常的大脑工作方式。 

---
# Mass-Scale Analysis of In-the-Wild Conversations Reveals Complexity Bounds on LLM Jailbreaking 

**Title (ZH)**: 大规模分析野生对话揭示了LLM Jailbreaking的复杂性界限 

**Authors**: Aldan Creo, Raul Castro Fernandez, Manuel Cebrian  

**Link**: [PDF](https://arxiv.org/pdf/2507.08014)  

**Abstract**: As large language models (LLMs) become increasingly deployed, understanding the complexity and evolution of jailbreaking strategies is critical for AI safety.
We present a mass-scale empirical analysis of jailbreak complexity across over 2 million real-world conversations from diverse platforms, including dedicated jailbreaking communities and general-purpose chatbots. Using a range of complexity metrics spanning probabilistic measures, lexical diversity, compression ratios, and cognitive load indicators, we find that jailbreak attempts do not exhibit significantly higher complexity than normal conversations. This pattern holds consistently across specialized jailbreaking communities and general user populations, suggesting practical bounds on attack sophistication. Temporal analysis reveals that while user attack toxicity and complexity remains stable over time, assistant response toxicity has decreased, indicating improving safety mechanisms. The absence of power-law scaling in complexity distributions further points to natural limits on jailbreak development.
Our findings challenge the prevailing narrative of an escalating arms race between attackers and defenders, instead suggesting that LLM safety evolution is bounded by human ingenuity constraints while defensive measures continue advancing. Our results highlight critical information hazards in academic jailbreak disclosure, as sophisticated attacks exceeding current complexity baselines could disrupt the observed equilibrium and enable widespread harm before defensive adaptation. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的不断部署，理解 Jailbreak 策略的复杂性和演变对于AI安全至关重要。我们对超过200万条来自各种平台的真实对话进行了大规模实证分析，包括专门的 Jailbreak 社区和通用聊天机器人。利用涵盖概率度量、词汇多样性、压缩比和认知负荷指标等多种复杂度度量，我们发现 Jailbreak 尝试的复杂度并不显著高于普通对话。这一模式在专门的 Jailbreak 社区和普通用户群体中保持一致，表明攻击复杂度存在实际上限。时间分析显示，虽然用户攻击毒性和复杂性保持稳定，辅助响应的毒性有所下降，表明安全机制在改进。复杂度分布中缺乏幂律分布进一步表明 Jailbreak 发展存在自然限制。我们的发现挑战了攻击者与防御者之间逐步升级的 Arms Race 模式，反而表明 LLM 安全进化受限于人类创新的限制，同时防御措施继续进步。我们的结果强调了学术界 Jailbreak 披露中的关键信息危害，因为超出当前复杂度基线的复杂攻击可能在防御适应之前破坏观察到的平衡，引发广泛危害。 

---
# MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model 

**Title (ZH)**: MedicalBERT：使用预训练的BERT模型增强生物医学自然语言处理 

**Authors**: K. Sahit Reddy, N. Ragavenderan, Vasanth K., Ganesh N. Naik, Vishalakshi Prabhu, Nagaraja G. S  

**Link**: [PDF](https://arxiv.org/pdf/2507.08013)  

**Abstract**: Recent advances in natural language processing (NLP) have been driven bypretrained language models like BERT, RoBERTa, T5, and GPT. Thesemodels excel at understanding complex texts, but biomedical literature, withits domain-specific terminology, poses challenges that models likeWord2Vec and bidirectional long short-term memory (Bi-LSTM) can't fullyaddress. GPT and T5, despite capturing context, fall short in tasks needingbidirectional understanding, unlike BERT. Addressing this, we proposedMedicalBERT, a pretrained BERT model trained on a large biomedicaldataset and equipped with domain-specific vocabulary that enhances thecomprehension of biomedical terminology. MedicalBERT model is furtheroptimized and fine-tuned to address diverse tasks, including named entityrecognition, relation extraction, question answering, sentence similarity, anddocument classification. Performance metrics such as the F1-score,accuracy, and Pearson correlation are employed to showcase the efficiencyof our model in comparison to other BERT-based models such as BioBERT,SciBERT, and ClinicalBERT. MedicalBERT outperforms these models onmost of the benchmarks, and surpasses the general-purpose BERT model by5.67% on average across all the tasks evaluated respectively. This work alsounderscores the potential of leveraging pretrained BERT models for medicalNLP tasks, demonstrating the effectiveness of transfer learning techniques incapturing domain-specific information.
(PDF) MedicalBERT: enhancing biomedical natural language processing using pretrained BERT-based model. Available from: this https URL [accessed Jul 06 2025]. 

**Abstract (ZH)**: MedicalBERT：利用预训练BERT模型增强 biomedical 自然语言处理 

---
# RepeaTTS: Towards Feature Discovery through Repeated Fine-Tuning 

**Title (ZH)**: RepeaTTS：通过重复微调实现特征发现 

**Authors**: Atli Sigurgeirsson, Simon King  

**Link**: [PDF](https://arxiv.org/pdf/2507.08012)  

**Abstract**: A Prompt-based Text-To-Speech model allows a user to control different aspects of speech, such as speaking rate and perceived gender, through natural language instruction. Although user-friendly, such approaches are on one hand constrained: control is limited to acoustic features exposed to the model during training, and too flexible on the other: the same inputs yields uncontrollable variation that are reflected in the corpus statistics.
We investigate a novel fine-tuning regime to address both of these issues at the same time by exploiting the uncontrollable variance of the model. Through principal component analysis of thousands of synthesised samples, we determine latent features that account for the highest proportion of the output variance and incorporate them as new labels for secondary fine-tuning. We evaluate the proposed methods on two models trained on an expressive Icelandic speech corpus, one with emotional disclosure and one without. In the case of the model without emotional disclosure, the method yields both continuous and discrete features that improve overall controllability of the model. 

**Abstract (ZH)**: 基于提示的文本转语音模型通过自然语言指令允许用户控制语音的不同方面，如发音速率和感知性别。虽然这种方法用户友好，但在控制方面仍然受限：控制仅限于模型训练期间暴露的声学特征，而在灵活性方面则不足：相同的输入导致不可控的变异，这些变异在语料库统计中显现出来。

我们研究了一种新的微调制度，同时解决这两方面的问题，通过利用模型的不可控变异。通过数千个合成样本的主成分分析，我们确定了能够解释输出变异最大比例的潜在特征，并将它们作为次要微调的新标签进行集成。我们在这两个分别基于富有表现力的冰岛语音语料库训练的模型上评估了所提出的方法，一个带有情感披露，另一个没有。在没有情感披露的模型的情况下，该方法产生了连续和离散特征，从而提高了模型的整体可控性。 

---
# Energy Management for Renewable-Colocated Artificial Intelligence Data Centers 

**Title (ZH)**: 可再生能源共址人工智能数据中心的能源管理 

**Authors**: Siying Li, Lang Tong, Timothy D. Mount  

**Link**: [PDF](https://arxiv.org/pdf/2507.08011)  

**Abstract**: We develop an energy management system (EMS) for artificial intelligence (AI) data centers with colocated renewable generation. Under a profit-maximizing framework, the EMS of renewable-colocated data center (RCDC) co-optimizes AI workload scheduling, on-site renewable utilization, and electricity market participation. Within both wholesale and retail market participation models, the economic benefit of the RCDC operation is maximized. Empirical evaluations using real-world traces of electricity prices, data center power consumption, and renewable generation demonstrate significant profit gains from renewable and AI data center colocations. 

**Abstract (ZH)**: 我们开发了一种针对共址可再生能源的 Artificial Intelligence 数据中心的能源管理系统 (EMS)。在利润最大化框架下，RCDC的EMS同时优化了AI工作负载调度、现场可再生能源利用以及电力市场参与。无论是在批发还是零售市场参与模型中，RCDC的操作经济收益都被最大化。使用实际的电力价格、数据中心电力消耗和可再生能源生成数据进行的实证评估表明，可再生能源与AI数据中心共址具有显著的经济效益。 

---
# Unraveling the Potential of Diffusion Models in Small Molecule Generation 

**Title (ZH)**: 探究扩散模型在小分子生成中的潜力 

**Authors**: Peining Zhang, Daniel Baker, Minghu Song, Jinbo Bi  

**Link**: [PDF](https://arxiv.org/pdf/2507.08005)  

**Abstract**: Generative AI presents chemists with novel ideas for drug design and facilitates the exploration of vast chemical spaces. Diffusion models (DMs), an emerging tool, have recently attracted great attention in drug R\&D. This paper comprehensively reviews the latest advancements and applications of DMs in molecular generation. It begins by introducing the theoretical principles of DMs. Subsequently, it categorizes various DM-based molecular generation methods according to their mathematical and chemical applications. The review further examines the performance of these models on benchmark datasets, with a particular focus on comparing the generation performance of existing 3D methods. Finally, it concludes by emphasizing current challenges and suggesting future research directions to fully exploit the potential of DMs in drug discovery. 

**Abstract (ZH)**: 生成式AI为化学家提供了新的药物设计思路，并促进了广泛化学空间的探索。扩散模型（DMs）作为一种新兴工具，近年来在药物研发（R&D）中引起了广泛关注。本文全面回顾了DMs在分子生成方面的最新进展和应用。首先介绍了DMs的理论原则。随后，根据其数学和化学应用对各种DM基分子生成方法进行了分类。回顾进一步分析了这些模型在基准数据集上的性能，特别关注了现有三维方法的生成性能比较。最后，本文强调了当前面临的挑战，并建议未来的研究方向，以充分挖掘DMs在药物发现中的潜力。 

---
# Human vs. LLM-Based Thematic Analysis for Digital Mental Health Research: Proof-of-Concept Comparative Study 

**Title (ZH)**: 人类分析师与基于LLM的主题分析在数字心理健康研究中的比较研究：概念验证研究 

**Authors**: Karisa Parkington, Bazen G. Teferra, Marianne Rouleau-Tang, Argyrios Perivolaris, Alice Rueda, Adam Dubrowski, Bill Kapralos, Reza Samavi, Andrew Greenshaw, Yanbo Zhang, Bo Cao, Yuqi Wu, Sirisha Rambhatla, Sridhar Krishnan, Venkat Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2507.08002)  

**Abstract**: Thematic analysis provides valuable insights into participants' experiences through coding and theme development, but its resource-intensive nature limits its use in large healthcare studies. Large language models (LLMs) can analyze text at scale and identify key content automatically, potentially addressing these challenges. However, their application in mental health interviews needs comparison with traditional human analysis. This study evaluates out-of-the-box and knowledge-base LLM-based thematic analysis against traditional methods using transcripts from a stress-reduction trial with healthcare workers. OpenAI's GPT-4o model was used along with the Role, Instructions, Steps, End-Goal, Narrowing (RISEN) prompt engineering framework and compared to human analysis in Dedoose. Each approach developed codes, noted saturation points, applied codes to excerpts for a subset of participants (n = 20), and synthesized data into themes. Outputs and performance metrics were compared directly. LLMs using the RISEN framework developed deductive parent codes similar to human codes, but humans excelled in inductive child code development and theme synthesis. Knowledge-based LLMs reached coding saturation with fewer transcripts (10-15) than the out-of-the-box model (15-20) and humans (90-99). The out-of-the-box LLM identified a comparable number of excerpts to human researchers, showing strong inter-rater reliability (K = 0.84), though the knowledge-based LLM produced fewer excerpts. Human excerpts were longer and involved multiple codes per excerpt, while LLMs typically applied one code. Overall, LLM-based thematic analysis proved more cost-effective but lacked the depth of human analysis. LLMs can transform qualitative analysis in mental healthcare and clinical research when combined with human oversight to balance participant perspectives and research resources. 

**Abstract (ZH)**: 大型语言模型在心理健康访谈中基于主题分析的应用：与传统方法的比较研究 

---
# Dual-Attention U-Net++ with Class-Specific Ensembles and Bayesian Hyperparameter Optimization for Precise Wound and Scale Marker Segmentation 

**Title (ZH)**: 双重注意U-Net++结合类特定集成和贝叶斯超参数优化的精确伤口和尺度标记分割 

**Authors**: Daniel Cieślak, Miriam Reca, Olena Onyshchenko, Jacek Rumiński  

**Link**: [PDF](https://arxiv.org/pdf/2507.05314)  

**Abstract**: Accurate segmentation of wounds and scale markers in clinical images remainsa significant challenge, crucial for effective wound management and automatedassessment. In this study, we propose a novel dual-attention U-Net++ archi-tecture, integrating channel-wise (SCSE) and spatial attention mechanisms toaddress severe class imbalance and variability in medical images this http URL, extensive benchmarking across diverse architectures and encoders via 5-fold cross-validation identified EfficientNet-B7 as the optimal encoder this http URL, we independently trained two class-specific models with tailoredpreprocessing, extensive data augmentation, and Bayesian hyperparameter tun-ing (WandB sweeps). The final model ensemble utilized Test Time Augmentationto further enhance prediction reliability. Our approach was evaluated on a bench-mark dataset from the NBC 2025 & PCBBE 2025 competition. Segmentationperformance was quantified using a weighted F1-score (75% wounds, 25% scalemarkers), calculated externally by competition organizers on undisclosed hard-ware. The proposed approach achieved an F1-score of 0.8640, underscoring itseffectiveness for complex medical segmentation tasks. 

**Abstract (ZH)**: 临床图像中伤口和量尺标记的准确分割仍然是一个重大挑战，对于有效的伤口管理和自动化评估至关重要。在本研究中，我们提出了一种新颖的双注意U-Net++架构，结合了通道级（SCSE）和空间注意机制以解决医学图像中的严重类别不平衡和变异性。我们的方法在NBC 2025 & PCBBE 2025竞赛的数据集上进行了评估，使用加权F1分数（75%伤口，25%量尺标记）进行了量化，该分数由竞赛组织者在外置未公开硬件上计算得出。所提出的方法实现了0.8640的F1分数，证明了其在复杂医学分割任务中的有效性。 

---
