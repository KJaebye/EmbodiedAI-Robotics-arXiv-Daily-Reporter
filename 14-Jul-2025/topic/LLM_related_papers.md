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
# From Language to Logic: A Bi-Level Framework for Structured Reasoning 

**Title (ZH)**: 从语言到逻辑：一种结构化推理的双层框架 

**Authors**: Keying Yang, Hao Wang, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08501)  

**Abstract**: Structured reasoning over natural language inputs remains a core challenge in artificial intelligence, as it requires bridging the gap between unstructured linguistic expressions and formal logical representations. In this paper, we propose a novel \textbf{bi-level framework} that maps language to logic through a two-stage process: high-level task abstraction and low-level logic generation. At the upper level, a large language model (LLM) parses natural language queries into intermediate structured representations specifying the problem type, objectives, decision variables, and symbolic constraints. At the lower level, the LLM uses these representations to generate symbolic workflows or executable reasoning programs for accurate and interpretable decision making. The framework supports modular reasoning, enforces explicit constraints, and generalizes across domains such as mathematical problem solving, question answering, and logical inference. We further optimize the framework with an end-to-end {bi-level} optimization approach that jointly refines both the high-level abstraction and low-level logic generation stages. Experiments on multiple realistic reasoning benchmarks demonstrate that our approach significantly outperforms existing baselines in accuracy, with accuracy gains reaching as high as 40\%. Moreover, the bi-level design enhances transparency and error traceability, offering a promising step toward trustworthy and systematic reasoning with LLMs. 

**Abstract (ZH)**: 基于自然语言输入的结构化推理仍是对人工智核心挑战，因为这要求在非结构化语义表达与正式逻辑表示之间建立桥梁。在本文中，我们提出了一种新颖的双层框架，通过两阶段过程将语言映射到逻辑：高层任务抽象和低层逻辑生成。在高层，一个大型语言模型（LLM）将自然语言查询解析为中间结构化表示，指定问题类型、目标、决策变量和符号约束。在低层，LLM 使用这些表示生成符号工作流或可执行推理程序，以实现准确和可解释的决策。该框架支持模块化推理、强制显式约束，并跨数学问题求解、问答和逻辑推理等领域进行泛化。进一步地，我们通过端到端的双层优化方法优化该框架，同时磨练高层抽象和低层逻辑生成阶段。在多个现实推理基准测试上的实验表明，我们的方法在准确性上显著优于现有基线，准确率提升高达40%。此外，双层设计增强了透明度和错误追踪能力，为可信和系统的LLM推理提供了一步前景。 

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
# Lumos-1: On Autoregressive Video Generation from a Unified Model Perspective 

**Title (ZH)**: Lumos-1：统一模型视角下的自回归视频生成 

**Authors**: Hangjie Yuan, Weihua Chen, Jun Cen, Hu Yu, Jingyun Liang, Shuning Chang, Zhihui Lin, Tao Feng, Pengwei Liu, Jiazheng Xing, Hao Luo, Jiasheng Tang, Fan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08801)  

**Abstract**: Autoregressive large language models (LLMs) have unified a vast range of language tasks, inspiring preliminary efforts in autoregressive video generation. Existing autoregressive video generators either diverge from standard LLM architectures, depend on bulky external text encoders, or incur prohibitive latency due to next-token decoding. In this paper, we introduce Lumos-1, an autoregressive video generator that retains the LLM architecture with minimal architectural modifications. To inject spatiotemporal correlations in LLMs, we identify the efficacy of incorporating 3D RoPE and diagnose its imbalanced frequency spectrum ranges. Therefore, we propose MM-RoPE, a RoPE scheme that preserves the original textual RoPE while providing comprehensive frequency spectra and scaled 3D positions for modeling multimodal spatiotemporal data. Moreover, Lumos-1 resorts to a token dependency strategy that obeys intra-frame bidirectionality and inter-frame temporal causality. Based on this dependency strategy, we identify the issue of frame-wise loss imbalance caused by spatial information redundancy and solve it by proposing Autoregressive Discrete Diffusion Forcing (AR-DF). AR-DF introduces temporal tube masking during training with a compatible inference-time masking policy to avoid quality degradation. By using memory-efficient training techniques, we pre-train Lumos-1 on only 48 GPUs, achieving performance comparable to EMU3 on GenEval, COSMOS-Video2World on VBench-I2V, and OpenSoraPlan on VBench-T2V. Code and models are available at this https URL. 

**Abstract (ZH)**: 基于自回归的大语言模型（LLMs）已统一了广泛的语言任务，并激发了初步的自回归视频生成努力。现有的自回归视频生成器要么偏离标准LLM架构，要么依赖庞大的外部文本编码器，要么因下一步解码导致显著的延迟。在本文中，我们引入了Lumos-1，这是一种保留LLM架构并进行了最小架构修改的自回归视频生成器。为了在LLMs中注入空时相关性，我们认识到3D RoPE的有效性，并诊断其不均衡的频谱范围。因此，我们提出了一种MM-RoPE方案，该方案保留了原始文本RoPE的同时，提供了全面的频谱范围和缩放的3D位置，以建模多模态空时数据。此外，Lumos-1采用了遵循帧内双向依赖性和帧间时序因果性的令牌依赖策略。基于此依赖策略，我们发现了由于空间信息冗余导致的帧间损失不平衡的问题，并提出了一种自回归离散扩散强迫（AR-DF）来解决这一问题。AR-DF在训练时引入了时空管状掩码，并配备了兼容的推理时掩码策略，以避免质量下降。通过使用高效的训练技术，我们仅在48个GPU上预训练了Lumos-1，实现了与EMU3在GenEval、COSMOS-Video2World在VBench-I2V以及OpenSoraPlan在VBench-T2V上的性能相当的表现。代码和模型可在以下链接获取。 

---
# KV Cache Steering for Inducing Reasoning in Small Language Models 

**Title (ZH)**: 针对诱导小语言模型进行推断的KV缓存定向方法 

**Authors**: Max Belitsky, Dawid J. Kopiczko, Michael Dorkenwald, M. Jehanzeb Mirza, Cees G. M. Snoek, Yuki M. Asano  

**Link**: [PDF](https://arxiv.org/pdf/2507.08799)  

**Abstract**: We propose cache steering, a lightweight method for implicit steering of language models via a one-shot intervention applied directly to the key-value cache. To validate its effectiveness, we apply cache steering to induce chain-of-thought reasoning in small language models. Our approach leverages GPT-4o-generated reasoning traces to construct steering vectors that shift model behavior toward more explicit, multi-step reasoning without fine-tuning or prompt modifications. Experimental evaluations on diverse reasoning benchmarks demonstrate that cache steering improves both the qualitative structure of model reasoning and quantitative task performance. Compared to prior activation steering techniques that require continuous interventions, our one-shot cache steering offers substantial advantages in terms of hyperparameter stability, inference-time efficiency, and ease of integration, making it a more robust and practical solution for controlled generation. 

**Abstract (ZH)**: 我们提出了一种轻量级的方法——缓存引导，该方法通过一次性的干预直接作用于键值缓存，从而实现语言模型的隐式引导。为了验证其有效性，我们将缓存引导应用于小语言模型，以诱导其进行链式思考推理。我们的方法利用GPT-4o生成的推理踪迹构建引导向量，从而在无需微调或修改提示的情况下，使模型行为更加符合显式、多步骤的推理。在多种推理基准上的实验评估表明，缓存引导可以提高模型推理的定性和定量表现。与需要连续干预的先前提取引导技术相比，我们的单次缓存引导在超参数稳定性、推理时间效率和集成简易性方面具有显著优势，使其成为更稳健且实用的受控生成解决方案。 

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
# KELPS: A Framework for Verified Multi-Language Autoformalization via Semantic-Syntactic Alignment 

**Title (ZH)**: KELPS：一种通过语义-语法对齐进行验证的多语言自动形式化框架 

**Authors**: Jiyao Zhang, Chengli Zhong, Hui Xu, Qige Li, Yi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.08665)  

**Abstract**: Modern large language models (LLMs) show promising progress in formalizing informal mathematics into machine-verifiable theorems. However, these methods still face bottlenecks due to the limited quantity and quality of multilingual parallel corpora. In this paper, we propose a novel neuro-symbolic framework KELPS (Knowledge-Equation based Logical Processing System) to address these problems. KELPS is an iterative framework for translating, synthesizing, and filtering informal data into multiple formal languages (Lean, Coq, and Isabelle). First, we translate natural language into Knowledge Equations (KEs), a novel language that we designed, theoretically grounded in assertional logic. Next, we convert them to target languages through rigorously defined rules that preserve both syntactic structure and semantic meaning. This process yielded a parallel corpus of over 60,000 problems. Our framework achieves 88.9% syntactic accuracy (pass@1) on MiniF2F, outperforming SOTA models such as Deepseek-V3 (81%) and Herald (81.3%) across multiple datasets. All datasets and codes are available in the supplementary materials. 

**Abstract (ZH)**: 现代大型语言模型在将非形式化数学转换为机器可验证定理方面展现了前景，但仍然受限于多语言平行语料库的数量和质量不足。本文提出了一种新的神经符号框架KELPS（基于知识-方程的逻辑处理系统）来解决这些问题。KELPS是一种迭代框架，用于将非形式化数据翻译、合成和过滤为多种正式语言（Lean、Coq和Isabelle）。首先，我们将自然语言转换为我们设计的一种新型语言知识方程（KEs），其理论基础是断言逻辑。然后，通过严格定义的规则将其转换为目标语言，同时保留其语义意义和语法结构。这一过程产生了超过60,000个问题的平行语料库。我们的框架在MiniF2F上的语法准确率（pass@1）达到了88.9%，在多个数据集上超过了包括Deepseek-V3（81%）和Herald（81.3%）在内的当前最佳模型。所有数据集和代码都包含在补充材料中。 

---
# A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1 

**Title (ZH)**: 基于LLM的论证分类综合研究：从LLAMA到GPT-4再到Deepseek-R1 

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski  

**Link**: [PDF](https://arxiv.org/pdf/2507.08621)  

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as this http URL and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings. 

**Abstract (ZH)**: Argument 矿工（AM）是跨逻辑、哲学、语言学、修辞学、法律、心理学和计算机科学的多学科研究领域，涉及自动识别和提取论据组件（如前提和主张）及其关系（如支持、反对或中立）。近年来，随着大型语言模型（LLMs）的发展，该领域取得了显著进展，与传统方法和其他深度学习模型相比，大大提高了论据语义分析的效率。虽然有众多基准测试 LLM 的质量和性能，但在公共可获得的论据分类数据库中，对这些模型的操作研究和结果仍然不足。本文研究了一种选择的 LLM，并使用如 this http URL 和 UKP 等多样化的数据集。测试的模型包括 GPT、Llama 和 DeepSeek 的版本，以及包含 Chain-of-Thoughts 算法的推理增强版本。结果显示，ChatGPT-4 在论据分类基准测试中表现最佳。对于具备推理能力的模型，Deepseek-R1 表现更优。然而，尽管表现出色，ChatGPT-4 和 Deepseek-R1 仍然会出错。针对所有模型，最常见错误进行了讨论。根据我们的知识，本研究是首次对所述数据集进行更全面的 LLM 和提示算法分析，展示了已知提示算法在论据分析中的某些局限性，指出了改进方向。工作的增益在于对可用的论据数据集进行了深入分析，并展示了它们的不足。 

---
# To Trade or Not to Trade: An Agentic Approach to Estimating Market Risk Improves Trading Decisions 

**Title (ZH)**: 要不要交易：一种代理方法来估计市场风险以改善交易决策 

**Authors**: Dimitrios Emmanoulopoulos, Ollie Olby, Justin Lyon, Namid R. Stillman  

**Link**: [PDF](https://arxiv.org/pdf/2507.08584)  

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic frameworks, in which prompts trigger complex tool-based analysis in pursuit of a goal. While these frameworks have shown promise across multiple domains including in finance, they typically lack a principled model-building step, relying instead on sentiment- or trend-based analysis. We address this gap by developing an agentic system that uses LLMs to iteratively discover stochastic differential equations for financial time series. These models generate risk metrics which inform daily trading decisions. We evaluate our system in both traditional backtests and using a market simulator, which introduces synthetic but causally plausible price paths and news events. We find that model-informed trading strategies outperform standard LLM-based agents, improving Sharpe ratios across multiple equities. Our results show that combining LLMs with agentic model discovery enhances market risk estimation and enables more profitable trading decisions. 

**Abstract (ZH)**: 大型语言模型在追求目标的代理框架中的迭代发现 stochastic 微分方程以生成金融时间序列的风险指标并指导每日交易决策：一种结合 LLMs 与代理模型发现的交易策略 

---
# FreeAudio: Training-Free Timing Planning for Controllable Long-Form Text-to-Audio Generation 

**Title (ZH)**: FreeAudio: 无需训练的定时规划以实现可控的长形式文本到语音生成 

**Authors**: Yuxuan Jiang, Zehua Chen, Zeqian Ju, Chang Li, Weibei Dou, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.08557)  

**Abstract**: Text-to-audio (T2A) generation has achieved promising results with the recent advances in generative models. However, because of the limited quality and quantity of temporally-aligned audio-text pairs, existing T2A methods struggle to handle the complex text prompts that contain precise timing control, e.g., "owl hooted at 2.4s-5.2s". Recent works have explored data augmentation techniques or introduced timing conditions as model inputs to enable timing-conditioned 10-second T2A generation, while their synthesis quality is still limited. In this work, we propose a novel training-free timing-controlled T2A framework, FreeAudio, making the first attempt to enable timing-controlled long-form T2A generation, e.g., "owl hooted at 2.4s-5.2s and crickets chirping at 0s-24s". Specifically, we first employ an LLM to plan non-overlapping time windows and recaption each with a refined natural language description, based on the input text and timing prompts. Then we introduce: 1) Decoupling and Aggregating Attention Control for precise timing control; 2) Contextual Latent Composition for local smoothness and Reference Guidance for global consistency. Extensive experiments show that: 1) FreeAudio achieves state-of-the-art timing-conditioned T2A synthesis quality among training-free methods and is comparable to leading training-based methods; 2) FreeAudio demonstrates comparable long-form generation quality with training-based Stable Audio and paves the way for timing-controlled long-form T2A synthesis. Demo samples are available at: this https URL 

**Abstract (ZH)**: 基于文本的无训练同步生成：FreeAudio框架 

---
# Pre-Training LLMs on a budget: A comparison of three optimizers 

**Title (ZH)**: 在预算范围内预训练大规模语言模型：三种优化器的比较 

**Authors**: Joel Schlotthauer, Christian Kroos, Chris Hinze, Viktor Hangya, Luzian Hahn, Fabian Küch  

**Link**: [PDF](https://arxiv.org/pdf/2507.08472)  

**Abstract**: Optimizers play a decisive role in reducing pre-training times for LLMs and achieving better-performing models. In this study, we compare three major variants: the de-facto standard AdamW, the simpler Lion, developed through an evolutionary search, and the second-order optimizer Sophia. For better generalization, we train with two different base architectures and use a single- and a multiple-epoch approach while keeping the number of tokens constant. Using the Maximal Update Parametrization and smaller proxy models, we tune relevant hyperparameters separately for each combination of base architecture and optimizer. We found that while the results from all three optimizers were in approximately the same range, Sophia exhibited the lowest training and validation loss, Lion was fastest in terms of training GPU hours but AdamW led to the best downstream evaluation results. 

**Abstract (ZH)**: 优化器在降低大语言模型预训练时间和提升模型性能方面起着决定性作用。本研究比较了三种主要变体：事实上的标准AdamW、通过进化搜索开发的简单Lion以及第二阶优化器Sophia。为了提高泛化能力，我们使用了两种不同的基架构进行训练，并采用单 epoch 和多 epoch 方法，同时保持token数量不变。利用Maximal Update Parametrization和较小的代理模型，我们分别对每种基架构和优化器组合进行相关超参数调整。研究发现，尽管所有三种优化器的结果大致在同一范围内，但Sophia在训练和验证损失方面最低，Lion在训练GPU小时内最快，而AdamW在下游评估结果方面表现最好。 

---
# CUE-RAG: Towards Accurate and Cost-Efficient Graph-Based RAG via Multi-Partite Graph and Query-Driven Iterative Retrieval 

**Title (ZH)**: CUE-RAG：通过多部图和查询驱动迭代检索实现准确高效的图基RAG 

**Authors**: Yaodong Su, Yixiang Fang, Yingli Zhou, Quanqing Xu, Chuanhui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08445)  

**Abstract**: Despite the remarkable progress of Large Language Models (LLMs), their performance in question answering (QA) remains limited by the lack of domain-specific and up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external information, often from graph-structured data. However, existing graph-based RAG methods suffer from poor graph quality due to incomplete extraction and insufficient utilization of query information during retrieval. To overcome these limitations, we propose CUE-RAG, a novel approach that introduces (1) a multi-partite graph index incorporates text Chunks, knowledge Units, and Entities to capture semantic content at multiple levels of granularity, (2) a hybrid extraction strategy that reduces LLM token usage while still producing accurate and disambiguated knowledge units, and (3) Q-Iter, a query-driven iterative retrieval strategy that enhances relevance through semantic search and constrained graph traversal. Experiments on three QA benchmarks show that CUE-RAG significantly outperforms state-of-the-art baselines, achieving up to 99.33% higher Accuracy and 113.51% higher F1 score while reducing indexing costs by 72.58%. Remarkably, CUE-RAG matches or outperforms baselines even without using an LLM for indexing. These results demonstrate the effectiveness and cost-efficiency of CUE-RAG in advancing graph-based RAG systems. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）取得了显著进展，但在问答（QA）任务上的表现仍受限于缺乏领域特定和时效性知识。检索增强生成（RAG）通过整合外部信息，通常是图结构化数据来解决这一问题。然而，现有的基于图的RAG方法由于检索过程中查询信息的提取不完整和利用不足，导致图质量较差。为了克服这些限制，我们提出了CUE-RAG这一新颖的方法，该方法引入了（1）一个多部图索引，结合文本片段、知识单元和实体以捕获多粒度级别的语义内容，（2）一种混合提取策略，减少LLM标记使用量同时仍能生成准确且去模糊的知识单元，以及（3）基于查询的迭代检索策略Q-Iter，通过语义搜索和受限图遍历提高相关性。在三个问答基准上的实验结果显示，CUE-RAG显著优于最先进的基线，准确率提高了99.33%，F1分数提高了113.51%，同时降低了72.58%的索引成本。尤为值得注意的是，即使不使用LLM进行索引，CUE-RAG也能匹配或超越基线。这些结果证明了CUE-RAG在推进基于图的RAG系统方面的有效性和成本效益。 

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
# A Practical Two-Stage Recipe for Mathematical LLMs: Maximizing Accuracy with SFT and Efficiency with Reinforcement Learning 

**Title (ZH)**: 一种实用的两阶段食谱：通过SFT最大化准确性并通过强化学习最大化效率的数学LLM 

**Authors**: Hiroshi Yoshihara, Taiki Yamaguchi, Yuichi Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2507.08267)  

**Abstract**: Enhancing the mathematical reasoning of Large Language Models (LLMs) is a pivotal challenge in advancing AI capabilities. While Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) are the dominant training paradigms, a systematic methodology for combining them to maximize both accuracy and efficiency remains largely unexplored. This paper introduces a practical and effective training recipe that strategically integrates extended SFT with RL from online inference (GRPO). We posit that these methods play complementary, not competing, roles: a prolonged SFT phase first pushes the model's accuracy to its limits, after which a GRPO phase dramatically improves token efficiency while preserving this peak performance. Our experiments reveal that extending SFT for as many as 10 epochs is crucial for performance breakthroughs, and that the primary role of GRPO in this framework is to optimize solution length. The efficacy of our recipe is rigorously validated through top-tier performance on challenging benchmarks, including a high rank among over 2,200 teams in the strictly leak-free AI Mathematical Olympiad (AIMO). This work provides the community with a battle-tested blueprint for developing state-of-the-art mathematical reasoners that are both exceptionally accurate and practically efficient. To ensure full reproducibility and empower future research, we will open-source our entire framework, including all code, model checkpoints, and training configurations at this https URL. 

**Abstract (ZH)**: 增强大型语言模型的数学推理能力是提升AI能力的关键挑战。尽管有监督微调(SFT)和强化学习(RL)是主要的训练范式，但如何系统地将二者结合起来以最大化准确性和效率仍待探索。本文介绍了一种实用且有效的训练方法，该方法战略性地将扩展的SFT与在线推断的RL (GRPO) 相结合。我们认为这些方法是互补而非竞争的：一个较长的SFT阶段首先将模型的准确性推至极限，随后的GRPO阶段则大幅提高 token 效率，同时保持这一峰值性能。我们的实验表明，将SFT扩展多达10个时期的性能突破至关重要，并且在此框架中，GRPO的主要作用是优化解决方案长度。通过在具有挑战性的基准测试中取得顶尖性能，包括在严格无泄漏的AI数学奥林匹克竞赛（AIMO）中名列前茅超过2200支队伍，我们严格验证了该方法的有效性。本工作为社区提供了实战验证的蓝图，用于开发既精确又高效的顶尖数学推理器。为了确保完全可再现并推动未来研究，我们将开源整个框架，包括所有代码、模型检查点和训练配置。 

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
# Compactor: Calibrated Query-Agnostic KV Cache Compression with Approximate Leverage Scores 

**Title (ZH)**: Compactor: 校准的免查询 KV 缓存压缩方法及其近似杠杆得分 

**Authors**: Vivek Chari, Benjamin Van Durme  

**Link**: [PDF](https://arxiv.org/pdf/2507.08143)  

**Abstract**: Modern Large Language Models (LLMs) are increasingly trained to support very large context windows. Unfortunately the ability to use long contexts in generation is complicated by the large memory requirement of the KV cache, which scales linearly with the context length. This memory footprint is often the dominant resource bottleneck in real-world deployments, limiting throughput and increasing serving cost. One way to address this is by compressing the KV cache, which can be done either with knowledge of the question being asked (query-aware) or without knowledge of the query (query-agnostic). We present Compactor, a parameter-free, query-agnostic KV compression strategy that uses approximate leverage scores to determine token importance. We show that Compactor can achieve the same performance as competing methods while retaining 1/2 the tokens in both synthetic and real-world context tasks, with minimal computational overhead. We further introduce a procedure for context-calibrated compression, which allows one to infer the maximum compression ratio a given context can support. Using context-calibrated compression, we show that Compactor achieves full KV performance on Longbench while reducing the KV memory burden by 63%, on average. To demonstrate the efficacy and generalizability of our approach, we apply Compactor to 27 synthetic and real-world tasks from RULER and Longbench, with models from both the Qwen 2.5 and Llama 3.1 families. 

**Abstract (ZH)**: 现代大规模语言模型（LLMs）越来越多地被训练以支持非常大的上下文窗口。不幸的是，在生成过程中使用长上下文受到了KV缓存的大量内存需求的复杂影响，这种内存需求与上下文长度成线性关系。这种内存足迹在现实世界部署中往往是主要的资源瓶颈，限制了吞吐量并增加了服务成本。一种解决办法是压缩KV缓存，这可以是有问题知识的（查询感知的）或没有问题知识的（查询无关的）。我们提出了Compactor，这是一种参数无关、查询无关的KV压缩策略，使用近似杠杆得分来确定token的重要性。我们展示了Compactor能够在合成和真实世界上下文任务中实现与竞争方法相同的效果，同时保留原有token数量的一半，并且几乎没有额外的计算开销。我们还引入了一种上下文校准压缩的程序，允许人们推断出给定上下文支持的最大压缩比。使用上下文校准压缩，我们展示了Compactor在Longbench上实现了完整的KV性能，同时平均将KV内存负担减少了63%。为了展示我们方法的有效性和通用性，我们在RULER和Longbench的27个合成和真实世界任务中应用了Compactor，使用了来自Qwen 2.5和Llama 3.1系列的模型。 

---
# Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models 

**Title (ZH)**: Audio Flamingo 3: 以完全开放的大规模音频语言模型促进音频智能 

**Authors**: Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, Bryan Catanzaro  

**Link**: [PDF](https://arxiv.org/pdf/2507.08128)  

**Abstract**: We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets. 

**Abstract (ZH)**: Audio Flamingo 3：一种面向语音、声效和音乐的全面开放的前沿大型音频语言模型 

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
# Towards Evaluating Robustness of Prompt Adherence in Text to Image Models 

**Title (ZH)**: 面向文本到图像模型提示遵从性robustness的评估研究 

**Authors**: Sujith Vemishetty, Advitiya Arora, Anupama Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2507.08039)  

**Abstract**: The advancements in the domain of LLMs in recent years have surprised many, showcasing their remarkable capabilities and diverse applications. Their potential applications in various real-world scenarios have led to significant research on their reliability and effectiveness. On the other hand, multimodal LLMs and Text-to-Image models have only recently gained prominence, especially when compared to text-only LLMs. Their reliability remains constrained due to insufficient research on assessing their performance and robustness. This paper aims to establish a comprehensive evaluation framework for Text-to-Image models, concentrating particularly on their adherence to prompts. We created a novel dataset that aimed to assess the robustness of these models in generating images that conform to the specified factors of variation in the input text prompts. Our evaluation studies present findings on three variants of Stable Diffusion models: Stable Diffusion 3 Medium, Stable Diffusion 3.5 Large, and Stable Diffusion 3.5 Large Turbo, and two variants of Janus models: Janus Pro 1B and Janus Pro 7B. We introduce a pipeline that leverages text descriptions generated by the gpt-4o model for our ground-truth images, which are then used to generate artificial images by passing these descriptions to the Text-to-Image models. We then pass these generated images again through gpt-4o using the same system prompt and compare the variation between the two descriptions. Our results reveal that these models struggle to create simple binary images with only two factors of variation: a simple geometric shape and its location. We also show, using pre-trained VAEs on our dataset, that they fail to generate images that follow our input dataset distribution. 

**Abstract (ZH)**: 近年来，Large Language Models (LLMs) 的发展令人惊讶，展示了其出色的能力和多种多样的应用。它们在各种现实场景中的潜在应用促使人们对其可靠性和有效性进行了大量研究。另一方面，多模态LLMs和Text-to-Image模型最近才受到重视，尤其是在与仅文本的LLMs相比时更为显著。由于对其性能和鲁棒性评估研究不足，这些模型的可靠性仍受到限制。本文旨在建立一个全面的Text-to-Image模型评估框架，特别关注它们对提示的遵守情况。我们创建了一个新的数据集，旨在评估这些模型在生成符合输入文本提示中指定变异因素的图像时的鲁棒性。我们的评估研究针对三种Stable Diffusion模型变体：Stable Diffusion 3 Medium、Stable Diffusion 3.5 Large 和 Stable Diffusion 3.5 Large Turbo，以及两种Janus模型变体：Janus Pro 1B 和 Janus Pro 7B。我们引入了一个模型管道，利用gpt-4o模型生成的文本描述作为我们的 ground-truth 图像，然后通过将这些描述传递给Text-to-Image模型生成人工图像。然后，我们再次将这些生成的图像通过gpt-4o系统提示处理，并比较两次描述之间的差异。我们的结果显示，这些模型难以生成仅有两个变异因素的简单二值图像：一个简单的几何形状及其位置。我们还使用在我们数据集上预训练的VAEs展示了它们无法生成遵循我们输入数据集分布的图像。 

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
# Human vs. LLM-Based Thematic Analysis for Digital Mental Health Research: Proof-of-Concept Comparative Study 

**Title (ZH)**: 人类分析师与基于LLM的主题分析在数字心理健康研究中的比较研究：概念验证研究 

**Authors**: Karisa Parkington, Bazen G. Teferra, Marianne Rouleau-Tang, Argyrios Perivolaris, Alice Rueda, Adam Dubrowski, Bill Kapralos, Reza Samavi, Andrew Greenshaw, Yanbo Zhang, Bo Cao, Yuqi Wu, Sirisha Rambhatla, Sridhar Krishnan, Venkat Bhat  

**Link**: [PDF](https://arxiv.org/pdf/2507.08002)  

**Abstract**: Thematic analysis provides valuable insights into participants' experiences through coding and theme development, but its resource-intensive nature limits its use in large healthcare studies. Large language models (LLMs) can analyze text at scale and identify key content automatically, potentially addressing these challenges. However, their application in mental health interviews needs comparison with traditional human analysis. This study evaluates out-of-the-box and knowledge-base LLM-based thematic analysis against traditional methods using transcripts from a stress-reduction trial with healthcare workers. OpenAI's GPT-4o model was used along with the Role, Instructions, Steps, End-Goal, Narrowing (RISEN) prompt engineering framework and compared to human analysis in Dedoose. Each approach developed codes, noted saturation points, applied codes to excerpts for a subset of participants (n = 20), and synthesized data into themes. Outputs and performance metrics were compared directly. LLMs using the RISEN framework developed deductive parent codes similar to human codes, but humans excelled in inductive child code development and theme synthesis. Knowledge-based LLMs reached coding saturation with fewer transcripts (10-15) than the out-of-the-box model (15-20) and humans (90-99). The out-of-the-box LLM identified a comparable number of excerpts to human researchers, showing strong inter-rater reliability (K = 0.84), though the knowledge-based LLM produced fewer excerpts. Human excerpts were longer and involved multiple codes per excerpt, while LLMs typically applied one code. Overall, LLM-based thematic analysis proved more cost-effective but lacked the depth of human analysis. LLMs can transform qualitative analysis in mental healthcare and clinical research when combined with human oversight to balance participant perspectives and research resources. 

**Abstract (ZH)**: 大型语言模型在心理健康访谈中基于主题分析的应用：与传统方法的比较研究 

---
