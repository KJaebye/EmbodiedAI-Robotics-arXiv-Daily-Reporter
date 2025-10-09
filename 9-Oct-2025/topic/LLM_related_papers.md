# SanDRA: Safe Large-Language-Model-Based Decision Making for Automated Vehicles Using Reachability Analysis 

**Title (ZH)**: SanDRA: 基于可达性分析的大型语言模型驱动的自动驾驶安全决策方法 

**Authors**: Yuanfei Lin, Sebastian Illing, Matthias Althoff  

**Link**: [PDF](https://arxiv.org/pdf/2510.06717)  

**Abstract**: Large language models have been widely applied to knowledge-driven decision-making for automated vehicles due to their strong generalization and reasoning capabilities. However, the safety of the resulting decisions cannot be ensured due to possible hallucinations and the lack of integrated vehicle dynamics. To address this issue, we propose SanDRA, the first safe large-language-model-based decision making framework for automated vehicles using reachability analysis. Our approach starts with a comprehensive description of the driving scenario to prompt large language models to generate and rank feasible driving actions. These actions are translated into temporal logic formulas that incorporate formalized traffic rules, and are subsequently integrated into reachability analysis to eliminate unsafe actions. We validate our approach in both open-loop and closed-loop driving environments using off-the-shelf and finetuned large language models, showing that it can provide provably safe and, where possible, legally compliant driving actions, even under high-density traffic conditions. To ensure transparency and facilitate future research, all code and experimental setups are publicly available at this http URL. 

**Abstract (ZH)**: 基于可达性分析的安全large语言模型驱动的自动驾驶决策框架SanDRA 

---
# NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents 

**Title (ZH)**: NewtonBench: 评估大规模语言模型代理通用科学定律发现能力的基准测试 

**Authors**: Tianshi Zheng, Kelvin Kiu-Wai Tam, Newt Hue-Nam K. Nguyen, Baixuan Xu, Zhaowei Wang, Jiayang Cheng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Tianqing Fang, Yangqiu Song, Ginny Y. Wong, Simon See  

**Link**: [PDF](https://arxiv.org/pdf/2510.07172)  

**Abstract**: Large language models are emerging as powerful tools for scientific law discovery, a foundational challenge in AI-driven science. However, existing benchmarks for this task suffer from a fundamental methodological trilemma, forcing a trade-off between scientific relevance, scalability, and resistance to memorization. Furthermore, they oversimplify discovery as static function fitting, failing to capture the authentic scientific process of uncovering embedded laws through the interactive exploration of complex model systems. To address these critical gaps, we introduce NewtonBench, a benchmark comprising 324 scientific law discovery tasks across 12 physics domains. Our design mitigates the evaluation trilemma by using metaphysical shifts - systematic alterations of canonical laws - to generate a vast suite of problems that are scalable, scientifically relevant, and memorization-resistant. Moreover, we elevate the evaluation from static function fitting to interactive model discovery, requiring agents to experimentally probe simulated complex systems to uncover hidden principles. Our extensive experiment reveals a clear but fragile capability for discovery in frontier LLMs: this ability degrades precipitously with increasing system complexity and exhibits extreme sensitivity to observational noise. Notably, we uncover a paradoxical effect of tool assistance: providing a code interpreter can hinder more capable models by inducing a premature shift from exploration to exploitation, causing them to satisfice on suboptimal solutions. These results demonstrate that robust, generalizable discovery in complex, interactive environments remains the core challenge. By providing a scalable, robust, and scientifically authentic testbed, NewtonBench offers a crucial tool for measuring true progress and guiding the development of next-generation AI agents capable of genuine scientific discovery. 

**Abstract (ZH)**: Large语言模型作为科学定律发现的强大工具正在 emergence，这是人工智能驱动科学的基础挑战。然而，现有任务基准在根本方法论三难中受到影响，被迫在科学相关性、可扩展性和抗记忆性之间做出权衡。此外，它们过度简化了发现过程，仅视为静态函数拟合，无法捕捉到通过探索复杂模型系统来揭示嵌入定律的真实科学过程。为了填补这些关键空白，我们引入了NewtonBench，这是一个涵盖12个物理学领域共324项科学定律发现任务的基准。我们的设计通过使用本体论转换——系统地改变经典定律——生成大量可扩展、科学相关且抗记忆性的问题。此外，我们将评估从静态函数拟合提升为交互式模型发现，要求代理对模拟的复杂系统进行实验性探索，以揭示隐藏的原则。我们的大量实验揭示了前沿LLM在发现方面的一种明确但易碎的能力：随着系统复杂性的增加，这种能力急剧下降，并对观测噪声表现出极端的敏感性。值得注意的是，我们发现了一个反常效应：提供代码解释器可能会妨碍更强大的模型，促使它们从探索转向利用，导致它们屈就于次优解决方案。这些结果表明，在复杂、交互式环境中实现稳健且通用的发现仍是最核心的挑战。通过提供一个可扩展、稳健且具有科学真实性的工作台，NewtonBench 提供了一个关键工具，用于衡量真实进展并引导下一代能够进行真正科学发现的AI代理的发展。 

---
# Integrating Domain Knowledge into Process Discovery Using Large Language Models 

**Title (ZH)**: 将领域知识集成到过程发现中使用大规模语言模型 

**Authors**: Ali Norouzifar, Humam Kourani, Marcus Dees, Wil van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2510.07161)  

**Abstract**: Process discovery aims to derive process models from event logs, providing insights into operational behavior and forming a foundation for conformance checking and process improvement. However, models derived solely from event data may not accurately reflect the real process, as event logs are often incomplete or affected by noise, and domain knowledge, an important complementary resource, is typically disregarded. As a result, the discovered models may lack reliability for downstream tasks. We propose an interactive framework that incorporates domain knowledge, expressed in natural language, into the process discovery pipeline using Large Language Models (LLMs). Our approach leverages LLMs to extract declarative rules from textual descriptions provided by domain experts. These rules are used to guide the IMr discovery algorithm, which recursively constructs process models by combining insights from both the event log and the extracted rules, helping to avoid problematic process structures that contradict domain knowledge. The framework coordinates interactions among the LLM, domain experts, and a set of backend services. We present a fully implemented tool that supports this workflow and conduct an extensive evaluation of multiple LLMs and prompt engineering strategies. Our empirical study includes a case study based on a real-life event log with the involvement of domain experts, who assessed the usability and effectiveness of the framework. 

**Abstract (ZH)**: 过程发现旨在从事件日志中推导出过程模型，提供对操作行为的洞察，并为合规检查和过程改进奠定基础。然而，仅从事件数据中推导出的模型可能无法准确反映实际过程，因为事件日志通常不完整且受到噪声影响，而重要的补充资源领域知识通常被忽视。因此，所发现的模型可能不适用于下游任务。我们提出了一种交互式框架，通过大型语言模型（LLMs）将自然语言表达的领域知识融入过程发现过程中。我们的方法利用LLMs从领域专家提供的文本描述中提取声明性规则，并利用这些规则引导IMr发现算法，该算法递归地通过结合事件日志和提取规则的洞见来构建过程模型，从而避免与领域知识矛盾的过程结构。该框架协调了LLMs、领域专家和一组后端服务之间的交互。我们提供了一个完全实现的工具来支持这一工作流程，并针对多种LLMs和提示工程策略进行了广泛评估。我们的实证研究包括一个基于实际事件日志的案例研究，领域专家参与评估了该框架的可用性和有效性。 

---
# The Cognitive Bandwidth Bottleneck: Shifting Long-Horizon Agent from Planning with Actions to Planning with Schemas 

**Title (ZH)**: 认知带宽瓶颈：从基于动作规划转向基于模式规划的长期 horizon 代理 

**Authors**: Baixuan Xu, Tianshi Zheng, Zhaowei Wang, Hong Ting Tsang, Weiqi Wang, Tianqing Fang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2510.07091)  

**Abstract**: Enabling LLMs to effectively operate long-horizon task which requires long-term planning and multiple interactions is essential for open-world autonomy. Conventional methods adopt planning with actions where a executable action list would be provided as reference. However, this action representation choice would be impractical when the environment action space is combinatorial exploded (e.g., open-ended real world). This naturally leads to a question: As environmental action space scales, what is the optimal action representation for long-horizon agents? In this paper, we systematically study the effectiveness of two different action representations. The first one is conventional planning with actions (PwA) which is predominantly adopted for its effectiveness on existing benchmarks. The other one is planning with schemas (PwS) which instantiate an action schema into action lists (e.g., "move [OBJ] to [OBJ]" -> "move apple to desk") to ensure concise action space and reliable scalability. This alternative is motivated by its alignment with human cognition and its compliance with environment-imposed action format restriction. We propose cognitive bandwidth perspective as a conceptual framework to qualitatively understand the differences between these two action representations and empirically observe a representation-choice inflection point between ALFWorld (~35 actions) and SciWorld (~500 actions), which serve as evidence of the need for scalable representations. We further conduct controlled experiments to study how the location of this inflection point interacts with different model capacities: stronger planning proficiency shifts the inflection rightward, whereas better schema instantiation shifts it leftward. Finally, noting the suboptimal performance of PwS agents, we provide an actionable guide for building more capable PwS agents for better scalable autonomy. 

**Abstract (ZH)**: 使大语言模型能够有效执行长期任务并进行长期规划和多次交互对于开放世界自主性至关重要。传统方法采用基于动作的规划方式，其中会提供可执行动作列表作为参考。然而，当环境动作空间出现组合爆炸（例如，开放的真实世界）时，这种动作表示形式的选择将变得不切实际。这自然引出了一个问题：随着环境动作空间的扩大，长期任务代理的最佳动作表示形式是什么？在本文中，我们系统地研究了两种不同动作表示形式的有效性。第一种是传统的基于动作的规划（PwA），因其在现有基准上的有效性而广泛采用。第二种是基于模式的规划（PwS），它将动作模式实例化为动作列表（例如，“将[OBJ]移动到[OBJ]” -> “将苹果移动到桌子”），以确保简洁的动作空间和可靠的可扩展性。这种替代方案的动力在于其与人类认知的契合以及对环境动作格式限制的合规性。我们提出认知带宽视角作为概念框架，以定性理解这两种动作表示形式之间的差异，并实证观察到认知带宽视角下的表示选择转折点，该转折点在ALFWorld（约35个动作）和SciWorld（约500个动作）之间，为需要可扩展表示形式的需求提供了证据。我们进一步进行了受控实验，研究这种转折点位置与不同模型能力的交互方式：更强的规划 proficiency使转折向右移动，而更好的模式实例化使转折向左移动。最后，注意到PwS代理的次优性能，我们提供了一条实用指南，以构建更强大且可扩展的PwS代理，从而实现更好的自主性。 

---
# VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems 

**Title (ZH)**: VRPAgent：基于LLM的车辆路线问题启发式操作发现 

**Authors**: André Hottung, Federico Berto, Chuanbo Hua, Nayeli Gast Zepeda, Daniel Wetzel, Michael Römer, Haoran Ye, Davide Zago, Michael Poli, Stefano Massaroli, Jinkyoo Park, Kevin Tierney  

**Link**: [PDF](https://arxiv.org/pdf/2510.07073)  

**Abstract**: Designing high-performing heuristics for vehicle routing problems (VRPs) is a complex task that requires both intuition and deep domain knowledge. Large language model (LLM)-based code generation has recently shown promise across many domains, but it still falls short of producing heuristics that rival those crafted by human experts. In this paper, we propose VRPAgent, a framework that integrates LLM-generated components into a metaheuristic and refines them through a novel genetic search. By using the LLM to generate problem-specific operators, embedded within a generic metaheuristic framework, VRPAgent keeps tasks manageable, guarantees correctness, and still enables the discovery of novel and powerful strategies. Across multiple problems, including the capacitated VRP, the VRP with time windows, and the prize-collecting VRP, our method discovers heuristic operators that outperform handcrafted methods and recent learning-based approaches while requiring only a single CPU core. To our knowledge, \VRPAgent is the first LLM-based paradigm to advance the state-of-the-art in VRPs, highlighting a promising future for automated heuristics discovery. 

**Abstract (ZH)**: 基于大型语言模型的VRP代理框架：整合生成组件并通过新颖的遗传搜索进行精化 

---
# Prompt Optimization Across Multiple Agents for Representing Diverse Human Populations 

**Title (ZH)**: 跨多个代理优化提示以表示多元人类群体 

**Authors**: Manh Hung Nguyen, Sebastian Tschiatschek, Adish Singla  

**Link**: [PDF](https://arxiv.org/pdf/2510.07064)  

**Abstract**: The difficulty and expense of obtaining large-scale human responses make Large Language Models (LLMs) an attractive alternative and a promising proxy for human behavior. However, prior work shows that LLMs often produce homogeneous outputs that fail to capture the rich diversity of human perspectives and behaviors. Thus, rather than trying to capture this diversity with a single LLM agent, we propose a novel framework to construct a set of agents that collectively capture the diversity of a given human population. Each agent is an LLM whose behavior is steered by conditioning on a small set of human demonstrations (task-response pairs) through in-context learning. The central challenge is therefore to select a representative set of LLM agents from the exponentially large space of possible agents. We tackle this selection problem from the lens of submodular optimization. In particular, we develop methods that offer different trade-offs regarding time complexity and performance guarantees. Extensive experiments in crowdsourcing and educational domains demonstrate that our approach constructs agents that more effectively represent human populations compared to baselines. Moreover, behavioral analyses on new tasks show that these agents reproduce the behavior patterns and perspectives of the students and annotators they are designed to represent. 

**Abstract (ZH)**: 大规模人类响应的获取难度和成本使得大型语言模型（LLMs）成为人类行为的有吸引力的替代方案和前景看好的代理。然而，先前的工作表明，LLMs经常产生同质化的输出，无法捕捉人类视角和行为的丰富多样性。因此，我们不试图通过单个LLM代理来捕捉这种多样性，而是提出了一种新的框架，构建一个集体制约来捕捉给定人类群体的多样性。每个代理是通过基于少量人类示范（任务-响应对）的上下文学习来引导其行为的LLM。因此，核心挑战是从可能的代理的指数空间中选择一个具有代表性的代理集。我们从亚模优化的角度来解决这个问题。特别是，我们开发了具有不同时间复杂性和性能保证的不同方法。在人群绘制和教育领域的广泛实验表明，我们方法构建的代理比基线方法更能有效地代表人类群体。此外，对新任务的行为分析表明，这些代理再现了设计它们来代表的学生和标注者的行为模式和视角。 

---
# Tool-Augmented Policy Optimization: Synergizing Reasoning and Adaptive Tool Use with Reinforcement Learning 

**Title (ZH)**: 工具增强的策略优化：强化学习中推理与自适应工具使用相结合 

**Authors**: Wenxun Wu, Yuanyang Li, Guhan Chen, Linyue Wang, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.07038)  

**Abstract**: Recent advances in large language models (LLMs) have popularized test-time scaling, where models generate additional reasoning tokens before producing final answers. These approaches have demonstrated significant performance improvements on benchmarks involving mathematical reasoning. However, language models relying solely on direct inference still struggle with tasks demanding up-to-date knowledge or computational tools such as calculators and code interpreters for complex arithmetic operations. To overcome these limitations, we propose Tool-Augmented Policy Optimization (TAPO), a novel reinforcement learning framework that systematically integrates multi-hop reasoning with adaptive tool-calling capabilities. Our approach employs a modified version of Dynamic Sampling Policy Optimization (DAPO), a recently developed RL paradigm, which we adapt specifically for tool invocation scenarios, enabling models to dynamically interleave complex reasoning with on-demand tool usage (including search APIs and Python interpreters).
To support this research, we introduce two new datasets: TAPO-easy-60K and TAPO-hard-18K, specifically designed to train and evaluate both fact-based reasoning and mathematical calculation capabilities. Our experiments on Qwen2.5-3B and Qwen2.5-7B models demonstrate the effectiveness of our approach, with both models achieving state-of-the-art performance on tasks requiring external knowledge and mathematical computation among methods with comparable parameters. Notably, TAPO achieves more efficient tool utilization than baseline methods while preventing excessive calls caused by reward hacking. These results highlight the significant potential of combining advanced reasoning with tool usage to enhance model performance in knowledge-intensive and computationally demanding tasks. 

**Abstract (ZH)**: 近期大型语言模型的发展推动了测试时扩展技术的普及，模型在生成最终答案之前会生成额外的推理令牌。尽管这些方法在涉及数学推理的基准测试中表现出显著的性能提升，但仅依赖直接推理的语言模型在处理需要最新知识或计算器、代码解释器等计算工具的任务时仍面临挑战，尤其是在复杂算术操作方面。为克服这些局限性，我们提出了工具增强策略优化（TAPO）框架，这是一种新颖的强化学习框架，系统地将多跳推理与适应性工具调用能力相结合。该方法采用了一种最近开发的RL范式——动态采样策略优化（DAPO）的修改版本，并针对工具调用场景进行了特定的适应，使模型能够在复杂推理与按需工具使用（包括搜索API和Python解释器）之间动态交织。

为了支持这项研究，我们引入了两个新的数据集：TAPO-easy-60K和TAPO-hard-18K，专门用于训练和评估基于事实的推理和数学计算能力。我们的实验表明，无论是Qwen2.5-3B还是Qwen2.5-7B模型，我们的方法都有效，同时在需要外部知识和数学计算的任务中达到了具有可比参数方法的最先进性能。值得注意的是，TAPO在工具利用效率上优于基线方法，同时防止了由于奖励黑客而导致的过度调用。这些结果突显了将高级推理与工具使用相结合以提高知识密集型和计算密集型任务模型性能的巨大潜力。 

---
# Revisiting the Uniform Information Density Hypothesis in LLM Reasoning Traces 

**Title (ZH)**: 重新审视大规模语言模型推理轨迹中的均匀信息密度假设 

**Authors**: Minju Gwak, Guijin Son, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.06953)  

**Abstract**: The Uniform Information Density (UID) hypothesis suggests that effective communication maintains a stable flow of information. In this work, we revisit this principle in the context of large language model (LLM) reasoning traces, asking whether step-level uniformity reflects reasoning quality. To this end, we propose an entropy-based stepwise information density metric and introduce two complementary measures of uniformity, local and global uniformity scores. Across the experiments on six different reasoning benchmarks, we find that step-level uniformity not only provides a strong theoretical lens but also yields practical performance benefits; for example, selecting reasoning traces with more uniform information density at the step-level improves accuracy by 10-32\% relative gains over baselines at AIME2025. Our analysis further reveals that correct reasoning traces tend to avoid sharp information density spikes, while incorrect traces exhibit irregular information bursts. These results demonstrate that UID-inspired information density measures outperform alternative internal signals as predictors of reasoning quality. Results highlight the uniformity of the information density as a robust diagnostic and selection criterion for building more reliable and accurate reasoning systems. 

**Abstract (ZH)**: 统一信息密度（UID）假设认为有效的通信保持信息流的稳定性。本文在大语言模型（LLM）推理追踪的背景下重新审视这一原则，询问步骤级的均匀性是否反映推理质量。为此，我们提出了一种基于熵的步骤级信息密度度量，并引入了局部和全局均匀性评分两种互补的一致性度量标准。通过对六个不同的推理基准的实验，我们发现步骤级的一致性不仅提供了强大的理论视角，还带来了实际性能优势；例如，选择步骤级信息密度更均匀的推理追踪可在AIME2025基准上实现10-32%的相对准确率提升。进一步的分析表明，正确推理追踪倾向于避免信息密度突跃，而不正确追踪则表现出不规则的信息突发。这些结果表明，基于UID的信息密度度量优于其他替代内部信号，作为推理质量的预测指标。结果强调信息密度的一致性是构建更可靠和准确推理系统的重要诊断和选择标准。 

---
# LLM-Assisted Modeling of Semantic Web-Enabled Multi-Agents Systems with AJAN 

**Title (ZH)**: 基于AJAN的语义网启用多智能体系统建模辅助方法 

**Authors**: Hacane Hechehouche, Andre Antakli, Matthias Klusch  

**Link**: [PDF](https://arxiv.org/pdf/2510.06911)  

**Abstract**: There are many established semantic Web standards for implementing multi-agent driven applications. The AJAN framework allows to engineer multi-agent systems based on these standards. In particular, agent knowledge is represented in RDF/RDFS and OWL, while agent behavior models are defined with Behavior Trees and SPARQL to access and manipulate this knowledge. However, the appropriate definition of RDF/RDFS and SPARQL-based agent behaviors still remains a major hurdle not only for agent modelers in practice. For example, dealing with URIs is very error-prone regarding typos and dealing with complex SPARQL queries in large-scale environments requires a high learning curve. In this paper, we present an integrated development environment to overcome such hurdles of modeling AJAN agents and at the same time to extend the user community for AJAN by the possibility to leverage Large Language Models for agent engineering. 

**Abstract (ZH)**: AJAN框架中基于RDF/RDFS和SPARQL的多智能体系统开发环境设计 

---
# TGPR: Tree-Guided Policy Refinement for Robust Self-Debugging of LLMs 

**Title (ZH)**: TGPR：基于树引导的策略细化方法以实现LLM的稳健自我调试 

**Authors**: Daria Ozerova, Ekaterina Trofimova  

**Link**: [PDF](https://arxiv.org/pdf/2510.06878)  

**Abstract**: Iterative refinement has been a promising paradigm to enable large language models (LLMs) to resolve difficult reasoning and problem-solving tasks. One of the key challenges, however, is how to effectively search through the enormous search space of possible refinements. Existing methods typically fall back on predefined heuristics, which are troubled by the exploration-exploitation dilemma and cannot adapt based on past refinement outcomes. We introduce Tree-Guided Policy Refinement (TGPR), a novel framework that combines GRPO with a Thompson-Sampling-based tree search. TGPR explores both failed and successful refinement paths actively, with denser training trajectories and more adaptive policies. On HumanEval, MBPP, and APPS benchmarks, our method achieves up to +4.2 percentage points absolute improvement in pass@1 (on MBPP) and up to +12.51 percentage points absolute improvement in pass@10 (on APPS) compared to a competitive GRPO baseline. Apart from debugging code, TGPR focuses on a principled approach to combining learned policies with structured search methods, offering a general framework for enhancing iterative refinement and stateful reasoning in LLMs. 

**Abstract (ZH)**: 基于树引导的策略精炼（Tree-Guided Policy Refinement）：一种结合GRPO和Thompson-Sampling树搜索的框架 

---
# Autoformalizer with Tool Feedback 

**Title (ZH)**: 自动形式化工具反馈系统 

**Authors**: Qi Guo, Jianing Wang, Jianfei Zhang, Deyang Kong, Xiangzhou Huang, Xiangyu Xi, Wei Wang, Jingang Wang, Xunliang Cai, Shikun Zhang, Wei Ye  

**Link**: [PDF](https://arxiv.org/pdf/2510.06857)  

**Abstract**: Autoformalization addresses the scarcity of data for Automated Theorem Proving (ATP) by translating mathematical problems from natural language into formal statements. Efforts in recent work shift from directly prompting large language models to training an end-to-end formalizer model from scratch, achieving remarkable advancements. However, existing formalizer still struggles to consistently generate valid statements that meet syntactic validity and semantic consistency. To address this issue, we propose the Autoformalizer with Tool Feedback (ATF), a novel approach that incorporates syntactic and consistency information as tools into the formalization process. By integrating Lean 4 compilers for syntax corrections and employing a multi-LLMs-as-judge approach for consistency validation, the model is able to adaptively refine generated statements according to the tool feedback, enhancing both syntactic validity and semantic consistency. The training of ATF involves a cold-start phase on synthetic tool-calling data, an expert iteration phase to improve formalization capabilities, and Direct Preference Optimization to alleviate ineffective revisions. Experimental results show that ATF markedly outperforms a range of baseline formalizer models, with its superior performance further validated by human evaluations. Subsequent analysis reveals that ATF demonstrates excellent inference scaling properties. Moreover, we open-source Numina-ATF, a dataset containing 750K synthetic formal statements to facilitate advancements in autoformalization and ATP research. 

**Abstract (ZH)**: 自动形式化通过将数学问题从自然语言翻译成形式化陈述来解决自动定理证明（ATP）中数据稀缺的问题。近年来的工作努力从直接提示大语言模型转变为从零开始训练端到端的形式化模型，取得了显著的进步。然而，现有的形式化模型仍然难以一贯生成符合语法有效性和语义一致性标准的有效陈述。为了解决这一问题，我们提出了一种新颖的方法——工具反馈自动形式化器（ATF），该方法将语法和一致性信息作为工具整合到形式化过程中。通过集成Lean 4编译器进行语法修正，并采用多LLM作为法官的方法进行一致性验证，模型可以根据工具反馈自适应地细化生成的陈述，从而增强语法有效性和语义一致性。ATF的训练包括一个基于合成工具调用数据的冷启动阶段、一个专家迭代阶段以提高形式化能力，以及直接偏好优化以缓解无效修订。实验结果表明，ATF显著优于多种基线形式化模型，其优越性进一步得到了人工评估的验证。后续分析显示，ATF表现出色的推理扩展性能。此外，我们开源了Numina-ATF数据集，包含750K个合成的形式化陈述，以促进自动形式化和自动定理证明研究。 

---
# Verifying Memoryless Sequential Decision-making of Large Language Models 

**Title (ZH)**: 验证大型语言模型的无记忆顺序决策 

**Authors**: Dennis Gross, Helge Spieker, Arnaud Gotlieb  

**Link**: [PDF](https://arxiv.org/pdf/2510.06756)  

**Abstract**: We introduce a tool for rigorous and automated verification of large language model (LLM)- based policies in memoryless sequential decision-making tasks. Given a Markov decision process (MDP) representing the sequential decision-making task, an LLM policy, and a safety requirement expressed as a PCTL formula, our approach incrementally constructs only the reachable portion of the MDP guided by the LLM's chosen actions. Each state is encoded as a natural language prompt, the LLM's response is parsed into an action, and reachable successor states by the policy are expanded. The resulting formal model is checked with Storm to determine whether the policy satisfies the specified safety property. In experiments on standard grid world benchmarks, we show that open source LLMs accessed via Ollama can be verified when deterministically seeded, but generally underperform deep reinforcement learning baselines. Our tool natively integrates with Ollama and supports PRISM-specified tasks, enabling continuous benchmarking in user-specified sequential decision-making tasks and laying a practical foundation for formally verifying increasingly capable LLMs. 

**Abstract (ZH)**: 我们介绍了一个工具，用于在无记忆顺序决策任务中对基于大型语言模型（LLM）的策略进行严格的自动化验证。给定一个表示顺序决策任务的马尔科夫决策过程（MDP）、一个LLM策略以及以PCTL公式表达的安全要求，我们的方法通过根据LLM选择的动作逐步构造MDP可达部分来验证策略。每个状态编码为自然语言提示，LLM的响应解析为动作，并展开策略可达的后继状态。最终形成的正式模型使用Storm进行检查，以确定策略是否满足指定的安全属性。在对标准网格世界基准的实验中，我们展示了通过Ollama访问的开源LLM在确定性初始化时可以进行验证，但总体上在深度强化学习基线方法下的表现较差。我们的工具原生支持与Ollama集成，并支持PRISM指定的任务，使得在用户指定的顺序决策任务中持续基准测试成为可能，并为正式验证日益强大的LLM奠定了实用基础。 

---
# Agent-in-the-Loop: A Data Flywheel for Continuous Improvement in LLM-based Customer Support 

**Title (ZH)**: Agent-in-the-Loop: 一种基于数据飞轮的LLM客户支持连续改进方法 

**Authors**: Zhao, Tiantian Zhang, Hanchen Su, Yufeng, Zhang, Shaowei Su, Mingzhi Xu, Wei Han, Jeremy Werner, Claire Na Cheng, Yashar Mehdad  

**Link**: [PDF](https://arxiv.org/pdf/2510.06674)  

**Abstract**: We introduce an Agent-in-the-Loop (AITL) framework that implements a continuous data flywheel for iteratively improving an LLM-based customer support system. Unlike standard offline approaches that rely on batch annotations, AITL integrates four key types of annotations directly into live customer operations: (1) pairwise response preferences, (2) agent adoption and rationales, (3) knowledge relevance checks, and (4) identification of missing knowledge. These feedback signals seamlessly feed back into models' updates, reducing retraining cycles from months to weeks. Our production pilot involving US-based customer support agents demonstrated significant improvements in retrieval accuracy (+11.7% recall@75, +14.8% precision@8), generation quality (+8.4% helpfulness) and agent adoption rates (+4.5%). These results underscore the effectiveness of embedding human feedback loops directly into operational workflows to continuously refine LLM-based customer support system. 

**Abstract (ZH)**: 基于代理的循环回路框架：连续数据飞轮及其在迭代改进基于LLM的客户服务系统中的应用 

---
# WebDART: Dynamic Decomposition and Re-planning for Complex Web Tasks 

**Title (ZH)**: WebDART：动态分解与重新规划用于复杂Web任务 

**Authors**: Jingbo Yang, Bairu Hou, Wei Wei, Shiyu Chang, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06587)  

**Abstract**: Large language model (LLM) agents are becoming competent at straightforward web tasks, such as opening an item page or submitting a form, but still struggle with objectives that require long horizon navigation, large scale information extraction, and reasoning under constraints. We present WebDART, a general framework that enables a single LLM to handle such complex chores. WebDART (i) dynamically decomposes each objective into three focused subtasks: navigation, information extraction, and execution, so the model concentrates on one skill at a time, and (ii) continuously replans the decomposition as new webpages are revealed, taking advantage of newly discovered filters or shortcuts and avoiding redundant exploration. Evaluated on WebChoreArena, WebDART lifts success rates by up to 13.7 percentage points over previous SOTA agents, while matching their performance on the easier WebArena suite and completing tasks with up to 14.7 fewer navigation steps. 

**Abstract (ZH)**: WebDART：一种通用框架，使大语言模型能够处理复杂的网络任务 

---
# Auto-Prompt Ensemble for LLM Judge 

**Title (ZH)**: 自动提示ensemblefor大规模语言模型法官 

**Authors**: Jiajie Li, Huayi Zhang, Peng Lin, Jinjun Xiong, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06538)  

**Abstract**: We present a novel framework that improves the reliability of LLM judges by selectively augmenting LLM with auxiliary evaluation dimensions. Existing LLM judges often miss crucial evaluation dimensions because they fail to recognize the implicit standards underlying human assessments. To address this challenge, we propose the Auto-Prompt Ensemble (APE), an adaptive framework that automatically learns evaluation dimensions from its failure cases. APE incorporates a confidence-based ensemble mechanism to decide when to adopt the judgments from additional evaluation dimensions through a novel confidence estimation approach called Collective Confidence. Extensive experiments demonstrate that APE improves the reliability of LLM Judge across diverse standard benchmarks. For instance, APE enhances GPT-4o agreement rate on Reward Bench from 87.2% to 90.5% in the zero-shot setting. Overall, APE provides a principled approach for LLM Judge to leverage test-time computation, and bridge the evaluation gap between human and LLM judges. 

**Abstract (ZH)**: 我们提出了一种新型框架，通过选择性地为LLM添加辅助评估维度来提高LLM评估者的可靠性。现有的LLM评估者常常忽视关键的评估维度，因为他们未能识别出人类评估背后的隐含标准。为了解决这一挑战，我们提出了一种自适应框架Auto-Prompt Ensemble (APE)，该框架能够从其失败案例中自动学习评估维度。APE结合了一种基于置信度的集成机制，通过一种新颖的集体置信估计方法来决定何时采用附加评估维度的判断。广泛的经验表明，APE在多种标准基准上提高了LLM评估者的可靠性。例如，在零样本设置下，APE将GPT-4o在Reward Bench上的一致性率从87.2%提升到90.5%。总体而言，APE为LLM评估者提供了一种原理性的方法，利用测试时计算，并弥合人类与LLM评估者之间的评价差距。 

---
# Beneficial Reasoning Behaviors in Agentic Search and Effective Post-training to Obtain Them 

**Title (ZH)**: 代理搜索中的有益推理行为及有效后训练以获得这些行为 

**Authors**: Jiahe Jin, Abhijay Paladugu, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2510.06534)  

**Abstract**: Agentic search leverages large language models (LLMs) to interpret complex user information needs and execute a multi-step process of planning, searching, and synthesizing information to provide answers. This paradigm introduces unique challenges for LLMs' reasoning and agentic capabilities when interacting with retrieval systems and the broader web. In this paper, we propose a reasoning-driven LLM-based pipeline to study effective reasoning behavior patterns in agentic search. Using this pipeline, we analyze successful agentic search trajectories and identify four beneficial reasoning behaviors: Information Verification, Authority Evaluation, Adaptive Search, and Error Recovery. Based on these findings, we propose a technique called Behavior Priming to train more effective agentic search models. It synthesizes agentic search trajectories that exhibit these four behaviors and integrates them into the agentic search model through supervised fine-tuning (SFT), followed by standard reinforcement learning (RL). Experiments on three benchmarks (GAIA, WebWalker, and HLE) demonstrate that behavior priming yields over 35% gains in Llama3.2-3B and Qwen3-1.7B compared to directly training agentic search models with RL. Crucially, we demonstrate that the desired reasoning behaviors in the SFT data, rather than the correctness of the final answer, is the critical factor for achieving strong final performance after RL: fine-tuning on trajectories with desirable reasoning behaviors but incorrect answers leads to better performance than fine-tuning on trajectories with correct answers. Our analysis further reveals the underlying mechanism: the introduced reasoning behaviors endow models with more effective exploration (higher pass@k and entropy) and test-time scaling (longer trajectories) capabilities, providing a strong foundation for RL. Our code will be released as open source. 

**Abstract (ZH)**: 基于推理的大语言模型驱动的代理搜索管道：研究有效的代理搜索推理行为模式及其训练技术 

---
# Off-Trajectory Reasoning: Can LLMs Collaborate on Reasoning Trajectory? 

**Title (ZH)**: 离轨推理：大规模语言模型能否协作进行推理轨迹？ 

**Authors**: Aochong Oliver Li, Tanya Goyal  

**Link**: [PDF](https://arxiv.org/pdf/2510.06410)  

**Abstract**: Reasoning LLMs are trained to verbalize their reasoning process, yielding strong gains on complex tasks. This transparency also opens a promising direction: multiple reasoners can directly collaborate on each other's thinking within a shared trajectory, yielding better inference efficiency and exploration. A key prerequisite, however, is the ability to assess the usefulness and build on another model's partial thinking -- we call this off-trajectory reasoning. Our paper investigates a critical question: can standard solo-reasoning training pipelines deliver desired off-trajectory behaviors? We propose twin tests that capture the two extremes of the off-trajectory spectrum, namely Recoverability, which tests whether LLMs can backtrack from "distractions" induced by misleading reasoning traces, and Guidability, which tests their ability to build upon correct reasoning from stronger collaborators. Our study evaluates 15 open-weight LLMs (1.5B-32B) and reveals a counterintuitive finding -- "stronger" LLMs on benchmarks are often more fragile under distraction. Moreover, all models tested fail to effectively leverage guiding steps from collaborators on problems beyond their inherent capabilities with solve rates remaining under 9.2%. Finally, we conduct control studies to isolate the effects of three factors in post-training on these behaviors: the choice of distillation teacher, the use of RL, and data selection strategy. Our results provide actionable insights for training natively strong reasoning collaborators; e.g., we find that suboptimal recoverability behaviors of teacher models are transferred to distilled students even if the distillation trajectories are correct. Taken together, this work lays the groundwork for evaluating multi-model collaborations in shared reasoning trajectories and highlights the limitations of off-the-shelf reasoning LLMs. 

**Abstract (ZH)**: 基于链路推理的透明性，LLMs被训练以 verbalize 其推理过程，从而在复杂任务上获得显著提升。这种透明性也为一个多推理器在共享轨迹上直接协作打开了新局面，从而提高推理效率和探索能力。然而，一个关键前提是对另一个模型部分推理的有效评估和利用能力——我们称之为离轨推理。本文探讨了一个关键问题：标准的单推理训练流程能否产生所需的离轨行为？我们提出了两种捕捉离轨推理光谱两端极端情况的测试方法：恢复性测试，检验LLMs能否从误导性推理痕迹引发的“偏离”中恢复；引导性测试，检验其利用更强合作伙伴正确推理构建的能力。我们的研究评估了15种开源权重LLM（1.5B-32B），揭示了一个反常识的发现——在基准测试中表现出优越性的LLM在面对误导性推理痕迹时往往更加脆弱。此外，所有测试的模型在外在能力范围之外的问题上未能有效利用合作者的引导步，解决率低于9.2%。最后，我们进行了控制实验，以分离这些行为在后续训练中受三个因素影响的效果：蒸馏教师的选择，使用RL，和数据选择策略。我们的研究成果为训练先天性强的推理合作者提供了可操作的见解；例如，我们发现，即使蒸馏轨迹是正确的，教师模型表现不佳的恢复行为也会传递给受蒸馏的学生模型。综上所述，这项工作为评估多模型在共享推理轨迹上的合作奠定了基础，并突显了现成推理LLM的局限性。 

---
# AlphaApollo: Orchestrating Foundation Models and Professional Tools into a Self-Evolving System for Deep Agentic Reasoning 

**Title (ZH)**: AlphaApollo: 将基础模型和专业工具 orchestrating 成一个自 evolution 系统以促进深度自主推理 

**Authors**: Zhanke Zhou, Chentao Cao, Xiao Feng, Xuan Li, Zongze Li, Xiangyu Lu, Jiangchao Yao, Weikai Huang, Linrui Xu, Tian Cheng, Guanyu Jiang, Yiming Zheng, Brando Miranda, Tongliang Liu, Sanmi Koyejo, Masashi Sugiyama, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2510.06261)  

**Abstract**: We present AlphaApollo, a self-evolving agentic reasoning system that aims to address two bottlenecks in foundation model (FM) reasoning-limited model-intrinsic capacity and unreliable test-time iteration. AlphaApollo orchestrates multiple models with professional tools to enable deliberate, verifiable reasoning. It couples (i) a computation tool (Python with numerical and symbolic libraries) and (ii) a retrieval tool (task-relevant external information) to execute exact calculations and ground decisions. The system further supports multi-round, multi-model solution evolution via a shared state map that records candidates, executable checks, and feedback for iterative refinement. In evaluations on AIME 2024/2025 across multiple models, AlphaApollo delivers consistent gains: +5.15% Average@32 and +23.34% Pass@32 for Qwen2.5-14B-Instruct, and +8.91% Average@32 with +26.67% Pass@32 for Llama-3.3-70B-Instruct. Tool-use analysis shows that more than 80% of tool calls are successfully executed, with consistent outperformance of non-tool baselines, thereby lifting the capability ceiling of FMs. More empirical results and implementation details will be updated at this https URL. 

**Abstract (ZH)**: AlphaApollo：一种自演进代理推理系统，旨在解决基础模型推理中的模型固有容量限制和测试时迭代不可靠性问题 

---
# Vibe Checker: Aligning Code Evaluation with Human Preference 

**Title (ZH)**: 代码评审器：使代码评估符合人类偏好 

**Authors**: Ming Zhong, Xiang Zhou, Ting-Yun Chang, Qingze Wang, Nan Xu, Xiance Si, Dan Garrette, Shyam Upadhyay, Jeremiah Liu, Jiawei Han, Benoit Schillings, Jiao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2510.07315)  

**Abstract**: Large Language Models (LLMs) have catalyzed vibe coding, where users leverage LLMs to generate and iteratively refine code through natural language interactions until it passes their vibe check. Vibe check is tied to real-world human preference and goes beyond functionality: the solution should feel right, read cleanly, preserve intent, and remain correct. However, current code evaluation remains anchored to pass@k and captures only functional correctness, overlooking the non-functional instructions that users routinely apply. In this paper, we hypothesize that instruction following is the missing piece underlying vibe check that represents human preference in coding besides functional correctness. To quantify models' code instruction following capabilities with measurable signals, we present VeriCode, a taxonomy of 30 verifiable code instructions together with corresponding deterministic verifiers. We use the taxonomy to augment established evaluation suites, resulting in Vibe Checker, a testbed to assess both code instruction following and functional correctness. Upon evaluating 31 leading LLMs, we show that even the strongest models struggle to comply with multiple instructions and exhibit clear functional regression. Most importantly, a composite score of functional correctness and instruction following correlates the best with human preference, with the latter emerging as the primary differentiator on real-world programming tasks. Our work identifies core factors of the vibe check, providing a concrete path for benchmarking and developing models that better align with user preferences in coding. 

**Abstract (ZH)**: 大型语言模型（LLMs）促进了情绪编码，用户通过自然语言互动利用LLMs生成并迭代 refining 代码，直至通过其情绪检查。情绪检查与现实世界的人类偏好相关，并超越了功能性：解决方案应感觉正确、读起来清爽、保持意图并保持正确。然而，当前的代码评估仍然局限于功能正确的通过率指标，忽略了用户常规应用的非功能性指令。在本文中，我们假设指令遵循是情绪检查背后的关键环节，情绪检查代表了编码中除功能性正确性之外的人类偏好。为量化模型的代码指令遵循能力并使用可量化的信号，我们提出了VeriCode，一种包含30条可验证代码指令及其相应确定性验证器的分类体系。我们使用分类体系扩展了现有的评估套件，从而得到Vibe Checker测试床，用于评估代码指令遵循和功能性正确性。通过对31个领先的大规模语言模型的评估，我们表明，即使是最强大的模型也难以遵守多个指令，并且表现出明显的功能性退化。最重要的是，功能性正确性和指令遵循的综合评分与人类偏好关联最强，后者成为了衡量现实世界编程任务的主要差异因子。我们的工作识别了情绪检查的核心因素，为基准测试和开发更好地符合用户编码偏好的模型提供了具体路径。 

---
# h1: Bootstrapping LLMs to Reason over Longer Horizons via Reinforcement Learning 

**Title (ZH)**: 通过强化学习]){
H1: 通过强化学习将LLMs扩展到更长的时间 horizon 上进行推理 

**Authors**: Sumeet Ramesh Motwani, Alesia Ivanova, Ziyang Cai, Philip Torr, Riashat Islam, Shital Shah, Christian Schroeder de Witt, Charles London  

**Link**: [PDF](https://arxiv.org/pdf/2510.07312)  

**Abstract**: Large language models excel at short-horizon reasoning tasks, but performance drops as reasoning horizon lengths increase. Existing approaches to combat this rely on inference-time scaffolding or costly step-level supervision, neither of which scales easily. In this work, we introduce a scalable method to bootstrap long-horizon reasoning capabilities using only existing, abundant short-horizon data. Our approach synthetically composes simple problems into complex, multi-step dependency chains of arbitrary length. We train models on this data using outcome-only rewards under a curriculum that automatically increases in complexity, allowing RL training to be scaled much further without saturating. Empirically, our method generalizes remarkably well: curriculum training on composed 6th-grade level math problems (GSM8K) boosts accuracy on longer, competition-level benchmarks (GSM-Symbolic, MATH-500, AIME) by up to 2.06x. Importantly, our long-horizon improvements are significantly higher than baselines even at high pass@k, showing that models can learn new reasoning paths under RL. Theoretically, we show that curriculum RL with outcome rewards achieves an exponential improvement in sample complexity over full-horizon training, providing training signal comparable to dense supervision. h1 therefore introduces an efficient path towards scaling RL for long-horizon problems using only existing data. 

**Abstract (ZH)**: 大规模语言模型在短视推理任务中表现出色，但随着推理视窗长度增加，性能下降。现有方法依赖于推理时的支架或昂贵的逐步骤监督，这两种方法都不容易扩展。在这种工作中，我们介绍了一种仅使用现有丰富短视数据来逐步增强长视窗推理能力的可扩展方法。我们的方法合成简单问题为任意长度的复杂多步依赖链。通过仅基于结果的奖励，在自适应增加复杂性的课程训练下训练模型，使得基于强化学习的训练能够大幅扩展而不会饱和。实验证明，我们的方法泛化效果出色：在合成的六年级水平数学问题（GSM8K）上进行课程训练，显著提高了更长、更高级基准（GSM-Symbolic、MATH-500、AIME）上的准确性最多2.06倍。重要的是，即使在高通过率下，我们的长视窗改进也显著高于基线，表明模型可以在强化学习下学习新的推理路径。理论上，我们证明了仅基于结果回报的课程训练在样本复杂性上实现了指数级改进，提供的训练信号与密集监督相当。因此，h1引入了一种仅使用现有数据扩展强化学习以应对长视窗问题的有效途径。 

---
# MLE-Smith: Scaling MLE Tasks with Automated Multi-Agent Pipeline 

**Title (ZH)**: MLE-Smith: 通过自动化多智能体管道扩展MLE任务 

**Authors**: Rushi Qiang, Yuchen Zhuang, Anikait Singh, Percy Liang, Chao Zhang, Sherry Yang, Bo Dai  

**Link**: [PDF](https://arxiv.org/pdf/2510.07307)  

**Abstract**: While Language Models (LMs) have made significant progress in automating machine learning engineering (MLE), the acquisition of high-quality MLE training data is significantly constrained. Current MLE benchmarks suffer from low scalability and limited applicability because they rely on static, manually curated tasks, demanding extensive time and manual effort to produce. We introduce MLE-Smith, a fully automated multi-agent pipeline, to transform raw datasets into competition-style MLE challenges through an efficient generate-verify-execute paradigm for scaling MLE tasks with verifiable quality, real-world usability, and rich diversity. The proposed multi-agent pipeline in MLE-Smith drives structured task design and standardized refactoring, coupled with a hybrid verification mechanism that enforces strict structural rules and high-level semantic soundness. It further validates empirical solvability and real-world fidelity through interactive execution. We apply MLE-Smith to 224 of real-world datasets and generate 606 tasks spanning multiple categories, objectives, and modalities, demonstrating that MLE-Smith can work effectively across a wide range of real-world datasets. Evaluation on the generated tasks shows that the performance of eight mainstream and cutting-edge LLMs on MLE-Smith tasks is strongly correlated with their performance on carefully human-designed tasks, highlighting the effectiveness of the MLE-Smith to scaling up MLE tasks, while maintaining task quality. 

**Abstract (ZH)**: 语言模型在自动化机器学习工程中的进展受限于高质量训练数据的获取。当前的机器学习工程基准因依赖静态的手动整理任务而缺乏可扩展性和适用性，需要大量时间和手动努力来生成。我们引入了MLE-Smith，这是一种完全自动化的多智能体流水线，通过高效生成-验证-执行范式将原始数据集转换为具有可验证质量、实际可用性和丰富多样性的竞赛风格的机器学习工程挑战。MLE-Smith中的提议多智能体流水线推动了结构化任务设计和标准化重构，并结合了一种混合验证机制，该机制强制执行严格的结构规则和高层次语义正确性。进一步通过交互执行验证其实证可解决性和现实世界的真实性。我们应用MLE-Smith处理了224个真实世界数据集并生成了涵盖多个类别、目标和模态的606个任务，证明了MLE-Smith可以在广泛的真实世界数据集上有效工作。对生成任务的评估表明，主流和前沿的语言模型在MLE-Smith任务上的性能与精心设计的人工任务上的性能之间高度相关，突显了MLE-Smith在扩展机器学习工程任务方面的有效性，同时保持任务质量。 

---
# AudioMarathon: A Comprehensive Benchmark for Long-Context Audio Understanding and Efficiency in Audio LLMs 

**Title (ZH)**: AudioMarathon: 长上下文音频理解与音频LLMs效率的综合基准 

**Authors**: Peize He, Zichen Wen, Yubo Wang, Yuxuan Wang, Xiaoqian Liu, Jiajie Huang, Zehui Lei, Zhuangcheng Gu, Xiangqi Jin, Jiabing Yang, Kai Li, Zhifei Liu, Weijia Li, Cunxiang Wang, Conghui He, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.07293)  

**Abstract**: Processing long-form audio is a major challenge for Large Audio Language models (LALMs). These models struggle with the quadratic cost of attention ($O(N^2)$) and with modeling long-range temporal dependencies. Existing audio benchmarks are built mostly from short clips and do not evaluate models in realistic long context settings. To address this gap, we introduce AudioMarathon, a benchmark designed to evaluate both understanding and inference efficiency on long-form audio. AudioMarathon provides a diverse set of tasks built upon three pillars: long-context audio inputs with durations ranging from 90.0 to 300.0 seconds, which correspond to encoded sequences of 2,250 to 7,500 audio tokens, respectively, full domain coverage across speech, sound, and music, and complex reasoning that requires multi-hop inference. We evaluate state-of-the-art LALMs and observe clear performance drops as audio length grows. We also study acceleration techniques and analyze the trade-offs of token pruning and KV cache eviction. The results show large gaps across current LALMs and highlight the need for better temporal reasoning and memory-efficient architectures. We believe AudioMarathon will drive the audio and multimodal research community to develop more advanced audio understanding models capable of solving complex audio tasks. 

**Abstract (ZH)**: 处理长格式音频是大型音频语言模型(LALMs)面临的重大挑战。AudioMarathon：一个用于评估长格式音频上的理解能力和推理效率的基准 

---
# Online Rubrics Elicitation from Pairwise Comparisons 

**Title (ZH)**: 基于成对比较的在线评分标准提取 

**Authors**: MohammadHossein Rezaei, Robert Vacareanu, Zihao Wang, Clinton Wang, Yunzhong He, Afra Feyza Akyürek  

**Link**: [PDF](https://arxiv.org/pdf/2510.07284)  

**Abstract**: Rubrics provide a flexible way to train LLMs on open-ended long-form answers where verifiable rewards are not applicable and human preferences provide coarse signals. Prior work shows that reinforcement learning with rubric-based rewards leads to consistent gains in LLM post-training. Most existing approaches rely on rubrics that remain static over the course of training. Such static rubrics, however, are vulnerable to reward-hacking type behaviors and fail to capture emergent desiderata that arise during training. We introduce Online Rubrics Elicitation (OnlineRubrics), a method that dynamically curates evaluation criteria in an online manner through pairwise comparisons of responses from current and reference policies. This online process enables continuous identification and mitigation of errors as training proceeds. Empirically, this approach yields consistent improvements of up to 8% over training exclusively with static rubrics across AlpacaEval, GPQA, ArenaHard as well as the validation sets of expert questions and rubrics. We qualitatively analyze the elicited criteria and identify prominent themes such as transparency, practicality, organization, and reasoning. 

**Abstract (ZH)**: 基于评分标准的动态 elicitation 方法在训练 LLMs 中的应用：连续改进和误差 mitigation 

---
# LeMAJ (Legal LLM-as-a-Judge): Bridging Legal Reasoning and LLM Evaluation 

**Title (ZH)**: LeMAJ (法律LLM法官): 融合法律推理与LLM评估 

**Authors**: Joseph Enguehard, Morgane Van Ermengem, Kate Atkinson, Sujeong Cha, Arijit Ghosh Chowdhury, Prashanth Kallur Ramaswamy, Jeremy Roghair, Hannah R Marlowe, Carina Suzana Negreanu, Kitty Boxall, Diana Mincu  

**Link**: [PDF](https://arxiv.org/pdf/2510.07243)  

**Abstract**: Evaluating large language model (LLM) outputs in the legal domain presents unique challenges due to the complex and nuanced nature of legal analysis. Current evaluation approaches either depend on reference data, which is costly to produce, or use standardized assessment methods, both of which have significant limitations for legal applications.
Although LLM-as-a-Judge has emerged as a promising evaluation technique, its reliability and effectiveness in legal contexts depend heavily on evaluation processes unique to the legal industry and how trustworthy the evaluation appears to the human legal expert. This is where existing evaluation methods currently fail and exhibit considerable variability.
This paper aims to close the gap: a) we break down lengthy responses into 'Legal Data Points' (LDPs), self-contained units of information, and introduce a novel, reference-free evaluation methodology that reflects how lawyers evaluate legal answers; b) we demonstrate that our method outperforms a variety of baselines on both our proprietary dataset and an open-source dataset (LegalBench); c) we show how our method correlates more closely with human expert evaluations and helps improve inter-annotator agreement; and finally d) we open source our Legal Data Points for a subset of LegalBench used in our experiments, allowing the research community to replicate our results and advance research in this vital area of LLM evaluation on legal question-answering. 

**Abstract (ZH)**: 评估法律领域的大型语言模型（LLM）输出面临着独特的挑战，因为法律分析具有复杂和微妙的性质。当前的评估方法要么依赖于成本高昂的参考数据，要么使用标准化评估方法，这两种方法在法律应用中都存在显著的局限性。

尽管已出现作为法官的大规模语言模型的评估技术，但其在法律情境下的可靠性和有效性很大程度上取决于独特的法律行业评估过程，以及这种评估对人类法律专家的信任程度。目前现有的评估方法在这方面存在缺陷，表现出显著的变异性。

本文旨在缩小这一差距：a) 我们将 lengthy responses 分解为“法律数据点”（LDPs），这是一种自包含的信息单元，并引入了一种新颖的、无参考的评估方法，反映了律师评估法律答案的方式；b) 我们证明了我们的方法在我们的专有数据集和开源数据集（LegalBench）上都优于多种基线方法；c) 我们展示了我们的方法与人类专家评估的相关性更高，并有助于提高标注者间的一致性；最后 d) 我们开源了用于实验的部分 LegalBench 法律数据点，使研究社区能够复制我们的结果并在这一关键领域的 LLM 法律问答评估研究中取得进展。 

---
# Benchmarking LLM Causal Reasoning with Scientifically Validated Relationships 

**Title (ZH)**: 基于科学验证的关系 Benchmarking 大型语言模型因果推理能力 

**Authors**: Donggyu Lee, Sungwon Park, Yerin Hwang, Hyunwoo Oh, Hyoshin Kim, Jungwon Kim, Meeyoung Cha, Sangyoon Park, Jihee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2510.07231)  

**Abstract**: Causal reasoning is fundamental for Large Language Models (LLMs) to understand genuine cause-and-effect relationships beyond pattern matching. Existing benchmarks suffer from critical limitations such as reliance on synthetic data and narrow domain coverage. We introduce a novel benchmark constructed from casually identified relationships extracted from top-tier economics and finance journals, drawing on rigorous methodologies including instrumental variables, difference-in-differences, and regression discontinuity designs. Our benchmark comprises 40,379 evaluation items covering five task types across domains such as health, environment, technology, law, and culture. Experimental results on eight state-of-the-art LLMs reveal substantial limitations, with the best model achieving only 57.6\% accuracy. Moreover, model scale does not consistently translate to superior performance, and even advanced reasoning models struggle with fundamental causal relationship identification. These findings underscore a critical gap between current LLM capabilities and demands of reliable causal reasoning in high-stakes applications. 

**Abstract (ZH)**: 因果推理是大型语言模型理解真正因果关系的基础，而不仅仅是模式匹配。现有基准存在关键局限性，如依赖合成数据和狭窄的领域覆盖率。我们提出了一个基于从顶级经济学和金融期刊中识别出的因果关系构建的新基准，采用了严格的因果方法，包括工具变量法、差分差异法和断点回归设计。该基准包含40,379个评估项目，覆盖健康、环境、技术、法律和文化等五个任务类型。在八种最先进的语言模型上的实验结果揭示了显著的局限性，最佳模型的准确率仅为57.6%。此外，模型规模并不一致地转化为更好的性能，即使是先进的推理模型也难以识别基本的因果关系。这些发现突显了当前大型语言模型能力和高标准的可靠因果推理之间的关键差距。 

---
# Where to Begin: Efficient Pretraining via Subnetwork Selection and Distillation 

**Title (ZH)**: 从何开始：通过子网络选择和知识蒸馏实现高效的预训练 

**Authors**: Arjun Krishnakumar, Rhea Sanjay Sukthanker, Hannan Javed Mahadik, Gabriela Kadlecová, Vladyslav Moroshan, Timur Carstensen, Frank Hutter, Aaron Klein  

**Link**: [PDF](https://arxiv.org/pdf/2510.07227)  

**Abstract**: Small Language models (SLMs) offer an efficient and accessible alternative to Large Language Models (LLMs), delivering strong performance while using far fewer resources. We introduce a simple and effective framework for pretraining SLMs that brings together three complementary ideas. First, we identify structurally sparse sub-network initializations that consistently outperform randomly initialized models of similar size under the same compute budget. Second, we use evolutionary search to automatically discover high-quality sub-network initializations, providing better starting points for pretraining. Third, we apply knowledge distillation from larger teacher models to speed up training and improve generalization. Together, these components make SLM pretraining substantially more efficient: our best model, discovered using evolutionary search and initialized with LLM weights, matches the validation perplexity of a comparable Pythia SLM while requiring 9.2x fewer pretraining tokens. We release all code and models at this https URL, offering a practical and reproducible path toward cost-efficient small language model development at scale. 

**Abstract (ZH)**: 小语言模型（SLMs）提供了一种在资源有限的情况下与大型语言模型（LLMs）竞争的有效替代方案，它们能在使用更少资源的同时取得强大的性能。我们提出了一个简单而有效的框架，用于预训练SLMs，该框架结合了三种互补的想法。首先，我们识别出结构上稀疏的子网络初始化方法，在相同的计算预算下，这些方法比随机初始化的模型表现更好。其次，我们使用进化搜索自动发现高质量的子网络初始化，为预训练提供更好的起点。第三，我们应用来自较大教师模型的知识蒸馏来加速训练并提高泛化能力。这些组件共同使得SLM预训练显著更加高效：我们使用进化搜索发现的最佳模型，在初始化时使用LLM权重，所要求的预训练标记数量仅为同类Pythia SLM的9.2倍，但在验证困惑度上表现相当。我们已在该链接发布了全部代码和模型，为大规模低成本小语言模型开发提供了可实践和可复制的路径。 

---
# Language Lives in Sparse Dimensions: Toward Interpretable and Efficient Multilingual Control for Large Language Models 

**Title (ZH)**: 语言存在于稀疏维度中： toward 可解释且高效的多语言控制大语言模型 

**Authors**: Chengzhi Zhong, Fei Cheng, Qianying Liu, Yugo Murawaki, Chenhui Chu, Sadao Kurohashi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07213)  

**Abstract**: Large language models exhibit strong multilingual capabilities despite limited exposure to non-English data. Prior studies show that English-centric large language models map multilingual content into English-aligned representations at intermediate layers and then project them back into target-language token spaces in the final layer. From this observation, we hypothesize that this cross-lingual transition is governed by a small and sparse set of dimensions, which occur at consistent indices across the intermediate to final layers. Building on this insight, we introduce a simple, training-free method to identify and manipulate these dimensions, requiring only as few as 50 sentences of either parallel or monolingual data. Experiments on a multilingual generation control task reveal the interpretability of these dimensions, demonstrating that the interventions in these dimensions can switch the output language while preserving semantic content, and that it surpasses the performance of prior neuron-based approaches at a substantially lower cost. 

**Abstract (ZH)**: 大型语言模型尽管具有有限的非英语数据暴露，但仍展现出强大的多语言能力。以往研究显示，以英语为中心的大规模语言模型将在中间层将多语言内容映射到英文对齐的表示，然后在最终层将其投影回目标语言的标记空间。基于这一观察，我们假设这种跨语言过渡是由一组小型且稀疏的维度控制的，这些维度在中间层到最终层中的一致索引位置出现。基于这一见解，我们提出了一种简单的、无需训练的方法来识别并操控这些维度，仅需少量（少至50句）平行或单一语言的数据。在多语言生成控制任务上的实验揭示了这些维度的可解释性，表明在这些维度上的干预可以切换输出语言同时保留语义内容，并且与先前的神经元基于的方法相比，在显著更低的成本下实现了更好的性能。 

---
# Comparing human and language models sentence processing difficulties on complex structures 

**Title (ZH)**: 比较人类与语言模型在处理复杂结构句子时的困难 

**Authors**: Samuel Joseph Amouyal, Aya Meltzer-Asscher, Jonathan Berant  

**Link**: [PDF](https://arxiv.org/pdf/2510.07141)  

**Abstract**: Large language models (LLMs) that fluently converse with humans are a reality - but do LLMs experience human-like processing difficulties? We systematically compare human and LLM sentence comprehension across seven challenging linguistic structures. We collect sentence comprehension data from humans and five families of state-of-the-art LLMs, varying in size and training procedure in a unified experimental framework. Our results show LLMs overall struggle on the target structures, but especially on garden path (GP) sentences. Indeed, while the strongest models achieve near perfect accuracy on non-GP structures (93.7% for GPT-5), they struggle on GP structures (46.8% for GPT-5). Additionally, when ranking structures based on average performance, rank correlation between humans and models increases with parameter count. For each target structure, we also collect data for their matched baseline without the difficult structure. Comparing performance on the target vs. baseline sentences, the performance gap observed in humans holds for LLMs, with two exceptions: for models that are too weak performance is uniformly low across both sentence types, and for models that are too strong the performance is uniformly high. Together, these reveal convergence and divergence in human and LLM sentence comprehension, offering new insights into the similarity of humans and LLMs. 

**Abstract (ZH)**: 大型语言模型在流畅地与人类交流方面已成为现实——但大型语言模型是否体验到类似人类的处理困难？我们系统地在七个具有挑战性的语言结构上比较了人类和大型语言模型的句子理解能力。我们在统一的实验框架下收集了人类和五大家族的最先进的大型语言模型的句子理解数据，这些模型在规模和训练过程上各不相同。结果显示，整体而言，大型语言模型在目标结构上表现挣扎，尤其是在“误判句”（GP句）上。事实上，虽然最强的模型在非GP句上几乎达到完美准确率（GPT-5达到93.7%），但在GP句上则表现挣扎（GPT-5仅为46.8%）。此外，根据平均表现对结构进行排序时，人类和模型之间的排名相关性随参数量增加而增强。在为每个目标结构收集对应的不含困难结构的基线数据后，将目标句子的表现与基线句子的表现进行比较，人类表现出的性能差距同样适用于大型语言模型，有两项例外：对于过弱的模型，两种句型的表现均低；对于过强的模型，两种句型的表现均高。这些结果揭示了人类和大型语言模型在句子理解上的趋同与分歧，为人类与大型语言模型的相似性提供了新的见解。 

---
# Opt-ICL at LeWiDi-2025: Maximizing In-Context Signal from Rater Examples via Meta-Learning 

**Title (ZH)**: Opt-ICL 在 LeWiDi-2025：通过元学习最大化标注示例的上下文信号 

**Authors**: Taylor Sorensen, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2510.07105)  

**Abstract**: Many natural language processing (NLP) tasks involve subjectivity, ambiguity, or legitimate disagreement between annotators. In this paper, we outline our system for modeling human variation. Our system leverages language models' (LLMs) in-context learning abilities, along with a two-step meta-learning training procedure for 1) post-training on many datasets requiring in-context learning and 2) specializing the model via in-context meta-learning to the particular data distribution of interest. We also evaluate the performance of our system submission to the Learning With Disagreements (LeWiDi) competition, where it was the overall winner on both tasks. Additionally, we perform an ablation study to measure the importance of each system component. We find that including rater examples in-context is crucial for our system's performance, dataset-specific fine-tuning is helpful on the larger datasets, post-training on other in-context datasets is helpful on one of the competition datasets, and that performance improves with model scale. 

**Abstract (ZH)**: 许多自然语言处理（NLP）任务涉及主观性、歧义或注释者之间的合理分歧。本文概述了我们的人类变异性建模系统。该系统利用了语言模型（LLMs）的上下文学习能力，并采用两步元学习训练程序，首先是经过多个需要上下文学习的數據集训练，其次是通过上下文元学习使模型专门化以适应特定的数据分布。另外，我们还评估了该系统在Learning With Disagreements（LeWiDi）竞赛中的表现，其中在两个任务上均获得了总体冠军。我们还进行了消融研究以衡量每个系统组件的重要性。研究发现，将评价者示例纳入上下文对于系统性能至关重要，在较大数据集上进行数据集特定微调是有帮助的，在竞赛数据集中有一个数据集的后训练有助于提高性能，随着模型规模的增加，性能会有所改善。 

---
# LuxInstruct: A Cross-Lingual Instruction Tuning Dataset For Luxembourgish 

**Title (ZH)**: LuxInstruct: 一种面向卢森堡语的跨语言指令调优数据集 

**Authors**: Fred Philippy, Laura Bernardy, Siwen Guo, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2510.07074)  

**Abstract**: Instruction tuning has become a key technique for enhancing the performance of large language models, enabling them to better follow human prompts. However, low-resource languages such as Luxembourgish face severe limitations due to the lack of high-quality instruction datasets. Traditional reliance on machine translation often introduces semantic misalignment and cultural inaccuracies. In this work, we address these challenges by creating a cross-lingual instruction tuning dataset for Luxembourgish, without resorting to machine-generated translations into it. Instead, by leveraging aligned data from English, French, and German, we build a high-quality dataset that preserves linguistic and cultural nuances. We provide evidence that cross-lingual instruction tuning not only improves representational alignment across languages but also the model's generative capabilities in Luxembourgish. This highlights how cross-lingual data curation can avoid the common pitfalls of machine-translated data and directly benefit low-resource language development. 

**Abstract (ZH)**: 跨语言指令调优：为卢森堡语创建高质量数据集并提高其生成能力 

---
# Search-R3: Unifying Reasoning and Embedding Generation in Large Language Models 

**Title (ZH)**: Search-R3: 在大型语言模型中统一推理和嵌入生成 

**Authors**: Yuntao Gui, James Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07048)  

**Abstract**: Despite their remarkable natural language understanding capabilities, Large Language Models (LLMs) have been underutilized for retrieval tasks. We present Search-R3, a novel framework that addresses this limitation by adapting LLMs to generate search embeddings as a direct output of their reasoning process. Our approach exploits LLMs' chain-of-thought capabilities, allowing them to produce more effective embeddings by reasoning step-by-step through complex semantic analyses. We implement this through three complementary mechanisms. (1) a supervised learning stage enables the model's ability to produce quality embeddings, (2) a reinforcement learning (RL) methodology that optimizes embedding generation alongside reasoning, and (3) a specialized RL environment that efficiently handles evolving embedding representations without requiring complete corpus re-encoding at each training iteration. Our extensive evaluations on diverse benchmarks demonstrate that Search-R3 significantly outperforms prior methods by unifying the reasoning and embedding generation processes. This integrated post-training approach represents a substantial advancement in handling complex knowledge-intensive tasks that require both sophisticated reasoning and effective information retrieval. Project page: this https URL 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具备非凡的自然语言理解能力，但它们在检索任务中的利用程度较低。我们提出了一种名为Search-R3的新型框架，通过将LLMs改编为直接生成搜索嵌入，从而弥补这一不足。我们的方法利用了LLMs的逐步推理能力，使其能够通过复杂的语义分析进行逐步推理，从而生成更有效的嵌入。我们通过三种互补机制实现这一目标。（1）监督学习阶段使模型能够生成高质量的嵌入；（2）结合推理优化嵌入生成的强化学习（RL）方法；（3）专门的RL环境，能够在每次训练迭代时高效处理不断变化的嵌入表示，而无需每次都重新编码整个语料库。我们在多种基准上的广泛评估表明，Search-R3显著优于先前的方法，通过将推理和嵌入生成过程统一起来。这种集成的后训练方法代表了处理需要复杂推理和有效信息检索的密集型知识任务的重大进展。项目页面：这个 https URL。 

---
# Mining the Mind: What 100M Beliefs Reveal About Frontier LLM Knowledge 

**Title (ZH)**: 探索心灵：1亿条信念揭示的前沿LLM知识 

**Authors**: Shrestha Ghosh, Luca Giordano, Yujia Hu, Tuan-Phong Nguyen, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.07024)  

**Abstract**: LLMs are remarkable artifacts that have revolutionized a range of NLP and AI tasks. A significant contributor is their factual knowledge, which, to date, remains poorly understood, and is usually analyzed from biased samples. In this paper, we take a deep tour into the factual knowledge (or beliefs) of a frontier LLM, based on GPTKB v1.5 (Hu et al., 2025a), a recursively elicited set of 100 million beliefs of one of the strongest currently available frontier LLMs, GPT-4.1. We find that the models' factual knowledge differs quite significantly from established knowledge bases, and that its accuracy is significantly lower than indicated by previous benchmarks. We also find that inconsistency, ambiguity and hallucinations are major issues, shedding light on future research opportunities concerning factual LLM knowledge. 

**Abstract (ZH)**: 大规模语言模型是革命性的成果，已重塑一系列NLP和AI任务。其事实知识是重要贡献者，至今仍 poorly understood，通常是从有偏样本中进行分析。本文基于GPTKB v1.5（Hu et al., 2025a），对前沿LLM的fact知识进行了深入探讨，GPTKB v1.5 是一个递归获取的包含1亿条信念的集合，来自于当前最强的前沿LLM之一GPT-4。我们发现，模型的事实知识与已建立的知识库存在显著差异，其准确性显著低于以往基准测试所显示的。我们还发现不一致、模糊性和幻觉是主要问题，这为未来关于事实LLM知识的研究提供了新的契机。 

---
# Native Hybrid Attention for Efficient Sequence Modeling 

**Title (ZH)**: 原生混合注意力机制高效序列建模 

**Authors**: Jusen Du, Jiaxi Hu, Tao Zhang, Weigao Sun, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.07019)  

**Abstract**: Transformers excel at sequence modeling but face quadratic complexity, while linear attention offers improved efficiency but often compromises recall accuracy over long contexts. In this work, we introduce Native Hybrid Attention (NHA), a novel hybrid architecture of linear and full attention that integrates both intra \& inter-layer hybridization into a unified layer design. NHA maintains long-term context in key-value slots updated by a linear RNN, and augments them with short-term tokens from a sliding window. A single \texttt{softmax attention} operation is then applied over all keys and values, enabling per-token and per-head context-dependent weighting without requiring additional fusion parameters. The inter-layer behavior is controlled through a single hyperparameter, the sliding window size, which allows smooth adjustment between purely linear and full attention while keeping all layers structurally uniform. Experimental results show that NHA surpasses Transformers and other hybrid baselines on recall-intensive and commonsense reasoning tasks. Furthermore, pretrained LLMs can be structurally hybridized with NHA, achieving competitive accuracy while delivering significant efficiency gains. Code is available at this https URL. 

**Abstract (ZH)**: Native Hybrid Attention: A Unified Hybrid Architecture of Linear and Full Attention 

---
# Pragyaan: Designing and Curating High-Quality Cultural Post-Training Datasets for Indian Languages 

**Title (ZH)**: Pragyaan: 设计和策展高质量的印度语言文化后训练数据集 

**Authors**: Neel Prabhanjan Rachamalla, Aravind Konakalla, Gautam Rajeev, Ashish Kulkarni, Chandra Khatri, Shubham Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.07000)  

**Abstract**: The effectiveness of Large Language Models (LLMs) depends heavily on the availability of high-quality post-training data, particularly instruction-tuning and preference-based examples. Existing open-source datasets, however, often lack multilingual coverage, cultural grounding, and suffer from task diversity gaps that are especially pronounced for Indian languages. We introduce a human-in-the-loop pipeline that combines translations with synthetic expansion to produce reliable and diverse Indic post-training data. Using this pipeline, we curate two datasets: Pragyaan-IT (22.5K) and Pragyaan-Align (100K) across 10 Indian languages covering 13 broad and 56 sub-categories, leveraging 57 diverse datasets. Our dataset protocol incorporates several often-overlooked dimensions and emphasize task diversity, multi-turn dialogue, instruction fidelity, safety alignment, and preservation of cultural nuance, providing a foundation for more inclusive and effective multilingual LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）的效果在很大程度上依赖于高质量的后训练数据，特别是指令调优和基于偏好的示例。现有的开源数据集往往缺乏多语言覆盖、文化基础，并且在任务多样性方面存在特别明显的缺口，尤其是对于印度语言。我们提出了一种人机协作流水线，结合翻译与合成扩展，生成可靠且多样的印度语言后训练数据。使用此流水线，我们构建了两个数据集：Pragyaan-IT（22,500）和Pragyaan-Align（100,000），涵盖10种印度语言，包括13个大类和56个子类别，利用57个不同的数据集。我们的数据集协议包含了多个常被忽视的维度，强调任务多样性、多轮对话、指令忠实度、安全性对齐以及文化细微差别的保存，为更包容和有效的多语言大型语言模型提供基础。 

---
# The Limits of Goal-Setting Theory in LLM-Driven Assessment 

**Title (ZH)**: 目标设定理论在驱动LLM评估中的局限性 

**Authors**: Mrityunjay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.06997)  

**Abstract**: Many users interact with AI tools like ChatGPT using a mental model that treats the system as human-like, which we call Model H. According to goal-setting theory, increased specificity in goals should reduce performance variance. If Model H holds, then prompting a chatbot with more detailed instructions should lead to more consistent evaluation behavior.
This paper tests that assumption through a controlled experiment in which ChatGPT evaluated 29 student submissions using four prompts with increasing specificity. We measured consistency using intra-rater reliability (Cohen's Kappa) across repeated runs.
Contrary to expectations, performance did not improve consistently with increased prompt specificity, and performance variance remained largely unchanged. These findings challenge the assumption that LLMs behave like human evaluators and highlight the need for greater robustness and improved input integration in future model development. 

**Abstract (ZH)**: 许多用户使用类似于ChatGPT的AI工具时，将其视为类人系统，我们称之为Model H。根据目标设定理论，目标设定的细化应减少性能的差异性。如果Model H成立，那么通过更详细的指令提示聊天机器人应该会导致更一致的评估行为。本文通过一项受控实验测试了这一假设，该实验让ChatGPT对29份学生提交的作品进行了评价，并通过重复运行内评分者信度（Cohen's Kappa）来衡量一致性。令人意外的是，随着提示的细化，性能并未一致提高，性能的差异性也没有显著变化。这些发现挑战了LLMs表现得像人类评估者的假设，强调了未来模型开发中需要增强鲁棒性和提高输入集成能力。 

---
# VelLMes: A high-interaction AI-based deception framework 

**Title (ZH)**: VelLMes：一种高交互人工智能欺骗框架 

**Authors**: Muris Sladić, Veronica Valeros, Carlos Catania, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2510.06975)  

**Abstract**: There are very few SotA deception systems based on Large Language Models. The existing ones are limited only to simulating one type of service, mainly SSH shells. These systems - but also the deception technologies not based on LLMs - lack an extensive evaluation that includes human attackers. Generative AI has recently become a valuable asset for cybersecurity researchers and practitioners, and the field of cyber-deception is no exception. Researchers have demonstrated how LLMs can be leveraged to create realistic-looking honeytokens, fake users, and even simulated systems that can be used as honeypots. This paper presents an AI-based deception framework called VelLMes, which can simulate multiple protocols and services such as SSH Linux shell, MySQL, POP3, and HTTP. All of these can be deployed and used as honeypots, thus VelLMes offers a variety of choices for deception design based on the users' needs. VelLMes is designed to be attacked by humans, so interactivity and realism are key for its performance. We evaluate the generative capabilities and the deception capabilities. Generative capabilities were evaluated using unit tests for LLMs. The results of the unit tests show that, with careful prompting, LLMs can produce realistic-looking responses, with some LLMs having a 100% passing rate. In the case of the SSH Linux shell, we evaluated deception capabilities with 89 human attackers. The results showed that about 30% of the attackers thought that they were interacting with a real system when they were assigned an LLM-based honeypot. Lastly, we deployed 10 instances of the SSH Linux shell honeypot on the Internet to capture real-life attacks. Analysis of these attacks showed us that LLM honeypots simulating Linux shells can perform well against unstructured and unexpected attacks on the Internet, responding correctly to most of the issued commands. 

**Abstract (ZH)**: 基于大语言模型的先进欺骗系统非常少。现有的系统仅限于模拟一种服务，主要是SSH终端。这些系统以及其他非基于大语言模型的欺骗技术缺乏包含人类攻击者的广泛评估。生成式AI最近已成为网络安全研究人员和 practitioner 的宝贵资产，欺骗技术领域也不例外。研究人员已经展示了如何利用大语言模型生成逼真的蜜罐、虚假用户，甚至是模拟系统作为蜜罐。本文提出了一种基于AI的欺骗框架VelLMes，它可以模拟多种协议和服务，如SSH Linux终端、MySQL、POP3和HTTP。这些都可以部署和使用为蜜罐，因此VelLMes提供了多种基于用户需求的欺骗设计选择。VelLMes设计为接受人类攻击，因此交互性和逼真性是其性能的关键。我们评估了生成能力和欺骗能力。生成能力通过大语言模型的单元测试进行评估。单元测试的结果显示，在精心提示下，大语言模型可以生成逼真的响应，有些模型通过率为100%。对于SSH Linux终端，我们使用89名人类攻击者评估了欺骗能力。结果显示，约30%的攻击者认为他们与基于大语言模型的蜜罐进行了真实的交互。最后，我们在互联网上部署了10个SSH Linux终端蜜罐实例以捕获真实攻击。这些攻击的分析表明，模拟Linux终端的LLM蜜罐在应对互联网上的非结构化和非预期攻击时表现良好，正确响应了大多数发出的命令。 

---
# EDUMATH: Generating Standards-aligned Educational Math Word Problems 

**Title (ZH)**: EDUMATH: 生成符合标准的教育数学文字题 

**Authors**: Bryan R. Christ, Penelope Molitz, Jonathan Kropko, Thomas Hartvigsen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06965)  

**Abstract**: Math word problems (MWPs) are critical K-12 educational tools, and customizing them to students' interests and ability levels can increase learning outcomes. However, teachers struggle to find time to customize MWPs for each student given large class sizes and increasing burnout. We propose that LLMs can support math education by generating MWPs customized to student interests and math education standards. To this end, we use a joint human expert-LLM judge approach to evaluate over 11,000 MWPs generated by open and closed LLMs and develop the first teacher-annotated dataset for standards-aligned educational MWP generation. We show the value of our data by using it to train a 12B open model that matches the performance of larger and more capable open models. We also use our teacher-annotated data to train a text classifier that enables a 30B open LLM to outperform existing closed baselines without any training. Next, we show our models' MWPs are more similar to human-written MWPs than those from existing models. We conclude by conducting the first study of customized LLM-generated MWPs with grade school students, finding they perform similarly on our models' MWPs relative to human-written MWPs but consistently prefer our customized MWPs. 

**Abstract (ZH)**: 数学文字问题（MWPs）是关键的K-12教育工具，将其个性化以适应学生的兴趣和能力水平可以提高学习成果。然而，在大班授课和不断增加的教学压力下，教师难以为每位学生量身定制MWPs。我们提出，大型语言模型（LLMs）可以通过生成符合学生兴趣和数学教育标准的MWPs来支持数学教育。为此，我们采用人工专家和LLM评判员联合评估了超过11,000个由开放和封闭的LLM生成的MWPs，并开发了首个针对标准对齐的教育MWPs生成的教师标注数据集。我们通过使用该数据集训练一个12亿参数的开放模型，证实了其性能与更大和更强大的开放模型相当。我们还使用教师标注的数据训练了一个文本分类器，使得一个30亿参数的开放LLM在没有任何训练的情况下超过了现有的封闭基线。此外，我们表明我们的模型生成的MWPs比现有模型生成的MWPs更类似于人工撰写的MWPs。最后，我们进行了首次针对小学生的研究，发现他们在我们的模型生成的MWPs上的表现与人工撰写的MWPs相当，但始终更偏好我们个性化生成的MWPs。 

---
# LongRM: Revealing and Unlocking the Context Boundary of Reward Modeling 

**Title (ZH)**: LongRM: 揭示和解锁奖励建模的上下文边界 

**Authors**: Zecheng Tang, Baibei Ji, Quantong Qiu, Haitian Wang, Xiaobo Liang, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06915)  

**Abstract**: Reward model (RM) plays a pivotal role in aligning large language model (LLM) with human preferences. As real-world applications increasingly involve long history trajectories, e.g., LLM agent, it becomes indispensable to evaluate whether a model's responses are not only high-quality but also grounded in and consistent with the provided context. Yet, current RMs remain confined to short-context settings and primarily focus on response-level attributes (e.g., safety or helpfulness), while largely neglecting the critical dimension of long context-response consistency. In this work, we introduce Long-RewardBench, a benchmark specifically designed for long-context RM evaluation, featuring both Pairwise Comparison and Best-of-N tasks. Our preliminary study reveals that even state-of-the-art generative RMs exhibit significant fragility in long-context scenarios, failing to maintain context-aware preference judgments. Motivated by the analysis of failure patterns observed in model outputs, we propose a general multi-stage training strategy that effectively scales arbitrary models into robust Long-context RMs (LongRMs). Experiments show that our approach not only substantially improves performance on long-context evaluation but also preserves strong short-context capability. Notably, our 8B LongRM outperforms much larger 70B-scale baselines and matches the performance of the proprietary Gemini 2.5 Pro model. 

**Abstract (ZH)**: Long-RewardBench: 一种专门用于长上下文强化评估的标准基准 

---
# OpenJAI-v1.0: An Open Thai Large Language Model 

**Title (ZH)**: OpenJAI-v1.0：一个开源泰语大型语言模型 

**Authors**: Pontakorn Trakuekul, Attapol T. Rutherford, Jullajak Karnjanaekarin, Narongkorn Panitsrisit, Sumana Sumanakul  

**Link**: [PDF](https://arxiv.org/pdf/2510.06847)  

**Abstract**: We introduce OpenJAI-v1.0, an open-source large language model for Thai and English, developed from the Qwen3-14B model. Our work focuses on boosting performance on practical tasks through carefully curated data across three key use cases: instruction following, long-context understanding, and tool use. Evaluation results show that OpenJAI-v1.0 improves on the capabilities of its base model and outperforms other leading open-source Thai models on a diverse suite of benchmarks, while avoiding catastrophic forgetting. OpenJAI-v1.0 is publicly released as another alternative NLP resource for the Thai AI community. 

**Abstract (ZH)**: 我们介绍OpenJAI-v1.0，一个基于Qwen3-14B模型的开源大型语言模型，适用于泰语和英语。我们的工作集中在通过精心选择的数据在三个关键应用场景上提高性能：指令跟随、长上下文理解以及工具使用。评估结果显示，OpenJAI-v1.0 在其基础模型的基础上提升性能，并在一系列基准测试中优于其他主要的开源泰语模型，同时避免了灾难性遗忘。OpenJAI-v1.0 公开发布，作为泰国人工智能社区的另一个自然语言处理资源。 

---
# SID: Multi-LLM Debate Driven by Self Signals 

**Title (ZH)**: SID: 由自我信号驱动的多语言模型辩论 

**Authors**: Xuhang Chen, Zhifan Song, Deyi Ji, Shuo Gao, Lanyun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06843)  

**Abstract**: Large Language Models (LLMs) have exhibited impressive capabilities across diverse application domains. Recent work has explored Multi-LLM Agent Debate (MAD) as a way to enhance performance by enabling multiple LLMs to discuss and refine responses iteratively. Nevertheless, existing MAD methods predominantly focus on utilizing external structures, such as debate graphs, using LLM-as-a-Judge, while neglecting the application of self signals, such as token logits and attention, that arise during generation. This omission leads to redundant computation and potential performance degradation. In this paper, we shift the focus to the self signals of multi-LLM debate and introduce a Self-Signals Driven Multi-LLM Debate (SID), which leverages two types of self-signals: model-level confidence and token-level semantic focus, to adaptively guide the debate process. Our approach enables high-confidence agents to exit early at the model level and compress the redundant debate contents based on the attention mechanism. We evaluate our method on various LLMs and Multimodal LLMs across multiple challenging benchmarks. Experimental results demonstrate that our method not only outperforms existing MAD techniques in accuracy but also reduces token consumption, highlighting the effectiveness of utilizing self signals in enhancing both the performance and efficiency of multi-agent debate systems. Our code will be available at~\href{this https URL}{\texttt{this https URL}}. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多个应用领域展示了出色的能力。近期研究表明，通过让多个LLM进行讨论和迭代修正回应的多LLM代理辩论（MAD）方法能够提升性能。然而，现有MAD方法主要侧重于使用外部结构，如辩论图，以及LLM作为裁判的方式，而忽视了生成过程中出现的自我信号，如令牌概率和注意力，这导致了重复计算和潜在性能下降。本文将重点放在多LLM辩论的自我信号上，提出了一种基于自我信号的多LLM代理辩论（SID），利用模型级别信心和令牌级别语义焦点两种自我信号，自适应地指导辩论过程。该方法允许高信心代理在模型级别提前退出，并基于注意力机制压缩冗余辩论内容。我们在多个LLM和多模态LLM上多项具有挑战性的基准测试上评估了该方法。实验结果表明，该方法不仅在准确率上优于现有MAD技术，还能减少令牌消耗，突出利用自我信号提高多代理辩论系统性能和效率的有效性。我们的代码将在\href{this https URL}{\texttt{this https URL}}提供。 

---
# Recurrence-Complete Frame-based Action Models 

**Title (ZH)**: 基于帧的循环完整动作模型 

**Authors**: Michael Keiblinger  

**Link**: [PDF](https://arxiv.org/pdf/2510.06828)  

**Abstract**: In recent years, attention-like mechanisms have been used to great success in the space of large language models, unlocking scaling potential to a previously unthinkable extent. "Attention Is All You Need" famously claims RNN cells are not needed in conjunction with attention. We challenge this view. In this paper, we point to existing proofs that architectures with fully parallelizable forward or backward passes cannot represent classes of problems specifically interesting for long-running agentic tasks. We further conjecture a critical time t beyond which non-recurrence-complete models fail to aggregate inputs correctly, with concrete implications for agentic systems (e.g., software engineering agents). To address this, we introduce a recurrence-complete architecture and train it on GitHub-derived action sequences. Loss follows a power law in the trained sequence length while the parameter count remains fixed. Moreover, longer-sequence training always amortizes its linearly increasing wall-time cost, yielding lower loss as a function of wall time. 

**Abstract (ZH)**: 近年来，注意力机制在大型语言模型领域取得了巨大的成功，解开了此前难以想象的扩展潜力。“Attention Is All You Need”著名地宣称，在注意力机制下RNN单元是不必要的。我们认为这一观点值得商榷。本文指出，具有完全并行可执行正向或反向传递的架构无法表示特定针对长时间运行智能任务的问题类。此外，我们推测存在一个关键时间点t，在此之后，非递归完备模型无法正确聚合输入，对智能系统（例如软件工程代理）具有具体影响。为解决这一问题，我们提出了一种递归完备架构，并在GitHub派生的操作序列上进行训练。损失遵循训练序列长度的幂律分布，而参数计数保持不变。此外，更长序列的训练总是能够摊薄线性增加的墙时间成本，以墙时间为函数，损失更低。 

---
# FURINA: A Fully Customizable Role-Playing Benchmark via Scalable Multi-Agent Collaboration Pipeline 

**Title (ZH)**: FURINA：一种基于可扩展多智能体协作pipeline的完全可定制角色扮演基准测试 

**Authors**: Haotian Wu, Shufan Jiang, Chios Chen, Yiyang Feng, Hehai Lin, Heqing Zou, Yao Shu, Yanran Li, Chengwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06800)  

**Abstract**: As large language models (LLMs) advance in role-playing (RP) tasks, existing benchmarks quickly become obsolete due to their narrow scope, outdated interaction paradigms, and limited adaptability across diverse application scenarios. To address this gap, we introduce FURINA-Builder, a novel multi-agent collaboration pipeline that automatically constructs fully customizable RP benchmarks at any scale. It enables evaluation of arbitrary characters across diverse scenarios and prompt formats, as the first benchmark builder in RP area for adaptable assessment. FURINA-Builder simulates dialogues between a test character and other characters drawn from a well-constructed character-scene pool, while an LLM judge selects fine-grained evaluation dimensions and adjusts the test character's responses into final test utterances. Using this pipeline, we build FURINA-Bench, a new comprehensive role-playing benchmark featuring both established and synthesized test characters, each assessed with dimension-specific evaluation criteria. Human evaluation and preliminary separability analysis justify our pipeline and benchmark design. We conduct extensive evaluations of cutting-edge LLMs and find that o3 and DeepSeek-R1 achieve the best performance on English and Chinese RP tasks, respectively. Across all models, established characters consistently outperform synthesized ones, with reasoning capabilities further amplifying this disparity. Interestingly, we observe that model scale does not monotonically reduce hallucinations. More critically, for reasoning LLMs, we uncover a novel trade-off: reasoning improves RP performance but simultaneously increases RP hallucinations. This trade-off extends to a broader Pareto frontier between RP performance and reliability for all LLMs. These findings demonstrate the effectiveness of FURINA-Builder and the challenge posed by FURINA-Bench. 

**Abstract (ZH)**: 随着大型语言模型在角色扮演任务中的进步，现有基准迅速变得过时，因为它们的范围狭窄、过时的交互范式以及在多样应用场景中的有限适应性。为了解决这一差距，我们引入了FURINA-Builder，这是一种新颖的多智能体协作流水线，可以自动构建任意规模的完全可定制的角色扮演基准。它能够评估各种场景和指令格式中的任意角色，是角色扮演领域中的首个可适应评估的基准构建器。FURINA-Builder模拟了测试角色与其他从精心构建的角色场景池中抽取的角色之间的对话，而一个语言模型裁判选取细致的评估维度并调整测试角色的回复以形成最终的测试话语。使用此流水线，我们构建了FURINA-Bench，一个新的全面的角色扮演基准，包含既有和合成的测试角色，并对每个角色使用特定维度的评估标准。通过人工评估和初步可分性分析，我们证明了我们的流水线和基准设计的有效性。我们对最先进的语言模型进行了广泛的评估，发现o3和DeepSeek-R1分别在英语和中文角色扮演任务中表现最佳。在所有模型中，既有角色始终优于合成角色，推理能力进一步放大了这一差异。有趣的是，我们发现模型规模并非单调减少涌现现象。更关键的是，对于推理语言模型，我们揭示了一种新的权衡：推理提高了角色扮演性能，但同时增加了角色扮演中的涌现现象。这种权衡扩展到了所有语言模型之间RP性能和可靠性之间的帕累托前沿。这些发现展示了FURINA-Builder的有效性和FURINA-Bench带来的挑战。 

---
# Foundations of LLM Knowledge Materialization: Termination, Reproducibility, Robustness 

**Title (ZH)**: LLM知识物质化的基础：终止性、可重复性、稳健性 

**Authors**: Luca Giordano, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2510.06780)  

**Abstract**: Large Language Models (LLMs) encode substantial factual knowledge, yet measuring and systematizing this knowledge remains challenging. Converting it into structured format, for example through recursive extraction approaches such as the GPTKB methodology (Hu et al., 2025b), is still underexplored. Key open questions include whether such extraction can terminate, whether its outputs are reproducible, and how robust they are to variations. We systematically study LLM knowledge materialization using miniGPTKBs (domain-specific, tractable subcrawls), analyzing termination, reproducibility, and robustness across three categories of metrics: yield, lexical similarity, and semantic similarity. We experiment with four variations (seed, language, randomness, model) and three illustrative domains (from history, entertainment, and finance). Our findings show (i) high termination rates, though model-dependent; (ii) mixed reproducibility; and (iii) robustness that varies by perturbation type: high for seeds and temperature, lower for languages and models. These results suggest that LLM knowledge materialization can reliably surface core knowledge, while also revealing important limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）蕴含大量的事实性知识，但测量和系统化这些知识依然具有挑战性。通过递归提取方法（如GPTKB方法，Hu等，2025b）将其转换为结构化格式仍然未被充分探索。关键的开放问题包括此类提取是否可以终止、其输出是否可重复以及对变化的鲁棒性如何。我们系统地研究了使用minigiptkb（领域特定、可处理的子抓取）来实现LLM知识物质化，分析了终止、可重复性和鲁棒性在三种类别指标下的表现：产出量、词汇相似性和语义相似性。我们实验了四种变体（种子、语言、随机性、模型）和三个示例领域（历史、娱乐和金融）。我们的研究结果表明：（i）高终止率，但取决于模型；（ii）重复性参差不齐；（iii）鲁棒性因扰动类型而异：对种子和温度高，对语言和模型低。这些结果表明，LLM知识物质化可以可靠地揭示核心知识，同时也揭示了重要的限制。 

---
# Evaluating LLMs for Historical Document OCR: A Methodological Framework for Digital Humanities 

**Title (ZH)**: 基于数字人文的方法论框架：评估语言模型在历史文档OCR中的应用 

**Authors**: Maria Levchenko  

**Link**: [PDF](https://arxiv.org/pdf/2510.06743)  

**Abstract**: Digital humanities scholars increasingly use Large Language Models for historical document digitization, yet lack appropriate evaluation frameworks for LLM-based OCR. Traditional metrics fail to capture temporal biases and period-specific errors crucial for historical corpus creation. We present an evaluation methodology for LLM-based historical OCR, addressing contamination risks and systematic biases in diplomatic transcription. Using 18th-century Russian Civil font texts, we introduce novel metrics including Historical Character Preservation Rate (HCPR) and Archaic Insertion Rate (AIR), alongside protocols for contamination control and stability testing. We evaluate 12 multimodal LLMs, finding that Gemini and Qwen models outperform traditional OCR while exhibiting over-historicization: inserting archaic characters from incorrect historical periods. Post-OCR correction degrades rather than improves performance. Our methodology provides digital humanities practitioners with guidelines for model selection and quality assessment in historical corpus digitization. 

**Abstract (ZH)**: 基于大型语言模型的历史文档光学字符识别评估方法：Addressing Contamination Risks and Systematic Biases in LLM-based Historical OCR 

---
# Are LLMs Reliable Rankers? Rank Manipulation via Two-Stage Token Optimization 

**Title (ZH)**: LLMs可靠吗？基于两阶段token优化的排名操控 

**Authors**: Tiancheng Xing, Jerry Li, Yixuan Du, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06732)  

**Abstract**: Large language models (LLMs) are increasingly used as rerankers in information retrieval, yet their ranking behavior can be steered by small, natural-sounding prompts. To expose this vulnerability, we present Rank Anything First (RAF), a two-stage token optimization method that crafts concise textual perturbations to consistently promote a target item in LLM-generated rankings while remaining hard to detect. Stage 1 uses Greedy Coordinate Gradient to shortlist candidate tokens at the current position by combining the gradient of the rank-target with a readability score; Stage 2 evaluates those candidates under exact ranking and readability losses using an entropy-based dynamic weighting scheme, and selects a token via temperature-controlled sampling. RAF generates ranking-promoting prompts token-by-token, guided by dual objectives: maximizing ranking effectiveness and preserving linguistic naturalness. Experiments across multiple LLMs show that RAF significantly boosts the rank of target items using naturalistic language, with greater robustness than existing methods in both promoting target items and maintaining naturalness. These findings underscore a critical security implication: LLM-based reranking is inherently susceptible to adversarial manipulation, raising new challenges for the trustworthiness and robustness of modern retrieval systems. Our code is available at: this https URL. 

**Abstract (ZH)**: 基于大型语言模型的检索重新排行为自然语言操控提供新挑战：Rank Anything First 

---
# Scaling LLM Multi-turn RL with End-to-end Summarization-based Context Management 

**Title (ZH)**: 基于摘要式上下文管理的大型语言模型多轮RL扩展研究 

**Authors**: Miao Lu, Weiwei Sun, Weihua Du, Zhan Ling, Xuesong Yao, Kang Liu, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.06727)  

**Abstract**: We study reinforcement learning (RL) fine-tuning of large language model (LLM) agents for long-horizon multi-turn tool use, where context length quickly becomes a fundamental bottleneck. Existing RL pipelines can suffer from degraded instruction following, excessive rollout costs, and most importantly, strict context limits. To address these challenges, we introduce summarization-based context management to training. In specific, it periodically compresses the tool using history by LLM-generated summaries that retain task-relevant information to keep a compact context while enabling the agent to scale beyond the fixed context window. Building on this formulation, we derive a policy gradient representation that seamlessly enables standard LLM RL infrastructures to optimize both tool-use behaviors as well as summarization strategies in an end-to-end fashion. We instantiate this framework with \underline{SU}mmarization augmented \underline{P}olicy \underline{O}ptimization (\texttt{SUPO}), an LLM RL algorithm that enables long-horizon training beyond a fixed context limit. Experiments on interactive function calling and searching tasks demonstrate that \texttt{SUPO} significantly improves the success rate while maintaining the same or even lower working context length compared to baselines. We also demonstrate that for complex searching tasks, \texttt{SUPO} can further improve the evaluation performance when scaling test-time maximum round of summarization beyond that of training time. Our results establish summarization-based context management as a principled and scalable approach for training RL agents beyond a fixed context length limit. 

**Abstract (ZH)**: 基于总结化的上下文管理的大语言模型强化学习 Fine-tuning 以实现长时 horizon 多轮工具使用 

---
# LLM Company Policies and Policy Implications in Software Organizations 

**Title (ZH)**: LLM 公司政策及软件组织中的政策影响 

**Authors**: Ranim Khojah, Mazen Mohamad, Linda Erlenhov, Francisco Gomes de Oliveira Neto, Philipp Leitner  

**Link**: [PDF](https://arxiv.org/pdf/2510.06718)  

**Abstract**: The risks associated with adopting large language model (LLM) chatbots in software organizations highlight the need for clear policies. We examine how 11 companies create these policies and the factors that influence them, aiming to help managers safely integrate chatbots into development workflows. 

**Abstract (ZH)**: 采用大型语言模型聊天机器人（LLM chatbots）在软件组织中面临的风险凸显了明确政策的必要性。我们研究了11家公司如何制定这些政策及其影响因素，旨在帮助管理人员安全地将聊天机器人集成到开发流程中。 

---
# AISysRev - LLM-based Tool for Title-abstract Screening 

**Title (ZH)**: AISysRev - 基于LLM的标题摘要筛查工具 

**Authors**: Aleksi Huotala, Miikka Kuutila, Olli-Pekka Turtio, Mika Mäntylä  

**Link**: [PDF](https://arxiv.org/pdf/2510.06708)  

**Abstract**: Systematic reviews are a standard practice for summarizing the state of evidence in software engineering. Conducting systematic reviews is laborious, especially during the screening or study selection phase, where the number of papers can be overwhelming. During this phase, papers are assessed against inclusion and exclusion criteria based on their titles and abstracts. Recent research has demonstrated that large language models (LLMs) can perform title-abstract screening at a level comparable to that of a master's student. While LLMs cannot be fully trusted, they can help, for example, in Rapid Reviews, which try to expedite the review process. Building on recent research, we developed AiSysRev, an LLM-based screening tool implemented as a web application running in a Docker container. The tool accepts a CSV file containing paper titles and abstracts. Users specify inclusion and exclusion criteria. One can use multiple LLMs for screening via OpenRouter. AiSysRev supports both zero-shot and few-shot screening, and also allows for manual screening through interfaces that display LLM results as guidance for human this http URL conducted a trial study with 137 papers using the tool. Our findings indicate that papers can be classified into four categories: Easy Includes, Easy Excludes, Boundary Includes, and Boundary Excludes. The Boundary cases, where LLMs are prone to errors, highlight the need for human intervention. While LLMs do not replace human judgment in systematic reviews, they can significantly reduce the burden of assessing large volumes of scientific literature. Video: this https URL Tool: this https URL 

**Abstract (ZH)**: 系统综述是软件工程中总结现有证据标准实践。进行系统综述是一项繁琐的工作，尤其是在筛选或研究选择阶段，此时需要评估的论文数量可能非常庞大。在这一阶段，论文依据标题和摘要评估是否符合纳入和排除标准。最近的研究表明，大型语言模型（LLMs）可以在标题-摘要筛选方面达到与硕士学生相当的水平。尽管LLMs不能完全信赖，但它们可以在快速综述（如尝试加快审查过程）等场景中发挥作用。基于最近的研究，我们开发了AiSysRev，这是一种基于LLM的筛选工具，实现为Docker容器中的Web应用。该工具接受包含论文标题和摘要的CSV文件。用户指定纳入和排除标准。可以使用OpenRouter的多个LLM进行筛选。AiSysRev支持零样本和少量样本筛选，并允许通过显示LLM结果以指导人工筛选的界面进行人工筛选。我们使用该工具对137篇论文进行了试用研究。研究结果表明，论文可以分为四类：易纳入、易排除、边界纳入和边界排除。边界情形下，LLMs容易出现错误，突显了人工干预的必要性。尽管LLMs不能替代系统综述中的人类判断，但它们可以显著减轻评估大量科学文献的负担。视频：[此链接]。工具：[此链接]。 

---
# Learning to Rewrite Prompts for Bootstrapping LLMs on Downstream Tasks 

**Title (ZH)**: 基于下游任务实训增强LLM的提示重写学习 

**Authors**: Qinhao Zhou, Xiang Xiang, Kun He, John E. Hopcroft  

**Link**: [PDF](https://arxiv.org/pdf/2510.06695)  

**Abstract**: In recent years, the growing interest in Large Language Models (LLMs) has significantly advanced prompt engineering, transitioning from manual design to model-based optimization. Prompts for LLMs generally comprise two components: the \textit{instruction}, which defines the task or objective, and the \textit{input}, which is tailored to the instruction type. In natural language generation (NLG) tasks such as machine translation, the \textit{input} component is particularly critical, while the \textit{instruction} component tends to be concise. Existing prompt engineering methods primarily focus on optimizing the \textit{instruction} component for general tasks, often requiring large-parameter LLMs as auxiliary tools. However, these approaches exhibit limited applicability for tasks like machine translation, where the \textit{input} component plays a more pivotal role. To address this limitation, this paper introduces a novel prompt optimization method specifically designed for machine translation tasks. The proposed approach employs a small-parameter model trained using a back-translation-based strategy, significantly reducing training overhead for single-task optimization while delivering highly effective performance. With certain adaptations, this method can also be extended to other downstream tasks. 

**Abstract (ZH)**: 近年来，对大规模语言模型（LLMs）日益增长的兴趣显著推动了提示工程的发展，从手动设计转向基于模型的优化。LLMs的提示通常由两个组件组成：\textit{指令}，定义任务或目标；和\textit{输入}，根据指令类型定制。在如机器翻译等自然语言生成（NLG）任务中，\textit{输入}组件尤为重要，而\textit{指令}组件通常较为简短。现有的提示工程方法主要侧重于通过使用大型参数量的LLMs对通用任务的\textit{指令}组件进行优化。然而，这些方法对于机器翻译等任务的适用性有限，在这些任务中，\textit{输入}组件发挥着更重要的作用。为解决这一局限性，本文提出了一种针对机器翻译任务的新型提示优化方法。该方法采用一种通过回译策略训练的小参数模型，在单任务优化中显著减少训练开销，并展现出高度有效的性能。经过适当的调整，该方法也可扩展应用于其他下游任务。 

---
# Heptapod: Language Modeling on Visual Signals 

**Title (ZH)**: Heptapod：基于视觉信号的语言模型 

**Authors**: Yongxin Zhu, Jiawei Chen, Yuanzhe Chen, Zhuo Chen, Dongya Jia, Jian Cong, Xiaobin Zhuang, Yuping Wang, Yuxuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06673)  

**Abstract**: We introduce Heptapod, an image autoregressive model that adheres to the foundational principles of language modeling. Heptapod employs \textbf{causal attention}, \textbf{eliminates reliance on CFG}, and \textbf{eschews the trend of semantic tokenizers}. Our key innovation is \textit{next 2D distribution prediction}: a causal Transformer with reconstruction-focused visual tokenizer, learns to predict the distribution over the entire 2D spatial grid of images at each timestep. This learning objective unifies the sequential modeling of autoregressive framework with the holistic self-supervised learning of masked autoencoding, enabling the model to capture comprehensive image semantics via generative training. On the ImageNet generation benchmark, Heptapod achieves an FID of $2.70$, significantly outperforming previous causal autoregressive approaches. We hope our work inspires a principled rethinking of language modeling on visual signals and beyond. 

**Abstract (ZH)**: 我们介绍了Heptapod，一种遵循语言模型基础原理的图像自回归模型。Heptapod采用因果注意力机制，消除对CFG的依赖，并摒弃语义分词趋势。我们的关键创新是“下一个2D分布预测”：一种以重构为重点的视觉分词因果Transformer，学习在每个时间步预测整个2D空间网格图像的概率分布。这一学习目标将自回归框架的序列建模与掩码自编码的整体自监督学习统一起来，使模型通过生成训练捕捉全面的图像语义。在ImageNet生成基准测试中，Heptapod的FID为2.70，显著优于先前的因果自回归方法。我们希望我们的工作能启发对视觉信号乃至更广泛领域语言模型原理性的重新思考。 

---
# Distilling Lightweight Language Models for C/C++ Vulnerabilities 

**Title (ZH)**: 轻量级语言模型压缩以识别C/C++漏洞 

**Authors**: Zhiyuan Wei, Xiaoxuan Yang, Jing Sun, Zijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.06645)  

**Abstract**: The increasing complexity of modern software systems exacerbates the prevalence of security vulnerabilities, posing risks of severe breaches and substantial economic loss. Consequently, robust code vulnerability detection is essential for software security. While Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing, their potential for automated code vulnerability detection remains underexplored. This paper presents FineSec, a novel framework that harnesses LLMs through knowledge distillation to enable efficient and precise vulnerability identification in C/C++ codebases. FineSec utilizes knowledge distillation to transfer expertise from large teacher models to compact student models, achieving high accuracy with minimal computational cost. By integrating data preparation, training, evaluation, and continuous learning into a unified, single-task workflow, FineSec offers a streamlined approach. Extensive evaluations on C/C++ codebases demonstrate its superiority over both base models and larger LLMs in identifying complex vulnerabilities and logical flaws, establishing FineSec as a practical and scalable solution for real-world software security. To facilitate reproducibility, the datasets, source code, and experimental results are made publicly available at: this https URL. 

**Abstract (ZH)**: 现代软件系统的日益复杂加剧了安全漏洞的普遍性，带来了严重泄露和巨大经济损失的风险。因此，强大的代码漏洞检测对于软件安全至关重要。尽管大型语言模型（LLMs）在自然语言处理方面展现了卓越的能力，但它们在自动化代码漏洞检测方面的潜力尚未被充分探索。本文提出了一种名为FineSec的新颖框架，通过知识蒸馏利用LLMs，以实现对C/C++代码库中的高效精准漏洞识别。FineSec利用知识蒸馏将大型教师模型的知识传递给紧凑的学生模型，以在较小的计算成本下实现高精度。通过将数据准备、训练、评估和持续学习整合到统一的一体化单任务工作流中，FineSec提供了一种简化的方法。对C/C++代码库的广泛评估表明，FineSec在识别复杂漏洞和逻辑缺陷方面优于基础模型和更大规模的LLMs，确立了FineSec作为实际软件安全问题中实用且可扩展的解决方案的地位。为便于重复性，相关数据集、源代码和实验结果已公开发布于此：this https URL。 

---
# Reading Between the Lines: Towards Reliable Black-box LLM Fingerprinting via Zeroth-order Gradient Estimation 

**Title (ZH)**: 细读其中之意：通过零阶梯度估计实现可靠的黑盒大模型指纹识别 

**Authors**: Shuo Shao, Yiming Li, Hongwei Yao, Yifei Chen, Yuchen Yang, Zhan Qin  

**Link**: [PDF](https://arxiv.org/pdf/2510.06605)  

**Abstract**: The substantial investment required to develop Large Language Models (LLMs) makes them valuable intellectual property, raising significant concerns about copyright protection. LLM fingerprinting has emerged as a key technique to address this, which aims to verify a model's origin by extracting an intrinsic, unique signature (a "fingerprint") and comparing it to that of a source model to identify illicit copies. However, existing black-box fingerprinting methods often fail to generate distinctive LLM fingerprints. This ineffectiveness arises because black-box methods typically rely on model outputs, which lose critical information about the model's unique parameters due to the usage of non-linear functions. To address this, we first leverage Fisher Information Theory to formally demonstrate that the gradient of the model's input is a more informative feature for fingerprinting than the output. Based on this insight, we propose ZeroPrint, a novel method that approximates these information-rich gradients in a black-box setting using zeroth-order estimation. ZeroPrint overcomes the challenge of applying this to discrete text by simulating input perturbations via semantic-preserving word substitutions. This operation allows ZeroPrint to estimate the model's Jacobian matrix as a unique fingerprint. Experiments on the standard benchmark show ZeroPrint achieves a state-of-the-art effectiveness and robustness, significantly outperforming existing black-box methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）的开发需要大量投资，使其成为有价值的知识产权，引发了关于版权保护的重要 concern。LLM 指纹识别已经成为解决这一问题的关键技术，旨在通过提取内在的独特签名（“指纹”）并与源模型进行比较来验证模型的起源，以识别非法副本。然而，现有的黑盒指纹识别方法往往无法生成独特的LLM 指纹。这种无效性源于黑盒方法通常依赖于模型输出，而线性函数的使用导致了关键信息关于模型独特参数的损失。为解决这一问题，我们首先利用费雪信息理论正式证明了模型输入的梯度比输出更能提供指纹识别信息。基于这一洞察，我们提出了零印（ZeroPrint）这一新颖方法，在黑盒环境中使用零阶估计近似这些信息丰富的梯度。零印通过语义保留的单词替代模拟输入扰动来克服将此应用于离散文本的挑战。这一操作使零印能够估计模型的雅可比矩阵作为独特指纹。在标准基准上的实验显示，零印达到了最先进的有效性和稳健性，显著优于现有黑盒方法。 

---
# The Algebra of Meaning: Why Machines Need Montague More Than Moore's Law 

**Title (ZH)**: 意义代数：机器为何需要蒙塔古而非摩尔定律 

**Authors**: Cheonkam Jeong, Sungdo Kim, Jewoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.06559)  

**Abstract**: Contemporary language models are fluent yet routinely mis-handle the types of meaning their outputs entail. We argue that hallucination, brittle moderation, and opaque compliance outcomes are symptoms of missing type-theoretic semantics rather than data or scale limitations. Building on Montague's view of language as typed, compositional algebra, we recast alignment as a parsing problem: natural-language inputs must be compiled into structures that make explicit their descriptive, normative, and legal dimensions under context.
We present Savassan, a neuro-symbolic architecture that compiles utterances into Montague-style logical forms and maps them to typed ontologies extended with deontic operators and jurisdictional contexts. Neural components extract candidate structures from unstructured inputs; symbolic components perform type checking, constraint reasoning, and cross-jurisdiction mapping to produce compliance-aware guidance rather than binary censorship. In cross-border scenarios, the system "parses once" (e.g., defect claim(product x, company y)) and projects the result into multiple legal ontologies (e.g., defamation risk in KR/JP, protected opinion in US, GDPR checks in EU), composing outcomes into a single, explainable decision.
This paper contributes: (i) a diagnosis of hallucination as a type error; (ii) a formal Montague-ontology bridge for business/legal reasoning; and (iii) a production-oriented design that embeds typed interfaces across the pipeline. We outline an evaluation plan using legal reasoning benchmarks and synthetic multi-jurisdiction suites. Our position is that trustworthy autonomy requires compositional typing of meaning, enabling systems to reason about what is described, what is prescribed, and what incurs liability within a unified algebra of meaning. 

**Abstract (ZH)**: 当代语言模型虽然流利，但经常错误处理其输出所蕴含的类型意义。我们认为，幻觉、脆弱的控制以及不透明的合规结果是缺少类型理论语义的症状，而非数据或规模限制的问题。基于蒙塔古关于语言为类型化和组合代数的观点，我们将对齐重新定义为一个解析问题：自然语言输入必须被编译成结构，这些结构在其上下文中明确地体现出描述性、规范性和法律性维度。

我们提出了Savassan，一个神经符号架构，将陈述编译为蒙塔古风格的逻辑形式，并映射到扩展了道义运算和管辖域上下文的类型化本体上。神经组件从无结构输入中提取候选结构；符号组件执行类型检查、约束推理和跨管辖域映射，以产生合规意识指导而非二元审查。在跨国场景中，该系统“一次解析”（例如，缺陷索赔（产品x，公司y））并将结果投影到多个法律本体中（例如，在KR/JP中判断诽谤风险，在US中判断保护意见，在EU中进行GDPR检查），将结果组合成一个可解释的决策。

本文贡献：（i）将幻觉诊断为类型错误；（ii）建立形式化的蒙塔古本体桥接以支持商业/法律推理；（iii）一个生产导向的设计，贯穿流水线嵌入类型化界面。我们提出了一个使用法律推理基准和合成跨国套件进行评估的计划。我们认为，可信赖的自主性需要语义的组合类型化，使系统能够在统一的意义代数中推理描述、规范和法律责任。 

---
# The Markovian Thinker 

**Title (ZH)**: 马尔可夫思维者 

**Authors**: Milad Aghajohari, Kamran Chitsaz, Amirhossein Kazemnejad, Sarath Chandar, Alessandro Sordoni, Aaron Courville, Siva Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2510.06557)  

**Abstract**: Reinforcement learning (RL) has recently become a strong recipe for training reasoning LLMs that produce long chains of thought (LongCoT). Yet the standard RL "thinking environment", where the state is the prompt plus all prior reasoning tokens, makes the state unbounded and forces attention-based policies to pay quadratic compute as thoughts lengthen. We revisit the environment itself. We propose Markovian Thinking, a paradigm in which the policy advances reasoning while conditioning on a constant-size state, decoupling thinking length from context size. As an immediate consequence this yields linear compute with constant memory. We instantiate this idea with Delethink, an RL environment that structures reasoning into fixed-size chunks. Within each chunk, the model thinks as usual; at the boundary, the environment resets the context and reinitializes the prompt with a short carryover. Through RL, the policy learns to write a textual state near the end of each chunk sufficient for seamless continuation of reasoning after reset. Trained in this environment, an R1-Distill 1.5B model reasons in 8K-token chunks yet thinks up to 24K tokens, matching or surpassing LongCoT-RL trained with a 24K budget. With test-time scaling, Delethink continues to improve where LongCoT plateaus. The effect of linear compute is substantial: we empirically estimate at 96K average thinking length LongCoT-RL costs 27 H100-months vs. 7 for Delethink. Analysis at RL initialization shows off-the-shelf reasoning models (1.5B-120B) often sample Markovian traces zero-shot across diverse benchmarks, providing positive samples that make RL effective at scale. Our results show that redesigning the thinking environment is a powerful lever: it enables very long reasoning without quadratic overhead and opens a path toward efficient, scalable reasoning LLMs. 

**Abstract (ZH)**: 标记化思维：一种线性计算的长期链条推理强化学习环境 

---
# Webscale-RL: Automated Data Pipeline for Scaling RL Data to Pretraining Levels 

**Title (ZH)**: webscale-RL：自动化数据流水线，用于将RL数据扩展到预训练水平 

**Authors**: Zhepeng Cen, Haolin Chen, Shiyu Wang, Zuxin Liu, Zhiwei Liu, Ding Zhao, Silvio Savarese, Caiming Xiong, Huan Wang, Weiran Yao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06499)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success through imitation learning on vast text corpora, but this paradigm creates a training-generation gap and limits robust reasoning. Reinforcement learning (RL) offers a more data-efficient solution capable of bridging this gap, yet its application has been constrained by a critical data bottleneck: existing RL datasets are orders of magnitude smaller and less diverse than web-scale pre-training corpora. To address this, we introduce the Webscale-RL pipeline, a scalable data engine that systematically converts large-scale pre-training documents into millions of diverse, verifiable question-answer pairs for RL. Using this pipeline, we construct the Webscale-RL dataset, containing 1.2 million examples across more than 9 domains. Our experiments show that the model trained on this dataset significantly outperforms continual pretraining and strong data refinement baselines across a suite of benchmarks. Notably, RL training with our dataset proves substantially more efficient, achieving the performance of continual pre-training with up to 100$\times$ fewer tokens. Our work presents a viable path toward scaling RL to pre-training levels, enabling more capable and efficient language models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过在大量文本语料库上进行模仿学习取得了显著的成功，但这一范式造成了训练-生成差距，并限制了稳健的推理能力。强化学习（RL）提供了一种更高效的数据解决方案，能够弥补这一差距，然而其应用受到一个关键技术数据瓶颈的限制：现有的RL数据集在规模和多样性上远逊于面向网络规模的预训练语料库。为了解决这个问题，我们引入了Webscale-RL管道，这是一种可扩展的数据引擎，能够系统地将大规模预训练文档转换为数以百万计的多样化、可验证的问题-答案对，用于强化学习。借助这一管道，我们构建了Webscale-RL数据集，包含超过9个领域的120万个示例。我们的实验表明，该数据集训练的模型在一系列基准测试中的表现显著优于持续预训练和强大的数据精炼基线。值得注意的是，使用我们数据集的RL训练表现出极大的效率，实现了与最多100倍少的令牌量的持续预训练相当的性能。我们的工作为将RL扩展到预训练水平提供了可行路径，有望促进更多强大和高效的语言模型的发展。 

---
# Valid Stopping for LLM Generation via Empirical Dynamic Formal Lift 

**Title (ZH)**: 基于经验动态形式提升的LLM生成有效终止方法 

**Authors**: Sanjeda Akter, Ibne Farabi Shihab, Anuj Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2510.06478)  

**Abstract**: We introduce Sequential-EDFL (Empirical Dynamic Formal Lift), applying anytime-valid sequential testing to language model generation stopping. Our approach tracks information lift -- the log-likelihood ratio between full models and deliberately weakened "skeleton" baselines -- using self-normalized empirical-Bernstein e-processes that provide formal delta-level error control regardless of stopping time. We handle unknown centering through online mean estimation, combine multiple parameters via mixture e-processes, and support adaptive resets under distributional drift. On six benchmarks, Sequential-EDFL reduces generation by 22-28% vs. sequential baselines while maintaining delta-level control with 12% computational overhead. We introduce automated skeletons (distilled submodels, randomized logits) and show robustness across skeleton families. Composing EDFL with a lightweight correctness gate (sentence boundaries + verifier) improves end-task correctness while preserving anytime-valid guarantees by only delaying stopping. Our certificates control information sufficiency, not factual correctness -- 10.9% of stopped sequences remain incorrect even with the gate (13.2-22.7% without it). EDFL serves as a first-stage filter reducing verification burden by 83%, not as a standalone solution for safety-critical domains. 

**Abstract (ZH)**: 我们引入了Sequential-EDFL（经验动态形式提升），将任意有效序列测试应用于语言模型生成停止。该方法使用自归一化经验伯恩斯坦e过程跟踪信息提升——即完整模型与故意削弱的“骨架”基线之间的对数似然比，并提供不受停止时间影响的正式delta级误差控制。通过在线均值估计处理未知中心化，通过混合e过程结合多个参数，并在分布漂移情况下支持自适应重置。在六个基准测试中，Sequential-EDFL相比序列基线将生成量减少了22-28%，同时通过12%的计算开销保持了delta级控制。我们引入了自动化骨架（精简子模型、随机化logits），并展示了其在不同骨架家族中的鲁棒性。将EDFL与轻量级正确性门（句子边界+验证器）结合，可以提高最终任务的正确性，同时通过仅延迟停止来保留任意有效保证。我们的证书控制信息充分性，而不是事实正确性，即使有门机制，仍有10.9%的停止序列是不正确的（没有门机制时，这一比例为13.2%-22.7%）。EDFL作为第一阶段过滤器可减少验证负担83%，而不是作为关键安全领域中的独立解决方案。 

---
# Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin 

**Title (ZH)**: Attention Sinks and Compression Valleys in LLMs Are Two Sides of the Same Coin 

**Authors**: Enrique Queipo-de-Llano, Álvaro Arroyo, Federico Barbero, Xiaowen Dong, Michael Bronstein, Yann LeCun, Ravid Shwartz-Ziv  

**Link**: [PDF](https://arxiv.org/pdf/2510.06477)  

**Abstract**: Attention sinks and compression valleys have attracted significant attention as two puzzling phenomena in large language models, but have been studied in isolation. In this work, we present a surprising connection between attention sinks and compression valleys, tracing both to the formation of massive activations in the residual stream. We prove theoretically that massive activations necessarily produce representational compression and establish bounds on the resulting entropy reduction. Through experiments across several models (410M-120B parameters), we confirm that when the beginning-of-sequence token develops extreme activation norms in the middle layers, both compression valleys and attention sinks emerge simultaneously. Targeted ablation studies validate our theoretical predictions. This unified view motivates us to propose the Mix-Compress-Refine theory of information flow, as an attempt to explain how LLMs organize their computation in depth by controlling attention and representational compression via massive activations. Specifically, we posit that Transformer-based LLMs process tokens in three distinct phases: (1) broad mixing in the early layers, (2) compressed computation with limited mixing in the middle layers, and (3) selective refinement in the late layers. Our framework helps explain why embedding tasks perform best at intermediate layers, whereas generation tasks benefit from full-depth processing, clarifying differences in task-dependent representations. 

**Abstract (ZH)**: 注意力陷阱和压缩谷值在大型语言模型中作为两个令人困惑的现象引起了广泛关注，但它们被单独研究。在这项工作中，我们提出了注意力陷阱和压缩谷值之间的意外联系，两者都追溯到残差流中巨大激活的形成。我们从理论上证明，巨大的激活必然导致表示压缩，并建立了其结果熵减少的边界。通过跨越多个模型（410M-120B参数）的实验，我们证实当序列起始标记在中间层发展出极端激活范数时，同时出现压缩谷值和注意力陷阱。目标消融实验验证了我们的理论预测。这一统一的视角促使我们提出了信息流的Mix-Compress-Refine理论，试图解释大型语言模型通过控制注意力和表示压缩来深度组织其计算的方式。具体而言，我们认为基于Transformer的大型语言模型在三个不同阶段处理标记：（1）早期层中的广泛混合，（2）中间层中的压缩计算伴随有限混合，以及（3）晚期层中的选择性精细处理。我们的框架有助于解释为什么嵌入任务在中间层表现最佳，而生成任务受益于全深度处理，澄清了任务依赖表示的差异。 

---
# A Survey on Agentic Security: Applications, Threats and Defenses 

**Title (ZH)**: 代理安全综述：应用、威胁与防御 

**Authors**: Asif Shahriar, Md Nafiu Rahman, Sadif Ahmed, Farig Sadeque, Md Rizwan Parvez  

**Link**: [PDF](https://arxiv.org/pdf/2510.06445)  

**Abstract**: The rapid shift from passive LLMs to autonomous LLM-agents marks a new paradigm in cybersecurity. While these agents can act as powerful tools for both offensive and defensive operations, the very agentic context introduces a new class of inherent security risks. In this work we present the first holistic survey of the agentic security landscape, structuring the field around three interdependent pillars: Applications, Threats, and Defenses. We provide a comprehensive taxonomy of over 150 papers, explaining how agents are used, the vulnerabilities they possess, and the countermeasures designed to protect them. A detailed cross-cutting analysis shows emerging trends in agent architecture while revealing critical research gaps in model and modality coverage. 

**Abstract (ZH)**: 快速从被动的大语言模型转向自主的大语言模型代理标志着网络安全领域的新范式。尽管这些代理可以作为强大的工具用于 Offensive 和 Defensive 操作，但自主性的引入带来了新的内在安全风险。在这项工作中，我们首次综述了代理安全景观，围绕三个相互依存的支柱：应用、威胁和防御构建领域。我们提供了一百余篇论文的全面分类体系，解释了代理的使用方式、它们所具备的漏洞以及设计的防护措施。详细的横向分析显示出代理架构的发展趋势，同时也揭示了模型和模态覆盖方面的关键研究缺口。 

---
# Reward Model Perspectives: Whose Opinions Do Reward Models Reward? 

**Title (ZH)**: 奖励模型视角：奖励模型奖励的是哪方意见？ 

**Authors**: Elle  

**Link**: [PDF](https://arxiv.org/pdf/2510.06391)  

**Abstract**: Reward models (RMs) are central to the alignment of language models (LMs). An RM often serves as a proxy for human preferences to guide downstream LM behavior. However, our understanding of RM behavior is limited. Our work (i) formalizes a framework for measuring the alignment of opinions captured by RMs, (ii) investigates the extent to which RMs demonstrate sociodemographic biases, and (iii) explores the effects of prompting to steer rewards towards the preferences of a target group. We study the subjective and diverse perspectives on controversial topics, which allows us to quantify RM perspectives in terms of their opinions, attitudes, and values. We show that RMs are poorly aligned with several demographic groups and can systematically reward harmful stereotypes, and steering alone is not enough to overcome these limitations. Our findings underscore the need for more careful consideration of RM behavior in model alignment during preference learning to prevent the propagation of unwanted social biases in the language technologies that we use. 

**Abstract (ZH)**: 奖励模型（RMs）是语言模型（LMs）对齐的核心。RMs通常作为人类偏好代理，引导下游LM的行为。然而，我们对RMs行为的理解有限。我们的工作（i）正式提出了一种衡量RMs捕捉意见对齐程度的框架，（ii）探究RMs表现社会人口偏见的程度，以及（iii）研究提示对引导奖励向目标群体偏好转变的影响。我们研究了争议性话题的主观和多样化观点，从而量化RMs观点在意见、态度和价值观方面的影响。我们发现，RMs与多个社会人口群体不契合，并且系统性地奖励有害刻板印象，单独调整不足以克服这些局限。我们的研究强调，在偏好学习过程中，对于RMs行为需要更加谨慎地考虑，以防止在语言技术中传播不希望的社会偏见。 

---
# Protecting De-identified Documents from Search-based Linkage Attacks 

**Title (ZH)**: 保护去标识化文档免受基于搜索的链接攻击 

**Authors**: Pierre Lison, Mark Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2510.06383)  

**Abstract**: While de-identification models can help conceal the identity of the individual(s) mentioned in a document, they fail to address linkage risks, defined as the potential to map the de-identified text back to its source. One straightforward way to perform such linkages is to extract phrases from the de-identified document and then check their presence in the original dataset. This paper presents a method to counter search-based linkage attacks while preserving the semantic integrity of the text. The method proceeds in two steps. We first construct an inverted index of the N-grams occurring in the document collection, making it possible to efficiently determine which N-grams appear in less than $k$ documents (either alone or in combination with other N-grams). An LLM-based rewriter is then iteratively queried to reformulate those spans until linkage is no longer possible. Experimental results on a collection of court cases show that the method is able to effectively prevent search-based linkages while remaining faithful to the original content. 

**Abstract (ZH)**: 尽管去识别模型可以帮助隐藏文档中提及的个体身份，但它们未能解决链接风险，即可以将去识别的文本重新映射回其源头的风险。一种简单的方法是提取去识别文档中的短语，然后检查这些短语在原始数据集中的存在。本文提出了一个方法来抵御基于搜索的链接攻击同时保持文本的语义完整性。该方法分为两步。我们首先构建文档集合中出现的N-grams的倒排索引，使得能够高效地确定哪些N-grams出现在少于$k$个文档中（单独或与其他N-grams组合）。然后，基于LLM的重写器逐步查询以重新表述这些片段，直到无法进行链接。实验结果表明，该方法能够有效防止基于搜索的链接，并忠实地保留原始内容。 

---
# Leveraging Large Language Models for Cybersecurity Risk Assessment -- A Case from Forestry Cyber-Physical Systems 

**Title (ZH)**: 利用大型语言模型进行网络安全风险评估——以林业 cyber-physical 系统为例 

**Authors**: Fikret Mert Gültekin, Oscar Lilja, Ranim Khojah, Rebekka Wohlrab, Marvin Damschen, Mazen Mohamad  

**Link**: [PDF](https://arxiv.org/pdf/2510.06343)  

**Abstract**: In safety-critical software systems, cybersecurity activities become essential, with risk assessment being one of the most critical. In many software teams, cybersecurity experts are either entirely absent or represented by only a small number of specialists. As a result, the workload for these experts becomes high, and software engineers would need to conduct cybersecurity activities themselves. This creates a need for a tool to support cybersecurity experts and engineers in evaluating vulnerabilities and threats during the risk assessment process. This paper explores the potential of leveraging locally hosted large language models (LLMs) with retrieval-augmented generation to support cybersecurity risk assessment in the forestry domain while complying with data protection and privacy requirements that limit external data sharing. We performed a design science study involving 12 experts in interviews, interactive sessions, and a survey within a large-scale project. The results demonstrate that LLMs can assist cybersecurity experts by generating initial risk assessments, identifying threats, and providing redundancy checks. The results also highlight the necessity for human oversight to ensure accuracy and compliance. Despite trust concerns, experts were willing to utilize LLMs in specific evaluation and assistance roles, rather than solely relying on their generative capabilities. This study provides insights that encourage the use of LLM-based agents to support the risk assessment process of cyber-physical systems in safety-critical domains. 

**Abstract (ZH)**: 在安全关键软件系统中，网络安全活动变得必不可少，风险评估是其中最关键的部分之一。在许多软件团队中，网络安全专家要么完全缺席，要么仅由少量专家代表。因此，这些专家的工作量增加，软件工程师需要自己进行网络安全活动。这产生了对工具的需求，以支持网络安全专家和工程师在风险评估过程中评估漏洞和威胁。本文探讨了利用本地托管的大语言模型（LLMs）及其检索增强生成技术，支持林业领域网络安全风险评估的可能性，同时遵守限制外部数据共享的数据保护和隐私要求。我们通过包含12位专家的访谈、互动会话和问卷调查进行了设计科学研究。研究结果表明，LLMs可以通过生成初始风险评估、识别威胁和提供冗余检查来协助网络安全专家。研究结果还强调了确保准确性和合规性的人工监督的必要性。尽管存在信任问题，专家们仍然愿意在特定评估和支持角色中使用LLMs，而不是完全依赖其生成能力。本文为利用LLM基础代理支持安全关键领域中的网络物理系统风险评估过程提供了见解。 

---
# VeriEquivBench: An Equivalence Score for Ground-Truth-Free Evaluation of Formally Verifiable Code 

**Title (ZH)**: VeriEquivBench：一种无需ground-truth的正式可验证代码等价性评分方法 

**Authors**: Lingfei Zeng, Fengdi Che, Xuhan Huang, Fei Ye, Xu Xu, Binhang Yuan, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06296)  

**Abstract**: Formal verification is the next frontier for ensuring the correctness of code generated by Large Language Models (LLMs). While methods that co-generate code and formal specifications in formal languages, like Dafny, can, in principle, prove alignment with user intent, progress is bottlenecked by specification quality evaluation. Current benchmarks rely on matching against ground-truth specifications, a manual and expertise-intensive process that has limited existing datasets to a few hundred simple problems and also suffers from a reliability issue. To address this, we introduce VeriEquivBench, a new benchmark with $2,389$ complex algorithmic problems that probe the limitations of current models in both code generation and formal reasoning. Our evaluation framework replaces ground-truth matching with a formally grounded metric, the equivalence score, and rigorously verifies the quality of generated specifications and code. Our results show that generating formally verifiable code remains a profound challenge for state-of-the-art LLMs. This underscores both the difficulty of the task and the need for benchmarks like VeriEquivBench to drive progress toward scalable and reliable coding agents. 

**Abstract (ZH)**: 正式验证是确保由大规模语言模型（LLMs）生成的代码正确性的下一个前沿领域。为了克服现有基准因依赖于匹配真实规格而产生的手动且耗费专业技能的过程限制，我们引入了VeriEquivBench这一新的基准，其中包括2,389个复杂算法问题，旨在测试当前模型在代码生成和形式推理方面的局限性。我们的评估框架使用了一个形式化的指标，等价分数，来验证生成的规格和代码的质量。结果表明，生成可形式验证的代码依然是当今最先进的LLMs面临的重大挑战。这不仅揭示了任务的复杂性，还强调了需要如VeriEquivBench这样的基准来推动可扩展和可靠编程代理的研究进展。 

---
# Reproducibility Study of "XRec: Large Language Models for Explainable Recommendation" 

**Title (ZH)**: XRec：可解释推荐的大语言模型再现性研究 

**Authors**: Ranjan Mishra, Julian I. Bibo, Quinten van Engelen, Henk Schaapman  

**Link**: [PDF](https://arxiv.org/pdf/2510.06275)  

**Abstract**: In this study, we reproduced the work done in the paper "XRec: Large Language Models for Explainable Recommendation" by Ma et al. (2024). The original authors introduced XRec, a model-agnostic collaborative instruction-tuning framework that enables large language models (LLMs) to provide users with comprehensive explanations of generated recommendations. Our objective was to replicate the results of the original paper, albeit using Llama 3 as the LLM for evaluation instead of GPT-3.5-turbo. We built on the source code provided by Ma et al. (2024) to achieve our goal. Our work extends the original paper by modifying the input embeddings or deleting the output embeddings of XRec's Mixture of Experts module. Based on our results, XRec effectively generates personalized explanations and its stability is improved by incorporating collaborative information. However, XRec did not consistently outperform all baseline models in every metric. Our extended analysis further highlights the importance of the Mixture of Experts embeddings in shaping the explanation structures, showcasing how collaborative signals interact with language modeling. Through our work, we provide an open-source evaluation implementation that enhances accessibility for researchers and practitioners alike. Our complete code repository can be found at this https URL. 

**Abstract (ZH)**: 本研究重现了Ma等（2024）在论文“XRec：可解释推荐的大语言模型”中的工作。原始作者引入了XRec模型，这是一种模型无关的合作指令调校框架，使大语言模型能够为用户提供生成推荐的全面解释。我们的目标是使用Llama 3替代GPT-3.5-turbo进行评估，重现原论文的结果。我们基于Ma等（2024）提供的源代码实现了这一目标。我们的工作通过修改XRec的专家混合模块的输入嵌入或删除输出嵌入，扩展了原论文。根据我们的结果，XRec能够有效生成个性化解释，并且通过集成合作信息提高了稳定性。然而，XRec在所有指标上并未始终优于基准模型。我们的扩展分析进一步突出了专家混合嵌入在塑造解释结构中的重要性，展示了协作信号与语言建模的交互。通过我们的工作，我们提供了一个开源评估实现，以提高研究人员和实践者的可访问性。我们的完整代码库可以在以下网址找到：this https URL。 

---
# MCCE: A Framework for Multi-LLM Collaborative Co-Evolution 

**Title (ZH)**: MCCE：多大型语言模型协同协进化框架 

**Authors**: Nian Ran, Zhongzheng Li, Yue Wang, Qingsong Ran, Xiaoyuan Zhang, Shikun Feng, Richard Allmendinger, Xiaoguang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2510.06270)  

**Abstract**: Multi-objective discrete optimization problems, such as molecular design, pose significant challenges due to their vast and unstructured combinatorial spaces. Traditional evolutionary algorithms often get trapped in local optima, while expert knowledge can provide crucial guidance for accelerating convergence. Large language models (LLMs) offer powerful priors and reasoning ability, making them natural optimizers when expert knowledge matters. However, closed-source LLMs, though strong in exploration, cannot update their parameters and thus cannot internalize experience. Conversely, smaller open models can be continually fine-tuned but lack broad knowledge and reasoning strength. We introduce Multi-LLM Collaborative Co-evolution (MCCE), a hybrid framework that unites a frozen closed-source LLM with a lightweight trainable model. The system maintains a trajectory memory of past search processes; the small model is progressively refined via reinforcement learning, with the two models jointly supporting and complementing each other in global exploration. Unlike model distillation, this process enhances the capabilities of both models through mutual inspiration. Experiments on multi-objective drug design benchmarks show that MCCE achieves state-of-the-art Pareto front quality and consistently outperforms baselines. These results highlight a new paradigm for enabling continual evolution in hybrid LLM systems, combining knowledge-driven exploration with experience-driven learning. 

**Abstract (ZH)**: 多目标离散优化问题，如分子设计，由于其庞大的无结构组合空间而面临重大挑战。传统进化算法往往陷入局部最优，而专家知识可以提供加速收敛的关键指导。大型语言模型（LLMs）具备强大的先验知识和推理能力，当专家知识至关重要时，它们是天然的优化器。然而，闭源的LLMs虽然在探索方面很强，但无法更新其参数，因此无法内化经验。相反，较小的开源模型可以持续微调，但缺乏广泛的知识和推理能力。我们介绍了多LLM协作共进化（MCCE）混合框架，该框架结合了一个冻结的闭源LLM和一个轻量级可训练模型。该系统维护了过去搜索过程的轨迹记忆；小型模型通过强化学习逐步优化，两种模型共同支持和补充彼此完成全局探索。与模型蒸馏不同，这一过程通过相互启发来增强两种模型的能力。在多目标药物设计基准测试上的实验表明，MCCE实现了最先进的帕累托前沿质量，并且一致地优于基线。这些结果突显了在一个结合了知识驱动探索和经验驱动学习的混合LLM系统中实现持续进化的新范式。 

---
# LLM-Driven Rubric-Based Assessment of Algebraic Competence in Multi-Stage Block Coding Tasks with Design and Field Evaluation 

**Title (ZH)**: 基于设计与现场评估的多阶段积木编码任务中的代数能力评分驱动评估 

**Authors**: Yong Oh Lee, Byeonghun Bang, Sejun Oh  

**Link**: [PDF](https://arxiv.org/pdf/2510.06253)  

**Abstract**: As online education platforms continue to expand, there is a growing need for assessment methods that not only measure answer accuracy but also capture the depth of students' cognitive processes in alignment with curriculum objectives. This study proposes and evaluates a rubric-based assessment framework powered by a large language model (LLM) for measuring algebraic competence, real-world-context block coding tasks. The problem set, designed by mathematics education experts, aligns each problem segment with five predefined rubric dimensions, enabling the LLM to assess both correctness and quality of students' problem-solving processes. The system was implemented on an online platform that records all intermediate responses and employs the LLM for rubric-aligned achievement evaluation. To examine the practical effectiveness of the proposed framework, we conducted a field study involving 42 middle school students engaged in multi-stage quadratic equation tasks with block coding. The study integrated learner self-assessments and expert ratings to benchmark the system's outputs. The LLM-based rubric evaluation showed strong agreement with expert judgments and consistently produced rubric-aligned, process-oriented feedback. These results demonstrate both the validity and scalability of incorporating LLM-driven rubric assessment into online mathematics and STEM education platforms. 

**Abstract (ZH)**: 随着在线教育平台的不断扩大，需要一种不仅可以衡量答案准确性，还能捕获学生认知过程深度并与课程目标保持一致的评估方法。本研究提出并评估了一种基于评分标准的评估框架，该框架借助大型语言模型（LLM）来衡量代数能力及基于现实情境的块式编程任务。由数学教育专家设计的问题集将每个问题部分与五个预设的评分标准维度对齐，使LLM能够评估学生解决问题过程的正确性和质量。该系统已在记录所有中间响应的在线平台上实现，并利用LLM进行评分标准对齐的表现评价。为了检验所提框架的实际有效性，我们在一项涉及42名中学生执行多阶段二次方程块式编程任务的实地研究中实施了该系统。该研究整合了学习者自我评估和专家评分，以衡量系统输出的基准。基于LLM的评分标准评估与专家判断高度一致，并持续提供与评分标准对齐的过程导向反馈。这些结果证明了将LLM驱动的评分标准评估集成到在线数学和STEM教育平台中的有效性和可扩展性。 

---
# Scalable multilingual PII annotation for responsible AI in LLMs 

**Title (ZH)**: 可扩展的多语言PII标注以实现LLM中的负责任人工智能 

**Authors**: Bharti Meena, Joanna Skubisz, Harshit Rajgarhia, Nand Dave, Kiran Ganesh, Shivali Dalmia, Abhishek Mukherji, Vasudevan Sundarababu, Olga Pospelova  

**Link**: [PDF](https://arxiv.org/pdf/2510.06250)  

**Abstract**: As Large Language Models (LLMs) gain wider adoption, ensuring their reliable handling of Personally Identifiable Information (PII) across diverse regulatory contexts has become essential. This work introduces a scalable multilingual data curation framework designed for high-quality PII annotation across 13 underrepresented locales, covering approximately 336 locale-specific PII types. Our phased, human-in-the-loop annotation methodology combines linguistic expertise with rigorous quality assurance, leading to substantial improvements in recall and false positive rates from pilot, training, and production phases. By leveraging inter-annotator agreement metrics and root-cause analysis, the framework systematically uncovers and resolves annotation inconsistencies, resulting in high-fidelity datasets suitable for supervised LLM fine-tuning. Beyond reporting empirical gains, we highlight common annotator challenges in multilingual PII labeling and demonstrate how iterative, analytics-driven pipelines can enhance both annotation quality and downstream model reliability. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的广泛应用，确保其在多种监管环境下可靠处理个人可识别信息（PII）已成为必要。本文介绍了一种针对13个代表性不足的地域、涵盖约336种地域特定PII类型的可扩展多语言数据整理框架。该框架的人机结合标注方法结合了语言学专长和严格的质量保证，从试点、训练和生产阶段显著提高了召回率和减少误报率。通过利用注标者间一致性指标和根本原因分析，框架系统地发现和解决了标注不一致问题，生成适用于监督微调的高保真数据集。除了报告实证收益，我们还强调了多语言PII标注中的常见注标者挑战，并展示了迭代的数据驱动管道如何提高标注质量和下游模型的可靠性。 

---
# TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B 

**Title (ZH)**: TRepLiNa: 层级CKA+REPINA对齐改善了Aya-23 8B低资源机器翻译 

**Authors**: Toshiki Nakai, Ravi Kiran Chikkala, Lena Sophie Oberkircher, Nicholas Jennings, Natalia Skachkova, Tatiana Anikina, Jesujoba Oluwadara Alabi  

**Link**: [PDF](https://arxiv.org/pdf/2510.06249)  

**Abstract**: The 2025 Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo) Language Challenge addresses one of India's most pressing linguistic gaps: the lack of resources for its diverse low-resource languages (LRLs). In this study, we investigate whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual large language model (LLM) can improve translation quality from LRL to high-resource language (HRL). Specifically, we combine Centered Kernel Alignment (CKA), a similarity metric that encourages representations of different languages to align, with REPINA, a regularization method that constrains parameter updates to remain close to the pretrained model, into a joint method we call TRepLiNa. In this research project, we experiment with zero-shot, few-shot, and fine-tuning settings using Aya-23 8B with QLoRA across MMLoSo shared task language pairs (Mundari, Santali, Bhili) with Hindi/English pivots. Our results show that aligning mid-level layers using TRepLiNa (CKA+REPINA) is a low-cost, practical approach to improving LRL translation, especially in data-scarce settings. 

**Abstract (ZH)**: 2025多模态模型在低资源语境和社会影响（MMLoSo）语言挑战中的低资源语言（LRL）资源不足问题研究：强制特定解码器内部层的跨语言相似性以提高LRL到高资源语言（HRL）翻译质量 

---
# CoT Referring: Improving Referring Expression Tasks with Grounded Reasoning 

**Title (ZH)**: 基于grounded reasoning改进指称表达任务 

**Authors**: Qihua Dong, Luis Figueroa, Handong Zhao, Kushal Kafle, Jason Kuen, Zhihong Ding, Scott Cohen, Yun Fu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06243)  

**Abstract**: Referring Expression Comprehension and Segmentation are critical tasks for assessing the integration of language understanding and image comprehension, serving as benchmarks for Multimodal Large Language Models (MLLMs) capabilities. To address these challenges, we propose a new strategy, CoT Referring, which enhances model reasoning across modalities through a structured, chain-of-thought training data structure. Our approach systematically parses textual structures to a sequential referring step, where in each step it identifies relationships and ensures consistent reference alignment, thereby improving accuracy in complex query scenarios. We restructure the training data to enforce a new output form, providing new annotations for existing datasets and compiling an evaluation benchmark from existing resources. This benchmark is designed explicitly for complex referring cases. We also integrate detection and segmentation capabilities into a unified MLLM framework, training it with a novel adaptive weighted loss to optimize performance. Experimental results on our curated benchmark and RefCOCO/+/g demonstrate the effectiveness of our approach, with a notable increase of 2.5%+ over baseline models. 

**Abstract (ZH)**: 参照表达理解和分割是评估语言理解与图像理解整合的关键任务，作为多模态大语言模型（MLLMs）能力的基准。为应对这些挑战，我们提出了一种新的策略CoT Referring，通过结构化的链式思考训练数据结构增强模型在多模态间的推理能力。我们的方法系统地将文本结构分解为顺序的参照步骤，在每一步中识别关系并确保一致的参照对齐，从而在复杂的查询场景中提高准确性。我们重构了训练数据以强制新的输出形式，并为现有数据集提供新的标注，同时整合现有资源构建了一个专门针对复杂参照情况的评估基准。我们还将检测和分割能力整合到统一的MLLM框架中，并使用新颖的自适应加权损失进行训练，以优化性能。在我们精心策划的基准和RefCOCO/+/g上的实验结果表明，我们的方法有效，相较于基线模型，准确率提高了2.5%以上。 

---
# Transparent Reference-free Automated Evaluation of Open-Ended User Survey Responses 

**Title (ZH)**: 开放-ended用户调查响应的透明无参考自动评估 

**Authors**: Subin An, Yugyeong Ji, Junyoung Kim, Heejin Kook, Yang Lu, Josh Seltzer  

**Link**: [PDF](https://arxiv.org/pdf/2510.06242)  

**Abstract**: Open-ended survey responses provide valuable insights in marketing research, but low-quality responses not only burden researchers with manual filtering but also risk leading to misleading conclusions, underscoring the need for effective evaluation. Existing automatic evaluation methods target LLM-generated text and inadequately assess human-written responses with their distinct characteristics. To address such characteristics, we propose a two-stage evaluation framework specifically designed for human survey responses. First, gibberish filtering removes nonsensical responses. Then, three dimensions-effort, relevance, and completeness-are evaluated using LLM capabilities, grounded in empirical analysis of real-world survey data. Validation on English and Korean datasets shows that our framework not only outperforms existing metrics but also demonstrates high practical applicability for real-world applications such as response quality prediction and response rejection, showing strong correlations with expert assessment. 

**Abstract (ZH)**: 开放式的调查反馈为市场营销研究提供了宝贵的见解，但低质量的反馈不仅增加了研究人员的手动筛选负担，还可能导致错误结论，强调了有效评估的必要性。现有自动评估方法主要针对LLM生成的文本，未能充分评估具有独特特征的人类撰写的反馈。为应对这些特征，我们提出了一种两阶段评估框架，专门设计用于评估人类调查反馈。首先，无意义内容过滤去除无意义的反馈。然后，通过基于实际调查数据的实证分析，使用LLM能力从努力程度、相关性和完整性三个维度进行评估。在英语和韩语数据集上的验证表明，我们的框架不仅在现有指标上表现更优，还展示了在真实应用中预测和拒绝反馈的强大实用性，并与专家评估高度相关。 

---
# A Multimodal GUI Architecture for Interfacing with LLM-Based Conversational Assistants 

**Title (ZH)**: 基于LLM的对话式助手的多模态GUI架构 

**Authors**: Hans G.W. van Dam  

**Link**: [PDF](https://arxiv.org/pdf/2510.06223)  

**Abstract**: Advances in large language models (LLMs) and real-time speech recognition now make it possible to issue any graphical user interface (GUI) action through natural language and receive the corresponding system response directly through the GUI. Most production applications were never designed with speech in mind. This article provides a concrete architecture that enables GUIs to interface with LLM-based speech-enabled assistants.
The architecture makes an application's navigation graph and semantics available through the Model Context Protocol (MCP). The ViewModel, part of the MVVM (Model-View-ViewModel) pattern, exposes the application's capabilities to the assistant by supplying both tools applicable to a currently visible view and application-global tools extracted from the GUI tree router. This architecture facilitates full voice accessibility while ensuring reliable alignment between spoken input and the visual interface, accompanied by consistent feedback across modalities. It future-proofs apps for upcoming OS super assistants that employ computer use agents (CUAs) and natively consume MCP if an application provides it.
To address concerns about privacy and data security, the practical effectiveness of locally deployable, open-weight LLMs for speech-enabled multimodal UIs is evaluated. Findings suggest that recent smaller open-weight models approach the performance of leading proprietary models in overall accuracy and require enterprise-grade hardware for fast responsiveness. 

**Abstract (ZH)**: 大型语言模型（LLMs）和实时语音识别的进步现在使通过自然语言发布任何图形用户界面（GUI）操作并通过GUI直接接收相应系统响应成为可能。大多数生产应用程序从未专门为语音设计。本文提供了一种具体的架构，使GUI能够与基于LLM的语音助手进行交互。
该架构通过模型上下文协议（MCP）使应用程序的导航图和语义可供应用。视图模型作为MVVM（模型-视图-视图模型）模式的一部分，通过提供适用于当前可见视图的工具以及从GUI树路由中提取的应用全局工具，向助手展示应用程序的能力。该架构促进了全面的语音访问，同时确保了口头输入与视觉界面之间的可靠对齐，并在不同模态中提供了一致的反馈。它为即将出现的使用计算使用代理（CUA）并原生消费MCP的OS超级助手做好了准备。
为了应对隐私和数据安全方面的担忧，评估了在语音增强型多模态UI中本地部署和开放权重的LLM的实际效果。研究发现，最近较小的开放权重模型在总体准确率方面接近领先专有模型的表现，并且需要企业级硬件才能实现快速响应。 

---
# WeatherArchive-Bench: Benchmarking Retrieval-Augmented Reasoning for Historical Weather Archives 

**Title (ZH)**: WeatherArchive-Bench: 基于历史天气档案的检索增强推理基准测试 

**Authors**: Yongan Yu, Xianda Du, Qingchen Hu, Jiahao Liang, Jingwei Ni, Dan Qiang, Kaiyu Huang, Grant McKenzie, Renee Sieber, Fengran Mo  

**Link**: [PDF](https://arxiv.org/pdf/2510.05336)  

**Abstract**: Historical archives on weather events are collections of enduring primary source records that offer rich, untapped narratives of how societies have experienced and responded to extreme weather events. These qualitative accounts provide insights into societal vulnerability and resilience that are largely absent from meteorological records, making them valuable for climate scientists to understand societal responses. However, their vast scale, noisy digitized quality, and archaic language make it difficult to transform them into structured knowledge for climate research. To address this challenge, we introduce WeatherArchive-Bench, the first benchmark for evaluating retrieval-augmented generation (RAG) systems on historical weather archives. WeatherArchive-Bench comprises two tasks: WeatherArchive-Retrieval, which measures a system's ability to locate historically relevant passages from over one million archival news segments, and WeatherArchive-Assessment, which evaluates whether Large Language Models (LLMs) can classify societal vulnerability and resilience indicators from extreme weather narratives. Extensive experiments across sparse, dense, and re-ranking retrievers, as well as a diverse set of LLMs, reveal that dense retrievers often fail on historical terminology, while LLMs frequently misinterpret vulnerability and resilience concepts. These findings highlight key limitations in reasoning about complex societal indicators and provide insights for designing more robust climate-focused RAG systems from archival contexts. The constructed dataset and evaluation framework are publicly available at this https URL. 

**Abstract (ZH)**: 历史天气档案作为持久的一手资料集合，提供了丰富的叙事，揭示了社会各界如何经历和应对极端天气事件。这些定性的记录为气候科学家理解社会响应提供了深刻的见解，但这些见解在气象记录中几乎是不存在的，使其对气候研究具有重要价值。然而，由于其庞大的规模、嘈杂的数字化质量和古老的语言，使其难以转换为结构化的知识用于气候研究。为解决这一挑战，我们引入了WeatherArchive-Bench，这是首个用于评估检索增强生成（RAG）系统在历史天气档案应用中的基准。WeatherArchive-Bench 包含两个任务：WeatherArchive-Retrieval，评估系统从超过一百万份档案新闻片段中找到相关历史段落的能力；以及WeatherArchive-Assessment，评估大型语言模型（LLMs）是否能够从极端天气叙事中分类出社会脆弱性和韧性的指标。广泛的实验表明，密集检索器在处理历史术语时经常失败，而大型语言模型频繁错误地解释脆弱性和韧性概念。这些发现强调了在复杂社会指标推理方面的重要限制，并为从档案视角设计更为稳健的气候聚焦RAG系统提供了指导。构建的数据集和评估框架可在<此链接>公开获取。 

---
