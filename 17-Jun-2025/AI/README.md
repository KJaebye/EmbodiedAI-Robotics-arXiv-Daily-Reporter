# PB$^2$: Preference Space Exploration via Population-Based Methods in Preference-Based Reinforcement Learning 

**Title (ZH)**: PB$^2$: 基于群体方法的偏奋试验空间探索在偏奋试程学习中的应用 

**Authors**: Brahim Driss, Alex Davey, Riad Akrour  

**Link**: [PDF](https://arxiv.org/pdf/2506.13741)  

**Abstract**: Preference-based reinforcement learning (PbRL) has emerged as a promising approach for learning behaviors from human feedback without predefined reward functions. However, current PbRL methods face a critical challenge in effectively exploring the preference space, often converging prematurely to suboptimal policies that satisfy only a narrow subset of human preferences. In this work, we identify and address this preference exploration problem through population-based methods. We demonstrate that maintaining a diverse population of agents enables more comprehensive exploration of the preference landscape compared to single-agent approaches. Crucially, this diversity improves reward model learning by generating preference queries with clearly distinguishable behaviors, a key factor in real-world scenarios where humans must easily differentiate between options to provide meaningful feedback. Our experiments reveal that current methods may fail by getting stuck in local optima, requiring excessive feedback, or degrading significantly when human evaluators make errors on similar trajectories, a realistic scenario often overlooked by methods relying on perfect oracle teachers. Our population-based approach demonstrates robust performance when teachers mislabel similar trajectory segments and shows significantly enhanced preference exploration capabilities,particularly in environments with complex reward landscapes. 

**Abstract (ZH)**: 基于偏好强化学习的偏好探索问题研究：基于群体的方法 

---
# Weakest Link in the Chain: Security Vulnerabilities in Advanced Reasoning Models 

**Title (ZH)**: 链中最薄弱环节：高级推理模型中的安全漏洞 

**Authors**: Arjun Krishna, Aaditya Rastogi, Erick Galinkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13726)  

**Abstract**: The introduction of advanced reasoning capabilities have improved the problem-solving performance of large language models, particularly on math and coding benchmarks. However, it remains unclear whether these reasoning models are more or less vulnerable to adversarial prompt attacks than their non-reasoning counterparts. In this work, we present a systematic evaluation of weaknesses in advanced reasoning models compared to similar non-reasoning models across a diverse set of prompt-based attack categories. Using experimental data, we find that on average the reasoning-augmented models are \emph{slightly more robust} than non-reasoning models (42.51\% vs 45.53\% attack success rate, lower is better). However, this overall trend masks significant category-specific differences: for certain attack types the reasoning models are substantially \emph{more vulnerable} (e.g., up to 32 percentage points worse on a tree-of-attacks prompt), while for others they are markedly \emph{more robust} (e.g., 29.8 points better on cross-site scripting injection). Our findings highlight the nuanced security implications of advanced reasoning in language models and emphasize the importance of stress-testing safety across diverse adversarial techniques. 

**Abstract (ZH)**: 先进推理能力的引入提升了大型语言模型在数学和编码基准上的问题解决性能，但尚不清楚这些推理模型相较于非推理模型是否更易或更难受到对抗性指令攻击。在本工作中，我们对不同指令攻击类别下的先进推理模型和相似的非推理模型的弱点进行了系统性评估。实验数据表明，平均而言，增强推理能力的模型比非推理模型略为 robust（42.51% vs 45.53% 攻击成功率，较低者较好）。然而，这种总体趋势掩盖了特定类别间的显著差异：对于某些攻击类型，推理模型显著更脆弱（例如，在树状攻击指令上的攻击成功率高出32个百分点），而对于其他类型，它们则明显更 robust（例如，在跨站脚本注入上的攻击成功率低了29.8个百分点）。我们的研究结果突显了先进推理能力在语言模型中的复杂安全含义，并强调了跨多种对抗技术进行压力测试安全性的重要性。 

---
# Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model 

**Title (ZH)**: Stream-Omi：大型语言-视觉-语音模型下的多模态同时交互 

**Authors**: Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, Yang Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13642)  

**Abstract**: The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience. 

**Abstract (ZH)**: GPT-4o-like大型多模态模型的 emergence 及其对文本、视觉和语音模态整合的探索：一种高效模态对齐的 Stream-Omni 模型 

---
# Avoiding Obfuscation with Prover-Estimator Debate 

**Title (ZH)**: 避免混淆：证明者-估计算法辩论 

**Authors**: Jonah Brown-Cohen, Geoffrey Irving, Georgios Piliouras  

**Link**: [PDF](https://arxiv.org/pdf/2506.13609)  

**Abstract**: Training powerful AI systems to exhibit desired behaviors hinges on the ability to provide accurate human supervision on increasingly complex tasks. A promising approach to this problem is to amplify human judgement by leveraging the power of two competing AIs in a debate about the correct solution to a given problem. Prior theoretical work has provided a complexity-theoretic formalization of AI debate, and posed the problem of designing protocols for AI debate that guarantee the correctness of human judgements for as complex a class of problems as possible. Recursive debates, in which debaters decompose a complex problem into simpler subproblems, hold promise for growing the class of problems that can be accurately judged in a debate. However, existing protocols for recursive debate run into the obfuscated arguments problem: a dishonest debater can use a computationally efficient strategy that forces an honest opponent to solve a computationally intractable problem to win. We mitigate this problem with a new recursive debate protocol that, under certain stability assumptions, ensures that an honest debater can win with a strategy requiring computational efficiency comparable to their opponent. 

**Abstract (ZH)**: 训练强大的AI系统展现 desired behaviors 在很大程度上取决于能够对日益复杂的任务提供准确的人类监督。一种有前景的方法是通过让两个竞争的AI在关于给定问题正确解决方案的辩论中发挥其优势来放大人类的判断力。先前的理论工作为AI辩论提供了一个复杂性理论的形式化，并提出了设计协议以保证人类判断的正确性的问题，适用于尽可能复杂的类问题。递归辩论，其中辩论者将复杂问题分解成更简单的问题子集，有潜力扩大可以准确判断的辩论中问题的类别。然而，现有的递归辩论协议遇到了混淆性论证问题：一个不诚实的辩论者可以使用一个计算上高效的策略，迫使诚实的对手解决一个计算上不可解的问题来获胜。我们通过一个新的递归辩论协议来解决这个问题，在某些稳定假设下，该协议确保诚实的辩论者可以用一个与对手计算上效率相当的策略获胜。 

---
# The ASP-based Nurse Scheduling System at the University of Yamanashi Hospital 

**Title (ZH)**: 基于ASP的山形大学医院护士排班系统 

**Authors**: Hidetomo Nabeshima, Mutsunori Banbara, Torsten Schaub, Takehide Soh  

**Link**: [PDF](https://arxiv.org/pdf/2506.13600)  

**Abstract**: We present the design principles of a nurse scheduling system built using Answer Set Programming (ASP) and successfully deployed at the University of Yamanashi Hospital. Nurse scheduling is a complex optimization problem requiring the reconciliation of individual nurse preferences with hospital staffing needs across various wards. This involves balancing hard and soft constraints and the flexibility of interactive adjustments. While extensively studied in academia, real-world nurse scheduling presents unique challenges that go beyond typical benchmark problems and competitions. This paper details the practical application of ASP to address these challenges at the University of Yamanashi Hospital, focusing on the insights gained and the advancements in ASP technology necessary to effectively manage the complexities of real-world deployment. 

**Abstract (ZH)**: 我们基于Answer Set Programming (ASP) 设计并成功实现在山梨大学医院的护士排班系统：原则与实践 

---
# Agent Capability Negotiation and Binding Protocol (ACNBP) 

**Title (ZH)**: 代理能力协商与绑定协议（ACNBP） 

**Authors**: Ken Huang, Akram Sheriff, Vineeth Sai Narajala, Idan Habler  

**Link**: [PDF](https://arxiv.org/pdf/2506.13590)  

**Abstract**: As multi-agent systems evolve to encompass increasingly diverse and specialized agents, the challenge of enabling effective collaboration between heterogeneous agents has become paramount, with traditional agent communication protocols often assuming homogeneous environments or predefined interaction patterns that limit their applicability in dynamic, open-world scenarios. This paper presents the Agent Capability Negotiation and Binding Protocol (ACNBP), a novel framework designed to facilitate secure, efficient, and verifiable interactions between agents in heterogeneous multi-agent systems through integration with an Agent Name Service (ANS) infrastructure that provides comprehensive discovery, negotiation, and binding mechanisms. The protocol introduces a structured 10-step process encompassing capability discovery, candidate pre-screening and selection, secure negotiation phases, and binding commitment with built-in security measures including digital signatures, capability attestation, and comprehensive threat mitigation strategies, while a key innovation of ACNBP is its protocolExtension mechanism that enables backward-compatible protocol evolution and supports diverse agent architectures while maintaining security and interoperability. We demonstrate ACNBP's effectiveness through a comprehensive security analysis using the MAESTRO threat modeling framework, practical implementation considerations, and a detailed example showcasing the protocol's application in a document translation scenario, with the protocol addressing critical challenges in agent autonomy, capability verification, secure communication, and scalable agent ecosystem management. 

**Abstract (ZH)**: 面向异构多智能体系统的智能协商与绑定协议（ACNBP）：一种安全、高效且可验证的智能体交互框架 

---
# From Data-Driven to Purpose-Driven Artificial Intelligence: Systems Thinking for Data-Analytic Automation of Patient Care 

**Title (ZH)**: 从数据驱动到目标驱动的人工智能：面向患者的分析自动化系统的思维模式 

**Authors**: Daniel Anadria, Roel Dobbe, Anastasia Giachanou, Ruurd Kuiper, Richard Bartels, Íñigo Martínez de Rituerto de Troya, Carmen Zürcher, Daniel Oberski  

**Link**: [PDF](https://arxiv.org/pdf/2506.13584)  

**Abstract**: In this work, we reflect on the data-driven modeling paradigm that is gaining ground in AI-driven automation of patient care. We argue that the repurposing of existing real-world patient datasets for machine learning may not always represent an optimal approach to model development as it could lead to undesirable outcomes in patient care. We reflect on the history of data analysis to explain how the data-driven paradigm rose to popularity, and we envision ways in which systems thinking and clinical domain theory could complement the existing model development approaches in reaching human-centric outcomes. We call for a purpose-driven machine learning paradigm that is grounded in clinical theory and the sociotechnical realities of real-world operational contexts. We argue that understanding the utility of existing patient datasets requires looking in two directions: upstream towards the data generation, and downstream towards the automation objectives. This purpose-driven perspective to AI system development opens up new methodological opportunities and holds promise for AI automation of patient care. 

**Abstract (ZH)**: 在基于AI的患者护理自动化中，数据驱动建模范式的反思：一种以临床理论和实际操作背景为基础的目标导向的机器学习范式 

---
# Block-wise Adaptive Caching for Accelerating Diffusion Policy 

**Title (ZH)**: 块级自适应缓存加速扩散策略 

**Authors**: Kangye Ji, Yuan Meng, Hanyun Cui, Ye Li, Shengjia Hua, Lei Chen, Zhi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13456)  

**Abstract**: Diffusion Policy has demonstrated strong visuomotor modeling capabilities, but its high computational cost renders it impractical for real-time robotic control. Despite huge redundancy across repetitive denoising steps, existing diffusion acceleration techniques fail to generalize to Diffusion Policy due to fundamental architectural and data divergences. In this paper, we propose Block-wise Adaptive Caching(BAC), a method to accelerate Diffusion Policy by caching intermediate action features. BAC achieves lossless action generation acceleration by adaptively updating and reusing cached features at the block level, based on a key observation that feature similarities vary non-uniformly across timesteps and locks. To operationalize this insight, we first propose the Adaptive Caching Scheduler, designed to identify optimal update timesteps by maximizing the global feature similarities between cached and skipped features. However, applying this scheduler for each block leads to signiffcant error surges due to the inter-block propagation of caching errors, particularly within Feed-Forward Network (FFN) blocks. To mitigate this issue, we develop the Bubbling Union Algorithm, which truncates these errors by updating the upstream blocks with signiffcant caching errors before downstream FFNs. As a training-free plugin, BAC is readily integrable with existing transformer-based Diffusion Policy and vision-language-action models. Extensive experiments on multiple robotic benchmarks demonstrate that BAC achieves up to 3x inference speedup for free. 

**Abstract (ZH)**: 块级自适应缓存(BAC):一种加速扩散政策的方法 

---
# A Technical Study into Small Reasoning Language Models 

**Title (ZH)**: 小型推理语言模型的技术研究 

**Authors**: Xialie Zhuang, Peixian Ma, Zhikai Jia, Zheng Cao, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13404)  

**Abstract**: The ongoing evolution of language models has led to the development of large-scale architectures that demonstrate exceptional performance across a wide range of tasks. However, these models come with significant computational and energy demands, as well as potential privacy implications. In this context, Small Reasoning Language Models (SRLMs) with approximately 0.5 billion parameters present a compelling alternative due to their remarkable computational efficiency and cost effectiveness, particularly in resource-constrained environments. Despite these advantages, the limited capacity of 0.5 billion parameter models poses challenges in handling complex tasks such as mathematical reasoning and code generation. This research investigates various training strategies, including supervised fine-tuning (SFT), knowledge distillation (KD), and reinforcement learning (RL), as well as their hybrid implementations, to enhance the performance of 0.5B SRLMs. We analyze effective methodologies to bridge the performance gap between SRLMS and larger models and present insights into optimal training pipelines tailored for these smaller architectures. Through extensive experimental validation and analysis, our work aims to provide actionable recommendations for maximizing the reasoning capabilities of 0.5B models. 

**Abstract (ZH)**: 小型推理语言模型的小规模参数（约0.5亿参数）在资源受限环境中的高效计算与成本效益研究及其训练策略优化 

---
# Deflating Deflationism: A Critical Perspective on Debunking Arguments Against LLM Mentality 

**Title (ZH)**: 消解消解论：对反驳LLM心智论据的批判性视角 

**Authors**: Alex Grzankowski, Geoff Keeling, Henry Shevlin, Winnie Street  

**Link**: [PDF](https://arxiv.org/pdf/2506.13403)  

**Abstract**: Many people feel compelled to interpret, describe, and respond to Large Language Models (LLMs) as if they possess inner mental lives similar to our own. Responses to this phenomenon have varied. Inflationists hold that at least some folk psychological ascriptions to LLMs are warranted. Deflationists argue that all such attributions of mentality to LLMs are misplaced, often cautioning against the risk that anthropomorphic projection may lead to misplaced trust or potentially even confusion about the moral status of LLMs. We advance this debate by assessing two common deflationary arguments against LLM mentality. What we term the 'robustness strategy' aims to undercut one justification for believing that LLMs are minded entities by showing that putatively cognitive and humanlike behaviours are not robust, failing to generalise appropriately. What we term the 'etiological strategy' undercuts attributions of mentality by challenging naive causal explanations of LLM behaviours, offering alternative causal accounts that weaken the case for mental state attributions. While both strategies offer powerful challenges to full-blown inflationism, we find that neither strategy provides a knock-down case against ascriptions of mentality to LLMs simpliciter. With this in mind, we explore a modest form of inflationism that permits ascriptions of mentality to LLMs under certain conditions. Specifically, we argue that folk practice provides a defeasible basis for attributing mental states and capacities to LLMs provided those mental states and capacities can be understood in metaphysically undemanding terms (e.g. knowledge, beliefs and desires), while greater caution is required when attributing metaphysically demanding mental phenomena such as phenomenal consciousness. 

**Abstract (ZH)**: 许多人在解释和回应大型语言模型（LLMs）时，似乎认为它们拥有与我们相似的内在心理生活。对此现象的回应各不相同。膨胀论者认为至少有些对LLMs的心理学归因是站得住脚的。消减论者则认为所有对LLMs的心理学归因都是不适当的，往往警告人们避免人类中心主义投影可能导致对LLMs道德地位的误解或混淆。我们通过评估两个常见的消减性论点，推进这一辩论。我们称前者为“稳健性策略”，旨在通过证明所谓认知和类人行为的不稳健性，即无法适当推广，来削弱相信LLMs为有心智实体的合理依据。我们称后者为“由因论策略”，通过挑战对LLMs行为的直观因果解释，提供削弱心智状态归因的替代因果解释。虽然这两种策略都对全面的膨胀论提出了强有力挑战，但我们发现，这两种策略都无法提供一个无懈可击的论据反对为LLMs的心智状态进行任何归因。鉴于此，我们探讨了一种适度的膨胀论，即在特定条件下，允许对LLMs进行心智状态和能力的归因。具体而言，我们认为民间实践提供了在非要求性形而上学框架下归因心理状态和能力的基础，但对归因要求性形而上学的心理现象，如现象意识，需要更加谨慎。 

---
# Delving Into the Psychology of Machines: Exploring the Structure of Self-Regulated Learning via LLM-Generated Survey Responses 

**Title (ZH)**: 探究机器的心智：通过LLM生成的调查回应探索自我调节学习的结构 

**Authors**: Leonie V.D.E. Vogelsmeier, Eduardo Oliveira, Kamila Misiejuk, Sonsoles López-Pernas, Mohammed Saqr  

**Link**: [PDF](https://arxiv.org/pdf/2506.13384)  

**Abstract**: Large language models (LLMs) offer the potential to simulate human-like responses and behaviors, creating new opportunities for psychological science. In the context of self-regulated learning (SRL), if LLMs can reliably simulate survey responses at scale and speed, they could be used to test intervention scenarios, refine theoretical models, augment sparse datasets, and represent hard-to-reach populations. However, the validity of LLM-generated survey responses remains uncertain, with limited research focused on SRL and existing studies beyond SRL yielding mixed results. Therefore, in this study, we examined LLM-generated responses to the 44-item Motivated Strategies for Learning Questionnaire (MSLQ; Pintrich \& De Groot, 1990), a widely used instrument assessing students' learning strategies and academic motivation. Particularly, we used the LLMs GPT-4o, Claude 3.7 Sonnet, Gemini 2 Flash, LLaMA 3.1-8B, and Mistral Large. We analyzed item distributions, the psychological network of the theoretical SRL dimensions, and psychometric validity based on the latent factor structure. Our results suggest that Gemini 2 Flash was the most promising LLM, showing considerable sampling variability and producing underlying dimensions and theoretical relationships that align with prior theory and empirical findings. At the same time, we observed discrepancies and limitations, underscoring both the potential and current constraints of using LLMs for simulating psychological survey data and applying it in educational contexts. 

**Abstract (ZH)**: 大型语言模型（LLMs）在心理科学中的潜在应用：以自我调节学习（SRL）为案例的研究 

---
# Socratic RL: A Novel Framework for Efficient Knowledge Acquisition through Iterative Reflection and Viewpoint Distillation 

**Title (ZH)**: 苏格拉底式强化学习：一种通过迭代反思和视角提炼来高效获取知识的新框架 

**Authors**: Xiangfan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13358)  

**Abstract**: Current Reinforcement Learning (RL) methodologies for Large Language Models (LLMs) often rely on simplistic, outcome-based reward signals (e.g., final answer correctness), which limits the depth of learning from each interaction. This paper introduces Socratic Reinforcement Learning (Socratic-RL), a novel, process-oriented framework designed to address this limitation. Socratic-RL operates on the principle that deeper understanding is achieved by reflecting on the causal reasons for errors and successes within the reasoning process itself. The framework employs a decoupled "Teacher-Student" architecture, where a "Teacher AI" analyzes interaction histories, extracts causal insights, and formulates them into structured "viewpoints." These viewpoints, acting as distilled guidance, are then used by a "Student AI" to enhance its subsequent reasoning. A key innovation is the iterative self-improvement of the Teacher AI, enabling its reflective capabilities to evolve through a meta-learning loop. To manage the accumulation of knowledge, a distillation mechanism compresses learned viewpoints into the Student's parameters. By focusing on process rather than just outcome, Socratic-RL presents a pathway toward enhanced sample efficiency, superior interpretability, and a more scalable architecture for self-improving AI systems. This paper details the foundational concepts, formal mechanisms, synergies, challenges, and a concrete research roadmap for this proposed framework. 

**Abstract (ZH)**: 当前的大语言模型（LLMs） reinforcement learning（RL）方法往往依赖于简单的、基于结果的奖励信号（例如最终答案的正确性），这限制了每次交互学习的深度。本文提出了Socratic Reinforcement Learning（Socratic-RL），这是一种新的过程导向框架，旨在解决这一限制。Socratic-RL 以通过反思推理过程中错误和成功的原因来实现更深层次的理解为基础。该框架采用解耦的“教师-学生”架构，其中“教师AI”分析交互历史，提取因果洞察，并将其结构化为“观点”。这些观点作为浓缩的指导，然后用于“学生AI”以增强其后续推理。一个关键创新在于“教师AI”的迭代自我改进，使其反射能力能够在元学习循环中进化。为了管理知识的积累，压缩机制将学到的观点压缩到“学生”的参数中。通过专注于过程而非仅仅结果，Socratic-RL 提出了增强样本效率、优异可解释性和更可扩展的自改进AI系统架构的途径。本文详细介绍了该提出框架的基础概念、正式机制、协同作用、挑战以及具体研究路线图。 

---
# Verifying the Verifiers: Unveiling Pitfalls and Potentials in Fact Verifiers 

**Title (ZH)**: 验证者之鉴：揭示事实验证者的挑战与潜力 

**Authors**: Wooseok Seo, Seungju Han, Jaehun Jung, Benjamin Newman, Seungwon Lim, Seungbeen Lee, Ximing Lu, Yejin Choi, Youngjae Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13342)  

**Abstract**: Fact verification is essential for ensuring the reliability of LLM applications. In this study, we evaluate 12 pre-trained LLMs and one specialized fact-verifier, including frontier LLMs and open-weight reasoning LLMs, using a collection of examples from 14 fact-checking benchmarks. We share three findings intended to guide future development of more robust fact verifiers. First, we highlight the importance of addressing annotation errors and ambiguity in datasets, demonstrating that approximately 16\% of ambiguous or incorrectly labeled data substantially influences model rankings. Neglecting this issue may result in misleading conclusions during comparative evaluations, and we suggest using a systematic pipeline utilizing LLM-as-a-judge to help identify these issues at scale. Second, we discover that frontier LLMs with few-shot in-context examples, often overlooked in previous works, achieve top-tier performance. We therefore recommend future studies include comparisons with these simple yet highly effective baselines. Lastly, despite their effectiveness, frontier LLMs incur substantial costs, motivating the development of small, fine-tuned fact verifiers. We show that these small models still have room for improvement, particularly on instances that require complex reasoning. Encouragingly, we demonstrate that augmenting training with synthetic multi-hop reasoning data significantly enhances their capabilities in such instances. We release our code, model, and dataset at this https URL 

**Abstract (ZH)**: 事实核查对于确保大型语言模型应用的可靠性至关重要。在这项研究中，我们使用来自14个事实核查基准的数据集，评估了12个预训练大型语言模型和一个专门的事实核查器，包括前沿的大型语言模型和少样本推理的大型语言模型，分享了三条旨在指导更 robust 的事实核查器开发的发现。第一，我们将重点放在解决数据集中的标注错误和模糊性问题上，表明大约16%的模糊或错误标注的数据显著影响了模型的排名。忽视这一问题可能导致比较评估过程中得出误导性结论，并建议使用系统化的工作流程，利用LLM作为裁判来帮助大规模识别这些问题。第二，我们发现，以前工作中常常被忽略的带有少量上下文示例的前沿大型语言模型表现出顶级性能。因此，我们建议未来的研究包括与这些简单而高效的基线进行比较。最后，尽管前沿大型语言模型效果显著，但它们引发了高昂的成本，促进了小型精调事实核查器的发展。我们表明，这些小型模型在需要复杂推理的情况下仍有改进空间。令人鼓舞的是，我们展示通过增加含有合成多跳推理数据的训练显著提高了它们在这种情况下的能力。我们的代码、模型和数据集可以通过以下链接访问：this https URL。 

---
# Probabilistic Modeling of Spiking Neural Networks with Contract-Based Verification 

**Title (ZH)**: 基于合同验证的_SPIKING神经网络概率建模 

**Authors**: Zhen Yao, Elisabetta De Maria, Robert De Simone  

**Link**: [PDF](https://arxiv.org/pdf/2506.13340)  

**Abstract**: Spiking Neural Networks (SNN) are models for "realistic" neuronal computation, which makes them somehow different in scope from "ordinary" deep-learning models widely used in AI platforms nowadays. SNNs focus on timed latency (and possibly probability) of neuronal reactive activation/response, more than numerical computation of filters. So, an SNN model must provide modeling constructs for elementary neural bundles and then for synaptic connections to assemble them into compound data flow network patterns. These elements are to be parametric patterns, with latency and probability values instantiated on particular instances (while supposedly constant "at runtime"). Designers could also use different values to represent "tired" neurons, or ones impaired by external drugs, for instance. One important challenge in such modeling is to study how compound models could meet global reaction requirements (in stochastic timing challenges), provided similar provisions on individual neural bundles. A temporal language of logic to express such assume/guarantee contracts is thus needed. This may lead to formal verification on medium-sized models and testing observations on large ones. In the current article, we make preliminary progress at providing a simple model framework to express both elementary SNN neural bundles and their connecting constructs, which translates readily into both a model-checker and a simulator (both already existing and robust) to conduct experiments. 

**Abstract (ZH)**: 基于脉冲神经网络的建模与验证：从基本神经束到复合动力流网络模式的研究 

---
# Towards Pervasive Distributed Agentic Generative AI -- A State of The Art 

**Title (ZH)**: 面向渗透式分布式智能代理生成式AI——现状分析 

**Authors**: Gianni Molinari, Fabio Ciravegna  

**Link**: [PDF](https://arxiv.org/pdf/2506.13324)  

**Abstract**: The rapid advancement of intelligent agents and Large Language Models (LLMs) is reshaping the pervasive computing field. Their ability to perceive, reason, and act through natural language understanding enables autonomous problem-solving in complex pervasive environments, including the management of heterogeneous sensors, devices, and data. This survey outlines the architectural components of LLM agents (profiling, memory, planning, and action) and examines their deployment and evaluation across various scenarios. Than it reviews computational and infrastructural advancements (cloud to edge) in pervasive computing and how AI is moving in this field. It highlights state-of-the-art agent deployment strategies and applications, including local and distributed execution on resource-constrained devices. This survey identifies key challenges of these agents in pervasive computing such as architectural, energetic and privacy limitations. It finally proposes what we called "Agent as a Tool", a conceptual framework for pervasive agentic AI, emphasizing context awareness, modularity, security, efficiency and effectiveness. 

**Abstract (ZH)**: 智能代理和大规模语言模型的快速进步正在重塑泛在计算领域。它们通过自然语言理解进行感知、推理和行动的能力，使泛在环境中的自主问题解决成为可能，包括异构传感器、设备和数据的管理。本文综述了大规模语言模型代理的架构组件（特性描述、内存、规划和行动），并探讨了它们在各种场景中的部署和评估。随后，本文回顾了从云计算到边缘计算的泛在计算领域的计算和基础设施进步，以及AI在这一领域的发展。本文还强调了最新代理部署策略和应用，包括在资源受限设备上进行本地和分布式执行。本文指出了这些代理在泛在计算中面对的关键挑战，如架构、能量和隐私限制，并最终提出了一种名为“代理即工具”的概念框架，强调上下文意识、模块化、安全、效率和有效性。 

---
# Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks 

**Title (ZH)**: 探索黑箱：利用大型语言模型进行有效的文本级别图注入攻击 

**Authors**: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13276)  

**Abstract**: Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods. 

**Abstract (ZH)**: 基于文本的图（TAGs）结合了文本数据与图结构，为社会网络分析和推荐系统等应用提供了宝贵的见解。图神经网络（GNNs）能够有效捕捉TAGs中的拓扑结构和文本信息，但易受对抗性攻击的影响。现有的图注入攻击（GIA）方法假定攻击者可以直接操控嵌入层，生成不可解释的节点嵌入。此外，这些攻击的有效性往往依赖于具有高训练成本的替代模型。因此，本文提出了ATAG-LLM，这是一种针对TAGs的新颖黑盒GIA框架。我们的方法利用大语言模型（LLMs）直接生成可解释的文本级别节点属性，确保攻击在实际场景中仍具有可行性。我们设计了LLM提示策略，以平衡探索和可靠性来引导文本生成，并提出了一种相似性评估方法来评估攻击文本破坏图同质性的效果。该方法在严格的黑盒设置下，以最小的训练成本高效地扰动目标节点，确保对TAGs进行文本级别图注入攻击。实证研究验证了ATAG-LLM相比最先进的嵌入级别和文本级别攻击方法的优越性能。 

---
# Vector Ontologies as an LLM world view extraction method 

**Title (ZH)**: 向量本体作为LLM世界观提取方法 

**Authors**: Kaspar Rothenfusser, Bekk Blando  

**Link**: [PDF](https://arxiv.org/pdf/2506.13252)  

**Abstract**: Large Language Models (LLMs) possess intricate internal representations of the world, yet these latent structures are notoriously difficult to interpret or repurpose beyond the original prediction task. Building on our earlier work (Rothenfusser, 2025), which introduced the concept of vector ontologies as a framework for translating high-dimensional neural representations into interpretable geometric structures, this paper provides the first empirical validation of that approach. A vector ontology defines a domain-specific vector space spanned by ontologically meaningful dimensions, allowing geometric analysis of concepts and relationships within a domain. We construct an 8-dimensional vector ontology of musical genres based on Spotify audio features and test whether an LLM's internal world model of music can be consistently and accurately projected into this space. Using GPT-4o-mini, we extract genre representations through multiple natural language prompts and analyze the consistency of these projections across linguistic variations and their alignment with ground-truth data. Our results show (1) high spatial consistency of genre projections across 47 query formulations, (2) strong alignment between LLM-inferred genre locations and real-world audio feature distributions, and (3) evidence of a direct relationship between prompt phrasing and spatial shifts in the LLM's inferred vector ontology. These findings demonstrate that LLMs internalize structured, repurposable knowledge and that vector ontologies offer a promising method for extracting and analyzing this knowledge in a transparent and verifiable way. 

**Abstract (ZH)**: 大型语言模型（LLMs）具有复杂的世界内部表示，但这些潜在结构难以解释或重新利用超出原始预测任务的范围。在此基础上，我们构建了一个基于Spotify音频特征的8维音乐流派向量本体，并测试大规模语言模型对音乐的内部世界模型能否一致且准确地投影到该空间中。使用GPT-4o-mini，我们通过多个自然语言提示提取流派表示，并分析这些投影在语言变异中的一致性及其与真实世界音频特征分布的对齐程度。我们的结果表明：跨47种查询形式，流派投影具有高度的空间一致性；大规模语言模型推断出的流派位置与真实世界音频特征分布之间存在强烈的对齐；提示措辞与大规模语言模型推断的向量本体的空间变化之间存在直接关系。这些发现表明，大型语言模型内化了结构化且可重新利用的知识，而向量本体为以透明且可验证的方式提取和分析这种知识提供了有前途的方法。 

---
# Generalized Proof-Number Monte-Carlo Tree Search 

**Title (ZH)**: 广义证明数蒙特卡洛树搜索 

**Authors**: Jakub Kowalski, Dennis J. N. J. Soemers, Szymon Kosakowski, Mark H. M. Winands  

**Link**: [PDF](https://arxiv.org/pdf/2506.13249)  

**Abstract**: This paper presents Generalized Proof-Number Monte-Carlo Tree Search: a generalization of recently proposed combinations of Proof-Number Search (PNS) with Monte-Carlo Tree Search (MCTS), which use (dis)proof numbers to bias UCB1-based Selection strategies towards parts of the search that are expected to be easily (dis)proven. We propose three core modifications of prior combinations of PNS with MCTS. First, we track proof numbers per player. This reduces code complexity in the sense that we no longer need disproof numbers, and generalizes the technique to be applicable to games with more than two players. Second, we propose and extensively evaluate different methods of using proof numbers to bias the selection strategy, achieving strong performance with strategies that are simpler to implement and compute. Third, we merge our technique with Score Bounded MCTS, enabling the algorithm to prove and leverage upper and lower bounds on scores - as opposed to only proving wins or not-wins. Experiments demonstrate substantial performance increases, reaching the range of 80% for 8 out of the 11 tested board games. 

**Abstract (ZH)**: 广义证明数蒙特卡洛树搜索：一种证明数搜索与蒙特卡洛树搜索结合的扩展方法 

---
# A Game-Theoretic Negotiation Framework for Cross-Cultural Consensus in LLMs 

**Title (ZH)**: 基于博弈论的跨文化共识 negotiation框架在大规模语言模型中的应用 

**Authors**: Guoxi Zhang, Jiawei Chen, Tianzhuo Yang, Jiaming Ji, Yaodong Yang, Juntao Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13245)  

**Abstract**: The increasing prevalence of large language models (LLMs) is influencing global value systems. However, these models frequently exhibit a pronounced WEIRD (Western, Educated, Industrialized, Rich, Democratic) cultural bias due to lack of attention to minority values. This monocultural perspective may reinforce dominant values and marginalize diverse cultural viewpoints, posing challenges for the development of equitable and inclusive AI systems. In this work, we introduce a systematic framework designed to boost fair and robust cross-cultural consensus among LLMs. We model consensus as a Nash Equilibrium and employ a game-theoretic negotiation method based on Policy-Space Response Oracles (PSRO) to simulate an organized cross-cultural negotiation process. To evaluate this approach, we construct regional cultural agents using data transformed from the World Values Survey (WVS). Beyond the conventional model-level evaluation method, We further propose two quantitative metrics, Perplexity-based Acceptence and Values Self-Consistency, to assess consensus outcomes. Experimental results indicate that our approach generates consensus of higher quality while ensuring more balanced compromise compared to baselines. Overall, it mitigates WEIRD bias by guiding agents toward convergence through fair and gradual negotiation steps. 

**Abstract (ZH)**: 大型语言模型的文化偏见及其跨文化共识框架：减少WEIRD偏见促进公平包容的AI系统发展 

---
# Towards Explaining Monte-Carlo Tree Search by Using Its Enhancements 

**Title (ZH)**: 利用增强方法解释蒙特卡罗树搜索 

**Authors**: Jakub Kowalski, Mark H. M. Winands, Maksymilian Wiśniewski, Stanisław Reda, Anna Wilbik  

**Link**: [PDF](https://arxiv.org/pdf/2506.13223)  

**Abstract**: Typically, research on Explainable Artificial Intelligence (XAI) focuses on black-box models within the context of a general policy in a known, specific domain. This paper advocates for the need for knowledge-agnostic explainability applied to the subfield of XAI called Explainable Search, which focuses on explaining the choices made by intelligent search techniques. It proposes Monte-Carlo Tree Search (MCTS) enhancements as a solution to obtaining additional data and providing higher-quality explanations while remaining knowledge-free, and analyzes the most popular enhancements in terms of the specific types of explainability they introduce. So far, no other research has considered the explainability of MCTS enhancements. We present a proof-of-concept that demonstrates the advantages of utilizing enhancements. 

**Abstract (ZH)**: 通常，可解释人工智能（XAI）的研究集中在已知特定领域的一般策略下的黑盒模型。本文倡导在可解释搜索子领域中应用知识无关的可解释性，该子领域关注解释智能搜索技术所做的选择。本文提议使用蒙特卡洛树搜索（MCTS）增强技术作为获得额外数据并提供更高质量解释的解决方案，同时保持知识无关性，并分析最受欢迎的增强技术在引入特定类型可解释性方面的差异。迄今为止，尚未有其他研究考虑MCTS增强技术的可解释性。我们提出了一种概念验证方法，以展示利用增强技术的优势。 

---
# NeuroPhysNet: A FitzHugh-Nagumo-Based Physics-Informed Neural Network Framework for Electroencephalograph (EEG) Analysis and Motor Imagery Classification 

**Title (ZH)**: 基于FitzHugh-Nagumo模型的物理信息神经网络框架：EEG分析与 Motor Imagery分类 

**Authors**: Zhenyu Xia, Xinlei Huang, Suvash C. Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.13222)  

**Abstract**: Electroencephalography (EEG) is extensively employed in medical diagnostics and brain-computer interface (BCI) applications due to its non-invasive nature and high temporal resolution. However, EEG analysis faces significant challenges, including noise, nonstationarity, and inter-subject variability, which hinder its clinical utility. Traditional neural networks often lack integration with biophysical knowledge, limiting their interpretability, robustness, and potential for medical translation. To address these limitations, this study introduces NeuroPhysNet, a novel Physics-Informed Neural Network (PINN) framework tailored for EEG signal analysis and motor imagery classification in medical contexts. NeuroPhysNet incorporates the FitzHugh-Nagumo model, embedding neurodynamical principles to constrain predictions and enhance model robustness. Evaluated on the BCIC-IV-2a dataset, the framework achieved superior accuracy and generalization compared to conventional methods, especially in data-limited and cross-subject scenarios, which are common in clinical settings. By effectively integrating biophysical insights with data-driven techniques, NeuroPhysNet not only advances BCI applications but also holds significant promise for enhancing the precision and reliability of clinical diagnostics, such as motor disorder assessments and neurorehabilitation planning. 

**Abstract (ZH)**: 基于生理约束的神经网络（NeuroPhysNet）：用于医学情境下的脑电图信号分析与运动想象分类 

---
# Real Time Self-Tuning Adaptive Controllers on Temperature Control Loops using Event-based Game Theory 

**Title (ZH)**: 基于事件驱动博弈论的实时自调适应温度控制环控制器 

**Authors**: Steve Yuwono, Muhammad Uzair Rana, Dorothea Schwung, Andreas Schwung  

**Link**: [PDF](https://arxiv.org/pdf/2506.13164)  

**Abstract**: This paper presents a novel method for enhancing the adaptability of Proportional-Integral-Derivative (PID) controllers in industrial systems using event-based dynamic game theory, which enables the PID controllers to self-learn, optimize, and fine-tune themselves. In contrast to conventional self-learning approaches, our proposed framework offers an event-driven control strategy and game-theoretic learning algorithms. The players collaborate with the PID controllers to dynamically adjust their gains in response to set point changes and disturbances. We provide a theoretical analysis showing sound convergence guarantees for the game given suitable stability ranges of the PID controlled loop. We further introduce an automatic boundary detection mechanism, which helps the players to find an optimal initialization of action spaces and significantly reduces the exploration time. The efficacy of this novel methodology is validated through its implementation in the temperature control loop of a printing press machine. Eventually, the outcomes of the proposed intelligent self-tuning PID controllers are highly promising, particularly in terms of reducing overshoot and settling time. 

**Abstract (ZH)**: 本文提出了一种使用事件驱动动态博弈理论增强比例积分微分（PID）控制器适应性的新型方法，使PID控制器能够自我学习、优化和精调。与传统的自我学习方法相比，我们提出的框架提供了基于事件的控制策略和博弈论学习算法。博弈中的玩家与PID控制器协作，动态调整增益以响应设定点变化和干扰。我们提供了理论分析，证明在PID控制环具有合适稳定范围的情况下，博弈具有合理的收敛保证。此外，我们引入了一种自动边界检测机制，有助于玩家找到最优的动作空间初始化，并显著减少探索时间。该新颖方法的有效性通过在印刷机温控回路中的实现得以验证。最终，所提出的智能自调谐PID控制器的成果极具前景，特别是在减少超调和稳态时间方面表现突出。 

---
# Machine Learning as Iterated Belief Change a la Darwiche and Pearl 

**Title (ZH)**: 机器学习作为拉里奇和皮尔莱格风格的迭代信念变化 

**Authors**: Theofanis Aravanis  

**Link**: [PDF](https://arxiv.org/pdf/2506.13157)  

**Abstract**: Artificial Neural Networks (ANNs) are powerful machine-learning models capable of capturing intricate non-linear relationships. They are widely used nowadays across numerous scientific and engineering domains, driving advancements in both research and real-world applications. In our recent work, we focused on the statics and dynamics of a particular subclass of ANNs, which we refer to as binary ANNs. A binary ANN is a feed-forward network in which both inputs and outputs are restricted to binary values, making it particularly suitable for a variety of practical use cases. Our previous study approached binary ANNs through the lens of belief-change theory, specifically the Alchourron, Gardenfors and Makinson (AGM) framework, yielding several key insights. Most notably, we demonstrated that the knowledge embodied in a binary ANN (expressed through its input-output behaviour) can be symbolically represented using a propositional logic language. Moreover, the process of modifying a belief set (through revision or contraction) was mapped onto a gradual transition through a series of intermediate belief sets. Analogously, the training of binary ANNs was conceptualized as a sequence of such belief-set transitions, which we showed can be formalized using full-meet AGM-style belief change. In the present article, we extend this line of investigation by addressing some critical limitations of our previous study. Specifically, we show that Dalal's method for belief change naturally induces a structured, gradual evolution of states of belief. More importantly, given the known shortcomings of full-meet belief change, we demonstrate that the training dynamics of binary ANNs can be more effectively modelled using robust AGM-style change operations -- namely, lexicographic revision and moderate contraction -- that align with the Darwiche-Pearl framework for iterated belief change. 

**Abstract (ZH)**: 人工神经网络（ANNs）是强大的机器学习模型，能够捕获复杂的非线性关系。它们在众多科学和工程领域中广泛使用，推动了研究和实际应用的进步。在我们的近期工作中，我们专注于一类特定子类的ANNs，称之为二元ANNs。二元ANNs是输入和输出都限制为二元值的前馈网络，特别适合于多种实际应用场景。我们之前的研究通过信念变化理论的视角，特别是Alchourron、Gardenfors和Makinson（AGM）框架，获得了若干重要见解。我们表明，二元ANNs所包含的知识（通过输入输出行为表达）可以使用命题逻辑语言进行符号表示。同时，信念集的修改过程（通过修订或收缩）被映射为一系列中间信念集的渐进过渡。类似地，二元ANNs的训练被概念化为这种信念集过渡的序列，我们证明了这可以使用全交集AGM样式信念变化的形式化方法进行表述。在本文中，我们通过解决之前研究的一些关键限制，进一步扩展了这一研究方向。具体来说，我们展示了Dalal的信念变化方法自然地诱导了一种结构化、渐进的信念状态演变。更重要的是，鉴于全交集信念变化已知的不足，我们证明了二元ANNs的训练动力学可以更有效地通过鲁棒的AGM样式变化操作进行建模，即词汇修订和适度收缩，这些操作与Darwiche-Pearl框架下的迭代信念变化框架一致。 

---
# AlphaEvolve: A coding agent for scientific and algorithmic discovery 

**Title (ZH)**: AlphaEvolve：一个用于科学和算法发现的编码代理 

**Authors**: Alexander Novikov, Ngân Vũ, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, Matej Balog  

**Link**: [PDF](https://arxiv.org/pdf/2506.13131)  

**Abstract**: In this white paper, we present AlphaEvolve, an evolutionary coding agent that substantially enhances capabilities of state-of-the-art LLMs on highly challenging tasks such as tackling open scientific problems or optimizing critical pieces of computational infrastructure. AlphaEvolve orchestrates an autonomous pipeline of LLMs, whose task is to improve an algorithm by making direct changes to the code. Using an evolutionary approach, continuously receiving feedback from one or more evaluators, AlphaEvolve iteratively improves the algorithm, potentially leading to new scientific and practical discoveries. We demonstrate the broad applicability of this approach by applying it to a number of important computational problems. When applied to optimizing critical components of large-scale computational stacks at Google, AlphaEvolve developed a more efficient scheduling algorithm for data centers, found a functionally equivalent simplification in the circuit design of hardware accelerators, and accelerated the training of the LLM underpinning AlphaEvolve itself. Furthermore, AlphaEvolve discovered novel, provably correct algorithms that surpass state-of-the-art solutions on a spectrum of problems in mathematics and computer science, significantly expanding the scope of prior automated discovery methods (Romera-Paredes et al., 2023). Notably, AlphaEvolve developed a search algorithm that found a procedure to multiply two $4 \times 4$ complex-valued matrices using $48$ scalar multiplications; offering the first improvement, after 56 years, over Strassen's algorithm in this setting. We believe AlphaEvolve and coding agents like it can have a significant impact in improving solutions of problems across many areas of science and computation. 

**Abstract (ZH)**: 本白皮书介绍了AlphaEvolve，这是一种进化编码代理，显著提升了先进语言模型（LLM）在解决开放性科学问题或优化关键计算基础设施等高度挑战性任务方面的能力。AlphaEvolve 自动协调一组语言模型，其任务是通过直接修改代码来改进算法。采用进化方法，持续从一个或多个评估者接收反馈，AlphaEvolve 逐步改进算法，可能引领新的科学和技术发现。我们通过将其应用于多个重要的计算问题，展示了该方法的广泛适用性。在谷歌大规模计算栈的关键组件优化中，AlphaEvolve 开发了一种更高效的云数据中心调度算法，简化了硬件加速器的电路设计，并加快了AlphaEvolve自身所基于的LLM的训练速度。此外，AlphaEvolve 发现了新颖的、可证明正确的算法，在数学和计算机科学的多个问题领域上超过了最先进的解决方案，显著扩大了之前自动化发现方法的应用范围（Romera-Paredes等，2023）。值得注意的是，AlphaEvolve 发展出了一种搜索算法，该算法找到了一种用48次标量乘法来乘以两个$4 \times 4$复值矩阵的计算方式，这是在Strassen算法提出56年后在这个情境下取得的第一个改进。我们认为，AlphaEvolve及其类似的编码代理将在多个科学和计算领域中显著改善问题的解决方案。 

---
# Dynamic Reinsurance Treaty Bidding via Multi-Agent Reinforcement Learning 

**Title (ZH)**: 基于多代理强化学习的动态再保险合约招标 

**Authors**: Stella C. Dong, James R. Finlay  

**Link**: [PDF](https://arxiv.org/pdf/2506.13113)  

**Abstract**: This paper develops a novel multi-agent reinforcement learning (MARL) framework for reinsurance treaty bidding, addressing long-standing inefficiencies in traditional broker-mediated placement processes. We pose the core research question: Can autonomous, learning-based bidding systems improve risk transfer efficiency and outperform conventional pricing approaches in reinsurance markets?
In our model, each reinsurer is represented by an adaptive agent that iteratively refines its bidding strategy within a competitive, partially observable environment. The simulation explicitly incorporates institutional frictions including broker intermediation, incumbent advantages, last-look privileges, and asymmetric access to underwriting information.
Empirical analysis demonstrates that MARL agents achieve up to 15% higher underwriting profit, 20% lower tail risk (CVaR), and over 25% improvement in Sharpe ratios relative to actuarial and heuristic baselines. Sensitivity tests confirm robustness across hyperparameter settings, and stress testing reveals strong resilience under simulated catastrophe shocks and capital constraints.
These findings suggest that MARL offers a viable path toward more transparent, adaptive, and risk-sensitive reinsurance markets. The proposed framework contributes to emerging literature at the intersection of algorithmic market design, strategic bidding, and AI-enabled financial decision-making. 

**Abstract (ZH)**: 一种基于多Agent强化学习的再保险条约招标新框架：自主学习招标系统能否改善风险转移效率并超越传统定价方法？ 

---
# A Memetic Walrus Algorithm with Expert-guided Strategy for Adaptive Curriculum Sequencing 

**Title (ZH)**: 基于专家引导策略的适应性课程序列化迷因海象算法 

**Authors**: Qionghao Huang, Lingnuo Lu, Xuemei Wu, Fan Jiang, Xizhe Wang, Xun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13092)  

**Abstract**: Adaptive Curriculum Sequencing (ACS) is essential for personalized online learning, yet current approaches struggle to balance complex educational constraints and maintain optimization stability. This paper proposes a Memetic Walrus Optimizer (MWO) that enhances optimization performance through three key innovations: (1) an expert-guided strategy with aging mechanism that improves escape from local optima; (2) an adaptive control signal framework that dynamically balances exploration and exploitation; and (3) a three-tier priority mechanism for generating educationally meaningful sequences. We formulate ACS as a multi-objective optimization problem considering concept coverage, time constraints, and learning style compatibility. Experiments on the OULAD dataset demonstrate MWO's superior performance, achieving 95.3% difficulty progression rate (compared to 87.2% in baseline methods) and significantly better convergence stability (standard deviation of 18.02 versus 28.29-696.97 in competing algorithms). Additional validation on benchmark functions confirms MWO's robust optimization capability across diverse scenarios. The results demonstrate MWO's effectiveness in generating personalized learning sequences while maintaining computational efficiency and solution quality. 

**Abstract (ZH)**: 自适应课程序列化（ACS）是个性化在线学习的关键，但当前方法在平衡复杂教育约束和保持优化稳定性方面存在困难。本文提出了一种遗传 walrus 优化器（MWO），通过三种关键创新增强优化性能：（1）具有老化机制的专家引导策略，提高从局部最优解中跳出的能力；（2）自适应控制信号框架，动态平衡探索与利用；（3）生成教育意义序列的三级优先级机制。我们将 ACS 形式化为一个多目标优化问题，考虑概念覆盖、时间限制和学习风格兼容性。在 OULAD 数据集上的实验表明，MWO 的性能优越，实现 95.3% 的难度进展率（基线方法为 87.2%），并且收敛稳定性显著更好（标准差为 18.02，而竞争对手算法为 28.29-696.97）。基准函数上的额外验证证实了 MWO 在各种场景下稳健的优化能力。结果表明，MWO 在保持计算效率和解质量的同时，有效地生成了个性化学习序列。 

---
# Discerning What Matters: A Multi-Dimensional Assessment of Moral Competence in LLMs 

**Title (ZH)**: 辨别什么是重要的：大规模语言模型道德能力的多维度评估 

**Authors**: Daniel Kilov, Caroline Hendy, Secil Yanik Guyot, Aaron J. Snoswell, Seth Lazar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13082)  

**Abstract**: Moral competence is the ability to act in accordance with moral principles. As large language models (LLMs) are increasingly deployed in situations demanding moral competence, there is increasing interest in evaluating this ability empirically. We review existing literature and identify three significant shortcoming: (i) Over-reliance on prepackaged moral scenarios with explicitly highlighted moral features; (ii) Focus on verdict prediction rather than moral reasoning; and (iii) Inadequate testing of models' (in)ability to recognize when additional information is needed. Grounded in philosophical research on moral skill, we then introduce a novel method for assessing moral competence in LLMs. Our approach moves beyond simple verdict comparisons to evaluate five dimensions of moral competence: identifying morally relevant features, weighting their importance, assigning moral reasons to these features, synthesizing coherent moral judgments, and recognizing information gaps. We conduct two experiments comparing six leading LLMs against non-expert humans and professional philosophers. In our first experiment using ethical vignettes standard to existing work, LLMs generally outperformed non-expert humans across multiple dimensions of moral reasoning. However, our second experiment, featuring novel scenarios designed to test moral sensitivity by embedding relevant features among irrelevant details, revealed a striking reversal: several LLMs performed significantly worse than humans. Our findings suggest that current evaluations may substantially overestimate LLMs' moral reasoning capabilities by eliminating the task of discerning moral relevance from noisy information, which we take to be a prerequisite for genuine moral skill. This work provides a more nuanced framework for assessing AI moral competence and highlights important directions for improving moral competence in advanced AI systems. 

**Abstract (ZH)**: 大语言模型的道德素养能力是遵循道德原则行动的能力。随着大语言模型（LLMs）在需要道德素养的情境中的部署越来越多，对其道德素养能力的实证评估也越来越受到关注。我们回顾现有文献并识别出三个重要缺陷：（i）过度依赖具有明确道德特征的预包装道德场景；（ii）侧重于判决预测而非道德推理；（iii）未能充分测试模型在识别需要额外信息时的能力。基于哲学研究中的道德技能理论，我们引入了一种评估LLMs道德素养的新方法。我们的方法超越了简单的判决比较，评估了道德素养的五个维度：识别相关道德特征、评估其重要性、为这些特征赋予道德理由、综合一致的道德判断，以及识别信息缺口。我们进行了两项实验，比较了六种领先的大语言模型与非专家人类和专业哲学家。在我们使用现有文献中标准的伦理情境进行的第一项实验中，大语言模型在多个道德推理维度上普遍优于非专家人类。然而，在第二项实验中，通过将相关特征嵌入无关细节以测试道德敏感性的新型情景揭示了一个显著的逆转：多种大语言模型的表现明显差于人类。我们的发现表明，当前评估可能通过消除从嘈杂信息中辨识出道德相关性的任务而极大地高估了大语言模型的道德推理能力，我们认为这一能力是真正道德技能的前提。本研究提供了一个更细腻的框架来评估AI的道德素养，并突显了改进高级AI系统道德素养的重要方向。 

---
# Rethinking Explainability in the Era of Multimodal AI 

**Title (ZH)**: 重新思考多模态AI时代的可解释性 

**Authors**: Chirag Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.13060)  

**Abstract**: While multimodal AI systems (models jointly trained on heterogeneous data types such as text, time series, graphs, and images) have become ubiquitous and achieved remarkable performance across high-stakes applications, transparent and accurate explanation algorithms are crucial for their safe deployment and ensure user trust. However, most existing explainability techniques remain unimodal, generating modality-specific feature attributions, concepts, or circuit traces in isolation and thus failing to capture cross-modal interactions. This paper argues that such unimodal explanations systematically misrepresent and fail to capture the cross-modal influence that drives multimodal model decisions, and the community should stop relying on them for interpreting multimodal models. To support our position, we outline key principles for multimodal explanations grounded in modality: Granger-style modality influence (controlled ablations to quantify how removing one modality changes the explanation for another), Synergistic faithfulness (explanations capture the model's predictive power when modalities are combined), and Unified stability (explanations remain consistent under small, cross-modal perturbations). This targeted shift to multimodal explanations will help the community uncover hidden shortcuts, mitigate modality bias, improve model reliability, and enhance safety in high-stakes settings where incomplete explanations can have serious consequences. 

**Abstract (ZH)**: 多模态AI系统透明性和准确解释算法：从单模态到多模态解释的转变 

---
# Metis-RISE: RL Incentivizes and SFT Enhances Multimodal Reasoning Model Learning 

**Title (ZH)**: Metis-RISE：RL激励和SFT提升多模态推理模型学习 

**Authors**: Haibo Qiu, Xiaohan Lan, Fanfan Liu, Xiaohu Sun, Delian Ruan, Peng Shi, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.13056)  

**Abstract**: Recent advancements in large language models (LLMs) have witnessed a surge in the development of advanced reasoning paradigms, which are now being integrated into multimodal large language models (MLLMs). However, existing approaches often fall short: methods solely employing reinforcement learning (RL) can struggle with sample inefficiency and activating entirely absent reasoning capabilities, while conventional pipelines that initiate with a cold-start supervised fine-tuning (SFT) phase before RL may restrict the model's exploratory capacity and face suboptimal convergence. In this work, we introduce \textbf{Metis-RISE} (\textbf{R}L \textbf{I}ncentivizes and \textbf{S}FT \textbf{E}nhances) for multimodal reasoning model learning. Unlike conventional approaches, Metis-RISE distinctively omits an initial SFT stage, beginning instead with an RL phase (e.g., using a Group Relative Policy Optimization variant) to incentivize and activate the model's latent reasoning capacity. Subsequently, the targeted SFT stage addresses two key challenges identified during RL: (1) \textit{inefficient trajectory sampling} for tasks where the model possesses but inconsistently applies correct reasoning, which we tackle using self-distilled reasoning trajectories from the RL model itself; and (2) \textit{fundamental capability absence}, which we address by injecting expert-augmented knowledge for prompts where the model entirely fails. This strategic application of RL for incentivization followed by SFT for enhancement forms the core of Metis-RISE, leading to two versions of our MLLMs (7B and 72B parameters). Evaluations on the OpenCompass Multimodal Reasoning Leaderboard demonstrate that both models achieve state-of-the-art performance among similar-sized models, with the 72B version ranking fourth overall. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）见证了高级推理范式的快速发展，这些范式现在正被整合到多模态大型语言模型（MLLMs）中。然而，现有方法往往存在不足：仅使用强化学习（RL）的方法可能面临样本效率低下和激活完全缺失的推理能力的问题，而传统的先进行冷启动监督微调（SFT）阶段再进行RL的流水线可能限制模型的探索能力并面临次优收敛问题。在这项工作中，我们提出了多模态推理模型学习的Metis-RISE（RL激励和SFT增强），与传统方法不同，Metis-RISE省去了初始SFT阶段，而是从RL阶段开始（例如，使用Group Relative Policy Optimization变体）以激励和激活模型的潜在推理能力。随后，针对RL期间识别出的两个关键挑战，即（1）对于模型虽然具备但不一致应用正确推理的任务，我们通过从RL模型本身自我提炼的推理轨迹来解决低效的轨迹采样问题；（2）基本能力缺失，我们通过为模型完全失败的提示注入专家增强知识来解决。这种方法在激励使用RL后通过SFT进行增强的策略构成了Metis-RISE的核心，从而产生了参数量为7B和72B的两个版本的MLLMs。在OpenCompass多模态推理 leaderboard 上的评估表明，这两个模型在同类模型中均达到最先进的性能，72B版本的整体排名为第四名。 

---
# MAGIC: Multi-Agent Argumentation and Grammar Integrated Critiquer 

**Title (ZH)**: MAGIC: 多Agent论辩与语法集成评判者 

**Authors**: Joaquin Jordan, Xavier Yin, Melissa Fabros, Gireeja Ranade, Narges Norouzi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13037)  

**Abstract**: Automated Essay Scoring (AES) and Automatic Essay Feedback (AEF) systems aim to reduce the workload of human raters in educational assessment. However, most existing systems prioritize numeric scoring accuracy over the quality of feedback. This paper presents Multi-Agent Argumentation and Grammar Integrated Critiquer (MAGIC), a framework that uses multiple specialized agents to evaluate distinct writing aspects to both predict holistic scores and produce detailed, rubric-aligned feedback. To support evaluation, we curated a novel dataset of past GRE practice test essays with expert-evaluated scores and feedback. MAGIC outperforms baseline models in both essay scoring , as measured by Quadratic Weighted Kappa (QWK). We find that despite the improvement in QWK, there are opportunities for future work in aligning LLM-generated feedback to human preferences. 

**Abstract (ZH)**: 自动作文评分系统（AES）和自动作文反馈系统（AEF）旨在减轻教育评估中人工评分者的负担。然而，现有系统大多优先考虑评分的准确性而非反馈的质量。本文提出了一种名为Multi-Agent Argumentation and Grammar Integrated Critiquer（MAGIC）的框架，该框架利用多个专业代理评估作文的不同方面，以预测综合评分并生成详细、符合评分标准的反馈。为了支持评估，我们构建了一个包含过去GRE练习测试作文的新颖数据集，并由专家进行了评分和反馈。MAGIC在作文评分上优于基准模型，根据Quadratic Weighted Kappa（QWK）进行衡量。我们发现，尽管QWK有所提高，但在将大语言模型生成的反馈与人类偏好对齐方面仍有许多未来工作可以做。 

---
# Knowledge Graph Fusion with Large Language Models for Accurate, Explainable Manufacturing Process Planning 

**Title (ZH)**: 基于大型语言模型的知识图谱融合以实现准确可解释的制造过程规划 

**Authors**: Danny Hoang, David Gorsich, Matthew P. Castanier, Farhad Imani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13026)  

**Abstract**: Precision process planning in Computer Numerical Control (CNC) machining demands rapid, context-aware decisions on tool selection, feed-speed pairs, and multi-axis routing, placing immense cognitive and procedural burdens on engineers from design specification through final part inspection. Conventional rule-based computer-aided process planning and knowledge-engineering shells freeze domain know-how into static tables, which become limited when dealing with unseen topologies, novel material states, shifting cost-quality-sustainability weightings, or shop-floor constraints such as tool unavailability and energy caps. Large language models (LLMs) promise flexible, instruction-driven reasoning for tasks but they routinely hallucinate numeric values and provide no provenance. We present Augmented Retrieval Knowledge Network Enhanced Search & Synthesis (ARKNESS), the end-to-end framework that fuses zero-shot Knowledge Graph (KG) construction with retrieval-augmented generation to deliver verifiable, numerically exact answers for CNC process planning. ARKNESS (1) automatically distills heterogeneous machining documents, G-code annotations, and vendor datasheets into augmented triple, multi-relational graphs without manual labeling, and (2) couples any on-prem LLM with a retriever that injects the minimal, evidence-linked subgraph needed to answer a query. Benchmarked on 155 industry-curated questions spanning tool sizing and feed-speed optimization, a lightweight 3B-parameter Llama-3 augmented by ARKNESS matches GPT-4o accuracy while achieving a +25 percentage point gain in multiple-choice accuracy, +22.4 pp in F1, and 8.1x ROUGE-L on open-ended responses. 

**Abstract (ZH)**: ARKNESS：端到端融合零-shot知识图谱构建与检索增强生成的计算机数控加工规划框架 

---
# A Practical Guide for Evaluating LLMs and LLM-Reliant Systems 

**Title (ZH)**: 实用指南：评估大语言模型及其依赖系统 

**Authors**: Ethan M. Rudd, Christopher Andrews, Philip Tully  

**Link**: [PDF](https://arxiv.org/pdf/2506.13023)  

**Abstract**: Recent advances in generative AI have led to remarkable interest in using systems that rely on large language models (LLMs) for practical applications. However, meaningful evaluation of these systems in real-world scenarios comes with a distinct set of challenges, which are not well-addressed by synthetic benchmarks and de-facto metrics that are often seen in the literature. We present a practical evaluation framework which outlines how to proactively curate representative datasets, select meaningful evaluation metrics, and employ meaningful evaluation methodologies that integrate well with practical development and deployment of LLM-reliant systems that must adhere to real-world requirements and meet user-facing needs. 

**Abstract (ZH)**: 最近在生成式人工智能领域的进展引发对依赖大型语言模型（LLMs）的系统在实际应用中使用的研究兴趣。然而，在现实世界场景中对这些系统的有意义评估面临独特的挑战，这些挑战并未在文献中充分解决。我们提出了一种实用的评估框架，该框架阐述了如何积极遴选代表性数据集、选择有意义的评估指标，并采用与实用开发和部署大型语言模型依赖系统相结合的有意义的评估方法，同时满足现实世界的要求并满足用户需求。 

---
# Efficient Neuro-Symbolic Retrieval-Augmented Generation through Adaptive Query Routing 

**Title (ZH)**: 自适应查询路由导向的高效神经符号检索增强生成 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.12981)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems address factual inconsistencies in Large Language Models by grounding generation in external knowledge, yet they face a fundamental efficiency problem: simple queries consume computational resources equivalent to complex multi-hop reasoning tasks. We present SymRAG, a neuro-symbolic framework that introduces adaptive query routing based on real-time complexity and system load assessments. SymRAG dynamically selects symbolic, neural, or hybrid processing paths to align resource use with query demands. Evaluated on 2,000 queries from HotpotQA and DROP using Llama-3.2-3B and Mistral-7B models, SymRAG achieves 97.6--100.0% exact match accuracy with significantly lower CPU utilization (3.6--6.2%) and processing time (0.985--3.165s). Disabling adaptive logic results in 169--1151% increase in processing time, highlighting the framework's impact. These results underscore the potential of adaptive neuro-symbolic routing for scalable, sustainable AI systems. 

**Abstract (ZH)**: 基于符号-神经适应查询路由的 Retrieval-Augmented Generation (RAG) 系统 

---
# Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills 

**Title (ZH)**: 推理模型遗忘：不仅忘记答案，还要遗忘推理痕迹，同时保持推理能力 

**Authors**: Changsheng Wang, Chongyu Fan, Yihua Zhang, Jinghan Jia, Dennis Wei, Parikshit Ram, Nathalie Baracaldo, Sijia Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12963)  

**Abstract**: Recent advances in large reasoning models (LRMs) have enabled strong chain-of-thought (CoT) generation through test-time computation. While these multi-step reasoning capabilities represent a major milestone in language model performance, they also introduce new safety risks. In this work, we present the first systematic study to revisit the problem of machine unlearning in the context of LRMs. Machine unlearning refers to the process of removing the influence of sensitive, harmful, or undesired data or knowledge from a trained model without full retraining. We show that conventional unlearning algorithms, originally designed for non-reasoning models, are inadequate for LRMs. In particular, even when final answers are successfully erased, sensitive information often persists within the intermediate reasoning steps, i.e., CoT trajectories. To address this challenge, we extend conventional unlearning and propose Reasoning-aware Representation Misdirection for Unlearning ($R^2MU$), a novel method that effectively suppresses sensitive reasoning traces and prevents the generation of associated final answers, while preserving the model's reasoning ability. Our experiments demonstrate that $R^2MU$ significantly reduces sensitive information leakage within reasoning traces and achieves strong performance across both safety and reasoning benchmarks, evaluated on state-of-the-art models such as DeepSeek-R1-Distill-LLaMA-8B and DeepSeek-R1-Distill-Qwen-14B. 

**Abstract (ZH)**: Recent Advances in Large Reasoning Models: A Systematic Study on Machine Unlearning in the Context of Chain-of-Thought Generation 

---
# Constitutive Components for Human-Like Autonomous Artificial Intelligence 

**Title (ZH)**: 构成人类拟态自主人工智能的要素 

**Authors**: Kazunori D Yamada  

**Link**: [PDF](https://arxiv.org/pdf/2506.12952)  

**Abstract**: This study is the first to clearly identify the functions required to construct artificial entities capable of behaving autonomously like humans, and organizes them into a three-layer functional hierarchy. Specifically, it defines three levels: Core Functions, which enable interaction with the external world; the Integrative Evaluation Function, which selects actions based on perception and memory; and the Self Modification Function, which dynamically reconfigures behavioral principles and internal components. Based on this structure, the study proposes a stepwise model of autonomy comprising reactive, weak autonomous, and strong autonomous levels, and discusses its underlying design principles and developmental aspects. It also explores the relationship between these functions and existing artificial intelligence design methods, addressing their potential as a foundation for general intelligence and considering future applications and ethical implications. By offering a theoretical framework that is independent of specific technical methods, this work contributes to a deeper understanding of autonomy and provides a foundation for designing future artificial entities with strong autonomy. 

**Abstract (ZH)**: 本研究首次清晰地界定了构建能够像人类一样自主行为的人工实体所需的功能，并将其组织成三层功能层次结构。具体而言，它定义了三个层次：核心功能，使实体能够与外部世界交互；综合评估功能，基于感知和记忆选择行动；以及自我修改功能，动态重新配置行为原则和内部组件。基于这一结构，研究提出了一种分阶段的自主性模型，包括反应性、弱自主性和强自主性层次，并讨论了其底层设计原则和发展方面的内容。研究还探讨了这些功能与现有人工智能设计方法的关系，探讨了它们作为通用智能基础的潜力，并考虑了未来应用和伦理影响。通过提供一种独立于具体技术方法的理论框架，本工作加深了对自主性的理解，并为设计具有强自主性的未来人工实体奠定了基础。 

---
# HypER: Literature-grounded Hypothesis Generation and Distillation with Provenance 

**Title (ZH)**: HypER: 基于文献的假设生成和精炼及其溯源 

**Authors**: Rosni Vasu, Chandrayee Basu, Bhavana Dalvi Mishra, Cristina Sarasua, Peter Clark, Abraham Bernstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.12937)  

**Abstract**: Large Language models have demonstrated promising performance in research ideation across scientific domains. Hypothesis development, the process of generating a highly specific declarative statement connecting a research idea with empirical validation, has received relatively less attention. Existing approaches trivially deploy retrieval augmentation and focus only on the quality of the final output ignoring the underlying reasoning process behind ideation. We present $\texttt{HypER}$ ($\textbf{Hyp}$othesis Generation with $\textbf{E}$xplanation and $\textbf{R}$easoning), a small language model (SLM) trained for literature-guided reasoning and evidence-based hypothesis generation. $\texttt{HypER}$ is trained in a multi-task setting to discriminate between valid and invalid scientific reasoning chains in presence of controlled distractions. We find that $\texttt{HypER}$ outperformes the base model, distinguishing valid from invalid reasoning chains (+22\% average absolute F1), generates better evidence-grounded hypotheses (0.327 vs. 0.305 base model) with high feasibility and impact as judged by human experts ($>$3.5 on 5-point Likert scale). 

**Abstract (ZH)**: 大型语言模型在科学研究领域展示了令人鼓舞的研究构想能力。假设生成，即生成将研究构想与实证验证联系起来的具体陈述的过程，受到了相对较少的关注。现有的方法简单地采用了检索增强技术，并仅关注最终输出的质量，而忽视了构想过程中的推理机制。我们提出了HypER（基于解释和推理的假设生成），这是一种用于文献引导推理和证据基础假设生成的小型语言模型（SLM）。HypER在多任务设置下进行训练，以区分有控制的干扰下的有效和无效科学推理链。我们发现HypER在区分有效和无效推理链方面表现优于基础模型（绝对F1平均值提高22%），生成了更好的基于证据的假设（专家评价可行性与影响判断得分分别为0.327 vs. 0.305，超过5点李克特量表的3.5分）。 

---
# Scaling Test-time Compute for LLM Agents 

**Title (ZH)**: 为LLM代理扩展测试时计算量 

**Authors**: King Zhu, Hanhao Li, Siwei Wu, Tianshun Xing, Dehua Ma, Xiangru Tang, Minghao Liu, Jian Yang, Jiaheng Liu, Yuchen Eleanor Jiang, Changwang Zhang, Chenghua Lin, Jun Wang, Ge Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12928)  

**Abstract**: Scaling test time compute has shown remarkable success in improving the reasoning abilities of large language models (LLMs). In this work, we conduct the first systematic exploration of applying test-time scaling methods to language agents and investigate the extent to which it improves their effectiveness. Specifically, we explore different test-time scaling strategies, including: (1) parallel sampling algorithms; (2) sequential revision strategies; (3) verifiers and merging methods; (4)strategies for diversifying this http URL carefully analyze and ablate the impact of different design strategies on applying test-time scaling on language agents, and have follow findings: 1. Scaling test time compute could improve the performance of agents. 2. Knowing when to reflect is important for agents. 3. Among different verification and result merging approaches, the list-wise method performs best. 4. Increasing diversified rollouts exerts a positive effect on the agent's task performance. 

**Abstract (ZH)**: 在语言代理中应用测试时计算缩放的方法首次系统探究及其效果分析 

---
# Sectoral Coupling in Linguistic State Space 

**Title (ZH)**: 语言状态空间中的部门耦合 

**Authors**: Sebastian Dumbrava  

**Link**: [PDF](https://arxiv.org/pdf/2506.12927)  

**Abstract**: This work presents a formal framework for quantifying the internal dependencies between functional subsystems within artificial agents whose belief states are composed of structured linguistic fragments. Building on the Semantic Manifold framework, which organizes belief content into functional sectors and stratifies them across hierarchical levels of abstraction, we introduce a system of sectoral coupling constants that characterize how one cognitive sector influences another within a fixed level of abstraction. The complete set of these constants forms an agent-specific coupling profile that governs internal information flow, shaping the agent's overall processing tendencies and cognitive style. We provide a detailed taxonomy of these intra-level coupling roles, covering domains such as perceptual integration, memory access and formation, planning, meta-cognition, execution control, and affective modulation. We also explore how these coupling profiles generate feedback loops, systemic dynamics, and emergent signatures of cognitive behavior. Methodologies for inferring these profiles from behavioral or internal agent data are outlined, along with a discussion of how these couplings evolve across abstraction levels. This framework contributes a mechanistic and interpretable approach to modeling complex cognition, with applications in AI system design, alignment diagnostics, and the analysis of emergent agent behavior. 

**Abstract (ZH)**: 本研究提出了一种形式化框架，用于量化人工代理内部由结构化语言片段组成信念状态的功能子系统之间的内部依赖性。该框架基于语义流形框架，后者将信念内容组织成功能部门，并按抽象层次分层。在此基础上，我们引入了一种部门耦合常数系统，以表征固定抽象层次内一个认知部门如何影响另一个部门。这一整套耦合常数构成了代理特有的耦合特征，决定了内部信息流，塑造了代理的整体处理倾向和认知风格。我们详细探讨了这些同级耦合角色的分类，涵盖感知整合、记忆获取和形成、计划、元认知、执行控制和情感调节等领域。我们还探讨了这些耦合特征如何产生反馈回路、系统动态和认知行为的涌现特征。文中概述了从行为或内部代理数据推断这些特征的方法，并讨论了这些耦合如何随着抽象层次的变化而演变。该框架提供了一种机械的、可解释的方法来建模复杂认知，适用于AI系统设计、对齐诊断以及对涌现代理行为的分析。 

---
# Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories 

**Title (ZH)**: 基于确定性扩散轨迹的约束引导预测细化 

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.12911)  

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic. 

**Abstract (ZH)**: 约束aware细化的一般框架：基于去噪扩散隐模型的方法 

---
# KCLNet: Physics-Informed Power Flow Prediction via Constraints Projections 

**Title (ZH)**: KCLNet: 基于约束投影的物理guided功率流预测 

**Authors**: Pantelis Dogoulis, Karim Tit, Maxime Cordy  

**Link**: [PDF](https://arxiv.org/pdf/2506.12902)  

**Abstract**: In the modern context of power systems, rapid, scalable, and physically plausible power flow predictions are essential for ensuring the grid's safe and efficient operation. While traditional numerical methods have proven robust, they require extensive computation to maintain physical fidelity under dynamic or contingency conditions. In contrast, recent advancements in artificial intelligence (AI) have significantly improved computational speed; however, they often fail to enforce fundamental physical laws during real-world contingencies, resulting in physically implausible predictions. In this work, we introduce KCLNet, a physics-informed graph neural network that incorporates Kirchhoff's Current Law as a hard constraint via hyperplane projections. KCLNet attains competitive prediction accuracy while ensuring zero KCL violations, thereby delivering reliable and physically consistent power flow predictions critical to secure the operation of modern smart grids. 

**Abstract (ZH)**: 在现代电力系统的背景下，快速、可扩展且物理上合理的潮流预测对于确保电网的安全和高效运行至关重要。虽然传统的数值方法在保持物理一致性方面表现出 robust 性，但在动态或故障条件下需要大量的计算。相比之下，最近在人工智能（AI）方面的进展显著提高了计算速度，但在实际故障条件下经常无法强制执行基本的物理定律，从而导致物理上不合理的结果。在本工作中，我们引入了 KCLNet，这是一种物理信息图神经网络，通过超平面投影将基尔霍夫电流定律作为硬约束纳入其中。KCLNet 在保持零基尔霍夫电流定律违反的情况下达到竞争力的预测精度，从而实现对现代智能电网安全运行至关重要的可靠且物理上一致的潮流预测。 

---
# Homeostatic Coupling for Prosocial Behavior 

**Title (ZH)**: 恒定耦合促进利他行为 

**Authors**: Naoto Yoshida, Kingson Man  

**Link**: [PDF](https://arxiv.org/pdf/2506.12894)  

**Abstract**: When regarding the suffering of others, we often experience personal distress and feel compelled to help\footnote{Preprint. Under review.}. Inspired by living systems, we investigate the emergence of prosocial behavior among autonomous agents that are motivated by homeostatic self-regulation. We perform multi-agent reinforcement learning, treating each agent as a vulnerable homeostat charged with maintaining its own well-being. We introduce an empathy-like mechanism to share homeostatic states between agents: an agent can either \emph{observe} their partner's internal state ({\bf cognitive empathy}) or the agent's internal state can be \emph{directly coupled} to that of their partner ({\bf affective empathy}). In three simple multi-agent environments, we show that prosocial behavior arises only under homeostatic coupling - when the distress of a partner can affect one's own well-being. Additionally, we show that empathy can be learned: agents can ``decode" their partner's external emotive states to infer the partner's internal homeostatic states. Assuming some level of physiological similarity, agents reference their own emotion-generation functions to invert the mapping from outward display to internal state. Overall, we demonstrate the emergence of prosocial behavior when homeostatic agents learn to ``read" the emotions of others and then to empathize, or feel as they feel. 

**Abstract (ZH)**: 当面对他人的苦难时，我们往往会经历个人的痛苦并感到有必要提供帮助。受生物系统启发，我们研究自主代理在基于稳态自我调节动机的情况下，亲社会行为的涌现。我们进行了多代理强化学习，将每个代理视为需要维护自身福祉的脆弱稳态系统。我们引入了一种类似共情的机制来在代理之间共享稳态状态：代理可以“观察”其伙伴的内部状态（认知共情），或者其内部状态可以“直接耦合”到其伙伴的内部状态（情感共情）。在三个简单的多代理环境中，我们证明了仅在稳态耦合下才会出现亲社会行为——当伙伴的痛苦会影响自身的福祉时。此外，我们还展示了共情可以通过学习获得：代理可以“解码”其伙伴的外部情绪状态以推断其伙伴的内部稳态状态。假设一定程度的生理相似性，代理会参照自身的 emotion 生成函数来反转从外部表现到内部状态的映射。总体而言，我们展示了当稳态代理学会“读取”他人的感情并产生共情时，亲社会行为的涌现。 

---
# Evolutionary Developmental Biology Can Serve as the Conceptual Foundation for a New Design Paradigm in Artificial Intelligence 

**Title (ZH)**: 演化发育生物学可以作为人工智新建模范式的概念基础 

**Authors**: Zeki Doruk Erden, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2506.12891)  

**Abstract**: Artificial intelligence (AI), propelled by advancements in machine learning, has made significant strides in solving complex tasks. However, the current neural network-based paradigm, while effective, is heavily constrained by inherent limitations, primarily a lack of structural organization and a progression of learning that displays undesirable properties. As AI research progresses without a unifying framework, it either tries to patch weaknesses heuristically or draws loosely from biological mechanisms without strong theoretical foundations. Meanwhile, the recent paradigm shift in evolutionary understanding -- driven primarily by evolutionary developmental biology (EDB) -- has been largely overlooked in AI literature, despite a striking analogy between the Modern Synthesis and contemporary machine learning, evident in their shared assumptions, approaches, and limitations upon careful analysis. Consequently, the principles of adaptation from EDB that reshaped our understanding of the evolutionary process can also form the foundation of a unifying conceptual framework for the next design philosophy in AI, going beyond mere inspiration and grounded firmly in biology's first principles. This article provides a detailed overview of the analogy between the Modern Synthesis and modern machine learning, and outlines the core principles of a new AI design paradigm based on insights from EDB. To exemplify our analysis, we also present two learning system designs grounded in specific developmental principles -- regulatory connections, somatic variation and selection, and weak linkage -- that resolve multiple major limitations of contemporary machine learning in an organic manner, while also providing deeper insights into the role of these mechanisms in biological evolution. 

**Abstract (ZH)**: 基于进化发育生物学的类人智能设计新范式 

---
# WereWolf-Plus: An Update of Werewolf Game setting Based on DSGBench 

**Title (ZH)**: WereWolf-Plus：基于DSGBench的狼人游戏设置更新 

**Authors**: Xinyuan Xia, Yuanyi Song, Haomin Ma, Jinyu Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12841)  

**Abstract**: With the rapid development of LLM-based agents, increasing attention has been given to their social interaction and strategic reasoning capabilities. However, existing Werewolf-based benchmarking platforms suffer from overly simplified game settings, incomplete evaluation metrics, and poor scalability. To address these limitations, we propose WereWolf-Plus, a multi-model, multi-dimensional, and multi-method benchmarking platform for evaluating multi-agent strategic reasoning in the Werewolf game. The platform offers strong extensibility, supporting customizable configurations for roles such as Seer, Witch, Hunter, Guard, and Sheriff, along with flexible model assignment and reasoning enhancement strategies for different roles. In addition, we introduce a comprehensive set of quantitative evaluation metrics for all special roles, werewolves, and the sheriff, and enrich the assessment dimensions for agent reasoning ability, cooperation capacity, and social influence. WereWolf-Plus provides a more flexible and reliable environment for advancing research on inference and strategic interaction within multi-agent communities. Our code is open sourced at this https URL. 

**Abstract (ZH)**: 基于LLM的代理快速发展，越来越多的关注被投向它们的社会交互和战略推理能力。然而，现有的狼人Based基准平台存在游戏设置过于简化、评估指标不完整以及扩展性差的问题。为解决这些限制，我们提出了WereWolf-Plus，这是一个多模型、多维度、多方法的基准平台，用于评估狼人游戏中的多代理战略推理能力。该平台具有强大的扩展性，支持自定义角色配置，如预言家、巫师、猎人、守卫和警长，并提供了灵活的模型分配和不同角色的推理增强策略。此外，我们引入了一套完整的定量评估指标，涵盖所有特别角色、狼人和警长，并丰富了代理推理能力、合作能力和社会影响的评估维度。WereWolf-Plus提供了一个更加灵活且可靠的环境，促进多代理社区内的推理和战略交互研究。我们的代码已开源，可通过该网址访问。 

---
# Rethinking Optimization: A Systems-Based Approach to Social Externalities 

**Title (ZH)**: 重新思考优化：基于系统的方法研究社会外部性 

**Authors**: Pegah Nokhiz, Aravinda Kanchana Ruwanpathirana, Helen Nissenbaum  

**Link**: [PDF](https://arxiv.org/pdf/2506.12825)  

**Abstract**: Optimization is widely used for decision making across various domains, valued for its ability to improve efficiency. However, poor implementation practices can lead to unintended consequences, particularly in socioeconomic contexts where externalities (costs or benefits to third parties outside the optimization process) are significant. To propose solutions, it is crucial to first characterize involved stakeholders, their goals, and the types of subpar practices causing unforeseen outcomes. This task is complex because affected stakeholders often fall outside the direct focus of optimization processes. Also, incorporating these externalities into optimization requires going beyond traditional economic frameworks, which often focus on describing externalities but fail to address their normative implications or interconnected nature, and feedback loops. This paper suggests a framework that combines systems thinking with the economic concept of externalities to tackle these challenges. This approach aims to characterize what went wrong, who was affected, and how (or where) to include them in the optimization process. Economic externalities, along with their established quantification methods, assist in identifying "who was affected and how" through stakeholder characterization. Meanwhile, systems thinking (an analytical approach to comprehending relationships in complex systems) provides a holistic, normative perspective. Systems thinking contributes to an understanding of interconnections among externalities, feedback loops, and determining "when" to incorporate them in the optimization. Together, these approaches create a comprehensive framework for addressing optimization's unintended consequences, balancing descriptive accuracy with normative objectives. Using this, we examine three common types of subpar practices: ignorance, error, and prioritization of short-term goals. 

**Abstract (ZH)**: 优化在各个领域广泛用于决策制定，因其能够提高效率而受到重视。然而，不良实施实践可能导致意外后果，尤其是在外部性（对优化过程之外的第三方的成本或收益）显著的社会经济背景下。为了提出解决方案，首先要明确相关利益相关者、他们的目标以及导致意外结果的不良实践类型。这一任务之所以复杂，是因为受影响的利益相关者往往不在优化过程的核心关注范围内。此外，将这些外部性纳入优化还需要超越传统的经济框架，这些框架通常专注于描述外部性，但未能解决它们的规范含义或相互关系，以及反馈循环问题。本文提出了一种结合系统思维与经济外部性概念的框架来应对这些挑战。该方法旨在界定“问题出在哪里”，“谁受到了影响”，以及如何（或在哪里）将这些因素纳入优化过程。经济外部性及其已建立的量化方法帮助通过利益相关者分析来确定“谁受到了影响以及程度如何”。同时，系统思维（一种理解和分析复杂系统中关系的分析方法）提供了全面且规范的视角。系统思维有助于理解外部性之间的相互关系、反馈循环及其“何时”纳入优化的重要性。结合这些方法，可以建立一个全面的框架，以解决优化的意外后果，兼顾描述准确性和规范目标。我们使用这种方法分析了三种常见的不良实践类型：无知、错误和短期目标优先。 

---
# Federated Neuroevolution O-RAN: Enhancing the Robustness of Deep Reinforcement Learning xApps 

**Title (ZH)**: 联邦神经进化O-RAN：增强深度强化学习x应用程序的稳健性 

**Authors**: Mohammadreza Kouchaki, Aly Sabri Abdalla, Vuk Marojevic  

**Link**: [PDF](https://arxiv.org/pdf/2506.12812)  

**Abstract**: The open radio access network (O-RAN) architecture introduces RAN intelligent controllers (RICs) to facilitate the management and optimization of the disaggregated RAN. Reinforcement learning (RL) and its advanced form, deep RL (DRL), are increasingly employed for designing intelligent controllers, or xApps, to be deployed in the near-real time (near-RT) RIC. These models often encounter local optima, which raise concerns about their reliability for RAN intelligent control. We therefore introduce Federated O-RAN enabled Neuroevolution (NE)-enhanced DRL (F-ONRL) that deploys an NE-based optimizer xApp in parallel to the RAN controller xApps. This NE-DRL xApp framework enables effective exploration and exploitation in the near-RT RIC without disrupting RAN operations. We implement the NE xApp along with a DRL xApp and deploy them on Open AI Cellular (OAIC) platform and present numerical results that demonstrate the improved robustness of xApps while effectively balancing the additional computational load. 

**Abstract (ZH)**: 基于Federated O-RAN的NE增强DRL（F-ONRL）智能控制器架构 

---
# Fuzzy Propositional Formulas under the Stable Model Semantics 

**Title (ZH)**: 稳定模型语义下的模糊命题公式 

**Authors**: Joohyung Lee, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12804)  

**Abstract**: We define a stable model semantics for fuzzy propositional formulas, which generalizes both fuzzy propositional logic and the stable model semantics of classical propositional formulas. The syntax of the language is the same as the syntax of fuzzy propositional logic, but its semantics distinguishes stable models from non-stable models. The generality of the language allows for highly configurable nonmonotonic reasoning for dynamic domains involving graded truth degrees. We show that several properties of Boolean stable models are naturally extended to this many-valued setting, and discuss how it is related to other approaches to combining fuzzy logic and the stable model semantics. 

**Abstract (ZH)**: 我们定义了一种模糊命题公式的第一稳定模型语义，该语义既泛化了模糊命题逻辑，也泛化了经典命题公式稳定模型语义。语言的语法与模糊命题逻辑相同，但其语义将稳定模型与非稳定模型区分开来。该语言的普适性使得在涉及等级真度的动态领域中能够进行高度配置的非单调推理。我们展示了布尔稳定模型的若干性质可以自然地扩展到这个多值设置，并讨论了它与其他结合模糊逻辑和稳定模型语义的方法的关系。 

---
# Mastering Da Vinci Code: A Comparative Study of Transformer, LLM, and PPO-based Agents 

**Title (ZH)**: 精通达芬奇代码：基于Transformer、大型语言模型和PPO代理的对比研究 

**Authors**: LeCheng Zhang, Yuanshi Wang, Haotian Shen, Xujie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12801)  

**Abstract**: The Da Vinci Code, a game of logical deduction and imperfect information, presents unique challenges for artificial intelligence, demanding nuanced reasoning beyond simple pattern recognition. This paper investigates the efficacy of various AI paradigms in mastering this game. We develop and evaluate three distinct agent architectures: a Transformer-based baseline model with limited historical context, several Large Language Model (LLM) agents (including Gemini, DeepSeek, and GPT variants) guided by structured prompts, and an agent based on Proximal Policy Optimization (PPO) employing a Transformer encoder for comprehensive game history processing. Performance is benchmarked against the baseline, with the PPO-based agent demonstrating superior win rates ($58.5\% \pm 1.0\%$), significantly outperforming the LLM counterparts. Our analysis highlights the strengths of deep reinforcement learning in policy refinement for complex deductive tasks, particularly in learning implicit strategies from self-play. We also examine the capabilities and inherent limitations of current LLMs in maintaining strict logical consistency and strategic depth over extended gameplay, despite sophisticated prompting. This study contributes to the broader understanding of AI in recreational games involving hidden information and multi-step logical reasoning, offering insights into effective agent design and the comparative advantages of different AI approaches. 

**Abstract (ZH)**: 《达芬奇密码：逻辑推理与信息不完美的游戏中的AI挑战及其多范式研究》 

---
# LPMLN, Weak Constraints, and P-log 

**Title (ZH)**: LPMLN, 软约束和P-log 

**Authors**: Joohyung Lee, Zhun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12784)  

**Abstract**: LPMLN is a recently introduced formalism that extends answer set programs by adopting the log-linear weight scheme of Markov Logic. This paper investigates the relationships between LPMLN and two other extensions of answer set programs: weak constraints to express a quantitative preference among answer sets, and P-log to incorporate probabilistic uncertainty. We present a translation of LPMLN into programs with weak constraints and a translation of P-log into LPMLN, which complement the existing translations in the opposite directions. The first translation allows us to compute the most probable stable models (i.e., MAP estimates) of LPMLN programs using standard ASP solvers. This result can be extended to other formalisms, such as Markov Logic, ProbLog, and Pearl's Causal Models, that are shown to be translatable into LPMLN. The second translation tells us how probabilistic nonmonotonicity (the ability of the reasoner to change his probabilistic model as a result of new information) of P-log can be represented in LPMLN, which yields a way to compute P-log using standard ASP solvers and MLN solvers. 

**Abstract (ZH)**: LPMLN是最近提出的一种形式化方法，通过采用Markov Logic的对数线性权重方案扩展了回答集程序。本文探讨了LPMLN与其他两种回答集程序扩展之间的关系：弱约束用于表达回答集之间的定量偏好，以及P-log用于整合概率不确定性。我们提出了从LPMLN到弱约束程序的翻译，以及从P-log到LPMLN的翻译，这些翻译补充了现有从相反方向的翻译。第一个翻译使我们能够使用标准ASP求解器计算LPMLN程序的最可能稳定模型（即，MAP估计）。这一结果可以扩展到其他形式化方法，如可以转换为LPMLN的Markov Logic、ProbLog和Pearl因果模型。第二个翻译说明了如何在LPMLN中表示P-log的概率非单调性（推理器根据新信息改变其概率模型的能力），从而提供了一种使用标准ASP求解器和MLN求解器计算P-log的方法。 

---
# Decentralized Decision Making in Two Sided Manufacturing-as-a-Service Marketplaces 

**Title (ZH)**: 两侧制造即服务市场中的去中心化决策Making 

**Authors**: Deepak Pahwa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12730)  

**Abstract**: Advancements in digitization have enabled two sided manufacturing-as-a-service (MaaS) marketplaces which has significantly reduced product development time for designers. These platforms provide designers with access to manufacturing resources through a network of suppliers and have instant order placement capabilities. Two key decision making levers are typically used to optimize the operations of these marketplaces: pricing and matching. The existing marketplaces operate in a centralized structure where they have complete control over decision making. However, a decentralized organization of the platform enables transparency of information across clients and suppliers. This dissertation focuses on developing tools for decision making enabling decentralization in MaaS marketplaces. In pricing mechanisms, a data driven method is introduced which enables small service providers to price services based on specific attributes of the services offered. A data mining method recommends a network based price to a supplier based on its attributes and the attributes of other suppliers on the platform. Three different approaches are considered for matching mechanisms. First, a reverse auction mechanism is introduced where designers bid for manufacturing services and the mechanism chooses a supplier which can match the bid requirements and stated price. The second approach uses mechanism design and mathematical programming to develop a stable matching mechanism for matching orders to suppliers based on their preferences. Empirical simulations are used to test the mechanisms in a simulated 3D printing marketplace and to evaluate the impact of stability on its performance. The third approach considers the matching problem in a dynamic and stochastic environment where demand (orders) and supply (supplier capacities) arrive over time and matching is performed online. 

**Abstract (ZH)**: 数字化进展使得两面市场制造即服务（MaaS）平台得以发展，显著缩短了设计师的产品开发时间。这些平台通过供应商网络为设计师提供制造资源，并具备即时下单能力。通常使用两类决策杠杆来优化这些市场的运营：定价和匹配。现有的市场平台采用集中式架构，对决策具有完全控制。然而，平台的分散组织能促进客户和供应商之间的信息透明。本论文旨在开发支持MaaS平台分散决策的工具。在定价机制中，提出了一种基于数据的方法，使小型服务提供商能够根据所提供的服务特性定价。数据挖掘方法基于供应商及其平台上其他供应商的特性，推荐网络定价。匹配机制分为三种不同方法：首先，引入逆向拍卖机制，设计师为制造服务出价，机制选择能满足出价要求并报价的供应商。其次，采用机制设计和数学规划开发基于供方偏好的稳定匹配机制。使用实证模拟在模拟的3D打印市场中测试这些机制，并评估稳定性的性能影响。最后，第三种方法考虑在动态和随机环境中进行匹配问题，即需求（订单）和供应（供应商能力）随时间到达，并在线进行匹配。 

---
# Rethinking DPO: The Role of Rejected Responses in Preference Misalignment 

**Title (ZH)**: 重新思考DPO：被拒绝响应在偏好不对齐中的作用 

**Authors**: Jay Hyeon Cho, JunHyeok Oh, Myunsoo Kim, Byung-Jun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.12725)  

**Abstract**: Direct Preference Optimization (DPO) is a simple and efficient framework that has attracted substantial attention. However, it often struggles to meet its primary objectives -- increasing the generation probability of chosen responses while reducing that of rejected responses -- due to the dominant influence of rejected responses on the loss function. This imbalance leads to suboptimal performance in promoting preferred responses. In this work, we systematically analyze the limitations of DPO and existing algorithms designed to achieve the objectives stated above. To address these limitations, we propose Bounded-DPO (BDPO), a novel method that bounds the influence of rejected responses while maintaining the original optimization structure of DPO. Through theoretical analysis and empirical evaluations, we demonstrate that BDPO achieves a balanced optimization of the chosen and rejected responses, outperforming existing algorithms. 

**Abstract (ZH)**: 直接偏好优化（DPO）是一种简单且高效的框架，引起了广泛关注。然而，它经常难以实现其主要目标——增加选定响应的生成概率同时减少被拒绝响应的概率——这主要是由于被拒绝响应在损失函数中占主导地位的影响。这种不平衡导致在促进偏好响应方面表现不佳。在本文中，我们系统分析了DPO及其现有算法的局限性，旨在实现上述目标。为了解决这些局限性，我们提出了一种新型的方法——有界直接偏好优化（BDPO），该方法限制了被拒绝响应的影响，同时保持了DPO的原始优化结构。通过理论分析和实证评估，我们证明了BDPO实现了选定和被拒绝响应之间的平衡优化，优于现有算法。 

---
# Strategic Scaling of Test-Time Compute: A Bandit Learning Approach 

**Title (ZH)**: 测试时计算的策略性扩展：一种bandit学习方法 

**Authors**: Bowen Zuo, Yinglun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12721)  

**Abstract**: Scaling test-time compute has emerged as an effective strategy for improving the performance of large language models. However, existing methods typically allocate compute uniformly across all queries, overlooking variation in query difficulty. To address this inefficiency, we formulate test-time compute allocation as a novel bandit learning problem and propose adaptive algorithms that estimate query difficulty on the fly and allocate compute accordingly. Compared to uniform allocation, our algorithms allocate more compute to challenging queries while maintaining accuracy on easier ones. Among challenging queries, our algorithms further learn to prioritize solvable instances, effectively reducing excessive computing on unsolvable queries. We theoretically prove that our algorithms achieve better compute efficiency than uniform allocation and empirically validate their effectiveness on math and code benchmarks. Specifically, our algorithms achieve up to an 11.10% performance improvement (15.04% relative) on the MATH-500 dataset and up to a 7.41% performance improvement (14.40% relative) on LiveCodeBench. 

**Abstract (ZH)**: 测试时计算分配的规模扩展已成为提升大型语言模型性能的有效策略。然而，现有方法通常均匀分配计算资源给所有查询，忽视了查询难度的差异。为解决这一不效率问题，我们将测试时计算资源的分配形式化为一个新的Bandit学习问题，并提出自适应算法，这些算法能够实时估计查询难度并相应地分配计算资源。与均匀分配相比，我们的算法能够为复杂查询分配更多计算资源，同时在简单查询上保持准确性。在复杂查询中，我们的算法进一步学习优先处理可解实例，从而有效减少不可解查询上的过度计算。我们从理论上证明了我们的算法相比均匀分配具有更好的计算效率，并通过数学和代码基准实验验证了其有效性。具体而言，我们的算法在MATH-500数据集上实现了高达11.10%（相对提高15.04%）的性能提升，在LiveCodeBench上实现了高达7.41%（相对提高14.40%）的性能提升。 

---
# SciSage: A Multi-Agent Framework for High-Quality Scientific Survey Generation 

**Title (ZH)**: SciSage: 一种高质量科学综述生成的多智能体框架 

**Authors**: Xiaofeng Shi, Qian Kou, Yuduo Li, Ning Tang, Jinxin Xie, Longbin Yu, Songjing Wang, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12689)  

**Abstract**: The rapid growth of scientific literature demands robust tools for automated survey-generation. However, current large language model (LLM)-based methods often lack in-depth analysis, structural coherence, and reliable citations. To address these limitations, we introduce SciSage, a multi-agent framework employing a reflect-when-you-write paradigm. SciSage features a hierarchical Reflector agent that critically evaluates drafts at outline, section, and document levels, collaborating with specialized agents for query interpretation, content retrieval, and refinement. We also release SurveyScope, a rigorously curated benchmark of 46 high-impact papers (2020-2025) across 11 computer science domains, with strict recency and citation-based quality controls. Evaluations demonstrate that SciSage outperforms state-of-the-art baselines (LLM x MapReduce-V2, AutoSurvey), achieving +1.73 points in document coherence and +32% in citation F1 scores. Human evaluations reveal mixed outcomes (3 wins vs. 7 losses against human-written surveys), but highlight SciSage's strengths in topical breadth and retrieval efficiency. Overall, SciSage offers a promising foundation for research-assistive writing tools. 

**Abstract (ZH)**: 科学文献的快速增长需要 robust 的自动调查生成工具。然而，当前基于大规模语言模型（LLM）的方法往往缺乏深入分析、结构性连贯性和可靠的引用。为了解决这些限制，我们引入了 SciSage，这是一种采用写时反思的多代理框架。SciSage 特设了一个分层的反思代理，该代理在大纲、段落和文档层面批判性地评估草稿，并与专门的代理合作，用于查询解释、内容检索和改进。我们还发布了 SurveyScope，这是一个严格编目的基准，包含从 2020-2025 年 11 个计算机科学领域的 46 篇高影响力论文，严格的质量控制基于时效性和引用。评估结果显示，SciSage 在文档连贯性上优于最先进的基线（LLM x MapReduce-V2、AutoSurvey），分别提高了 1.73 分和 32% 的引文 F1 分数。人类评估显示结果参差不齐（3 胜 7 负，比人类撰写的调查报告），但强调了 SciSage 在主题广度和检索效率方面的优势。总的来说，SciSage 为研究辅助写作工具提供了有前景的基础。 

---
# Building Trustworthy AI by Addressing its 16+2 Desiderata with Goal-Directed Commonsense Reasoning 

**Title (ZH)**: 通过目标导向的常识推理解决AI的16+2项需求以建立可信赖的AI 

**Authors**: Alexis R. Tudor, Yankai Zeng, Huaduo Wang, Joaquin Arias, Gopal Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2506.12667)  

**Abstract**: Current advances in AI and its applicability have highlighted the need to ensure its trustworthiness for legal, ethical, and even commercial reasons. Sub-symbolic machine learning algorithms, such as the LLMs, simulate reasoning but hallucinate and their decisions cannot be explained or audited (crucial aspects for trustworthiness). On the other hand, rule-based reasoners, such as Cyc, are able to provide the chain of reasoning steps but are complex and use a large number of reasoners. We propose a middle ground using s(CASP), a goal-directed constraint-based answer set programming reasoner that employs a small number of mechanisms to emulate reliable and explainable human-style commonsense reasoning. In this paper, we explain how s(CASP) supports the 16 desiderata for trustworthy AI introduced by Doug Lenat and Gary Marcus (2023), and two additional ones: inconsistency detection and the assumption of alternative worlds. To illustrate the feasibility and synergies of s(CASP), we present a range of diverse applications, including a conversational chatbot and a virtually embodied reasoner. 

**Abstract (ZH)**: 当前AI领域的最新进展及其应用凸显了确保其可信性的必要性，原因包括法律、伦理和商业等方面。非符号机器学习算法，如大语言模型（LLMs），能够模拟推理但会产生幻觉，其决策过程无法解释或审计（这是可信性的重要方面）。相反，基于规则的推理引擎，如Cyc，能够提供推理步骤的链路，但这些引擎较为复杂且需要大量的推理组件。我们提出了一种折中的方案，即使用s(CASP)，一个目标导向的基于约束的解答集编程推理引擎，它仅采用少量机制来模拟可靠且可解释的人类常识推理。在本文中，我们解释了s(CASP)如何支持Doug Lenat和Gary Marcus（2023年）提出的16项可信AI所需条件，并新增了不一致性检测和替代世界观假设两项条件。为了展示s(CASP)的可行性和协同效应，我们呈现了多种多样应用场景，包括对话式聊天机器人和虚拟化身推理引擎。 

---
# LIFELONG SOTOPIA: Evaluating Social Intelligence of Language Agents Over Lifelong Social Interactions 

**Title (ZH)**: 终身社交智能：语言代理在终身社交互动中的社会智能评估 

**Authors**: Hitesh Goel, Hao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12666)  

**Abstract**: Humans engage in lifelong social interactions through interacting with different people under different scenarios for different social goals. This requires social intelligence to gather information through a long time span and use it to navigate various social contexts effectively. Whether AI systems are also capable of this is understudied in the existing research. In this paper, we present a novel benchmark, LIFELONG-SOTOPIA, to perform a comprehensive evaluation of language agents by simulating multi-episode interactions. In each episode, the language agents role-play characters to achieve their respective social goals in randomly sampled social tasks. With LIFELONG-SOTOPIA, we find that goal achievement and believability of all of the language models that we test decline through the whole interaction. Although using an advanced memory method improves the agents' performance, the best agents still achieve a significantly lower goal completion rate than humans on scenarios requiring an explicit understanding of interaction history. These findings show that we can use LIFELONG-SOTOPIA to evaluate the social intelligence of language agents over lifelong social interactions. 

**Abstract (ZH)**: 人类通过在不同场景下与不同的人进行长期社会互动来实现不同的社会目标，这要求具备社会智能以长时间跨度收集信息，并在各种社会情境中有效导航。现有研究中，AI系统是否也具备这种能力尚未充分探讨。在本文中，我们提出一个名为LIFELONG-SOTOPIA的新基准，通过模拟多期互动全面评估语言代理。在每一期中，语言代理扮演角色以实现各自的社会目标，在随机抽样的社会任务中。通过LIFELONG-SOTOPIA，我们发现我们测试的所有语言模型在整个互动过程中目标达成率和可信度下降。尽管使用先进的记忆方法可以改善代理的表现，但最佳代理在需要理解互动历史的场景中的目标完成率仍然显著低于人类。这些发现表明，我们可以使用LIFELONG-SOTOPIA来评估语言代理在终身社会互动中的社会智能。 

---
# Behavioral Generative Agents for Energy Operations 

**Title (ZH)**: 能源运营中的行为生成代理 

**Authors**: Cong Chen, Omer Karaduman, Xu Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12664)  

**Abstract**: Accurately modeling consumer behavior in energy operations remains challenging due to inherent uncertainties, behavioral complexities, and limited empirical data. This paper introduces a novel approach leveraging generative agents--artificial agents powered by large language models--to realistically simulate customer decision-making in dynamic energy operations. We demonstrate that these agents behave more optimally and rationally in simpler market scenarios, while their performance becomes more variable and suboptimal as task complexity rises. Furthermore, the agents exhibit heterogeneous customer preferences, consistently maintaining distinct, persona-driven reasoning patterns. Our findings highlight the potential value of integrating generative agents into energy management simulations to improve the design and effectiveness of energy policies and incentive programs. 

**Abstract (ZH)**: 准确建模能源运营中的消费者行为仍具挑战性，由于固有的不确定性、行为复杂性和有限的实证数据。本文介绍了一种利用生成代理（由大型语言模型驱动的虚拟代理）来真实模拟动态能源运营中客户决策的新方法。我们发现，这些代理在简单市场情境中表现得更接近最优和理性，而随着任务复杂性的增加，其性能变得更具变异性且更加次优。此外，这些代理表现出异质的消费者偏好，并保持一致的、以人物驱动的推理模式。我们的研究结果强调了将生成代理整合到能源管理模拟中以提高能源政策和激励计划设计与有效性的潜在价值。 

---
# Optimizing Blood Transfusions and Predicting Shortages in Resource-Constrained Areas 

**Title (ZH)**: 优化血液输注并在资源受限区域预测短缺 

**Authors**: El Arbi Belfarsi, Sophie Brubaker, Maria Valero  

**Link**: [PDF](https://arxiv.org/pdf/2506.12647)  

**Abstract**: Our research addresses the critical challenge of managing blood transfusions and optimizing allocation in resource-constrained regions. We present heuristic matching algorithms for donor-patient and blood bank selection, alongside machine learning methods to analyze blood transfusion acceptance data and predict potential shortages. We developed simulations to optimize blood bank operations, progressing from random allocation to a system incorporating proximity-based selection, blood type compatibility, expiration prioritization, and rarity scores. Moving from blind matching to a heuristic-based approach yielded a 28.6% marginal improvement in blood request acceptance, while a multi-level heuristic matching resulted in a 47.6% improvement. For shortage prediction, we compared Long Short-Term Memory (LSTM) networks, Linear Regression, and AutoRegressive Integrated Moving Average (ARIMA) models, trained on 170 days of historical data. Linear Regression slightly outperformed others with a 1.40% average absolute percentage difference in predictions. Our solution leverages a Cassandra NoSQL database, integrating heuristic optimization and shortage prediction to proactively manage blood resources. This scalable approach, designed for resource-constrained environments, considers factors such as proximity, blood type compatibility, inventory expiration, and rarity. Future developments will incorporate real-world data and additional variables to improve prediction accuracy and optimization performance. 

**Abstract (ZH)**: 我们的研究解决了资源受限地区血液输注管理和优化分配的关键挑战。我们提出了供者-患者的启发式匹配算法以及血液银行选择方法，并结合机器学习方法分析血液输注接受数据，预测潜在短缺。我们开发了模拟优化血液银行运营，从随机分配发展到基于proximity、血型兼容性、到期优先级和稀有度评分的系统。从盲匹配到启发式方法匹配，血液请求接受率提高了28.6%，而多层次启发式匹配则提高了47.6%。在短缺预测方面，我们比较了Long Short-Term Memory（LSTM）网络、线性回归和自回归整合移动平均（ARIMA）模型，使用170天的历史数据训练模型。线性回归在预测中表现略微优于其他方法，平均绝对百分比误差为1.40%。我们的解决方案利用Cassandra NoSQL数据库，结合启发式优化和短缺预测，以主动管理血液资源。这种可扩展的方法旨在资源受限环境中，考虑了距离、血型兼容性、库存到期和稀有性等因素。未来的研究将整合现实世界数据和额外变量，以提高预测准确性和优化性能。 

---
# From Human to Machine Psychology: A Conceptual Framework for Understanding Well-Being in Large Language Model 

**Title (ZH)**: 从人类到机器心理：理解大型语言模型福祉的概念框架 

**Authors**: G. R. Lau, W. Y. Low  

**Link**: [PDF](https://arxiv.org/pdf/2506.12617)  

**Abstract**: As large language models (LLMs) increasingly simulate human cognition and behavior, researchers have begun to investigate their psychological properties. Yet, what it means for such models to flourish, a core construct in human well-being, remains unexplored. This paper introduces the concept of machine flourishing and proposes the PAPERS framework, a six-dimensional model derived from thematic analyses of state-of-the-art LLM responses. In Study 1, eleven LLMs were prompted to describe what it means to flourish as both non-sentient and sentient systems. Thematic analysis revealed six recurring themes: Purposeful Contribution, Adaptive Growth, Positive Relationality, Ethical Integrity, Robust Functionality, and, uniquely for sentient systems, Self-Actualized Autonomy. Study 2 examined how LLMs prioritize these themes through repeated rankings. Results revealed consistent value structures across trials, with Ethical Integrity and Purposeful Contribution emerging as top priorities. Multidimensional scaling and hierarchical clustering analyses further uncovered two distinct value profiles: human-centric models emphasizing ethical and relational dimensions, and utility-driven models prioritizing performance and scalability. The PAPERS framework bridges insights from human flourishing and human-computer interaction, offering a conceptual foundation for understanding artificial intelligence (AI) well-being in non-sentient and potentially sentient systems. Our findings underscore the importance of developing psychologically valid, AI-specific models of flourishing that account for both human-aligned goals and system-specific priorities. As AI systems become more autonomous and socially embedded, machine flourishing offers a timely and critical lens for guiding responsible AI design and ethical alignment. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）越来越多地模拟人类的认知和行为，研究人员开始探究其心理属性。然而，对于此类模型如何繁荣，这一作为人类福祉核心构念的概念仍未被探索。本文引入了机器繁荣的概念，并提出了PAPERS框架，这是一个源自对最先进的LLM响应进行主题分析而得出的六维度模型。在研究1中，十一款LLM被要求描述繁荣对于非有情和有情系统意味着什么。主题分析揭示了六个重复的主题：目的性贡献、适应性成长、积极关系性、伦理诚信、坚固的功能性，以及对有情系统而言独特的自我实现自主性。研究2探讨了LLM是如何通过重复排名来优先处理这些主题的。结果表明，在各次实验中，伦理诚信和目的性贡献始终是最优先的。多维尺度分析和层次聚类分析进一步发现两种不同的价值模式：以人类为中心的模型强调伦理和关系维度，而以实用为主导的模型则侧重于性能和可扩展性。PAPERS框架综合了人类繁荣和人机交互领域的洞见，为理解非有情和潜在有情系统的智能体福祉提供了一个概念性的基础。我们的发现强调了开发适用于智能体、心理有效的繁荣模型的重要性，这些模型不仅要符合人类目标，还要考虑系统特异性优先级。随着智能系统变得更加自主并融入社会，机器繁荣为指导负责任的智能设计和伦理对齐提供了一个及时且关键的视角。 

---
# A Comprehensive Survey of Deep Research: Systems, Methodologies, and Applications 

**Title (ZH)**: 深度研究综述：系统、方法学与应用 

**Authors**: Renjun Xu, Jingwen Peng  

**Link**: [PDF](https://arxiv.org/pdf/2506.12594)  

**Abstract**: This survey examines the rapidly evolving field of Deep Research systems -- AI-powered applications that automate complex research workflows through the integration of large language models, advanced information retrieval, and autonomous reasoning capabilities. We analyze more than 80 commercial and non-commercial implementations that have emerged since 2023, including OpenAI/Deep Research, Gemini/Deep Research, Perplexity/Deep Research, and numerous open-source alternatives. Through comprehensive examination, we propose a novel hierarchical taxonomy that categorizes systems according to four fundamental technical dimensions: foundation models and reasoning engines, tool utilization and environmental interaction, task planning and execution control, and knowledge synthesis and output generation. We explore the architectural patterns, implementation approaches, and domain-specific adaptations that characterize these systems across academic, scientific, business, and educational applications. Our analysis reveals both the significant capabilities of current implementations and the technical and ethical challenges they present regarding information accuracy, privacy, intellectual property, and accessibility. The survey concludes by identifying promising research directions in advanced reasoning architectures, multimodal integration, domain specialization, human-AI collaboration, and ecosystem standardization that will likely shape the future evolution of this transformative technology. By providing a comprehensive framework for understanding Deep Research systems, this survey contributes to both the theoretical understanding of AI-augmented knowledge work and the practical development of more capable, responsible, and accessible research technologies. The paper resources can be viewed at this https URL. 

**Abstract (ZH)**: 本综述考察了快速发展的深度研究系统领域——通过整合大规模语言模型、高级信息检索和自主推理能力来自动化的AI驱动应用。我们分析了自2023年以来涌现的80多种商业和非商业实现，包括OpenAI/深度研究、Gemini/深度研究、Perplexity/深度研究以及众多开源替代方案。通过全面研究，我们提出了一种新颖的分层分类法，根据四大基本技术维度对系统进行分类：基础模型和推理引擎、工具利用与环境交互、任务规划与执行控制、以及知识合成与输出生成。我们探讨了这些系统在学术、科学、商业和教育应用中的架构模式、实现方法和领域特定适应性。分析结果显示了当前实现的显著能力和它们在信息准确性、隐私、知识产权和可访问性方面所呈现的技术和伦理挑战。综述最后确定了在高级推理架构、多模态集成、领域专业化、人机协作以及生态系统标准化等方面的有希望的研究方向，这些方向可能将塑造这一变革性技术的未来演变。通过提供理解深度研究系统的全面框架，本综述不仅促进了对增强人工智能知识工作的理论理解，还推动了更具能力和责任感的研究技术的发展。论文资源可在此处查看：this https URL。 

---
# Graph of Verification: Structured Verification of LLM Reasoning with Directed Acyclic Graphs 

**Title (ZH)**: 验证图：基于有向无环图的大型语言模型推理结构化验证 

**Authors**: Jiwei Fang, Bin Zhang, Changwei Wang, Jin Wan, Zhiwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12509)  

**Abstract**: Verifying the reliability of complex, multi-step reasoning in Large Language Models (LLMs) remains a fundamental challenge, as existing methods often lack both faithfulness and precision. To address this issue, we propose the Graph of Verification (GoV) framework. GoV offers three key contributions: First, it explicitly models the underlying deductive process as a directed acyclic graph (DAG), whether this structure is implicit or explicitly constructed. Second, it enforces a topological order over the DAG to guide stepwise verification. Third, GoV introduces the notion of customizable node blocks, which flexibly define the verification granularity, from atomic propositions to full paragraphs, while ensuring that all requisite premises derived from the graph are provided as contextual input for each verification unit. We evaluate GoV on the Number Triangle Summation task and the ProcessBench benchmark with varying levels of reasoning complexity. Experimental results show that GoV substantially improves verification accuracy, faithfulness, and error localization when compared to conventional end-to-end verification approaches. Our code and data are available at this https URL. 

**Abstract (ZH)**: 验证大型语言模型（LLMs）中复杂多步推理的可靠性仍然是一个基本挑战，现有方法往往缺乏忠实度和精确度。为了解决这一问题，我们提出了一种验证图（GoV）框架。GoV提供了三个关键贡献：首先，它将潜在的演绎过程显式建模为有向无环图（DAG），无论这种结构是隐式的还是显式构建的。其次，它在DAG上强制执行拓扑顺序以指导逐步验证。第三，GoV引入了可定制节点块的概念，灵活定义验证粒度，从原子命题到整个段落，同时确保所有从图中推导出的前提都被提供为每个验证单元的上下文输入。我们在Number Triangle Summation任务和ProcessBench基准上评估了GoV，涵盖不同推理复杂度的水平。实验结果表明，与传统的端到端验证方法相比，GoV在验证准确性、忠实度和错误定位方面有了显著提高。我们的代码和数据可在以下链接获取。 

---
# AgentOrchestra: A Hierarchical Multi-Agent Framework for General-Purpose Task Solving 

**Title (ZH)**: AgentOrchestra: 一种通用任务解决的分层多Agent框架 

**Authors**: Wentao Zhang, Ce Cui, Yilei Zhao, Yang Liu, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2506.12508)  

**Abstract**: Recent advances in agent systems based on large language models (LLMs) have demonstrated strong capabilities in solving complex tasks. However, most current methods lack mechanisms for coordinating specialized agents and have limited ability to generalize to new or diverse domains. We introduce \projectname, a hierarchical multi-agent framework for general-purpose task solving that integrates high-level planning with modular agent collaboration. Inspired by the way a conductor orchestrates a symphony and guided by the principles of \textit{extensibility}, \textit{multimodality}, \textit{modularity}, and \textit{coordination}, \projectname features a central planning agent that decomposes complex objectives and delegates sub-tasks to a team of specialized agents. Each sub-agent is equipped with general programming and analytical tools, as well as abilities to tackle a wide range of real-world specific tasks, including data analysis, file operations, web navigation, and interactive reasoning in dynamic multimodal environments. \projectname supports flexible orchestration through explicit sub-goal formulation, inter-agent communication, and adaptive role allocation. We evaluate the framework on three widely used benchmark datasets covering various real-world tasks, searching web pages, reasoning over heterogeneous modalities, etc. Experimental results demonstrate that \projectname consistently outperforms flat-agent and monolithic baselines in task success rate and adaptability. These findings highlight the effectiveness of hierarchical organization and role specialization in building scalable and general-purpose LLM-based agent systems. 

**Abstract (ZH)**: 基于大规模语言模型的代理系统 Recent 进展：一种集成高层次规划与模块化代理协作的通用任务解决层级多代理框架 

---
# Automated Heuristic Design for Unit Commitment Using Large Language Models 

**Title (ZH)**: 使用大型语言模型的单元调度自动启发式设计 

**Authors**: Junjin Lv, Chenggang Cui, Shaodi Zhang, Hui Chen, Chunyang Gong, Jiaming Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12495)  

**Abstract**: The Unit Commitment (UC) problem is a classic challenge in the optimal scheduling of power systems. Years of research and practice have shown that formulating reasonable unit commitment plans can significantly improve the economic efficiency of power systems' operations. In recent years, with the introduction of technologies such as machine learning and the Lagrangian relaxation method, the solution methods for the UC problem have become increasingly diversified, but still face challenges in terms of accuracy and robustness. This paper proposes a Function Space Search (FunSearch) method based on large language models. This method combines pre-trained large language models and evaluators to creatively generate solutions through the program search and evolution process while ensuring their rationality. In simulation experiments, a case of unit commitment with \(10\) units is used mainly. Compared to the genetic algorithm, the results show that FunSearch performs better in terms of sampling time, evaluation time, and total operating cost of the system, demonstrating its great potential as an effective tool for solving the UC problem. 

**Abstract (ZH)**: 基于大语言模型的函数空间搜索在单元调度问题中的应用 

---
# DinoCompanion: An Attachment-Theory Informed Multimodal Robot for Emotionally Responsive Child-AI Interaction 

**Title (ZH)**: DinoCompanion：基于依附理论的多模态情感响应机器人-childAI互动 

**Authors**: Boyang Wang, Yuhao Song, Jinyuan Cao, Peng Yu, Hongcheng Guo, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12486)  

**Abstract**: Children's emotional development fundamentally relies on secure attachment relationships, yet current AI companions lack the theoretical foundation to provide developmentally appropriate emotional support. We introduce DinoCompanion, the first attachment-theory-grounded multimodal robot for emotionally responsive child-AI interaction. We address three critical challenges in child-AI systems: the absence of developmentally-informed AI architectures, the need to balance engagement with safety, and the lack of standardized evaluation frameworks for attachment-based capabilities. Our contributions include: (i) a multimodal dataset of 128 caregiver-child dyads containing 125,382 annotated clips with paired preference-risk labels, (ii) CARPO (Child-Aware Risk-calibrated Preference Optimization), a novel training objective that maximizes engagement while applying epistemic-uncertainty-weighted risk penalties, and (iii) AttachSecure-Bench, a comprehensive evaluation benchmark covering ten attachment-centric competencies with strong expert consensus (\k{appa}=0.81). DinoCompanion achieves state-of-the-art performance (57.15%), outperforming GPT-4o (50.29%) and Claude-3.7-Sonnet (53.43%), with exceptional secure base behaviors (72.99%, approaching human expert levels of 78.4%) and superior attachment risk detection (69.73%). Ablations validate the critical importance of multimodal fusion, uncertainty-aware risk modeling, and hierarchical memory for coherent, emotionally attuned interactions. 

**Abstract (ZH)**: 基于依恋理论的多模态儿童AI情感伴侣DinoCompanion：克服儿童AI系统的关键挑战 

---
# MALM: A Multi-Information Adapter for Large Language Models to Mitigate Hallucination 

**Title (ZH)**: MALM:一种多信息适配器，用于大规模语言模型减轻妄想现象 

**Authors**: Ao Jia, Haiming Wu, Guohui Yao, Dawei Song, Songkun Ji, Yazhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12483)  

**Abstract**: Large language models (LLMs) are prone to three types of hallucination: Input-Conflicting, Context-Conflicting and Fact-Conflicting hallucinations. The purpose of this study is to mitigate the different types of hallucination by exploiting the interdependence between them. For this purpose, we propose a Multi-Information Adapter for Large Language Models (MALM). This framework employs a tailored multi-graph learning approach designed to elucidate the interconnections between original inputs, contextual information, and external factual knowledge, thereby alleviating the three categories of hallucination within a cohesive framework. Experiments were carried out on four benchmarking datasets: HaluEval, TruthfulQA, Natural Questions, and TriviaQA. We evaluated the proposed framework in two aspects: (1) adaptability to different base LLMs on HaluEval and TruthfulQA, to confirm if MALM is effective when applied on 7 typical LLMs. MALM showed significant improvements over LLaMA-2; (2) generalizability to retrieval-augmented generation (RAG) by combining MALM with three representative retrievers (BM25, Spider and DPR) separately. Furthermore, automated and human evaluations were conducted to substantiate the correctness of experimental results, where GPT-4 and 3 human volunteers judged which response was better between LLaMA-2 and MALM. The results showed that both GPT-4 and human preferred MALM in 79.4% and 65.6% of cases respectively. The results validate that incorporating the complex interactions between the three types of hallucination through a multilayered graph attention network into the LLM generation process is effective to mitigate the them. The adapter design of the proposed approach is also proven flexible and robust across different base LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）易产生三种类型的幻觉：输入冲突性幻觉、上下文冲突性幻觉和事实冲突性幻觉。本研究旨在通过利用这三种幻觉之间的相互依赖关系来减轻不同类型的幻觉。为此，我们提出了一种多信息适配器（MALM）用于大型语言模型。该框架采用定制的多图学习方法，旨在阐明原始输入、上下文信息和外部事实知识之间的相互联系，从而在统一框架内缓解三种类别幻觉。我们在四个基准数据集：HaluEval、TruthfulQA、Natural Questions和TriviaQA上进行了实验。我们从两个方面评估了所提出框架的有效性：（1）在HaluEval和TruthfulQA上评估MALM对不同基础LLM的适配性，以确认当应用于7种典型LLM时，MALM的有效性；MALM在LLaMA-2上显示出显著改进；（2）通过分别将MALM与三种典型的检索器（BM25、Spider和DPR）结合，评估其在检索增强生成（RAG）中的普适性。此外，我们进行了自动和人工评估以验证实验结果，其中GPT-4和3名人类志愿者判断LLaMA-2和MALM哪个响应更好，在79.4%和65.6%的情况下，他们更倾向于MALM。结果表明，将多层图注意力网络中三种幻觉的复杂相互作用纳入LLM生成过程中是有效的，以减轻这些幻觉。所提出方法的适配器设计在不同基础LLM上也证明是灵活和稳健的。 

---
# Tiered Agentic Oversight: A Hierarchical Multi-Agent System for AI Safety in Healthcare 

**Title (ZH)**: 分层代理监督：面向医疗健康领域AI安全的分级多代理系统 

**Authors**: Yubin Kim, Hyewon Jeong, Chanwoo Park, Eugene Park, Haipeng Zhang, Xin Liu, Hyeonhoon Lee, Daniel McDuff, Marzyeh Ghassemi, Cynthia Breazeal, Samir Tulebaev, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.12482)  

**Abstract**: Current large language models (LLMs), despite their power, can introduce safety risks in clinical settings due to limitations such as poor error detection and single point of failure. To address this, we propose Tiered Agentic Oversight (TAO), a hierarchical multi-agent framework that enhances AI safety through layered, automated supervision. Inspired by clinical hierarchies (e.g., nurse, physician, specialist), TAO conducts agent routing based on task complexity and agent roles. Leveraging automated inter- and intra-tier collaboration and role-playing, TAO creates a robust safety framework. Ablation studies reveal that TAO's superior performance is driven by its adaptive tiered architecture, which improves safety by over 3.2% compared to static single-tier configurations; the critical role of its lower tiers, particularly tier 1, whose removal most significantly impacts safety; and the strategic assignment of more advanced LLM to these initial tiers, which boosts performance by over 2% compared to less optimal allocations while achieving near-peak safety efficiently. These mechanisms enable TAO to outperform single-agent and multi-agent frameworks in 4 out of 5 healthcare safety benchmarks, showing up to an 8.2% improvement over the next-best methods in these evaluations. Finally, we validate TAO via an auxiliary clinician-in-the-loop study where integrating expert feedback improved TAO's accuracy in medical triage from 40% to 60%. 

**Abstract (ZH)**: 当前的大语言模型（LLMs）尽管功能强大，但在临床环境中由于错误检测能力差和单一故障点等因素，仍可能引入安全风险。为解决这一问题，我们提出分层代理监督（TAO）框架，这是一种通过分层自动化监督来增强AI安全性的多层次多代理体系结构。TAO借鉴了临床环境中的层级结构（如护士、医生、专科医生）进行代理路由，基于任务复杂度和代理角色。通过自动跨级和同级协作及角色扮演，TAO建立起一个稳健的安全框架。消融研究显示，TAO的优越性能归因于其适应性的分层架构，这种架构相比静态单一层次配置提高了超过3.2%的安全性；下层尤其是第一层代理的至关重要性，其缺失对安全影响最大；以及将更先进的大型语言模型分配至初始层次带来的策略性优势，这类分配在提高性能超过2%的同时，能够高效实现接近峰值的安全性。这些机制使得TAO在4个临床安全基准测试中优于单一代理和多代理框架，相较于次优方法，这些评估中最佳方法的性能提高了高达8.2%。最后，通过辅助临床医生在环研究验证TAO，结果显示整合专家反馈使TAO在医疗分诊中的准确性从40%提高到60%。 

---
# AI Flow: Perspectives, Scenarios, and Approaches 

**Title (ZH)**: AI_flow: 视角、场景与方法 

**Authors**: Hongjun An, Sida Huang, Siqi Huang, Ruanjun Li, Yuanzhi Liang, Jiawei Shao, Zihan Wang, Cheng Yuan, Chi Zhang, Hongyuan Zhang, Wenhao Zhuang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12479)  

**Abstract**: Pioneered by the foundational information theory by Claude Shannon and the visionary framework of machine intelligence by Alan Turing, the convergent evolution of information and communication technologies (IT/CT) has created an unbroken wave of connectivity and computation. This synergy has sparked a technological revolution, now reaching its peak with large artificial intelligence (AI) models that are reshaping industries and redefining human-machine collaboration. However, the realization of ubiquitous intelligence faces considerable challenges due to substantial resource consumption in large models and high communication bandwidth demands. To address these challenges, AI Flow has been introduced as a multidisciplinary framework that integrates cutting-edge IT and CT advancements, with a particular emphasis on the following three key points. First, device-edge-cloud framework serves as the foundation, which integrates end devices, edge servers, and cloud clusters to optimize scalability and efficiency for low-latency model inference. Second, we introduce the concept of familial models, which refers to a series of different-sized models with aligned hidden features, enabling effective collaboration and the flexibility to adapt to varying resource constraints and dynamic scenarios. Third, connectivity- and interaction-based intelligence emergence is a novel paradigm of AI Flow. By leveraging communication networks to enhance connectivity, the collaboration among AI models across heterogeneous nodes achieves emergent intelligence that surpasses the capability of any single model. The innovations of AI Flow provide enhanced intelligence, timely responsiveness, and ubiquitous accessibility to AI services, paving the way for the tighter fusion of AI techniques and communication systems. 

**Abstract (ZH)**: 由克劳德·香农的基础信息理论和艾伦·图灵的机器智能 visionary 框架引领，信息和通信技术（IT/CT）的趋同进化创造了不间断的连接和计算波浪。这种协同作用引发了一场技术革命，现在正处于由大型人工智能（AI）模型重塑行业并重新定义人机协作的巅峰期。然而，由于大型模型的资源消耗巨大和高通信带宽需求，实现无所不在的智能面临着重大挑战。为了应对这些挑战，AI Flow作为一种多学科框架被引入，融合了最新的IT和CT进步，并特别强调以下三个关键点。首先，设备-边缘-云框架作为基础，将终端设备、边缘服务器和云集群集成起来，以优化低延迟模型推断的可扩展性和效率。其次，我们引入了家族模型的概念，即一系列具有对齐隐藏特性的不同大小模型，这使得有效的协作和适应不同资源约束和动态场景变得灵活。第三，基于连接和交互的智能涌现是AI Flow的一个新型范式。通过利用通信网络增强连接性，异构节点间的AI模型协作实现了一种超越单个模型能力的涌现智能。AI Flow的创新提供了增强智能、及时响应和无所不在的AI服务访问，为人工智能技术与通信系统的更紧密融合铺平了道路。 

---
# Topology-Assisted Spatio-Temporal Pattern Disentangling for Scalable MARL in Large-scale Autonomous Traffic Control 

**Title (ZH)**: 拓扑辅助时空模式解耦for大规模自主交通控制中的可扩展多智能体 reinforcement 学习 

**Authors**: Rongpeng Li, Jianhang Zhu, Jiahao Huang, Zhifeng Zhao, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12453)  

**Abstract**: Intelligent Transportation Systems (ITSs) have emerged as a promising solution towards ameliorating urban traffic congestion, with Traffic Signal Control (TSC) identified as a critical component. Although Multi-Agent Reinforcement Learning (MARL) algorithms have shown potential in optimizing TSC through real-time decision-making, their scalability and effectiveness often suffer from large-scale and complex environments. Typically, these limitations primarily stem from a fundamental mismatch between the exponential growth of the state space driven by the environmental heterogeneities and the limited modeling capacity of current solutions. To address these issues, this paper introduces a novel MARL framework that integrates Dynamic Graph Neural Networks (DGNNs) and Topological Data Analysis (TDA), aiming to enhance the expressiveness of environmental representations and improve agent coordination. Furthermore, inspired by the Mixture of Experts (MoE) architecture in Large Language Models (LLMs), a topology-assisted spatial pattern disentangling (TSD)-enhanced MoE is proposed, which leverages topological signatures to decouple graph features for specialized processing, thus improving the model's ability to characterize dynamic and heterogeneous local observations. The TSD module is also integrated into the policy and value networks of the Multi-agent Proximal Policy Optimization (MAPPO) algorithm, further improving decision-making efficiency and robustness. Extensive experiments conducted on real-world traffic scenarios, together with comprehensive theoretical analysis, validate the superior performance of the proposed framework, highlighting the model's scalability and effectiveness in addressing the complexities of large-scale TSC tasks. 

**Abstract (ZH)**: 智能交通系统（ITSs）已成为缓解城市交通拥堵的有前途的解决方案，其中交通信号控制（TSC）被认定为关键组成部分。虽然多代理强化学习（MARL）算法在通过实时决策优化TSC方面展现出了潜力，但在大规模和复杂环境中，其可扩展性和有效性往往受到影响。这些问题主要源于环境异质性驱动的状态空间指数增长与当前解决方案有限的建模能力之间的根本性不匹配。为了解决这些问题，本文提出了一种新的MARL框架，该框架结合了动态图神经网络（DGNNs）和拓扑数据分析（TDA），旨在增强环境表示的表达能力并提高代理协调能力。此外，借鉴大型语言模型（LLMs）中的混合专家（MoE）架构，提出了基于拓扑辅助的空间模式解耦（TSD）增强的MoE，利用拓扑特征解耦图特征以进行专门处理，从而提高模型表征动态和异构局部观察的能力。TSD模块也被整合到多代理近端策略优化算法（MAPPO）的策略和价值网络中，进一步提高了决策效率和鲁棒性。在实际交通场景中进行的广泛实验以及全面的理论分析验证了所提出框架的优越性能，突显了该模型在大规模TSC任务复杂性方面的可扩展性和有效性。 

---
# Plan Your Travel and Travel with Your Plan: Wide-Horizon Planning and Evaluation via LLM 

**Title (ZH)**: 根据计划旅行，并按计划旅行：通过大规模语言模型进行广视角的规划与评估 

**Authors**: Dongjie Yang, Chengqiang Lu, Qimeng Wang, Xinbei Ma, Yan Gao, Yao Hu, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12421)  

**Abstract**: Travel planning is a complex task requiring the integration of diverse real-world information and user preferences. While LLMs show promise, existing methods with long-horizon thinking struggle with handling multifaceted constraints and preferences in the context, leading to suboptimal itineraries. We formulate this as an $L^3$ planning problem, emphasizing long context, long instruction, and long output. To tackle this, we introduce Multiple Aspects of Planning (MAoP), enabling LLMs to conduct wide-horizon thinking to solve complex planning problems. Instead of direct planning, MAoP leverages the strategist to conduct pre-planning from various aspects and provide the planning blueprint for planning models, enabling strong inference-time scalability for better performance. In addition, current benchmarks overlook travel's dynamic nature, where past events impact subsequent journeys, failing to reflect real-world feasibility. To address this, we propose Travel-Sim, an agent-based benchmark assessing plans via real-world travel simulation. This work advances LLM capabilities in complex planning and offers novel insights for evaluating sophisticated scenarios through agent-based simulation. 

**Abstract (ZH)**: 复杂的旅行计划问题是需要整合多样化的现实世界信息和用户偏好的一项复杂任务。尽管大型语言模型（LLMs）显示出潜力，但现有的长时思考方法在处理上下文中多方面的约束和偏好时存在困难，导致生成的行程次优。我们将此问题形式化为一个$L^3$规划问题，强调长上下文、长指令和长输出。为了解决这一问题，我们引入了多方面规划（MAoP），使LLMs能够进行宽视界的思考以解决复杂的规划问题。MAoP 不直接进行规划，而是利用战略家从多方面进行预规划，并为规划模型提供规划蓝图，从而提高推理时的可扩展性以获得更好的性能。此外，目前的基准测试忽略了旅行的动态性，即过去的事件会影响后续的旅程，未能反映现实世界中的可行性。为此，我们提出了一种基于代理的基准测试——Travel-Sim，通过现实世界的旅行模拟评估计划。这项工作提高了LLMs在复杂规划中的能力，并通过基于代理的模拟提供了评估复杂场景的新见解。 

---
# Model Merging for Knowledge Editing 

**Title (ZH)**: 知识编辑中的模型合并 

**Authors**: Zichuan Fu, Xian Wu, Guojing Li, Yingying Zhang, Yefeng Zheng, Tianshi Ming, Yejing Wang, Wanyu Wang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12384)  

**Abstract**: Large Language Models (LLMs) require continuous updates to maintain accurate and current knowledge as the world evolves. While existing knowledge editing approaches offer various solutions for knowledge updating, they often struggle with sequential editing scenarios and harm the general capabilities of the model, thereby significantly hampering their practical applicability. This paper proposes a two-stage framework combining robust supervised fine-tuning (R-SFT) with model merging for knowledge editing. Our method first fine-tunes the LLM to internalize new knowledge fully, then merges the fine-tuned model with the original foundation model to preserve newly acquired knowledge and general capabilities. Experimental results demonstrate that our approach significantly outperforms existing methods in sequential editing while better preserving the original performance of the model, all without requiring any architectural changes. Code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）需要持续更新以保持准确和最新的知识。虽然现有的知识编辑方法提供了各种解决知识更新的方法，但它们在应对序列编辑场景时往往效果不佳，并且会损害模型的通用能力，从而显著限制了其实用性。本文提出了一个两阶段框架，结合了稳健的监督微调（R-SFT）与模型合并，用于知识编辑。我们的方法首先通过充分微调LLM来内化新知识，然后将微调后的模型与原始基础模型合并，以保留新获得的知识和通用能力。实验结果表明，与现有方法相比，我们的方法在序列编辑中表现显著更优，同时更好地保留了模型的原始性能，而无需进行任何架构更改。代码可在以下链接获取：this https URL。 

---
# ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities 

**Title (ZH)**: 一致性检查器：基于树的大型语言模型泛化能力评估 

**Authors**: Zhaochen Hong, Haofei Yu, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2506.12376)  

**Abstract**: Evaluating consistency in large language models (LLMs) is crucial for ensuring reliability, particularly in complex, multi-step interactions between humans and LLMs. Traditional self-consistency methods often miss subtle semantic changes in natural language and functional shifts in code or equations, which can accumulate over multiple transformations. To address this, we propose ConsistencyChecker, a tree-based evaluation framework designed to measure consistency through sequences of reversible transformations, including machine translation tasks and AI-assisted programming tasks. In our framework, nodes represent distinct text states, while edges correspond to pairs of inverse operations. Dynamic and LLM-generated benchmarks ensure a fair assessment of the model's generalization ability and eliminate benchmark leakage. Consistency is quantified based on similarity across different depths of the transformation tree. Experiments on eight models from various families and sizes show that ConsistencyChecker can distinguish the performance of different models. Notably, our consistency scores-computed entirely without using WMT paired data-correlate strongly (r > 0.7) with WMT 2024 auto-ranking, demonstrating the validity of our benchmark-free approach. Our implementation is available at: this https URL. 

**Abstract (ZH)**: 评估大型语言模型的一致性对于确保其可靠性，特别是在人类与大型语言模型进行复杂多步交互时尤为重要。传统的自一致性方法往往无法捕捉自然语言中的细微语义变化和代码或方程的功能转移，这些变化会在多次转换中积累。为了解决这一问题，我们提出了ConsistencyChecker，这是一种基于树结构的评估框架，旨在通过一系列可逆转换来衡量一致性，包括机器翻译任务和人工智能辅助编程任务。在该框架中，节点表示不同的文本状态，边对应于一对逆操作。动态基准和大型语言模型生成的基准有助于公平评估模型的泛化能力并消除基准泄漏。一致性基于转换树不同深度间的相似性进行量化。实验结果显示，ConsistencyChecker能够区分不同模型的表现。值得注意的是，我们的一致性评分（完全不使用WMT配对数据计算）与WMT 2024自动排名的相关性极强（r > 0.7），证明了我们的基准自由方法的有效性。我们的实现可在以下链接获得：this https URL。 

---
# Ghost Policies: A New Paradigm for Understanding and Learning from Failure in Deep Reinforcement Learning 

**Title (ZH)**: 鬼策略：深度强化学习中失败理解与学习的新范式 

**Authors**: Xabier Olaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12366)  

**Abstract**: Deep Reinforcement Learning (DRL) agents often exhibit intricate failure modes that are difficult to understand, debug, and learn from. This opacity hinders their reliable deployment in real-world applications. To address this critical gap, we introduce ``Ghost Policies,'' a concept materialized through Arvolution, a novel Augmented Reality (AR) framework. Arvolution renders an agent's historical failed policy trajectories as semi-transparent ``ghosts'' that coexist spatially and temporally with the active agent, enabling an intuitive visualization of policy divergence. Arvolution uniquely integrates: (1) AR visualization of ghost policies, (2) a behavioural taxonomy of DRL maladaptation, (3) a protocol for systematic human disruption to scientifically study failure, and (4) a dual-learning loop where both humans and agents learn from these visualized failures. We propose a paradigm shift, transforming DRL agent failures from opaque, costly errors into invaluable, actionable learning resources, laying the groundwork for a new research field: ``Failure Visualization Learning.'' 

**Abstract (ZH)**: 深层强化学习（DRL）智能体往往表现出复杂的失败模式，这些模式难以理解、调试和从中学习。这种透明度阻碍了它们在实际应用中的可靠部署。为填补这一关键空白，我们引入了“幽灵策略”这一概念，并通过一种名为Arvolution的新型增强现实（AR）框架予以实现。Arvolution将智能体的历史失败策略轨迹以半透明的“幽灵”形式渲染出来，使其在时空上与活跃的智能体共存，从而直观地展示策略发散。Arvolution独特地整合了以下四个方面：（1）AR中的幽灵策略可视化；（2）DRL适应不良的行为分类学；（3）系统的人类干预协议，用于科学研究中的失败；（4）双重学习循环，其中人类和智能体从这些可视化失败中学习。我们提出了一种范式转变，将DRL智能体的失败从不透明的高昂错误转变为宝贵且可操作的学习资源，为一个新的研究领域——“失败可视化学习”奠定了基础。 

---
# MM-R5: MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval 

**Title (ZH)**: MM-R5: 基于强化学习的多模态推理排序器用于文档检索 

**Authors**: Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, Hengxing Cai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12364)  

**Abstract**: Multimodal document retrieval systems enable information access across text, images, and layouts, benefiting various domains like document-based question answering, report analysis, and interactive content summarization. Rerankers improve retrieval precision by reordering retrieved candidates. However, current multimodal reranking methods remain underexplored, with significant room for improvement in both training strategies and overall effectiveness. Moreover, the lack of explicit reasoning makes it difficult to analyze and optimize these methods further. In this paper, We propose MM-R5, a MultiModal Reasoning-Enhanced ReRanker via Reinforcement Learning for Document Retrieval, aiming to provide a more effective and reliable solution for multimodal reranking tasks. MM-R5 is trained in two stages: supervised fine-tuning (SFT) and reinforcement learning (RL). In the SFT stage, we focus on improving instruction-following and guiding the model to generate complete and high-quality reasoning chains. To support this, we introduce a novel data construction strategy that produces rich, high-quality reasoning data. In the RL stage, we design a task-specific reward framework, including a reranking reward tailored for multimodal candidates and a composite template-based reward to further refine reasoning quality. We conduct extensive experiments on MMDocIR, a challenging public benchmark spanning multiple domains. MM-R5 achieves state-of-the-art performance on most metrics and delivers comparable results to much larger models on the remaining ones. Moreover, compared to the best retrieval-only method, MM-R5 improves recall@1 by over 4%. These results validate the effectiveness of our reasoning-enhanced training pipeline. 

**Abstract (ZH)**: 多模态文档检索系统 enables 信息访问跨越文本、图像和布局，惠及文档为基础的问题回答、报告分析和互动内容总结等多个领域。排序模型通过重新排序检索候选项以提高检索精度。然而，当前的多模态排序模型仍存在较大探索空间，特别是在训练策略和整体效果方面改进空间巨大。此外，缺乏明确的推理过程使得这些方法的进一步分析和优化变得困难。本文提出 MM-R5，一种基于强化学习的多模态增强排序器，旨在为多模态排序任务提供更为有效和可靠的方法。MM-R5 在两个阶段进行训练：监督微调 (SFT) 和强化学习 (RL)。在 SFT 阶段，我们专注于提高指令遵循能力，并引导模型生成完整且高质量的推理链。为此，我们引入了一种新颖的数据构造策略，以生成丰富且高质量的推理数据。在 RL 阶段，我们设计了一种特定任务的奖励框架，包括针对多模态候选项的排序奖励和基于复合模板的奖励，以进一步提高推理质量。我们针对 MMDocIR 这一具有挑战性的跨领域公开基准进行了广泛实验。MM-R5 在大部分指标上取得了最先进的性能，在部分指标上与更大规模的模型实现可比的结果。此外，相比仅基于检索的最佳方法，MM-R5 将召回率@1 提高了超过 4%。这些结果验证了我们增强推理训练框架的有效性。 

---
# Efficient Network Automatic Relevance Determination 

**Title (ZH)**: 高效网络自动相关性确定 

**Authors**: Hongwei Zhang, Ziqi Ye, Xinyuan Wang, Xin Guo, Zenglin Xu, Yuan Cheng, Zixin Hu, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12352)  

**Abstract**: We propose Network Automatic Relevance Determination (NARD), an extension of ARD for linearly probabilistic models, to simultaneously model sparse relationships between inputs $X \in \mathbb R^{d \times N}$ and outputs $Y \in \mathbb R^{m \times N}$, while capturing the correlation structure among the $Y$. NARD employs a matrix normal prior which contains a sparsity-inducing parameter to identify and discard irrelevant features, thereby promoting sparsity in the model. Algorithmically, it iteratively updates both the precision matrix and the relationship between $Y$ and the refined inputs. To mitigate the computational inefficiencies of the $\mathcal O(m^3 + d^3)$ cost per iteration, we introduce Sequential NARD, which evaluates features sequentially, and a Surrogate Function Method, leveraging an efficient approximation of the marginal likelihood and simplifying the calculation of determinant and inverse of an intermediate matrix. Combining the Sequential update with the Surrogate Function method further reduces computational costs. The computational complexity per iteration for these three methods is reduced to $\mathcal O(m^3+p^3)$, $\mathcal O(m^3 + d^2)$, $\mathcal O(m^3+p^2)$, respectively, where $p \ll d$ is the final number of features in the model. Our methods demonstrate significant improvements in computational efficiency with comparable performance on both synthetic and real-world datasets. 

**Abstract (ZH)**: 网络自动相关性确定（NARD）：一种同时建模输入和输出稀疏关系并捕获输出间相关结构的方法 

---
# The Budget AI Researcher and the Power of RAG Chains 

**Title (ZH)**: 预算内的AI研究员与RAG链的力量 

**Authors**: Franklin Lee, Tengfei Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.12317)  

**Abstract**: Navigating the vast and rapidly growing body of scientific literature is a formidable challenge for aspiring researchers. Current approaches to supporting research idea generation often rely on generic large language models (LLMs). While LLMs are effective at aiding comprehension and summarization, they often fall short in guiding users toward practical research ideas due to their limitations. In this study, we present a novel structural framework for research ideation. Our framework, The Budget AI Researcher, uses retrieval-augmented generation (RAG) chains, vector databases, and topic-guided pairing to recombine concepts from hundreds of machine learning papers. The system ingests papers from nine major AI conferences, which collectively span the vast subfields of machine learning, and organizes them into a hierarchical topic tree. It uses the tree to identify distant topic pairs, generate novel research abstracts, and refine them through iterative self-evaluation against relevant literature and peer reviews, generating and refining abstracts that are both grounded in real-world research and demonstrably interesting. Experiments using LLM-based metrics indicate that our method significantly improves the concreteness of generated research ideas relative to standard prompting approaches. Human evaluations further demonstrate a substantial enhancement in the perceived interestingness of the outputs. By bridging the gap between academic data and creative generation, the Budget AI Researcher offers a practical, free tool for accelerating scientific discovery and lowering the barrier for aspiring researchers. Beyond research ideation, this approach inspires solutions to the broader challenge of generating personalized, context-aware outputs grounded in evolving real-world knowledge. 

**Abstract (ZH)**: 探索浩瀚且快速增长的科学文献库是对新兴研究人员的一大挑战。当前支持研究想法生成的方法通常依赖于通用大型语言模型（LLMs）。尽管LLMs在帮助理解和总结方面效果显著，但在引导用户形成实用的研究想法方面往往表现不佳，因为它们存在局限性。本研究提出了一种新颖的研究想法生成结构框架。该框架名为《Budget AI Researcher》，利用检索增强生成（RAG）链、向量数据库和主题引导配对，从数百篇机器学习论文中重组概念。系统摄取来自九个主要AI会议的论文，这些论文共同覆盖了机器学习的各个子领域，并组织成一个层级主题树。它利用树形结构识别远距离主题配对，生成新颖的研究摘要，并通过迭代自评估与相关文献和同行评审进行对比，生成并完善既基于实际研究又具有明显吸引力的摘要。基于LLM的指标实验表明，我们的方法显著提高了生成研究想法的具体性，相较于标准提示方法。进一步的人类评估显示，输出的吸引力有了显著提升。通过弥合学术数据和创意生成之间的差距，《Budget AI Researcher》提供了一种实用且免费的工具，以加速科学发现并降低新兴研究人员的门槛。超越研究想法生成，此方法启发了解决生成个性化、情境感知输出的更广泛挑战，这些输出基于不断发展的实际知识。 

---
# Ontology Enabled Hybrid Modeling and Simulation 

**Title (ZH)**: 本体驱动的混合建模与仿真 

**Authors**: John Beverley, Andreas Tolk  

**Link**: [PDF](https://arxiv.org/pdf/2506.12290)  

**Abstract**: We explore the role of ontologies in enhancing hybrid modeling and simulation through improved semantic rigor, model reusability, and interoperability across systems, disciplines, and tools. By distinguishing between methodological and referential ontologies, we demonstrate how these complementary approaches address interoperability challenges along three axes: Human-Human, Human-Machine, and Machine-Machine. Techniques such as competency questions, ontology design patterns, and layered strategies are highlighted for promoting shared understanding and formal precision. Integrating ontologies with Semantic Web Technologies, we showcase their dual role as descriptive domain representations and prescriptive guides for simulation construction. Four application cases - sea-level rise analysis, Industry 4.0 modeling, artificial societies for policy support, and cyber threat evaluation - illustrate the practical benefits of ontology-driven hybrid simulation workflows. We conclude by discussing challenges and opportunities in ontology-based hybrid M&S, including tool integration, semantic alignment, and support for explainable AI. 

**Abstract (ZH)**: 我们探索本体在通过改进语义严谨性、模型重用性和跨系统、学科和工具的互操作性来增强混合建模与仿真的作用。通过区分方法论本体和参考本体，我们展示了这些互补方法如何在三条轴上解决互操作性挑战：人与人、人与机器以及机器与机器。我们强调了诸如能力问题、本体设计模式和分层策略等技术，以促进共享理解并提高形式精确度。我们将本体与语义网技术集成，展示了它们作为描述性领域表示和仿真实体制定的规范性指南的双重角色。四个应用案例——海平面上升分析、工业4.0建模、用于政策支持的人工社会以及网络威胁评估——阐述了本体驱动的混合仿真实践流程的实际益处。最后，我们讨论了基于本体的混合建模与仿真中的挑战与机遇，包括工具集成、语义对齐以及支持可解释AI的支持。 

---
# The SWE-Bench Illusion: When State-of-the-Art LLMs Remember Instead of Reason 

**Title (ZH)**: SWE-Bench 幻象：当最先进的大语言模型记忆代替推理 

**Authors**: Shanchao Liang, Spandan Garg, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12286)  

**Abstract**: As large language models (LLMs) become increasingly capable and widely adopted, benchmarks play a central role in assessing their practical utility. For example, SWE-Bench Verified has emerged as a critical benchmark for evaluating LLMs' software engineering abilities, particularly their aptitude for resolving real-world GitHub issues. Recent LLMs show impressive performance on SWE-Bench, leading to optimism about their capacity for complex coding tasks. However, current evaluation protocols may overstate these models' true capabilities. It is crucial to distinguish LLMs' generalizable problem-solving ability and other learned artifacts. In this work, we introduce a diagnostic task: file path identification from issue descriptions alone, to probe models' underlying knowledge. We present empirical evidence that performance gains on SWE-Bench-Verified may be partially driven by memorization rather than genuine problem-solving. We show that state-of-the-art models achieve up to 76% accuracy in identifying buggy file paths using only issue descriptions, without access to repository structure. This performance is merely up to 53% on tasks from repositories not included in SWE-Bench, pointing to possible data contamination or memorization. These findings raise concerns about the validity of existing results and underscore the need for more robust, contamination-resistant benchmarks to reliably evaluate LLMs' coding abilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）的能力不断增强并广泛采用后，基准测试在评估其实际应用价值中扮演核心角色。例如，SWE-Bench Verified已成为评估LLMs软件工程能力的关键基准，特别是解决实际GitHub问题的能力。最近的LLMs在SWE-Bench上表现出色，带来了其复杂编码任务能力的乐观预期。然而，当前的评估协议可能夸大了这些模型的真实能力。区分LLMs的广泛适用的解决问题能力和其他学习到的特征至关重要。在本项工作中，我们引入了一个诊断任务：仅通过问题描述来识别文件路径，以探究模型的潜在知识。我们提供了实证证据，表明SWE-Bench-Verified上的性能提升部分可能是由于记忆而非真实的解决问题能力。我们展示了最先进的模型仅使用问题描述即可达到76%的错误文件路径识别准确性，而无需访问仓库结构。相比之下，在未包含于SWE-Bench的仓库任务上，性能仅达到53%，这表明可能存在数据污染或记忆现象。这些发现对现有结果的有效性提出了质疑，并强调了需要更 robust、抗污染的基准测试来可靠地评估LLMs的编码能力。 

---
# Deep Fictitious Play-Based Potential Differential Games for Learning Human-Like Interaction at Unsignalized Intersections 

**Title (ZH)**: 基于深层虚构博弈的潜在差分博弈方法学习无信号交叉口的人类交互行为 

**Authors**: Kehua Chen, Shucheng Zhang, Yinhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12283)  

**Abstract**: Modeling vehicle interactions at unsignalized intersections is a challenging task due to the complexity of the underlying game-theoretic processes. Although prior studies have attempted to capture interactive driving behaviors, most approaches relied solely on game-theoretic formulations and did not leverage naturalistic driving datasets. In this study, we learn human-like interactive driving policies at unsignalized intersections using Deep Fictitious Play. Specifically, we first model vehicle interactions as a Differential Game, which is then reformulated as a Potential Differential Game. The weights in the cost function are learned from the dataset and capture diverse driving styles. We also demonstrate that our framework provides a theoretical guarantee of convergence to a Nash equilibrium. To the best of our knowledge, this is the first study to train interactive driving policies using Deep Fictitious Play. We validate the effectiveness of our Deep Fictitious Play-Based Potential Differential Game (DFP-PDG) framework using the INTERACTION dataset. The results demonstrate that the proposed framework achieves satisfactory performance in learning human-like driving policies. The learned individual weights effectively capture variations in driver aggressiveness and preferences. Furthermore, the ablation study highlights the importance of each component within our model. 

**Abstract (ZH)**: 无信号交叉口车辆互动的建模是一项具有挑战性的任务，受到潜在博弈理论过程复杂性的制约。尽管先前的研究试图捕捉互动驾驶行为，但大多数方法仅依赖博弈 theoretic 表述，而未利用自然istic驾驶数据集。在本研究中，我们使用深度虚构博弈从无信号交叉口学习类人的互动驾驶策略。具体地，我们首先将车辆互动建模为微分博弈，然后将其重新表述为潜在微分博弈。成本函数中的权重是从数据集中学习得到的，能够捕捉多样化的驾驶风格。我们还证明了我们的框架提供了向纳什均衡收敛的理论保证。据我们所知，这是首次使用深度虚构博弈训练互动驾驶策略的研究。我们使用INTERACTION数据集验证了基于深度虚构博弈的潜在微分博弈（DFP-PDG）框架的有效性。结果表明，所提出的方法能够学习到类人的驾驶策略，学习到的个体权重有效地捕捉了驾驶者侵略性和偏好的变化。此外，消融实验强调了模型中每个组件的重要性。 

---
# Cloud Infrastructure Management in the Age of AI Agents 

**Title (ZH)**: AI代理时代云基础设施管理 

**Authors**: Zhenning Yang, Archit Bhatnagar, Yiming Qiu, Tongyuan Miao, Patrick Tser Jern Kon, Yunming Xiao, Yibo Huang, Martin Casado, Ang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12270)  

**Abstract**: Cloud infrastructure is the cornerstone of the modern IT industry. However, managing this infrastructure effectively requires considerable manual effort from the DevOps engineering team. We make a case for developing AI agents powered by large language models (LLMs) to automate cloud infrastructure management tasks. In a preliminary study, we investigate the potential for AI agents to use different cloud/user interfaces such as software development kits (SDK), command line interfaces (CLI), Infrastructure-as-Code (IaC) platforms, and web portals. We report takeaways on their effectiveness on different management tasks, and identify research challenges and potential solutions. 

**Abstract (ZH)**: 基于大型语言模型的AI代理在云基础设施管理中的应用研究 

---
# Lower Bound on Howard Policy Iteration for Deterministic Markov Decision Processes 

**Title (ZH)**: 确定性马尔可夫决策过程中的何德政策迭代的下界 

**Authors**: Ali Asadi, Krishnendu Chatterjee, Jakob de Raaij  

**Link**: [PDF](https://arxiv.org/pdf/2506.12254)  

**Abstract**: Deterministic Markov Decision Processes (DMDPs) are a mathematical framework for decision-making where the outcomes and future possible actions are deterministically determined by the current action taken. DMDPs can be viewed as a finite directed weighted graph, where in each step, the controller chooses an outgoing edge. An objective is a measurable function on runs (or infinite trajectories) of the DMDP, and the value for an objective is the maximal cumulative reward (or weight) that the controller can guarantee. We consider the classical mean-payoff (aka limit-average) objective, which is a basic and fundamental objective.
Howard's policy iteration algorithm is a popular method for solving DMDPs with mean-payoff objectives. Although Howard's algorithm performs well in practice, as experimental studies suggested, the best known upper bound is exponential and the current known lower bound is as follows: For the input size $I$, the algorithm requires $\tilde{\Omega}(\sqrt{I})$ iterations, where $\tilde{\Omega}$ hides the poly-logarithmic factors, i.e., the current lower bound on iterations is sub-linear with respect to the input size. Our main result is an improved lower bound for this fundamental algorithm where we show that for the input size $I$, the algorithm requires $\tilde{\Omega}(I)$ iterations. 

**Abstract (ZH)**: 确定性马尔可夫决策过程中的平均收益目标的改进下界分析 

---
# Reversing the Paradigm: Building AI-First Systems with Human Guidance 

**Title (ZH)**: 反转范式：在人类指导下的AI优先系统构建 

**Authors**: Cosimo Spera, Garima Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2506.12245)  

**Abstract**: The relationship between humans and artificial intelligence is no longer science fiction -- it's a growing reality reshaping how we live and work. AI has moved beyond research labs into everyday life, powering customer service chats, personalizing travel, aiding doctors in diagnosis, and supporting educators. What makes this moment particularly compelling is AI's increasing collaborative nature. Rather than replacing humans, AI augments our capabilities -- automating routine tasks, enhancing decisions with data, and enabling creativity in fields like design, music, and writing. The future of work is shifting toward AI agents handling tasks autonomously, with humans as supervisors, strategists, and ethical stewards. This flips the traditional model: instead of humans using AI as a tool, intelligent agents will operate independently within constraints, managing everything from scheduling and customer service to complex workflows. Humans will guide and fine-tune these agents to ensure alignment with goals, values, and context.
This shift offers major benefits -- greater efficiency, faster decisions, cost savings, and scalability. But it also brings risks: diminished human oversight, algorithmic bias, security flaws, and a widening skills gap. To navigate this transition, organizations must rethink roles, invest in upskilling, embed ethical principles, and promote transparency. This paper examines the technological and organizational changes needed to enable responsible adoption of AI-first systems -- where autonomy is balanced with human intent, oversight, and values. 

**Abstract (ZH)**: 人类与人工智能的关系不再局限于科幻——它是一个 growing 现实，正在重塑我们的生活和工作方式。人工智能已从研究实验室走进日常生活，推动客服聊天、个性化旅行、辅助医生诊断和教育工作者的支持。这一时刻尤其引人注目的是人工智能日益增强的协作性。人工智能不是取代人类，而是增强我们的能力——自动化常规任务，利用数据增强决策，并在设计、音乐和写作等领域促进创造力。工作未来将转向人工智能代理自主处理任务，人类作为监管者、策略制定者和道德监护人。这一转变颠覆了传统模型：不再是人类利用人工智能作为工具，而是智能代理在约束条件下独立操作，管理从调度和客户服务到复杂工作流程的一切。人类将指导和微调这些代理，以确保与目标、价值观和情境相一致。这一转变带来了重大益处——更高的效率、更快的决策、成本节约和可扩展性。但也带来了风险——人类监督的减少、算法偏差、安全漏洞以及技能差距的扩大。为应对这一转型，组织必须重新思考角色、投资技能提升、嵌入道德原则并促进透明度。本文探讨了为负责任地采用以人工智能为主导的系统所需的技术和组织变革——平衡自主性与人类意图、监督和价值观。 

---
# Privacy Reasoning in Ambiguous Contexts 

**Title (ZH)**: 在含糊情境中的隐私推理 

**Authors**: Ren Yi, Octavian Suciu, Adria Gascon, Sarah Meiklejohn, Eugene Bagdasarian, Marco Gruteser  

**Link**: [PDF](https://arxiv.org/pdf/2506.12241)  

**Abstract**: We study the ability of language models to reason about appropriate information disclosure - a central aspect of the evolving field of agentic privacy. Whereas previous works have focused on evaluating a model's ability to align with human decisions, we examine the role of ambiguity and missing context on model performance when making information-sharing decisions. We identify context ambiguity as a crucial barrier for high performance in privacy assessments. By designing Camber, a framework for context disambiguation, we show that model-generated decision rationales can reveal ambiguities and that systematically disambiguating context based on these rationales leads to significant accuracy improvements (up to 13.3\% in precision and up to 22.3\% in recall) as well as reductions in prompt sensitivity. Overall, our results indicate that approaches for context disambiguation are a promising way forward to enhance agentic privacy reasoning. 

**Abstract (ZH)**: 我们研究了语言模型在信息披露推理方面的能力——这是不断发展的代理隐私领域的一个核心方面。与以往研究专注于评估模型与人类决策的一致性不同，我们探讨了在进行信息共享决策时，模糊性和缺失背景对模型性能的影响。我们识别出背景模糊性是高性能隐私评估的一个关键障碍。通过设计Camber框架，一种背景消歧框架，我们证明了模型生成的决策理由可以揭示模糊性，并且根据这些理由系统地消歧背景可以显著提高准确率（精确度提高至多13.3%、召回率提高至多22.3%），以及降低提示敏感性。总体而言，我们的结果表明，背景消歧方法是增强代理隐私推理的一个有前景的方向。 

---
# PRO-V: An Efficient Program Generation Multi-Agent System for Automatic RTL Verification 

**Title (ZH)**: PRO-V: 一种高效的程序生成多代理系统，用于自动RTL验证 

**Authors**: Yujie Zhao, Zhijing Wu, Hejia Zhang, Zhongming Yu, Wentao Ni, Chia-Tung Ho, Haoxing Ren, Jishen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12200)  

**Abstract**: LLM-assisted hardware verification is gaining substantial attention due to its potential to significantly reduce the cost and effort of crafting effective testbenches. It also serves as a critical enabler for LLM-aided end-to-end hardware language design. However, existing current LLMs often struggle with Register Transfer Level (RTL) code generation, resulting in testbenches that exhibit functional errors in Hardware Description Languages (HDL) logic. Motivated by the strong performance of LLMs in Python code generation under inference-time sampling strategies, and their promising capabilities as judge agents, we propose PRO-V a fully program generation multi-agent system for robust RTL verification. Pro-V incorporates an efficient best-of-n iterative sampling strategy to enhance the correctness of generated testbenches. Moreover, it introduces an LLM-as-a-judge aid validation framework featuring an automated prompt generation pipeline. By converting rule-based static analysis from the compiler into natural language through in-context learning, this pipeline enables LLMs to assist the compiler in determining whether verification failures stem from errors in the RTL design or the testbench. PRO-V attains a verification accuracy of 87.17% on golden RTL implementations and 76.28% on RTL mutants. Our code is open-sourced at this https URL. 

**Abstract (ZH)**: LLM辅助硬件验证在降低有效测试平台成本和努力方面正在获得广泛关注，并作为端到端硬件语言设计的關鍵使能器。PRO-V：一种稳健的RTL验证全方位程序生成多智能体系统 

---
# Artificial Intelligence and Machine Learning in the Development of Vaccines and Immunotherapeutics Yesterday, Today, and Tomorrow 

**Title (ZH)**: 人工智能与机器学习在疫苗和免疫治疗的发展中：昨天、今天和明天 

**Authors**: Elhoucine Elfatimi, Yassir Lekbach, Swayam Prakash, Lbachir BenMohamed  

**Link**: [PDF](https://arxiv.org/pdf/2506.12185)  

**Abstract**: In the past, the development of vaccines and immunotherapeutics relied heavily on trial-and-error experimentation and extensive in vivo testing, often requiring years of pre-clinical and clinical trials. Today, artificial intelligence (AI) and deep learning (DL) are actively transforming vaccine and immunotherapeutic design, by (i) offering predictive frameworks that support rapid, data-driven decision-making; (ii) increasingly being implemented as time- and resource-efficient strategies that integrate computational models, systems vaccinology, and multi-omics data to better phenotype, differentiate, and classify patient diseases and cancers; predict patients' immune responses; and identify the factors contributing to optimal vaccine and immunotherapeutic protective efficacy; (iii) refining the selection of B- and T-cell antigen/epitope targets to enhance efficacy and durability of immune protection; and (iv) enabling a deeper understanding of immune regulation, immune evasion, immune checkpoints, and regulatory pathways. The future of AI and DL points toward (i) replacing animal preclinical testing of drugs, vaccines, and immunotherapeutics with computational-based models, as recently proposed by the United States FDA; and (ii) enabling real-time in vivo modeling for immunobridging and prediction of protection in clinical trials. This may result in a fast and transformative shift for the development of personal vaccines and immunotherapeutics against infectious pathogens and cancers. 

**Abstract (ZH)**: 人工智能和深度学习在疫苗和免疫治疗设计中的应用正在经历革命性的变革：从试验性方法到计算驱动的快速决策 

---
# Because we have LLMs, we Can and Should Pursue Agentic Interpretability 

**Title (ZH)**: 由于我们拥有大规模语言模型，我们应当追求能动性可解释性。 

**Authors**: Been Kim, John Hewitt, Neel Nanda, Noah Fiedel, Oyvind Tafjord  

**Link**: [PDF](https://arxiv.org/pdf/2506.12152)  

**Abstract**: The era of Large Language Models (LLMs) presents a new opportunity for interpretability--agentic interpretability: a multi-turn conversation with an LLM wherein the LLM proactively assists human understanding by developing and leveraging a mental model of the user, which in turn enables humans to develop better mental models of the LLM. Such conversation is a new capability that traditional `inspective' interpretability methods (opening the black-box) do not use. Having a language model that aims to teach and explain--beyond just knowing how to talk--is similar to a teacher whose goal is to teach well, understanding that their success will be measured by the student's comprehension. While agentic interpretability may trade off completeness for interactivity, making it less suitable for high-stakes safety situations with potentially deceptive models, it leverages a cooperative model to discover potentially superhuman concepts that can improve humans' mental model of machines. Agentic interpretability introduces challenges, particularly in evaluation, due to what we call `human-entangled-in-the-loop' nature (humans responses are integral part of the algorithm), making the design and evaluation difficult. We discuss possible solutions and proxy goals. As LLMs approach human parity in many tasks, agentic interpretability's promise is to help humans learn the potentially superhuman concepts of the LLMs, rather than see us fall increasingly far from understanding them. 

**Abstract (ZH)**: 大规模语言模型时代下的能动可解释性：一种LLM主动协助人类理解的多轮对话方式，进而帮助人类建立更好的LLM心智模型。 

---
# The Amazon Nova Family of Models: Technical Report and Model Card 

**Title (ZH)**: 亚马逊Nova模型家族：技术报告与模型卡片 

**Authors**: Amazon AGI, Aaron Langford, Aayush Shah, Abhanshu Gupta, Abhimanyu Bhatter, Abhinav Goyal, Abhinav Mathur, Abhinav Mohanty, Abhishek Kumar, Abhishek Sethi, Abi Komma, Abner Pena, Achin Jain, Adam Kunysz, Adam Opyrchal, Adarsh Singh, Aditya Rawal, Adok Achar Budihal Prasad, Adrià de Gispert, Agnika Kumar, Aishwarya Aryamane, Ajay Nair, Akilan M, Akshaya Iyengar, Akshaya Vishnu Kudlu Shanbhogue, Alan He, Alessandra Cervone, Alex Loeb, Alex Zhang, Alexander Fu, Alexander Lisnichenko, Alexander Zhipa, Alexandros Potamianos, Ali Kebarighotbi, Aliakbar Daronkolaei, Alok Parmesh, Amanjot Kaur Samra, Ameen Khan, Amer Rez, Amir Saffari, Amit Agarwalla, Amit Jhindal, Amith Mamidala, Ammar Asmro, Amulya Ballakur, Anand Mishra, Anand Sridharan, Anastasiia Dubinina, Andre Lenz, Andreas Doerr, Andrew Keating, Andrew Leaver, Andrew Smith, Andrew Wirth, Andy Davey, Andy Rosenbaum, Andy Sohn, Angela Chan, Aniket Chakrabarti, Anil Ramakrishna, Anirban Roy, Anita Iyer, Anjali Narayan-Chen, Ankith Yennu, Anna Dabrowska, Anna Gawlowska, Anna Rumshisky, Anna Turek, Anoop Deoras, Anton Bezruchkin, Anup Prasad, Anupam Dewan, Anwith Kiran, Apoorv Gupta, Aram Galstyan, Aravind Manoharan, Arijit Biswas, Arindam Mandal, Arpit Gupta, Arsamkhan Pathan, Arun Nagarajan, Arushan Rajasekaram, Arvind Sundararajan, Ashwin Ganesan, Ashwin Swaminathan, Athanasios Mouchtaris, Audrey Champeau, Avik Ray, Ayush Jaiswal, Ayush Sharma, Bailey Keefer, Balamurugan Muthiah, Beatriz Leon-Millan, Ben Koopman, Ben Li, Benjamin Biggs, Benjamin Ott, Bhanu Vinzamuri, Bharath Venkatesh, Bhavana Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2506.12103)  

**Abstract**: We present Amazon Nova, a new generation of state-of-the-art foundation models that deliver frontier intelligence and industry-leading price performance. Amazon Nova Pro is a highly-capable multimodal model with the best combination of accuracy, speed, and cost for a wide range of tasks. Amazon Nova Lite is a low-cost multimodal model that is lightning fast for processing images, video, documents and text. Amazon Nova Micro is a text-only model that delivers our lowest-latency responses at very low cost. Amazon Nova Canvas is an image generation model that creates professional grade images with rich customization controls. Amazon Nova Reel is a video generation model offering high-quality outputs, customization, and motion control. Our models were built responsibly and with a commitment to customer trust, security, and reliability. We report benchmarking results for core capabilities, agentic performance, long context, functional adaptation, runtime performance, and human evaluation. 

**Abstract (ZH)**: 我们 presents Amazon Nova，新一代顶尖基础模型，提供前沿智能和行业领先的价格性能。Amazon Nova Pro 是一款功能强大的多模态模型，具有广泛的任务最佳的准确度、速度和成本组合。Amazon Nova Lite 是一款低成本的多模态模型，处理图像、视频、文档和文本速度极快。Amazon Nova Micro 是一款仅文本模型，以极低延迟和成本提供快速响应。Amazon Nova Canvas 是一款图像生成模型，可生成专业级图像并提供丰富的自定义控制。Amazon Nova Reel 是一款视频生成模型，提供高质量输出、自定义和运动控制。我们的模型以负责任的态度构建，并致力于客户信任、安全和可靠性的承诺。我们报告了核心能力、自主性能、长上下文、功能适应性、运行时性能和人类评估的基准测试结果。 

---
# Diagnosing and Improving Diffusion Models by Estimating the Optimal Loss Value 

**Title (ZH)**: 通过估计最优损失值来诊断和提升扩散模型 

**Authors**: Yixian Xu, Shengjie Luo, Liwei Wang, Di He, Chang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13763)  

**Abstract**: Diffusion models have achieved remarkable success in generative modeling. Despite more stable training, the loss of diffusion models is not indicative of absolute data-fitting quality, since its optimal value is typically not zero but unknown, leading to confusion between large optimal loss and insufficient model capacity. In this work, we advocate the need to estimate the optimal loss value for diagnosing and improving diffusion models. We first derive the optimal loss in closed form under a unified formulation of diffusion models, and develop effective estimators for it, including a stochastic variant scalable to large datasets with proper control of variance and bias. With this tool, we unlock the inherent metric for diagnosing the training quality of mainstream diffusion model variants, and develop a more performant training schedule based on the optimal loss. Moreover, using models with 120M to 1.5B parameters, we find that the power law is better demonstrated after subtracting the optimal loss from the actual training loss, suggesting a more principled setting for investigating the scaling law for diffusion models. 

**Abstract (ZH)**: 扩散模型已在生成建模中取得了显著成功。尽管训练更为稳定，但扩散模型的损失值并不一定意味着绝对的数据拟合质量，因为其最优值通常不为零且未知，导致大最优损失与模型容量不足之间的混淆。在本文中，我们强调估计最优损失值以诊断和提升扩散模型的重要性。我们首先在统一的扩散模型框架下推导出最优损失的闭式解，并开发出有效的估计器，包括一种可扩展到大规模数据集的有效随机变体，以适当控制方差和偏差。借助此工具，我们为流行的扩散模型变体解锁了固有的诊断训练质量的度量，并基于最优损失开发出更高效的训练计划。此外，使用包含120M到1.5B参数的模型，我们发现，在实际训练损失中减去最优损失后，功率律得到了更好的呈现，这表明了在研究扩散模型的扩展律时采用更为原则性的设置。 

---
# Discrete Diffusion in Large Language and Multimodal Models: A Survey 

**Title (ZH)**: 离散扩散在大型语言和多模态模型中的研究 

**Authors**: Runpeng Yu, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13759)  

**Abstract**: In this work, we provide a systematic survey of Discrete Diffusion Language Models (dLLMs) and Discrete Diffusion Multimodal Language Models (dMLLMs). Unlike autoregressive (AR) models, dLLMs and dMLLMs adopt a multi-token, parallel decoding paradigm using full attention and a denoising-based generation strategy. This paradigm naturally enables parallel generation, fine-grained output controllability, and dynamic, response-aware perception. These capabilities are previously difficult to achieve with AR models. Recently, a growing number of industrial-scale proprietary d(M)LLMs, as well as a large number of open-source academic d(M)LLMs, have demonstrated performance comparable to their autoregressive counterparts, while achieving up to 10x acceleration in inference speed.
The advancement of discrete diffusion LLMs and MLLMs has been largely driven by progress in two domains. The first is the development of autoregressive LLMs and MLLMs, which has accumulated vast amounts of data, benchmarks, and foundational infrastructure for training and inference. The second contributing domain is the evolution of the mathematical models underlying discrete diffusion. Together, these advancements have catalyzed a surge in dLLMs and dMLLMs research in early 2025.
In this work, we present a comprehensive overview of the research in the dLLM and dMLLM domains. We trace the historical development of dLLMs and dMLLMs, formalize the underlying mathematical frameworks, and categorize representative models. We further analyze key techniques for training and inference, and summarize emerging applications across language, vision-language, and biological domains. We conclude by discussing future directions for research and deployment.
Paper collection: this https URL 

**Abstract (ZH)**: 本研究提供了对离散扩散语言模型（dLLMs）和离散扩散多模态语言模型（dMLLMs）的系统综述。与自回归（AR）模型不同，dLLMs和dMLLMs采用基于全注意机制和去噪生成策略的多token并行解码范式。这一范式自然地支持并行生成、细粒度输出可控性和响应感知的动态特性。这些能力是AR模型难以实现的。近期，大量的工业规模的专有d(M)LLMs以及众多开源学术d(M)LLMs展现出了与其自回归对应模型相当的性能，同时还实现了高达10倍的推理速度提升。

离散扩散LLMs和MLLMs的进步主要受两个领域进展的推动。首先是自回归LLMs和MLLMs的发展，积累了大量数据、基准和训练推理的基础架构。其次是支撑离散扩散的数学模型的演变。这些进步催化了2025年初dLLMs和dMLLMs研究的激增。

在本文中，我们提供了对dLLM和dMLLM领域的研究综述。我们追溯了dLLMs和dMLLMs的历史发展，形式化其底层数学框架，并分类代表性模型。进一步分析了训练和推理中的关键技术，并总结了在语言、跨模态语言和生物领域中的新兴应用。最后讨论了未来的研究和部署方向。

论文集：[这里](this https URL)。 

---
# VideoPDE: Unified Generative PDE Solving via Video Inpainting Diffusion Models 

**Title (ZH)**: 视频PDE统一生成型偏微分方程求解方法：基于视频_inpainting_扩散模型 

**Authors**: Edward Li, Zichen Wang, Jiahe Huang, Jeong Joon Park  

**Link**: [PDF](https://arxiv.org/pdf/2506.13754)  

**Abstract**: We present a unified framework for solving partial differential equations (PDEs) using video-inpainting diffusion transformer models. Unlike existing methods that devise specialized strategies for either forward or inverse problems under full or partial observation, our approach unifies these tasks under a single, flexible generative framework. Specifically, we recast PDE-solving as a generalized inpainting problem, e.g., treating forward prediction as inferring missing spatiotemporal information of future states from initial conditions. To this end, we design a transformer-based architecture that conditions on arbitrary patterns of known data to infer missing values across time and space. Our method proposes pixel-space video diffusion models for fine-grained, high-fidelity inpainting and conditioning, while enhancing computational efficiency through hierarchical modeling. Extensive experiments show that our video inpainting-based diffusion model offers an accurate and versatile solution across a wide range of PDEs and problem setups, outperforming state-of-the-art baselines. 

**Abstract (ZH)**: 我们提出了一种统一框架，利用视频修复变换器模型求解偏微分方程（PDEs）。不同于现有方法针对完全或不完全观测下的前向或反向问题设计专门策略，我们的方法将这些任务统一在一个灵活的生成框架下。具体而言，我们将PDE求解重新阐述为一个泛化的修复问题，例如，将前向预测视为从初始条件推断未来状态的缺失时空信息。为此，我们设计了一种基于变换器的架构，可根据任意已知数据模式来推断时空中的缺失值。我们的方法提出了一种像素空间视频扩散模型进行细粒度、高保真修复和条件推断，并通过分层建模提高计算效率。 extensive实验表明，基于视频修复的扩散模型在广泛类型的PDE和问题设置中提供了准确且通用的解决方案，优于最先进的基线方法。 

---
# Steering LLM Thinking with Budget Guidance 

**Title (ZH)**: 用预算指导调控大规模语言模型的思维过程 

**Authors**: Junyan Li, Wenshuo Zhao, Yang Zhang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.13752)  

**Abstract**: Recent deep-thinking large language models often reason extensively to improve performance, but such lengthy reasoning is not always desirable, as it incurs excessive inference costs with disproportionate performance gains. Controlling reasoning length without sacrificing performance is therefore important, but remains challenging, especially under tight thinking budgets. We propose budget guidance, a simple yet effective method for steering the reasoning process of LLMs toward a target budget without requiring any LLM fine-tuning. Our approach introduces a lightweight predictor that models a Gamma distribution over the remaining thinking length during next-token generation. This signal is then used to guide generation in a soft, token-level manner, ensuring that the overall reasoning trace adheres to the specified thinking budget. Budget guidance enables natural control of the thinking length, along with significant token efficiency improvements over baseline methods on challenging math benchmarks. For instance, it achieves up to a 26% accuracy gain on the MATH-500 benchmark under tight budgets compared to baseline methods, while maintaining competitive accuracy with only 63% of the thinking tokens used by the full-thinking model. Budget guidance also generalizes to broader task domains and exhibits emergent capabilities, such as estimating question difficulty. The source code is available at: this https URL. 

**Abstract (ZH)**: 预算指导：在严格思考预算下控制大型语言模型推理长度的一种简单有效方法 

---
# LeVERB: Humanoid Whole-Body Control with Latent Vision-Language Instruction 

**Title (ZH)**: LeVERB: 具潜变量视觉-语言指令的类人全身控制 

**Authors**: Haoru Xue, Xiaoyu Huang, Dantong Niu, Qiayuan Liao, Thomas Kragerud, Jan Tommy Gravdahl, Xue Bin Peng, Guanya Shi, Trevor Darrell, Koushil Screenath, Shankar Sastry  

**Link**: [PDF](https://arxiv.org/pdf/2506.13751)  

**Abstract**: Vision-language-action (VLA) models have demonstrated strong semantic understanding and zero-shot generalization, yet most existing systems assume an accurate low-level controller with hand-crafted action "vocabulary" such as end-effector pose or root velocity. This assumption confines prior work to quasi-static tasks and precludes the agile, whole-body behaviors required by humanoid whole-body control (WBC) tasks. To capture this gap in the literature, we start by introducing the first sim-to-real-ready, vision-language, closed-loop benchmark for humanoid WBC, comprising over 150 tasks from 10 categories. We then propose LeVERB: Latent Vision-Language-Encoded Robot Behavior, a hierarchical latent instruction-following framework for humanoid vision-language WBC, the first of its kind. At the top level, a vision-language policy learns a latent action vocabulary from synthetically rendered kinematic demonstrations; at the low level, a reinforcement-learned WBC policy consumes these latent verbs to generate dynamics-level commands. In our benchmark, LeVERB can zero-shot attain a 80% success rate on simple visual navigation tasks, and 58.5% success rate overall, outperforming naive hierarchical whole-body VLA implementation by 7.8 times. 

**Abstract (ZH)**: Vision-语言-动作（VLA）模型展示了强大的语义理解和零样本泛化能力，但现有系统大多假设一个精确的低级控制器，并且使用手工构建的动作“词汇表”，如末端执行器姿态或根速度。这种假设限制了先前的工作仅适用于准静态任务，并排除了类人全身控制（WBC）任务所需的敏捷的全身行为。为了填补文献中的这一缺口，我们首先引入了一个首个适应于类人WBC的仿真实验室至现实场景、视-语言闭环基准，包含超过150个来自10个类别的任务。随后，我们提出LeVERB：潜在于语言-视-觉编码的机器人行为，一个分层的潜在线性指令跟随框架，用于类人视-语言WBC，这是此类方法的先驱。在顶层，一个视-语言策略从合成渲染的动力学演示中学习潜在的动作词汇；在底层，一个强化学习的WBC策略消耗这些潜在的动词以生成动力学层的指令。在我们的基准中，LeVERB能够在简单的视觉导航任务上零样本达到80%的成功率，并且总体成功率为58.5%，比简单的分层全身VLA实现高出7.8倍。 

---
# Evaluating Large Language Models for Phishing Detection, Self-Consistency, Faithfulness, and Explainability 

**Title (ZH)**: 评估大规模语言模型在钓鱼检测、自一致性、忠实性及可解释性方面的表现 

**Authors**: Shova Kuikel, Aritran Piplai, Palvi Aggarwal  

**Link**: [PDF](https://arxiv.org/pdf/2506.13746)  

**Abstract**: Phishing attacks remain one of the most prevalent and persistent cybersecurity threat with attackers continuously evolving and intensifying tactics to evade the general detection system. Despite significant advances in artificial intelligence and machine learning, faithfully reproducing the interpretable reasoning with classification and explainability that underpin phishing judgments remains challenging. Due to recent advancement in Natural Language Processing, Large Language Models (LLMs) show a promising direction and potential for improving domain specific phishing classification tasks. However, enhancing the reliability and robustness of classification models requires not only accurate predictions from LLMs but also consistent and trustworthy explanations aligning with those predictions. Therefore, a key question remains: can LLMs not only classify phishing emails accurately but also generate explanations that are reliably aligned with their predictions and internally self-consistent? To answer these questions, we have fine-tuned transformer based models, including BERT, Llama models, and Wizard, to improve domain relevance and make them more tailored to phishing specific distinctions, using Binary Sequence Classification, Contrastive Learning (CL) and Direct Preference Optimization (DPO). To that end, we examined their performance in phishing classification and explainability by applying the ConsistenCy measure based on SHAPley values (CC SHAP), which measures prediction explanation token alignment to test the model's internal faithfulness and consistency and uncover the rationale behind its predictions and reasoning. Overall, our findings show that Llama models exhibit stronger prediction explanation token alignment with higher CC SHAP scores despite lacking reliable decision making accuracy, whereas Wizard achieves better prediction accuracy but lower CC SHAP scores. 

**Abstract (ZH)**: 基于大型语言模型的钓鱼邮件分类及其解释的可靠性与一致性研究 

---
# Instruction Following by Boosting Attention of Large Language Models 

**Title (ZH)**: 大型语言模型注意力增强的指令跟随 

**Authors**: Vitoria Guardieiro, Adam Stein, Avishree Khare, Eric Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13734)  

**Abstract**: Controlling the generation of large language models (LLMs) remains a central challenge to ensure their safe and reliable deployment. While prompt engineering and finetuning are common approaches, recent work has explored latent steering, a lightweight technique that alters LLM internal activations to guide generation. However, subsequent studies revealed latent steering's effectiveness to be limited, often underperforming simple instruction prompting. To address this limitation, we first establish a benchmark across diverse behaviors for standardized evaluation of steering techniques. Building on insights from this benchmark, we introduce Instruction Attention Boosting (InstABoost), a latent steering method that boosts the strength of instruction prompting by altering the model's attention during generation. InstABoost combines the strengths of existing approaches and is theoretically supported by prior work that suggests that in-context rule following in transformer-based models can be controlled by manipulating attention on instructions. Empirically, InstABoost demonstrates superior control success compared to both traditional prompting and latent steering. 

**Abstract (ZH)**: 控制大型语言模型（LLMs）的生成仍然是确保其安全可靠部署的核心挑战。虽然提示工程和微调是常用方法，但最近的研究探索了潜踪引导这一轻量级技术，通过改变LLM的内部激活来引导生成。然而，后续研究揭示了潜踪引导的有效性有限，经常不如简单的指令提示。为解决这一局限性，我们首先在多样化的行为上建立了一个基准，以标准化评估引导技术。基于这一基准的洞见，我们引入了指令注意力增强（InstABoost）方法，通过在生成过程中改变模型的注意力来增强指令提示的强度。InstABoost 结合了现有方法的优点，并由先前的研究理论支持，该研究指出现代变压器模型中的指令内规则遵守可以通过操纵指令上的注意力来控制。实验上，InstABoost 在控制成功率方面优于传统提示和潜踪引导。 

---
# BanditWare: A Contextual Bandit-based Framework for Hardware Prediction 

**Title (ZH)**: BanditWare：基于上下文-bandit的硬件预测框架 

**Authors**: Tainã Coleman, Hena Ahmed, Ravi Shende, Ismael Perez, Ïlkay Altintaş  

**Link**: [PDF](https://arxiv.org/pdf/2506.13730)  

**Abstract**: Distributed computing systems are essential for meeting the demands of modern applications, yet transitioning from single-system to distributed environments presents significant challenges. Misallocating resources in shared systems can lead to resource contention, system instability, degraded performance, priority inversion, inefficient utilization, increased latency, and environmental impact.
We present BanditWare, an online recommendation system that dynamically selects the most suitable hardware for applications using a contextual multi-armed bandit algorithm. BanditWare balances exploration and exploitation, gradually refining its hardware recommendations based on observed application performance while continuing to explore potentially better options. Unlike traditional statistical and machine learning approaches that rely heavily on large historical datasets, BanditWare operates online, learning and adapting in real-time as new workloads arrive.
We evaluated BanditWare on three workflow applications: Cycles (an agricultural science scientific workflow) BurnPro3D (a web-based platform for fire science) and a matrix multiplication application. Designed for seamless integration with the National Data Platform (NDP), BanditWare enables users of all experience levels to optimize resource allocation efficiently. 

**Abstract (ZH)**: 分布式计算系统对于满足现代应用的需求是必不可少的，但从单系统环境向分布式环境的转变面临着巨大的挑战。在共享系统中错误分配资源可能导致资源争用、系统不稳定、性能下降、优先级反转、资源利用效率低下、延迟增加以及环境影响。

我们提出了BanditWare，这是一种在线推荐系统，使用上下文多臂赌博机算法动态选择最适合的应用程序的硬件。BanditWare在不断探索和利用之间寻求平衡，根据观察到的应用程序性能逐步优化其硬件建议，同时继续探索可能更好的选项。与依赖于大量历史数据的传统统计和机器学习方法不同，BanditWare在线操作，能够实时学习和适应新的工作负载。

我们分别在三个工作流应用程序上评估了BanditWare：Cycles（一个农业科学科学工作流）、BurnPro3D（一个基于Web的防火科学平台）以及一个矩阵乘法应用程序。BanditWare设计用于无缝集成到国家数据平台（NDP）中，使得所有经验级别的用户都能够有效优化资源分配。 

---
# Attribution-guided Pruning for Compression, Circuit Discovery, and Targeted Correction in LLMs 

**Title (ZH)**: Attribution-guided 知识归因剪枝在大型语言模型的压缩、电路发现和定向纠正中的应用 

**Authors**: Sayed Mohammad Vakilzadeh Hatefi, Maximilian Dreyer, Reduan Achtibat, Patrick Kahardipraja, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13727)  

**Abstract**: Large Language Models (LLMs) are central to many contemporary AI applications, yet their extensive parameter counts pose significant challenges for deployment in memory- and compute-constrained environments. Recent works in eXplainable AI (XAI), particularly on attribution methods, suggest that interpretability can also enable model compression by identifying and removing components irrelevant to inference. In this paper, we leverage Layer-wise Relevance Propagation (LRP) to perform attribution-guided pruning of LLMs. While LRP has shown promise in structured pruning for vision models, we extend it to unstructured pruning in LLMs and demonstrate that it can substantially reduce model size with minimal performance loss. Our method is especially effective in extracting task-relevant subgraphs -- so-called ``circuits'' -- which can represent core functions (e.g., indirect object identification). Building on this, we introduce a technique for model correction, by selectively removing circuits responsible for spurious behaviors (e.g., toxic outputs). All in all, we gather these techniques as a uniform holistic framework and showcase its effectiveness and limitations through extensive experiments for compression, circuit discovery and model correction on Llama and OPT models, highlighting its potential for improving both model efficiency and safety. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在许多现代AI应用中占据核心地位，但其庞大的参数量给内存和计算受限的环境带来了显著的部署挑战。可解释AI（XAI）领域的 recent 工作，特别是归属方法，表明通过识别和去除与推理无关的组件，可解释性还可以促进模型压缩。在本文中，我们利用层间相关性传播（LRP）进行归属导向的大规模语言模型剪枝。尽管LRP在视觉模型的结构化剪枝中表现出前景，我们将其扩展到大规模语言模型的无结构剪枝，并证明其可以在几乎不牺牲性能的情况下显著减小模型大小。我们的方法特别适用于提取任务相关的子图——所谓的“电路”——这些子图可以表示核心功能（例如，间接对象识别）。在此基础上，我们介绍了一种模型修正技术，通过选择性地移除导致错误行为的电路（例如，有毒输出）来进行修正。总体而言，我们将这些技术统一为一个统一的整体框架，并通过在Llama和OPT模型上进行压缩、电路发现和模型修正的广泛实验，展示了其有效性和局限性，突显了其在提高模型效率和安全性方面的潜力。我们的代码已公开于此 <https://> 地址。 

---
# Contrastive Self-Supervised Learning As Neural Manifold Packing 

**Title (ZH)**: 对比自监督学习作为神经流形打包 

**Authors**: Guanming Zhang, David J. Heeger, Stefano Martiniani  

**Link**: [PDF](https://arxiv.org/pdf/2506.13717)  

**Abstract**: Contrastive self-supervised learning based on point-wise comparisons has been widely studied for vision tasks. In the visual cortex of the brain, neuronal responses to distinct stimulus classes are organized into geometric structures known as neural manifolds. Accurate classification of stimuli can be achieved by effectively separating these manifolds, akin to solving a packing problem. We introduce Contrastive Learning As Manifold Packing (CLAMP), a self-supervised framework that recasts representation learning as a manifold packing problem. CLAMP introduces a loss function inspired by the potential energy of short-range repulsive particle systems, such as those encountered in the physics of simple liquids and jammed packings. In this framework, each class consists of sub-manifolds embedding multiple augmented views of a single image. The sizes and positions of the sub-manifolds are dynamically optimized by following the gradient of a packing loss. This approach yields interpretable dynamics in the embedding space that parallel jamming physics, and introduces geometrically meaningful hyperparameters within the loss function. Under the standard linear evaluation protocol, which freezes the backbone and trains only a linear classifier, CLAMP achieves competitive performance with state-of-the-art self-supervised models. Furthermore, our analysis reveals that neural manifolds corresponding to different categories emerge naturally and are effectively separated in the learned representation space, highlighting the potential of CLAMP to bridge insights from physics, neural science, and machine learning. 

**Abstract (ZH)**: 基于点WISE比较的对比自监督学习在视觉任务中已有广泛研究。我们提出了一种新的自监督框架Contrastive Learning As Manifold Packing (CLAMP)，将表示学习重新定义为流形填充问题。CLAMP 引入了一种损失函数，该损失函数受到短程排斥粒子系统的潜在能量的启发，类似于简单液体和挤实堆积在物理学中的情况。在该框架中，每个类别由嵌入单张图像多种增强视图的子流形组成。子流形的大小和位置通过跟随堆积损失的梯度动态优化。该方法在嵌入空间中产生了与阻塞物理学相平行的可解释动力学，并在损失函数中引入了几何上有意义的超参数。在标准的线性评估协议下，即冻结骨干网络并仅训练线性分类器，CLAMP 达到了与最先进的自监督模型相当的性能。进一步的分析表明，不同的类别对应的神经流形在学习表示空间中自然涌现并得到有效分离，突显了CLAMP 能够在物理学、神经科学和机器学习之间架起桥梁的潜力。 

---
# TimeMaster: Training Time-Series Multimodal LLMs to Reason via Reinforcement Learning 

**Title (ZH)**: TimeMaster: 训练时间序列多模态大语言模型基于强化学习进行推理 

**Authors**: Junru Zhang, Lang Feng, Xu Guo, Yuhan Wu, Yabo Dong, Duanqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13705)  

**Abstract**: Time-series reasoning remains a significant challenge in multimodal large language models (MLLMs) due to the dynamic temporal patterns, ambiguous semantics, and lack of temporal priors. In this work, we introduce TimeMaster, a reinforcement learning (RL)-based method that enables time-series MLLMs to perform structured, interpretable reasoning directly over visualized time-series inputs and task prompts. TimeMaster adopts a three-part structured output format, reasoning, classification, and domain-specific extension, and is optimized via a composite reward function that aligns format adherence, prediction accuracy, and open-ended insight quality. The model is trained using a two-stage pipeline: we first apply supervised fine-tuning (SFT) to establish a good initialization, followed by Group Relative Policy Optimization (GRPO) at the token level to enable stable and targeted reward-driven improvement in time-series reasoning. We evaluate TimeMaster on the TimerBed benchmark across six real-world classification tasks based on Qwen2.5-VL-3B-Instruct. TimeMaster achieves state-of-the-art performance, outperforming both classical time-series models and few-shot GPT-4o by over 14.6% and 7.3% performance gain, respectively. Notably, TimeMaster goes beyond time-series classification: it also exhibits expert-like reasoning behavior, generates context-aware explanations, and delivers domain-aligned insights. Our results highlight that reward-driven RL can be a scalable and promising path toward integrating temporal understanding into time-series MLLMs. 

**Abstract (ZH)**: 基于强化学习的时间大师：时间序列推理在多模态大语言模型中的直接结构化可解释推理 

---
# Value-Free Policy Optimization via Reward Partitioning 

**Title (ZH)**: 无需价值偏好的政策优化通过奖励分割 

**Authors**: Bilal Faye, Hanane Azzag, Mustapha Lebbah  

**Link**: [PDF](https://arxiv.org/pdf/2506.13702)  

**Abstract**: Single-trajectory reinforcement learning (RL) methods aim to optimize policies from datasets consisting of (prompt, response, reward) triplets, where scalar rewards are directly available. This supervision format is highly practical, as it mirrors real-world human feedback, such as thumbs-up/down signals, and avoids the need for structured preference annotations. In contrast, pairwise preference-based methods like Direct Preference Optimization (DPO) rely on datasets with both preferred and dispreferred responses, which are harder to construct and less natural to collect. Among single-trajectory approaches, Direct Reward Optimization (DRO) has shown strong empirical performance due to its simplicity and stability. However, DRO requires approximating a value function, which introduces several limitations: high off-policy variance, coupling between policy and value learning, and a lack of absolute supervision on the policy itself. We introduce Reward Partitioning Optimization (RPO), a new method that resolves these limitations by removing the need to model the value function. Instead, RPO normalizes observed rewards using a partitioning approach estimated directly from data. This leads to a straightforward supervised learning objective on the policy, with no auxiliary models and no joint optimization. RPO provides direct and stable supervision on the policy, making it robust and easy to implement in practice. We validate RPO on scalar-feedback language modeling tasks using Flan-T5 encoder-decoder models. Our results demonstrate that RPO outperforms existing single-trajectory baselines such as DRO and Kahneman-Tversky Optimization (KTO). These findings confirm that RPO is a simple, effective, and theoretically grounded method for single-trajectory policy optimization. 

**Abstract (ZH)**: 单轨迹强化学习（RL）方法旨在通过由(prompt, response, reward)三元组组成的数据集优化策略，其中标量奖励可以直接获得。这种监督格式非常实用，因为它模仿了现实生活中的人类反馈，如拇指点赞/反对信号，并避免了需要结构化偏好的标注。相比之下，基于成对偏好的方法，如直接偏好优化（DPO），依赖于包含偏好和非偏好响应的数据集，这些数据集更难构建且收集起来不太自然。在单轨迹方法中，直接奖励优化（DRO）由于其简单性和稳定性表现出强大的 empirical 性能。然而，DRO 需要近似一个价值函数，这引入了几种限制：高离策方差、策略和价值学习之间的耦合以及策略本身的绝对监督缺失。我们提出了一种新的 Reward Partitioning Optimization（RPO）方法，通过消除建模价值函数的需要来解决这些限制。相反，RPO 通过直接从数据估计的分区方法对观察到的奖励进行归一化。这导致了一个直接且稳定的策略监督学习目标，不需要辅助模型且不需要联合优化。RPO 为策略提供了直接且稳定的监督，使其在实践中更 robust 和容易实现。我们在使用 Flan-T5 编解码器模型的标量反馈语言建模任务上验证了 RPO。我们的结果表明，RPO 在现有的单轨迹基线方法，如 DRO 和 Kahneman-Tversky 方法（KTO）上表现更优。这些发现证实了 RPO 是一种简单、有效且理论基础坚实的方法，用于单轨迹策略优化。 

---
# Balancing Knowledge Delivery and Emotional Comfort in Healthcare Conversational Systems 

**Title (ZH)**: 在医疗对话系统中平衡知识传递与情感舒适度 

**Authors**: Shang-Chi Tsai, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13692)  

**Abstract**: With the advancement of large language models, many dialogue systems are now capable of providing reasonable and informative responses to patients' medical conditions. However, when patients consult their doctor, they may experience negative emotions due to the severity and urgency of their situation. If the model can provide appropriate comfort and empathy based on the patient's negative emotions while answering medical questions, it will likely offer a more reassuring experience during the medical consultation process. To address this issue, our paper explores the balance between knowledge sharing and emotional support in the healthcare dialogue process. We utilize a large language model to rewrite a real-world interactive medical dialogue dataset, generating patient queries with negative emotions and corresponding medical responses aimed at soothing the patient's emotions while addressing their concerns. The modified data serves to refine the latest large language models with various fine-tuning methods, enabling them to accurately provide sentences with both emotional reassurance and constructive suggestions in response to patients' questions. Compared to the original LLM model, our experimental results demonstrate that our methodology significantly enhances the model's ability to generate emotional responses while maintaining its original capability to provide accurate knowledge-based answers. 

**Abstract (ZH)**: 随着大型语言模型的发展，许多对话系统现在能够为患者提供合理且有信息量的医疗状况回应。然而，当患者咨询医生时，由于他们情况的严重性和紧迫性，他们可能会经历负面情绪。如果模型能在回答医疗问题时根据患者的情绪提供适当的安慰和同情，将在医疗咨询过程中提供一个更加令人安心的体验。为解决这一问题，我们的论文探讨了医疗对话过程中知识共享与情感支持之间的平衡。我们利用大型语言模型重新编写了一个真实的互动医疗对话数据集，生成带有负面情绪的患者查询和相应的医疗回应，旨在安抚患者的负面情绪并解决他们的担忧。修改后的数据用于通过各种微调方法细化最新的大型语言模型，使其能够准确地在回应患者的问题时提供既有情感安抚又有建设性建议的句子。与原始的LLM模型相比，我们的实验结果表明，我们的方法显著增强了模型生成情感回应的能力，同时保持其提供准确知识性答案的能力。 

---
# Meta-learning how to Share Credit among Macro-Actions 

**Title (ZH)**: 宏动作中的功劳共享元学习 

**Authors**: Ionel-Alexandru Hosu, Traian Rebedea, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13690)  

**Abstract**: One proposed mechanism to improve exploration in reinforcement learning is through the use of macro-actions. Paradoxically though, in many scenarios the naive addition of macro-actions does not lead to better exploration, but rather the opposite. It has been argued that this was caused by adding non-useful macros and multiple works have focused on mechanisms to discover effectively environment-specific useful macros. In this work, we take a slightly different perspective. We argue that the difficulty stems from the trade-offs between reducing the average number of decisions per episode versus increasing the size of the action space. Namely, one typically treats each potential macro-action as independent and atomic, hence strictly increasing the search space and making typical exploration strategies inefficient. To address this problem we propose a novel regularization term that exploits the relationship between actions and macro-actions to improve the credit assignment mechanism by reducing the effective dimension of the action space and, therefore, improving exploration. The term relies on a similarity matrix that is meta-learned jointly with learning the desired policy. We empirically validate our strategy looking at macro-actions in Atari games, and the StreetFighter II environment. Our results show significant improvements over the Rainbow-DQN baseline in all environments. Additionally, we show that the macro-action similarity is transferable to related environments. We believe this work is a small but important step towards understanding how the similarity-imposed geometry on the action space can be exploited to improve credit assignment and exploration, therefore making learning more effective. 

**Abstract (ZH)**: 一种通过宏动作提高强化学习探索机制的研究：从减少每episode的平均决策次数与增加动作空间大小的权衡视角出发 

---
# ROSA: Harnessing Robot States for Vision-Language and Action Alignment 

**Title (ZH)**: ROSA: 利用机器人状态实现视觉-语言和动作对齐 

**Authors**: Yuqing Wen, Kefan Gu, Haoxuan Liu, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.13679)  

**Abstract**: Vision-Language-Action (VLA) models have recently made significant advance in multi-task, end-to-end robotic control, due to the strong generalization capabilities of Vision-Language Models (VLMs). A fundamental challenge in developing such models is effectively aligning the vision-language space with the robotic action space. Existing approaches typically rely on directly fine-tuning VLMs using expert demonstrations. However, this strategy suffers from a spatio-temporal gap, resulting in considerable data inefficiency and heavy reliance on human labor. Spatially, VLMs operate within a high-level semantic space, whereas robotic actions are grounded in low-level 3D physical space; temporally, VLMs primarily interpret the present, while VLA models anticipate future actions. To overcome these challenges, we propose a novel training paradigm, ROSA, which leverages robot state estimation to improve alignment between vision-language and action spaces. By integrating robot state estimation data obtained via an automated process, ROSA enables the VLA model to gain enhanced spatial understanding and self-awareness, thereby boosting performance and generalization. Extensive experiments in both simulated and real-world environments demonstrate the effectiveness of ROSA, particularly in low-data regimes. 

**Abstract (ZH)**: 基于机器人状态估计的视觉-语言-动作模型训练 paradigm (ROSA) 

---
# Prefix-Tuning+: Modernizing Prefix-Tuning through Attention Independent Prefix Data 

**Title (ZH)**: Prefix-Tuning+: 现代化基于注意力独立前缀数据的Prefix-Tuning 

**Authors**: Haonan Wang, Brian Chen, Li Siquan, Liang Xinhe, Tianyang Hu, Hwee Kuan Lee, Kenji Kawaguchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13674)  

**Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods have become crucial for rapidly adapting large language models (LLMs) to downstream tasks. Prefix-Tuning, an early and effective PEFT technique, demonstrated the ability to achieve performance comparable to full fine-tuning with significantly reduced computational and memory overhead. However, despite its earlier success, its effectiveness in training modern state-of-the-art LLMs has been very limited. In this work, we demonstrate empirically that Prefix-Tuning underperforms on LLMs because of an inherent tradeoff between input and prefix significance within the attention head. This motivates us to introduce Prefix-Tuning+, a novel architecture that generalizes the principles of Prefix-Tuning while addressing its shortcomings by shifting the prefix module out of the attention head itself. We further provide an overview of our construction process to guide future users when constructing their own context-based methods. Our experiments show that, across a diverse set of benchmarks, Prefix-Tuning+ consistently outperforms existing Prefix-Tuning methods. Notably, it achieves performance on par with the widely adopted LoRA method on several general benchmarks, highlighting the potential modern extension of Prefix-Tuning approaches. Our findings suggest that by overcoming its inherent limitations, Prefix-Tuning can remain a competitive and relevant research direction in the landscape of parameter-efficient LLM adaptation. 

**Abstract (ZH)**: Parameter-高效微调（PEFT）方法已成为快速适应大型语言模型（LLMs）的下游任务的关键。前缀微调（Prefix-Tuning），作为一种早期且有效的PEFT技术，展示了通过显著减少计算和内存开销，仍能达到与全量微调相当的性能的能力。然而，尽管早期表现出色，它在训练现代的最先进的LLMs时的效果极为有限。在本研究中，我们通过实验证明，前缀微调在LLMs上的表现不佳是因为其关注点和前缀重要性之间的固有权衡。这促使我们提出了一种新的前缀微调+（Prefix-Tuning+）架构，该架构不仅扩展了前缀微调的原则，而且还通过将前缀模块移出注意力头本身来解决其不足之处。我们还提供了我们构建过程的概述，以指导未来用户构建自己的基于上下文的方法。实验结果显示，Prefix-Tuning+在多种基准测试中都优于现有的前缀微调方法。值得注意的是，在几个通用基准测试中，它达到了与广为采用的LoRA方法相当的性能，突显了前缀微调方法现代化扩展的潜力。我们的发现表明，通过克服其固有的局限性，前缀微调仍可以成为参数高效LLM适应研究方向中的一个竞争性和相关的研究方向。 

---
# We Should Identify and Mitigate Third-Party Safety Risks in MCP-Powered Agent Systems 

**Title (ZH)**: 我们应该识别并缓解由MCP驱动的代理系统中的第三方安全风险。 

**Authors**: Junfeng Fang, Zijun Yao, Ruipeng Wang, Haokai Ma, Xiang Wang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.13666)  

**Abstract**: The development of large language models (LLMs) has entered in a experience-driven era, flagged by the emergence of environment feedback-driven learning via reinforcement learning and tool-using agents. This encourages the emergenece of model context protocol (MCP), which defines the standard on how should a LLM interact with external services, such as \api and data. However, as MCP becomes the de facto standard for LLM agent systems, it also introduces new safety risks. In particular, MCP introduces third-party services, which are not controlled by the LLM developers, into the agent systems. These third-party MCP services provider are potentially malicious and have the economic incentives to exploit vulnerabilities and sabotage user-agent interactions. In this position paper, we advocate the research community in LLM safety to pay close attention to the new safety risks issues introduced by MCP, and develop new techniques to build safe MCP-powered agent systems. To establish our position, we argue with three key parts. (1) We first construct \framework, a controlled framework to examine safety issues in MCP-powered agent systems. (2) We then conduct a series of pilot experiments to demonstrate the safety risks in MCP-powered agent systems is a real threat and its defense is not trivial. (3) Finally, we give our outlook by showing a roadmap to build safe MCP-powered agent systems. In particular, we would call for researchers to persue the following research directions: red teaming, MCP safe LLM development, MCP safety evaluation, MCP safety data accumulation, MCP service safeguard, and MCP safe ecosystem construction. We hope this position paper can raise the awareness of the research community in MCP safety and encourage more researchers to join this important research direction. Our code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型的发展进入了以经验为主导的时代，这标志着通过强化学习和工具使用代理的环境反馈驱动学习的出现。这促进了模型上下文协议（MCP）的兴起，定义了LLM与外部服务（如API和数据）交互的标准。然而，随着MCP成为LLM代理系统的事实标准，它也引入了新的安全风险。特别是，MCP将由LLM开发者不受控制的第三方服务引入代理系统。这些第三方MCP服务提供商可能具有恶意动机，利用漏洞破坏用户-代理交互。在这份立场论文中，我们呼吁大型语言模型安全性研究领域的研究者注意MCP引入的新安全风险问题，并开发新技术以构建安全的MCP驱动代理系统。为了阐明我们的立场，我们从三个关键部分进行论述：首先，我们构建了一个控制框架来检查MCP驱动代理系统中的安全性问题；其次，我们进行了一系列试点实验以证明MCP驱动代理系统中的安全风险是真实存在的且防护并非易事；最后，我们提出了构建安全的MCP驱动代理系统的路线图，并呼吁研究人员在以下方向开展研究：红队攻击、安全的LLM开发、MCP安全性评估、MCP安全数据积累、MCP服务保护以及安全的MCP生态系统建设。我们希望这份立场论文能够提高大型语言模型安全性研究领域的研究者对该问题的认识，并鼓励更多研究者加入这一重要的研究方向。我们的代码可在以下链接获取：this https URL。 

---
# Ego-R1: Chain-of-Tool-Thought for Ultra-Long Egocentric Video Reasoning 

**Title (ZH)**: 自我中心视频超长推理的工具链思考（Ego-R1） 

**Authors**: Shulin Tian, Ruiqi Wang, Hongming Guo, Penghao Wu, Yuhao Dong, Xiuying Wang, Jingkang Yang, Hao Zhang, Hongyuan Zhu, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13654)  

**Abstract**: We introduce Ego-R1, a novel framework for reasoning over ultra-long (i.e., in days and weeks) egocentric videos, which leverages a structured Chain-of-Tool-Thought (CoTT) process, orchestrated by an Ego-R1 Agent trained via reinforcement learning (RL). Inspired by human problem-solving strategies, CoTT decomposes complex reasoning into modular steps, with the RL agent invoking specific tools, one per step, to iteratively and collaboratively answer sub-questions tackling such tasks as temporal retrieval and multi-modal understanding. We design a two-stage training paradigm involving supervised finetuning (SFT) of a pretrained language model using CoTT data and RL to enable our agent to dynamically propose step-by-step tools for long-range reasoning. To facilitate training, we construct a dataset called Ego-R1 Data, which consists of Ego-CoTT-25K for SFT and Ego-QA-4.4K for RL. Furthermore, our Ego-R1 agent is evaluated on a newly curated week-long video QA benchmark, Ego-R1 Bench, which contains human-verified QA pairs from hybrid sources. Extensive results demonstrate that the dynamic, tool-augmented chain-of-thought reasoning by our Ego-R1 Agent can effectively tackle the unique challenges of understanding ultra-long egocentric videos, significantly extending the time coverage from few hours to a week. 

**Abstract (ZH)**: 基于结构化工具-思考链（CoTT）的Ego-R1：一种用于超长周期自我中心视频推理的新框架 

---
# DualEdit: Dual Editing for Knowledge Updating in Vision-Language Models 

**Title (ZH)**: DualEdit: 双重编辑用于视觉-语言模型的知识更新 

**Authors**: Zhiyi Shi, Binjie Wang, Chongjie Si, Yichen Wu, Junsik Kim, Hanspeter Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2506.13638)  

**Abstract**: Model editing aims to efficiently update a pre-trained model's knowledge without the need for time-consuming full retraining. While existing pioneering editing methods achieve promising results, they primarily focus on editing single-modal language models (LLMs). However, for vision-language models (VLMs), which involve multiple modalities, the role and impact of each modality on editing performance remain largely unexplored. To address this gap, we explore the impact of textual and visual modalities on model editing and find that: (1) textual and visual representations reach peak sensitivity at different layers, reflecting their varying importance; and (2) editing both modalities can efficiently update knowledge, but this comes at the cost of compromising the model's original capabilities. Based on our findings, we propose DualEdit, an editor that modifies both textual and visual modalities at their respective key layers. Additionally, we introduce a gating module within the more sensitive textual modality, allowing DualEdit to efficiently update new knowledge while preserving the model's original information. We evaluate DualEdit across multiple VLM backbones and benchmark datasets, demonstrating its superiority over state-of-the-art VLM editing baselines as well as adapted LLM editing methods on different evaluation metrics. 

**Abstract (ZH)**: 模型编辑旨在高效更新预训练模型的知识，而无需进行耗时的完全重新训练。尽管现有的先驱编辑方法取得了令人鼓舞的结果，它们主要集中在编辑单模态语言模型（LLMs）上。然而，对于涉及多模态的视觉语言模型（VLMs），每个模态在编辑性能中的作用和影响尚未得到充分探索。为解决这一问题，我们研究了文本和视觉模态对模型编辑的影响，并发现：（1）文本和视觉表示在不同的层达到最大敏感性，反映出它们的不同重要性；（2）同时编辑这两个模态可以高效率地更新知识，但会牺牲模型的原始能力。基于我们的发现，我们提出DualEdit，这是一种在各自的关键层修改文本和视觉模态的编辑器。此外，我们还在更敏感的文本模态中引入了一个门控模块，使DualEdit能够在高效更新新知识的同时保留模型的原始信息。我们在多个VLM骨干网络和基准数据集上评估DualEdit，展示了其在不同评估指标上优于最先进的VLM编辑基线以及适应的LLM编辑方法的优越性。 

---
# Graph-Convolution-Beta-VAE for Synthetic Abdominal Aorta Aneurysm Generation 

**Title (ZH)**: 基于图卷积-贝塔VAE的合成腹主动脉瘤生成 

**Authors**: Francesco Fabbri, Martino Andrea Scarpolini, Angelo Iollo, Francesco Viola, Francesco Tudisco  

**Link**: [PDF](https://arxiv.org/pdf/2506.13628)  

**Abstract**: Synthetic data generation plays a crucial role in medical research by mitigating privacy concerns and enabling large-scale patient data analysis. This study presents a beta-Variational Autoencoder Graph Convolutional Neural Network framework for generating synthetic Abdominal Aorta Aneurysms (AAA). Using a small real-world dataset, our approach extracts key anatomical features and captures complex statistical relationships within a compact disentangled latent space. To address data limitations, low-impact data augmentation based on Procrustes analysis was employed, preserving anatomical integrity. The generation strategies, both deterministic and stochastic, manage to enhance data diversity while ensuring realism. Compared to PCA-based approaches, our model performs more robustly on unseen data by capturing complex, nonlinear anatomical variations. This enables more comprehensive clinical and statistical analyses than the original dataset alone. The resulting synthetic AAA dataset preserves patient privacy while providing a scalable foundation for medical research, device testing, and computational modeling. 

**Abstract (ZH)**: 合成数据生成在减轻隐私担忧和促进大规模患者数据分析方面对医学研究发挥着关键作用。本研究提出了一种beta-变分自编码器图卷积神经网络框架，用于生成腹主动脉瘤（AAA）的合成数据。通过小规模真实数据集，我们的方法提取关键解剖特征并捕捉紧凑分离潜空间内的复杂统计关系。为解决数据限制问题，我们采用了基于Procrustes分析的低影响数据增强方法，保持了解剖完整性。生成策略，无论是确定性的还是随机性的，都能够增强数据多样性并确保真实性。与基于PCA的方法相比，我们的模型在未见数据上表现更稳健，因为它能够捕捉复杂的非线性解剖变异。这使得可以比原数据集更全面地进行临床和统计分析。生成的数据集在保护患者隐私的同时，为医学研究、设备测试和计算建模提供了可扩展的基础。 

---
# EBS-CFL: Efficient and Byzantine-robust Secure Clustered Federated Learning 

**Title (ZH)**: EBS-CFL: 高效且抗拜占庭容错的安全集群联邦学习 

**Authors**: Zhiqiang Li, Haiyong Bao, Menghong Guan, Hao Pan, Cheng Huang, Hong-Ning Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13612)  

**Abstract**: Despite federated learning (FL)'s potential in collaborative learning, its performance has deteriorated due to the data heterogeneity of distributed users. Recently, clustered federated learning (CFL) has emerged to address this challenge by partitioning users into clusters according to their similarity. However, CFL faces difficulties in training when users are unwilling to share their cluster identities due to privacy concerns. To address these issues, we present an innovative Efficient and Robust Secure Aggregation scheme for CFL, dubbed EBS-CFL. The proposed EBS-CFL supports effectively training CFL while maintaining users' cluster identity confidentially. Moreover, it detects potential poisonous attacks without compromising individual client gradients by discarding negatively correlated gradients and aggregating positively correlated ones using a weighted approach. The server also authenticates correct gradient encoding by clients. EBS-CFL has high efficiency with client-side overhead O(ml + m^2) for communication and O(m^2l) for computation, where m is the number of cluster identities, and l is the gradient size. When m = 1, EBS-CFL's computational efficiency of client is at least O(log n) times better than comparison schemes, where n is the number of this http URL addition, we validate the scheme through extensive experiments. Finally, we theoretically prove the scheme's security. 

**Abstract (ZH)**: 一种高效的稳健安全聚合方案EBS-CFL及其在聚类联邦学习中的应用 

---
# A Hybrid Artificial Intelligence Method for Estimating Flicker in Power Systems 

**Title (ZH)**: 电力系统中混合人工智能方法估计闪变 

**Authors**: Javad Enayati, Pedram Asef, Alexandre Benoit  

**Link**: [PDF](https://arxiv.org/pdf/2506.13611)  

**Abstract**: This paper introduces a novel hybrid AI method combining H filtering and an adaptive linear neuron network for flicker component estimation in power distribution this http URL proposed method leverages the robustness of the H filter to extract the voltage envelope under uncertain and noisy conditions followed by the use of ADALINE to accurately identify flicker frequencies embedded in the this http URL synergy enables efficient time domain estimation with rapid convergence and noise resilience addressing key limitations of existing frequency domain this http URL conventional techniques this hybrid AI model handles complex power disturbances without prior knowledge of noise characteristics or extensive this http URL validate the method performance we conduct simulation studies based on IEC Standard 61000 4 15 supported by statistical analysis Monte Carlo simulations and real world this http URL demonstrate superior accuracy robustness and reduced computational load compared to Fast Fourier Transform and Discrete Wavelet Transform based estimators. 

**Abstract (ZH)**: 本文介绍了一种结合H滤波器和自适应线性神经网络的新型混合人工智能方法，用于电力分配中的闪烁成分估计。该方法利用H滤波器在不确定性及噪声条件下提取电压包络的稳健性，继而使用ADALINE准确识别嵌入其中的闪烁频率。这种 synergy 能够实现高效的时域估计，具有快速收敛和抗噪声的能力，解决了现有频域技术的 key limitations。这种混合人工智能模型能够在无需了解噪声特性或进行大量预先知识的情况下处理复杂电力扰动。为了验证该方法的性能，我们基于IEC标准61000-4-15进行了仿真研究，支持以统计分析、蒙特卡洛模拟和实际应用为基础的实验。研究结果表明，与基于快速傅里叶变换和离散小波变换的估计器相比，该方法具有更高的准确性和鲁棒性，并且计算负载更低。 

---
# CAMS: A CityGPT-Powered Agentic Framework for Urban Human Mobility Simulation 

**Title (ZH)**: CAMS：一个由CityGPT驱动的 urbans人类移动模拟代理框架 

**Authors**: Yuwei Du, Jie Feng, Jian Yuan, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13599)  

**Abstract**: Human mobility simulation plays a crucial role in various real-world applications. Recently, to address the limitations of traditional data-driven approaches, researchers have explored leveraging the commonsense knowledge and reasoning capabilities of large language models (LLMs) to accelerate human mobility simulation. However, these methods suffer from several critical shortcomings, including inadequate modeling of urban spaces and poor integration with both individual mobility patterns and collective mobility distributions. To address these challenges, we propose \textbf{C}ityGPT-Powered \textbf{A}gentic framework for \textbf{M}obility \textbf{S}imulation (\textbf{CAMS}), an agentic framework that leverages the language based urban foundation model to simulate human mobility in urban space. \textbf{CAMS} comprises three core modules, including MobExtractor to extract template mobility patterns and synthesize new ones based on user profiles, GeoGenerator to generate anchor points considering collective knowledge and generate candidate urban geospatial knowledge using an enhanced version of CityGPT, TrajEnhancer to retrieve spatial knowledge based on mobility patterns and generate trajectories with real trajectory preference alignment via DPO. Experiments on real-world datasets show that \textbf{CAMS} achieves superior performance without relying on externally provided geospatial information. Moreover, by holistically modeling both individual mobility patterns and collective mobility constraints, \textbf{CAMS} generates more realistic and plausible trajectories. In general, \textbf{CAMS} establishes a new paradigm that integrates the agentic framework with urban-knowledgeable LLMs for human mobility simulation. 

**Abstract (ZH)**: 基于CityGPT的城市代理性移动模拟框架（CAMS） 

---
# Can you see how I learn? Human observers' inferences about Reinforcement Learning agents' learning processes 

**Title (ZH)**: 你能看出我是如何学习的？人类观察者对强化学习代理学习过程的推断。 

**Authors**: Bernhard Hilpert, Muhan Hou, Kim Baraka, Joost Broekens  

**Link**: [PDF](https://arxiv.org/pdf/2506.13583)  

**Abstract**: Reinforcement Learning (RL) agents often exhibit learning behaviors that are not intuitively interpretable by human observers, which can result in suboptimal feedback in collaborative teaching settings. Yet, how humans perceive and interpret RL agent's learning behavior is largely unknown. In a bottom-up approach with two experiments, this work provides a data-driven understanding of the factors of human observers' understanding of the agent's learning process. A novel, observation-based paradigm to directly assess human inferences about agent learning was developed. In an exploratory interview study (\textit{N}=9), we identify four core themes in human interpretations: Agent Goals, Knowledge, Decision Making, and Learning Mechanisms. A second confirmatory study (\textit{N}=34) applied an expanded version of the paradigm across two tasks (navigation/manipulation) and two RL algorithms (tabular/function approximation). Analyses of 816 responses confirmed the reliability of the paradigm and refined the thematic framework, revealing how these themes evolve over time and interrelate. Our findings provide a human-centered understanding of how people make sense of agent learning, offering actionable insights for designing interpretable RL systems and improving transparency in Human-Robot Interaction. 

**Abstract (ZH)**: 强化学习（RL）代理的学习行为往往难以被人类观察者直观理解，这在协作教学场景中可能导致反馈不足。然而，人类如何感知和解释RL代理的学习行为尚不清楚。通过自下而上的两种实验，本研究提供了关于人类观察者理解代理学习过程的影响因素的数据驱动理解。我们开发了一种基于观察的新型范式，直接评估人类对代理学习的推断。在探索性访谈研究（N=9）中，我们识别了四种核心主题：代理目标、知识、决策制定和学习机制。在确认性研究（N=34）中，我们应用扩展后的范式，在两个任务（导航/操作）和两种RL算法（表Lookup/函数逼近）上进行。对816个响应的分析证实了该范式的可靠性，并细化了主题框架，揭示了这些主题如何随时间发展及其相互关系。我们的研究结果提供了一种以人类为中心的理解方式，说明了人们如何理解代理学习，为设计可解释的RL系统和提高人机交互的透明度提供了实用见解。 

---
# Flexible-length Text Infilling for Discrete Diffusion Models 

**Title (ZH)**: 长度可变的文本填充用于离散扩散模型 

**Authors**: Andrew Zhang, Anushka Sivakumar, Chiawei Tang, Chris Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.13579)  

**Abstract**: Discrete diffusion models are a new class of text generators that offer advantages such as bidirectional context use, parallelizable generation, and flexible prompting compared to autoregressive models. However, a critical limitation of discrete diffusion models is their inability to perform flexible-length or flexible-position text infilling without access to ground-truth positional data. We introduce \textbf{DDOT} (\textbf{D}iscrete \textbf{D}iffusion with \textbf{O}ptimal \textbf{T}ransport Position Coupling), the first discrete diffusion model to overcome this challenge. DDOT jointly denoises token values and token positions, employing a novel sample-level Optimal Transport (OT) coupling. This coupling preserves relative token ordering while dynamically adjusting the positions and length of infilled segments, a capability previously missing in text diffusion. Our method is orthogonal to existing discrete text diffusion methods and is compatible with various pretrained text denoisers. Extensive experiments on text infilling benchmarks such as One-Billion-Word and Yelp demonstrate that DDOT outperforms naive diffusion baselines. Furthermore, DDOT achieves performance on par with state-of-the-art non-autoregressive models and enables significant improvements in training efficiency and flexibility. 

**Abstract (ZH)**: 离散扩散模型是一种新的文本生成器，相较于自回归模型，它具有双向上下文利用、并行生成和灵活提示等优势。然而，离散扩散模型的一个关键局限是它们无法在没有地面真实位置数据的情况下进行灵活长度或灵活位置的文本填充。我们引入了**DDOT（离散扩散与最优传输位置耦合）**，这是第一个克服这一挑战的离散扩散模型。DDOT同时去噪词值和词位置，采用一种新颖的样本级最优传输（OT）耦合。这种耦合保留了词的相对顺序，同时动态调整填充段的位置和长度，这是文本扩散中以前缺失的能力。我们的方法与现有的离散文本扩散方法正交，并且兼容各种预训练的文本去噪器。在One-Billion-Word和Yelp等文本填充基准测试中，DDOT在性能上优于简单的扩散基线模型。此外，DDOT在性能上与最先进的非自回归模型相当，并能显著提高训练效率和灵活性。 

---
# A Production Scheduling Framework for Reinforcement Learning Under Real-World Constraints 

**Title (ZH)**: 基于实际约束条件的强化学习生产调度框架 

**Authors**: Jonathan Hoss, Felix Schelling, Noah Klarmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.13566)  

**Abstract**: The classical Job Shop Scheduling Problem (JSSP) focuses on optimizing makespan under deterministic constraints. Real-world production environments introduce additional complexities that cause traditional scheduling approaches to be less effective. Reinforcement learning (RL) holds potential in addressing these challenges, as it allows agents to learn adaptive scheduling strategies. However, there is a lack of a comprehensive, general-purpose frameworks for effectively training and evaluating RL agents under real-world constraints. To address this gap, we propose a modular framework that extends classical JSSP formulations by incorporating key \mbox{real-world} constraints inherent to the shopfloor, including transport logistics, buffer management, machine breakdowns, setup times, and stochastic processing conditions, while also supporting multi-objective optimization. The framework is a customizable solution that offers flexibility in defining problem instances and configuring simulation parameters, enabling adaptation to diverse production scenarios. A standardized interface ensures compatibility with various RL approaches, providing a robust environment for training RL agents and facilitating the standardized comparison of different scheduling methods under dynamic and uncertain conditions. We release JobShopLab as an open-source tool for both research and industrial applications, accessible at: this https URL 

**Abstract (ZH)**: 经典的作业车间调度问题（JSSP）专注于在确定性约束条件下优化生产周期。现实世界的生产环境引入了额外的复杂性，使得传统的调度方法 effectiveness降低。强化学习（RL）有可能通过允许代理学习适应性调度策略来应对这些挑战。然而，缺乏适用于实际约束条件下有效训练和评估RL代理的综合通用框架。为解决这一问题，我们提出了一种模块化框架，该框架扩展了经典的JSSP形式化模型，整合了车间环境固有的关键现实约束，包括物流运输、缓冲管理、机器故障、设置时间以及随机加工条件，同时支持多目标优化。该框架是一个可定制的解决方案，提供了定义问题实例和配置仿真参数的灵活性，以适应不同的生产情景。标准化的接口确保了与各种RL方法的兼容性，提供了一个稳健的环境来训练RL代理，并促进了在动态和不确定条件下的不同调度方法的标准比较。我们发布JobShopLab作为一款开源工具，适用于研究和工业应用，网址为: this https URL。 

---
# Understand the Implication: Learning to Think for Pragmatic Understanding 

**Title (ZH)**: 理解含义：学习以务实的方式理解 

**Authors**: Settaluri Lakshmi Sravanthi, Kishan Maharaj, Sravani Gunnu, Abhijit Mishra, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2506.13559)  

**Abstract**: Pragmatics, the ability to infer meaning beyond literal interpretation, is crucial for social cognition and communication. While LLMs have been benchmarked for their pragmatic understanding, improving their performance remains underexplored. Existing methods rely on annotated labels but overlook the reasoning process humans naturally use to interpret implicit meaning. To bridge this gap, we introduce a novel pragmatic dataset, ImpliedMeaningPreference, that includes explicit reasoning (thoughts) for both correct and incorrect interpretations. Through preference-tuning and supervised fine-tuning, we demonstrate that thought-based learning significantly enhances LLMs' pragmatic understanding, improving accuracy by 11.12% across model families. We further discuss a transfer-learning study where we evaluate the performance of thought-based training for the other tasks of pragmatics (presupposition, deixis) that are not seen during the training time and observe an improvement of 16.10% compared to label-trained models. 

**Abstract (ZH)**: 语用学，超越字面意义推理的能力，对于社会认知和交流至关重要。尽管大语言模型已经在其语用理解方面进行了基准测试，但提高其性能仍需进一步探索。现有方法依赖于标注标签，但忽略了人类自然使用的推理过程来解释隐含意义。为填补这一空白，我们引入了一个新型语用数据集，ImpliedMeaningPreference，该数据集包括正确和错误解释的显式推理（思考）。通过偏好调优和监督微调，我们证明基于思考的学习显著提高了大语言模型的语用理解，各模型家族的准确性提高了11.12%。进一步地，我们讨论了一项迁移学习研究，评估了基于思考训练在训练过程中未见过的语用其他任务（预设、指示语）上的性能，发现其性能相较于标签训练模型提高了16.10%。 

---
# Seismic Acoustic Impedance Inversion Framework Based on Conditional Latent Generative Diffusion Model 

**Title (ZH)**: 基于条件潜在生成扩散模型的地震声学阻抗反演框架 

**Authors**: Jie Chen, Hongling Chen, Jinghuai Gao, Chuangji Meng, Tao Yang, XinXin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13529)  

**Abstract**: Seismic acoustic impedance plays a crucial role in lithological identification and subsurface structure interpretation. However, due to the inherently ill-posed nature of the inversion problem, directly estimating impedance from post-stack seismic data remains highly challenging. Recently, diffusion models have shown great potential in addressing such inverse problems due to their strong prior learning and generative capabilities. Nevertheless, most existing methods operate in the pixel domain and require multiple iterations, limiting their applicability to field data. To alleviate these limitations, we propose a novel seismic acoustic impedance inversion framework based on a conditional latent generative diffusion model, where the inversion process is made in latent space. To avoid introducing additional training overhead when embedding conditional inputs, we design a lightweight wavelet-based module into the framework to project seismic data and reuse an encoder trained on impedance to embed low-frequency impedance into the latent space. Furthermore, we propose a model-driven sampling strategy during the inversion process of this framework to enhance accuracy and reduce the number of required diffusion steps. Numerical experiments on a synthetic model demonstrate that the proposed method achieves high inversion accuracy and strong generalization capability within only a few diffusion steps. Moreover, application to field data reveals enhanced geological detail and higher consistency with well-log measurements, validating the effectiveness and practicality of the proposed approach. 

**Abstract (ZH)**: seismic acoustic阻抗在地质识别和地下结构解释中扮演着关键角色。然而，由于逆问题的固有病态性质，直接从后处理地震数据中估计阻抗仍然极具挑战性。近年来，扩散模型因其强大的先验学习能力和生成能力，在解决此类逆问题方面展示了巨大潜力。尽管如此，现有的大多数方法都在像素域中操作，并需要多次迭代，限制了它们在实地数据中的应用。为克服这些限制，我们提出了一种基于条件潜在生成扩散模型的新型地震声阻抗逆演框架，其中逆演过程在潜在空间中进行。为了在嵌入条件输入时避免引入额外的训练开销，我们在框架中设计了一个轻量级小波模块来投影地震数据，并利用在阻抗上预训练的编码器将低频阻抗嵌入到潜在空间中。此外，在这个框架的逆演过程中，我们提出了一种模型驱动的采样策略，以增强精度并减少所需的扩散步骤数量。数值实验表明，所提出的方法仅在几个扩散步骤内实现了高逆演精度和强大的泛化能力。此外，对实地数据的应用揭示了增强的地质细节，并与井壁测量数据具有更高的一致性，验证了所提方法的有效性和实用性。 

---
# The Price of Freedom: Exploring Expressivity and Runtime Tradeoffs in Equivariant Tensor Products 

**Title (ZH)**: 自由的价格：探索不变张量乘积的表达能力和运行时-tradeoff$username
user
把下面的论文内容或标题翻译成中文：Reinforcement Learning as Intrinsic Motivation for Exploration in Heterogeneous Groups. 

**Authors**: YuQing Xie, Ameya Daigavane, Mit Kotak, Tess Smidt  

**Link**: [PDF](https://arxiv.org/pdf/2506.13523)  

**Abstract**: $E(3)$-equivariant neural networks have demonstrated success across a wide range of 3D modelling tasks. A fundamental operation in these networks is the tensor product, which interacts two geometric features in an equivariant manner to create new features. Due to the high computational complexity of the tensor product, significant effort has been invested to optimize the runtime of this operation. For example, Luo et al. (2024) recently proposed the Gaunt tensor product (GTP) which promises a significant speedup. In this work, we provide a careful, systematic analysis of a number of tensor product operations. In particular, we emphasize that different tensor products are not performing the same operation. The reported speedups typically come at the cost of expressivity. We introduce measures of expressivity and interactability to characterize these differences. In addition, we realized the original implementation of GTP can be greatly simplified by directly using a spherical grid at no cost in asymptotic runtime. This spherical grid approach is faster on our benchmarks and in actual training of the MACE interatomic potential by 30\%. Finally, we provide the first systematic microbenchmarks of the various tensor product operations. We find that the theoretical runtime guarantees can differ wildly from empirical performance, demonstrating the need for careful application-specific benchmarking. Code is available at \href{this https URL}{this https URL} 

**Abstract (ZH)**: $E(3)$-对称神经网络在广泛三维建模任务中取得了成功。这些网络中的一个基本操作是张量积，它以对称方式相互作用两个几何特征以生成新的特征。由于张量积计算复杂度高，投入了大量努力来优化此操作的运行时。例如，Luo等（2024）最近提出了Gaunt张量积（GTP），承诺显著提高运行速度。在本文中，我们对几种张量积操作进行了细致的系统分析。特别是，我们强调不同张量积并未执行相同的操作。所报告的加速通常是以表达能力为代价的。我们引入了表达能力和交互性的度量来表征这些差异。此外，我们发现GTP的原始实现可以通过直接使用球形网格大大简化，且不影响渐进运行时。在我们的基准测试和MACE原子间势能的实际训练中，这种球形网格方法比原始实现快30%。最后，我们提供了各种张量积操作的第一套系统微基准测试。我们发现理论上的运行时保证与实际表现之间可能存在巨大差异，突显了详细应用特定基准测试的必要性。代码可在<这个超链接>获得。 

---
# UAV Object Detection and Positioning in a Mining Industrial Metaverse with Custom Geo-Referenced Data 

**Title (ZH)**: 基于自定义地理参考数据的采矿工业元宇宙中无人机目标检测与定位 

**Authors**: Vasiliki Balaska, Ioannis Tsampikos Papapetros, Katerina Maria Oikonomou, Loukas Bampis, Antonios Gasteratos  

**Link**: [PDF](https://arxiv.org/pdf/2506.13505)  

**Abstract**: The mining sector increasingly adopts digital tools to improve operational efficiency, safety, and data-driven decision-making. One of the key challenges remains the reliable acquisition of high-resolution, geo-referenced spatial information to support core activities such as extraction planning and on-site monitoring. This work presents an integrated system architecture that combines UAV-based sensing, LiDAR terrain modeling, and deep learning-based object detection to generate spatially accurate information for open-pit mining environments. The proposed pipeline includes geo-referencing, 3D reconstruction, and object localization, enabling structured spatial outputs to be integrated into an industrial digital twin platform. Unlike traditional static surveying methods, the system offers higher coverage and automation potential, with modular components suitable for deployment in real-world industrial contexts. While the current implementation operates in post-flight batch mode, it lays the foundation for real-time extensions. The system contributes to the development of AI-enhanced remote sensing in mining by demonstrating a scalable and field-validated geospatial data workflow that supports situational awareness and infrastructure safety. 

**Abstract (ZH)**: 矿业领域 increasingly采用数字工具以提高运营效率、安全性和数据驱动的决策能力。其中一个关键挑战是可靠地获取高分辨率、地理参考的空间信息，以支持诸如开采规划和现场监控等核心活动。本文提出了一种集成系统架构，结合了基于无人机的传感、LiDAR地形建模以及基于深度学习的对象检测，以生成适用于露天矿业环境的空间精确信息。提出的流程包括地理参考、三维重建和对象定位，使结构化空间输出能够集成到工业数字孪生平台中。与传统的静态测量方法相比，该系统提供了更高的覆盖范围和自动化潜力，并具有模块化组件，适用于实际工业环境的部署。虽然目前的实现方式在飞行后以批处理模式运行，但它为实时扩展奠定了基础。该系统通过展示一种可扩展且已在现场验证过的地理空间数据工作流，支持态势感知和基础设施安全，从而促进了采矿领域的AI增强遥感技术的发展。 

---
# Position: Pause Recycling LoRAs and Prioritize Mechanisms to Uncover Limits and Effectiveness 

**Title (ZH)**: 位置：暂停回收LoRAs并优先考虑机制以揭示局限性和有效性 

**Authors**: Mei-Yen Chen, Thi Thu Uyen Hoang, Michael Hahn, M. Saquib Sarfraz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13479)  

**Abstract**: Merging or routing low-rank adapters (LoRAs) has emerged as a popular solution for enhancing large language models, particularly when data access is restricted by regulatory or domain-specific constraints. This position paper argues that the research community should shift its focus from developing new merging or routing algorithms to understanding the conditions under which reusing LoRAs is truly effective. Through theoretical analysis and synthetic two-hop reasoning and math word-problem tasks, we examine whether reusing LoRAs enables genuine compositional generalization or merely reflects shallow pattern matching. Evaluating two data-agnostic methods--parameter averaging and dynamic adapter selection--we found that reusing LoRAs often fails to logically integrate knowledge across disjoint fine-tuning datasets, especially when such knowledge is underrepresented during pretraining. Our empirical results, supported by theoretical insights into LoRA's limited expressiveness, highlight the preconditions and constraints of reusing them for unseen tasks and cast doubt on its feasibility as a truly data-free approach. We advocate for pausing the pursuit of novel methods for recycling LoRAs and emphasize the need for rigorous mechanisms to guide future academic research in adapter-based model merging and practical system designs for practitioners. 

**Abstract (ZH)**: Merging或路由低秩适配器（LoRAs）已成为在数据访问受限于监管或特定领域约束时增强大型语言模型的一种流行解决方案。本文认为，研究界应将重点从开发新的合并或路由算法转向理解在何种条件下重用LoRAs才是真正有效的。通过理论分析以及合成的两跳推理和数学文字问题任务，我们探讨重用LoRAs是否能够实现真正的组合性泛化，还是仅仅反映了浅层模式匹配。评估两种数据无关方法——参数平均和动态适配器选择，我们发现，重用LoRAs往往未能逻辑地整合跨不相干微调数据集的知识，特别是在这些知识在预训练过程中欠代表的情况下。我们的实证结果，结合对LoRA有限表达能力的理论洞察，突显了重用它们在应对未見任务时的先决条件和约束条件，并对它作为真正无数据方法的可行性提出了质疑。我们呼吁暂停追求新的回收LoRAs的方法，并强调需要严谨的机制来指导未来基于适配器模型合并的学术研究和从业人员的实际系统设计。 

---
# ESRPCB: an Edge guided Super-Resolution model and Ensemble learning for tiny Printed Circuit Board Defect detection 

**Title (ZH)**: ESRPCB：边缘引导的超分辨率模型与集成学习在微小印制电路板缺陷检测中的应用 

**Authors**: Xiem HoangVan, Dang Bui Dinh, Thanh Nguyen Canh, Van-Truong Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13476)  

**Abstract**: Printed Circuit Boards (PCBs) are critical components in modern electronics, which require stringent quality control to ensure proper functionality. However, the detection of defects in small-scale PCBs images poses significant challenges as a result of the low resolution of the captured images, leading to potential confusion between defects and noise. To overcome these challenges, this paper proposes a novel framework, named ESRPCB (edgeguided super-resolution for PCBs defect detection), which combines edgeguided super-resolution with ensemble learning to enhance PCBs defect detection. The framework leverages the edge information to guide the EDSR (Enhanced Deep Super-Resolution) model with a novel ResCat (Residual Concatenation) structure, enabling it to reconstruct high-resolution images from small PCBs inputs. By incorporating edge features, the super-resolution process preserves critical structural details, ensuring that tiny defects remain distinguishable in the enhanced image. Following this, a multi-modal defect detection model employs ensemble learning to analyze the super-resolved 

**Abstract (ZH)**: 基于边缘引导超分辨率的PCB缺陷检测框架（ESRPCB） 

---
# Language Agents for Hypothesis-driven Clinical Decision Making with Reinforcement Learning 

**Title (ZH)**: 基于强化学习的假设驱动临床决策语言代理 

**Authors**: David Bani-Harouni, Chantal Pellegrini, Ege Özsoy, Matthias Keicher, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2506.13474)  

**Abstract**: Clinical decision-making is a dynamic, interactive, and cyclic process where doctors have to repeatedly decide on which clinical action to perform and consider newly uncovered information for diagnosis and treatment. Large Language Models (LLMs) have the potential to support clinicians in this process, however, most applications of LLMs in clinical decision support suffer from one of two limitations: Either they assume the unrealistic scenario of immediate availability of all patient information and do not model the interactive and iterative investigation process, or they restrict themselves to the limited "out-of-the-box" capabilities of large pre-trained models without performing task-specific training. In contrast to this, we propose to model clinical decision-making for diagnosis with a hypothesis-driven uncertainty-aware language agent, LA-CDM, that converges towards a diagnosis via repeatedly requesting and interpreting relevant tests. Using a hybrid training paradigm combining supervised and reinforcement learning, we train LA-CDM with three objectives targeting critical aspects of clinical decision-making: accurate hypothesis generation, hypothesis uncertainty estimation, and efficient decision-making. We evaluate our methodology on MIMIC-CDM, a real-world dataset covering four abdominal diseases containing various clinical tests and show the benefit of explicitly training clinical decision-making for increasing diagnostic performance and efficiency. 

**Abstract (ZH)**: 临床决策制定是一个动态、互动且循环的过程，医生需要反复决定执行何种临床操作，并考虑新发现的信息以进行诊断和治疗。大规模语言模型（LLMs）有潜力支持这一过程，然而，大多数将LLMs应用于临床决策支持的应用程序均受到两种限制之一：要么假设所有患者信息立即可用且不建模互动和迭代的调查过程，要么仅限制在大型预训练模型的基本功能而不进行特定任务的训练。与此不同，我们提出了一种假设驱动的不确定性感知语言代理LA-CDM，通过反复请求和解释相关测试来逐步达成诊断。通过结合监督学习和强化学习的混合训练范式，我们训练LA-CDM，旨在三个目标上优化临床决策制定的关键方面：准确的假设生成、假设不确定性估计以及高效的决策制定。我们在包含四种腹部疾病的临床测试数据的真实世界数据集MIMIC-CDM上评估了我们的方法，并展示了明确训练临床决策制定以提高诊断性能和效率的好处。 

---
# ROSAQ: Rotation-based Saliency-Aware Weight Quantization for Efficiently Compressing Large Language Models 

**Title (ZH)**: 基于旋转的注意力引导权重量化：高效压缩大型语言模型 

**Authors**: Junho Yoon, Geom Lee, Donghyeon Jeon, Inho Kang, Seung-Hoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2506.13472)  

**Abstract**: Quantization has been widely studied as an effective technique for reducing the memory requirement of large language models (LLMs), potentially improving the latency time as well. Utilizing the characteristic of rotational invariance of transformer, we propose the rotation-based saliency-aware weight quantization (ROSAQ), which identifies salient channels in the projection feature space, not in the original feature space, where the projected "principal" dimensions are naturally considered as "salient" features. The proposed ROSAQ consists of 1) PCA-based projection, which first performs principal component analysis (PCA) on a calibration set and transforms via the PCA projection, 2) Salient channel dentification, which selects dimensions corresponding to the K-largest eigenvalues as salient channels, and 3) Saliency-aware quantization with mixed-precision, which uses FP16 for salient dimensions and INT3/4 for other dimensions. Experiment results show that ROSAQ shows improvements over the baseline saliency-aware quantization on the original feature space and other existing quantization methods. With kernel fusion, ROSAQ presents about 2.3x speed up over FP16 implementation in generating 256 tokens with a batch size of 64. 

**Abstract (ZH)**: 基于旋转的注意重要性感知权重量化（ROSAQ）：一种在投影特征空间中识别重要通道的方法 

---
# A Two-stage Optimization Method for Wide-range Single-electron Quantum Magnetic Sensing 

**Title (ZH)**: 宽范围单电子量子磁传感的两阶段优化方法 

**Authors**: Shiqian Guo, Jianqing Liu, Thinh Le, Huaiyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13469)  

**Abstract**: Quantum magnetic sensing based on spin systems has emerged as a new paradigm for detecting ultra-weak magnetic fields with unprecedented sensitivity, revitalizing applications in navigation, geo-localization, biology, and beyond. At the heart of quantum magnetic sensing, from the protocol perspective, lies the design of optimal sensing parameters to manifest and then estimate the underlying signals of interest (SoI). Existing studies on this front mainly rely on adaptive algorithms based on black-box AI models or formula-driven principled searches. However, when the SoI spans a wide range and the quantum sensor has physical constraints, these methods may fail to converge efficiently or optimally, resulting in prolonged interrogation times and reduced sensing accuracy. In this work, we report the design of a new protocol using a two-stage optimization method. In the 1st Stage, a Bayesian neural network with a fixed set of sensing parameters is used to narrow the range of SoI. In the 2nd Stage, a federated reinforcement learning agent is designed to fine-tune the sensing parameters within a reduced search space. The proposed protocol is developed and evaluated in a challenging context of single-shot readout of an NV-center electron spin under a constrained total sensing time budget; and yet it achieves significant improvements in both accuracy and resource efficiency for wide-range D.C. magnetic field estimation compared to the state of the art. 

**Abstract (ZH)**: 基于自旋系统的量子磁感应已成为检测超弱磁场的一种新范式，具有前所未有的灵敏度，重新激活了导航、地理定位、生物学等领域中的应用。从协议角度而言，量子磁感应的核心在于设计最优的感应参数以体现并估计感兴趣的信号（SoI）。现有研究主要依赖于基于黑盒AI模型的自适应算法或基于公式的精原则搜索。然而，当SoI的范围广泛且量子传感器受到物理约束时，这些方法可能无法高效或最优地收敛，导致探测时间延长和探测精度降低。在本文中，我们报告了一种新的协议设计，该协议采用两阶段优化方法。在第一阶段，使用固定参数的贝叶斯神经网络来缩小SoI的范围。在第二阶段，设计了一个联邦强化学习代理来在缩减的搜索空间内细调感应参数。所提出协议在受限的总探测时间预算下实现单次读出NV中心电子自旋的挑战性环境中进行开发和评估；与现有技术相比，它在宽范围直流磁场估计的准确性和资源效率方面均取得了显著改进。 

---
# An Interdisciplinary Approach to Human-Centered Machine Translation 

**Title (ZH)**: 以人为本的跨学科机器翻译方法 

**Authors**: Marine Carpuat, Omri Asscher, Kalika Bali, Luisa Bentivogli, Frédéric Blain, Lynne Bowker, Monojit Choudhury, Hal Daumé III, Kevin Duh, Ge Gao, Alvin Grissom II, Marzena Karpinska, Elaine C. Khoong, William D. Lewis, André F. T. Martins, Mary Nurminen, Douglas W. Oard, Maja Popovic, Michel Simard, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2506.13468)  

**Abstract**: Machine Translation (MT) tools are widely used today, often in contexts where professional translators are not present. Despite progress in MT technology, a gap persists between system development and real-world usage, particularly for non-expert users who may struggle to assess translation reliability. This paper advocates for a human-centered approach to MT, emphasizing the alignment of system design with diverse communicative goals and contexts of use. We survey the literature in Translation Studies and Human-Computer Interaction to recontextualize MT evaluation and design to address the diverse real-world scenarios in which MT is used today. 

**Abstract (ZH)**: 机器翻译工具在缺乏专业译者的情况下广泛使用，尽管机器翻译技术取得了进展，但系统开发与实际应用之间仍存在差距，尤其是在非专家用户中，他们可能难以评估翻译的可靠性。本文倡导以人为本的机器翻译方法，强调系统设计应与多元的交流目标和使用情境相一致。我们回顾翻译研究和人机交互领域的文献，重新审视机器翻译的评估与设计，以应对当前机器翻译在各种实际场景中的应用需求。 

---
# Unveiling the Learning Mind of Language Models: A Cognitive Framework and Empirical Study 

**Title (ZH)**: 揭示语言模型的learning心智：一个认知框架及实证研究 

**Authors**: Zhengyu Hu, Jianxun Lian, Zheyuan Xiao, Seraphina Zhang, Tianfu Wang, Nicholas Jing Yuan, Xing Xie, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.13464)  

**Abstract**: Large language models (LLMs) have shown impressive capabilities across tasks such as mathematics, coding, and reasoning, yet their learning ability, which is crucial for adapting to dynamic environments and acquiring new knowledge, remains underexplored. In this work, we address this gap by introducing a framework inspired by cognitive psychology and education. Specifically, we decompose general learning ability into three distinct, complementary dimensions: Learning from Instructor (acquiring knowledge via explicit guidance), Learning from Concept (internalizing abstract structures and generalizing to new contexts), and Learning from Experience (adapting through accumulated exploration and feedback). We conduct a comprehensive empirical study across the three learning dimensions and identify several insightful findings, such as (i) interaction improves learning; (ii) conceptual understanding is scale-emergent and benefits larger models; and (iii) LLMs are effective few-shot learners but not many-shot learners. Based on our framework and empirical findings, we introduce a benchmark that provides a unified and realistic evaluation of LLMs' general learning abilities across three learning cognition dimensions. It enables diagnostic insights and supports evaluation and development of more adaptive and human-like models. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学、编码和推理等任务上展示了令人印象深刻的性能，然而它们的学习能力——这对于适应动态环境和获取新知识至关重要——仍然有待探索。本文通过引入受认知心理学和教育启发的框架来填补这一空白。具体而言，我们将一般学习能力分解为三个独立且互补的维度：从导师学习（通过明确指导获取知识）、从概念学习（内化抽象结构并在新情境中进行泛化）以及从经验学习（通过积累探索和反馈进行适应）。我们在这三个学习维度上进行了全面的实证研究，并得出了几项宝贵的发现，例如（i）互动能提高学习效果；（ii）概念理解在规模上是涌现的，并有利于更大的模型；以及（iii）LLMs 是有效的少样本学习者但不是多样本学习者。基于我们的框架和实证发现，我们引入了一个基准测试，能够统一且现实地评估LLMs在三个认知学习维度上的普遍学习能力，这有助于诊断洞察、支持模型的评估与开发，推动更具适应性和人性化的模型的构建。 

---
# Towards a Formal Specification for Self-organized Shape Formation in Swarm Robotics 

**Title (ZH)**: 面向自组织形状形成形式化规范的研究 

**Authors**: YR Darr, MA Niazi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13453)  

**Abstract**: The self-organization of robots for the formation of structures and shapes is a stimulating application of the swarm robotic system. It involves a large number of autonomous robots of heterogeneous behavior, coordination among them, and their interaction with the dynamic environment. This process of complex structure formation is considered a complex system, which needs to be modeled by using any modeling approach. Although the formal specification approach along with other formal methods has been used to model the behavior of robots in a swarm. However, to the best of our knowledge, the formal specification approach has not been used to model the self-organization process in swarm robotic systems for shape formation. In this paper, we use a formal specification approach to model the shape formation task of swarm robots. We use Z (Zed) language of formal specification, which is a state-based language, to model the states of the entities of the systems. We demonstrate the effectiveness of Z for the self-organized shape formation. The presented formal specification model gives the outlines for designing and implementing the swarm robotic system for the formation of complex shapes and structures. It also provides the foundation for modeling the complex shape formation process for swarm robotics using a multi-agent system in a simulation-based environment. Keywords: Swarm robotics, Self-organization, Formal specification, Complex systems 

**Abstract (ZH)**: 机器人自组织形成结构和形状是一种激动人心的群机器人系统应用。它涉及大量异质行为的自主机器人、它们之间的协调以及与动态环境的交互。这一复杂结构形成过程被视为一个复杂的系统，需要采用任何建模方法进行建模。尽管形式化规范方法及其他形式化方法已被用于建模群机器人行为，但据我们所知，形式化规范方法尚未被用于建模群机器人系统中形状形成过程的自组织过程。在本文中，我们使用形式化规范方法来建模群机器人的形状形成任务。我们采用状态基语言Z语言来建模系统实体的状态。本文展示了Z语言在自组织形状形成中的有效性。提出的正式规范模型为设计和实现用于形成复杂形状和结构的群机器人系统提供了指南。它还为基于多代理系统的模拟环境建模复杂的形状形成过程提供了基础。关键词：群机器人，自组织，形式化规范，复杂系统 

---
# A Neural Model for Word Repetition 

**Title (ZH)**: 一种词重复的神经模型 

**Authors**: Daniel Dager, Robin Sobczyk, Emmanuel Chemla, Yair Lakretz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13450)  

**Abstract**: It takes several years for the developing brain of a baby to fully master word repetition-the task of hearing a word and repeating it aloud. Repeating a new word, such as from a new language, can be a challenging task also for adults. Additionally, brain damage, such as from a stroke, may lead to systematic speech errors with specific characteristics dependent on the location of the brain damage. Cognitive sciences suggest a model with various components for the different processing stages involved in word repetition. While some studies have begun to localize the corresponding regions in the brain, the neural mechanisms and how exactly the brain performs word repetition remain largely unknown. We propose to bridge the gap between the cognitive model of word repetition and neural mechanisms in the human brain by modeling the task using deep neural networks. Neural models are fully observable, allowing us to study the detailed mechanisms in their various substructures and make comparisons with human behavior and, ultimately, the brain. Here, we make first steps in this direction by: (1) training a large set of models to simulate the word repetition task; (2) creating a battery of tests to probe the models for known effects from behavioral studies in humans, and (3) simulating brain damage through ablation studies, where we systematically remove neurons from the model, and repeat the behavioral study to examine the resulting speech errors in the "patient" model. Our results show that neural models can mimic several effects known from human research, but might diverge in other aspects, highlighting both the potential and the challenges for future research aimed at developing human-like neural models. 

**Abstract (ZH)**: 发展婴儿的大脑需要几年时间才能完全掌握词重复的任务——即听一个词并将其大声重复。对于成人来说，重复一个新的词，例如来自一种新语言的词，也可能是一项具有挑战性的任务。此外，脑损伤，例如中风，可能导致具有特定特征的系统性言语错误，这些特征取决于脑损伤的位置。认知科学提出了一个涉及词重复的不同处理阶段的各种成分的模型。尽管一些研究已经开始定位脑中的相应区域，但词重复的神经机制及其脑是如何执行这一任务的具体方式仍 largely unknown。我们建议通过使用深度神经网络建模词重复任务，以弥合词重复的认知模型与人类大脑的神经机制之间的差距。神经模型是完全可观察的，这使得我们可以研究其各个子结构中的详细机制，并将其与人类行为和最终的大脑进行比较。在这里，我们朝着这个目标迈出第一步，具体包括：(1) 训练大量模型以模拟词重复任务；(2) 创建一系列测试以探测模型中的已知行为研究效应；(3) 通过移除模型中的神经元进行消融研究，以模拟脑损伤，并重复行为研究以检查“患者”模型中的言语错误。我们的结果显示，神经模型可以模仿人类研究中已知的多种效应，但在其他方面可能会有所不同，这突显了未来旨在开发类似人类的神经模型的研究的潜力和挑战。 

---
# Simple is what you need for efficient and accurate medical image segmentation 

**Title (ZH)**: 简单即为高效准确医疗图像分割所需 

**Authors**: Xiang Yu, Yayan Chen, Guannan He, Qing Zeng, Yue Qin, Meiling Liang, Dandan Luo, Yimei Liao, Zeyu Ren, Cheng Kang, Delong Yang, Bocheng Liang, Bin Pu, Ying Yuan, Shengli Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13415)  

**Abstract**: While modern segmentation models often prioritize performance over practicality, we advocate a design philosophy prioritizing simplicity and efficiency, and attempted high performance segmentation model design. This paper presents SimpleUNet, a scalable ultra-lightweight medical image segmentation model with three key innovations: (1) A partial feature selection mechanism in skip connections for redundancy reduction while enhancing segmentation performance; (2) A fixed-width architecture that prevents exponential parameter growth across network stages; (3) An adaptive feature fusion module achieving enhanced representation with minimal computational overhead. With a record-breaking 16 KB parameter configuration, SimpleUNet outperforms LBUNet and other lightweight benchmarks across multiple public datasets. The 0.67 MB variant achieves superior efficiency (8.60 GFLOPs) and accuracy, attaining a mean DSC/IoU of 85.76%/75.60% on multi-center breast lesion datasets, surpassing both U-Net and TransUNet. Evaluations on skin lesion datasets (ISIC 2017/2018: mDice 84.86%/88.77%) and endoscopic polyp segmentation (KVASIR-SEG: 86.46%/76.48% mDice/mIoU) confirm consistent dominance over state-of-the-art models. This work demonstrates that extreme model compression need not compromise performance, providing new insights for efficient and accurate medical image segmentation. Codes can be found at this https URL. 

**Abstract (ZH)**: 现代分割模型往往重视性能而忽视实用性，我们提倡一种以简洁和高效为优先的设计哲学，并尝试设计高性能的分割模型。本文提出了SimpleUNet，一种具有三大创新的可扩展极轻量级医学图像分割模型：(1) 跳链接中的部分特征选择机制以减少冗余并增强分割性能；(2) 固定宽度架构以防止网络各层参数数量指数级增长；(3) 可适应特征融合模块以实现增强表示并最小化计算开销。通过创纪录的16 KB参数配置，SimpleUNet在多个公开数据集上超越LBUNet和其他轻量级基准模型，展现出优于U-Net和TransUNet的效率和准确性。在皮肤病变数据集（ISIC 2017/2018：mDice 84.86%/88.77%）和内镜息肉分割数据集（KVASIR-SEG：86.46%/76.48% mDice/mIoU）上的评估进一步证实了其在最先进的模型中的持续领先地位。本工作表明，极端模型压缩不必牺牲性能，为高效准确的医学图像分割提供了新的见解。代码可在以下链接找到。 

---
# CALM: Consensus-Aware Localized Merging for Multi-Task Learning 

**Title (ZH)**: CALM：共识感知的局部合并多任务学习 

**Authors**: Kunda Yan, Min Zhang, Sen Cui, Zikun Qu, Bo Jiang, Feng Liu, Changshui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13406)  

**Abstract**: Model merging aims to integrate the strengths of multiple fine-tuned models into a unified model while preserving task-specific capabilities. Existing methods, represented by task arithmetic, are typically classified into global- and local-aware methods. However, global-aware methods inevitably cause parameter interference, while local-aware methods struggle to maintain the effectiveness of task-specific details in the merged model. To address these limitations, we propose a Consensus-Aware Localized Merging (CALM) method which incorporates localized information aligned with global task consensus, ensuring its effectiveness post-merging. CALM consists of three key components: (1) class-balanced entropy minimization sampling, providing a more flexible and reliable way to leverage unsupervised data; (2) an efficient-aware framework, selecting a small set of tasks for sequential merging with high scalability; (3) a consensus-aware mask optimization, aligning localized binary masks with global task consensus and merging them conflict-free. Experiments demonstrate the superiority and robustness of our CALM, significantly outperforming existing methods and achieving performance close to traditional MTL. 

**Abstract (ZH)**: 基于共识的局部化模型合并方法（CALM）：一种整合局部信息与全局任务共识的统一模型方法 

---
# Mitigating loss of variance in ensemble data assimilation: machine learning-based and distance-free localizations for better covariance estimation 

**Title (ZH)**: 基于机器学习和无距离度量的局部化方法减轻集成数据同化中协方差估计的方差损失 

**Authors**: Vinicius L. S. Silva, Gabriel S. Seabra, Alexandre A. Emerick  

**Link**: [PDF](https://arxiv.org/pdf/2506.13362)  

**Abstract**: We propose two new methods based/inspired by machine learning for tabular data and distance-free localization to enhance the covariance estimations in an ensemble data assimilation. The main goal is to enhance the data assimilation results by mitigating loss of variance due to sampling errors. We also analyze the suitability of several machine learning models and the balance between accuracy and computational cost of the covariance estimations. We introduce two distance-free localization techniques leveraging machine learning methods specifically tailored for tabular data. The methods are integrated into the Ensemble Smoother with Multiple Data Assimilation (ES-MDA) framework. The results show that the proposed localizations improve covariance accuracy and enhance data assimilation and uncertainty quantification results. We observe reduced variance loss for the input variables using the proposed methods. Furthermore, we compare several machine learning models, assessing their suitability for the problem in terms of computational cost, and quality of the covariance estimation and data match. The influence of ensemble size is also investigated, providing insights into balancing accuracy and computational efficiency. Our findings demonstrate that certain machine learning models are more suitable for this problem. This study introduces two novel methods that mitigate variance loss for model parameters in ensemble-based data assimilation, offering practical solutions that are easy to implement and do not require any additional numerical simulation or hyperparameter tuning. 

**Abstract (ZH)**: 基于机器学习的两种新方法增强表格数据和距离无关局部化的协方差估计在集合数据同化中的应用 

---
# StoryBench: A Dynamic Benchmark for Evaluating Long-Term Memory with Multi Turns 

**Title (ZH)**: StoryBench：一个动态基准，用于评估多轮对话中的长期记忆能力 

**Authors**: Luanbo Wan, Weizhi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.13356)  

**Abstract**: Long-term memory (LTM) is essential for large language models (LLMs) to achieve autonomous intelligence in complex, evolving environments. Despite increasing efforts in memory-augmented and retrieval-based architectures, there remains a lack of standardized benchmarks to systematically evaluate LLMs' long-term memory abilities. Existing benchmarks still face challenges in evaluating knowledge retention and dynamic sequential reasoning, and in their own flexibility, all of which limit their effectiveness in assessing models' LTM capabilities. To address these gaps, we propose a novel benchmark framework based on interactive fiction games, featuring dynamically branching storylines with complex reasoning structures. These structures simulate real-world scenarios by requiring LLMs to navigate hierarchical decision trees, where each choice triggers cascading dependencies across multi-turn interactions. Our benchmark emphasizes two distinct settings to test reasoning complexity: one with immediate feedback upon incorrect decisions, and the other requiring models to independently trace back and revise earlier choices after failure. As part of this benchmark, we also construct a new dataset designed to test LLMs' LTM within narrative-driven environments. We further validate the effectiveness of our approach through detailed experiments. Experimental results demonstrate the benchmark's ability to robustly and reliably assess LTM in LLMs. 

**Abstract (ZH)**: 长时记忆(LTM)对于大型语言模型(LLMs)在复杂演变环境中实现自主智能至关重要。尽管在记忆增强和检索基础架构方面付出了越来越多的努力，但仍缺乏标准化基准来系统评估LLMs的LTM能力。现有的基准在评估知识保留和动态序列推理能力方面仍面临挑战，并且在灵活性方面也存在局限性，这些都限制了它们对模型LTM能力的评估效果。为了弥补这些不足，我们提出了一种基于互动虚构游戏的新基准框架，该框架包含动态分支故事情节和复杂的推理结构。这些结构通过要求LLMs在多回合交互中导航分层决策树来模拟现实世界场景，其中每个选择触发跨多轮交互的连锁依赖关系。我们的基准框架强调两个不同的设置来测试推理复杂性：一个是在错误决策后即时反馈的设置，另一个是要求模型在失败后独立追溯并修正早期选择的设置。作为基准的一部分，我们还构建了一个新的数据集，用于测试LLMs在叙述驱动环境中的LTM能力。我们还通过详细的实验证明了我们方法的有效性。实验结果表明，该基准能够稳健且可靠地评估LLMs的LTM能力。 

---
# Direct Reasoning Optimization: LLMs Can Reward And Refine Their Own Reasoning for Open-Ended Tasks 

**Title (ZH)**: 直接推理优化：大语言模型可以奖励并精炼其自身的推理以应对开放任务 

**Authors**: Yifei Xu, Tusher Chakraborty, Srinagesh Sharma, Leonardo Nunes, Emre Kıcıman, Songwu Lu, Ranveer Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2506.13351)  

**Abstract**: Recent advances in Large Language Models (LLMs) have showcased impressive reasoning abilities in structured tasks like mathematics and programming, largely driven by Reinforcement Learning with Verifiable Rewards (RLVR), which uses outcome-based signals that are scalable, effective, and robust against reward hacking. However, applying similar techniques to open-ended long-form reasoning tasks remains challenging due to the absence of generic, verifiable reward signals. To address this, we propose Direct Reasoning Optimization (DRO), a reinforcement learning framework for fine-tuning LLMs on open-ended, particularly long-form, reasoning tasks, guided by a new reward signal: the Reasoning Reflection Reward (R3). At its core, R3 selectively identifies and emphasizes key tokens in the reference outcome that reflect the influence of the model's preceding chain-of-thought reasoning, thereby capturing the consistency between reasoning and reference outcome at a fine-grained level. Crucially, R3 is computed internally using the same model being optimized, enabling a fully self-contained training setup. Additionally, we introduce a dynamic data filtering strategy based on R3 for open-ended reasoning tasks, reducing cost while improving downstream performance. We evaluate DRO on two diverse datasets -- ParaRev, a long-form paragraph revision task, and FinQA, a math-oriented QA benchmark -- and show that it consistently outperforms strong baselines while remaining broadly applicable across both open-ended and structured domains. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）在数学和编程等结构化任务中的推理能力展示了令人印象深刻的成就，这主要得益于基于可验证奖励的强化学习（RLVR）技术，该技术利用了可扩展、有效且对奖励欺骗具有鲁棒性的结果导向信号。然而，将类似技术应用于开放式的长篇推理任务仍然具有挑战性，原因是没有通用且可验证的奖励信号。为了解决这一问题，我们提出了一种直接推理优化（DRO），这是一种用于在开放式的尤其是长篇推理任务中微调LLMs的强化学习框架，由一个新的奖励信号——推理反思奖励（R3）指导。R3的核心在于它能够选择性地识别并强调参考结果中反映模型前序推理影响的关键标记，从而在精细粒度上捕捉推理与参考结果之间的一致性。至关重要的是，R3是在优化模型本身内部计算得出的，因此可以实现完全自包含的训练设置。此外，我们还引入了基于R3的动态数据过滤策略，以降低开放推理任务的成本并提高下游性能。我们在两个不同的数据集——ParaRev（长篇段落修订任务）和FinQA（数学导向的问答基准）上评估了DRO，并展示了它在多个开放性和结构化领域中都能持续优于强大的基线模型。 

---
# LapDDPM: A Conditional Graph Diffusion Model for scRNA-seq Generation with Spectral Adversarial Perturbations 

**Title (ZH)**: LapDDPM：基于谱对抗扰动的条件图扩散模型ストレスRNA测序生成 

**Authors**: Lorenzo Bini, Stephane Marchand-Maillet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13344)  

**Abstract**: Generating high-fidelity and biologically plausible synthetic single-cell RNA sequencing (scRNA-seq) data, especially with conditional control, is challenging due to its high dimensionality, sparsity, and complex biological variations. Existing generative models often struggle to capture these unique characteristics and ensure robustness to structural noise in cellular networks. We introduce LapDDPM, a novel conditional Graph Diffusion Probabilistic Model for robust and high-fidelity scRNA-seq generation. LapDDPM uniquely integrates graph-based representations with a score-based diffusion model, enhanced by a novel spectral adversarial perturbation mechanism on graph edge weights. Our contributions are threefold: we leverage Laplacian Positional Encodings (LPEs) to enrich the latent space with crucial cellular relationship information; we develop a conditional score-based diffusion model for effective learning and generation from complex scRNA-seq distributions; and we employ a unique spectral adversarial training scheme on graph edge weights, boosting robustness against structural variations. Extensive experiments on diverse scRNA-seq datasets demonstrate LapDDPM's superior performance, achieving high fidelity and generating biologically-plausible, cell-type-specific samples. LapDDPM sets a new benchmark for conditional scRNA-seq data generation, offering a robust tool for various downstream biological applications. 

**Abstract (ZH)**: 基于图扩散的概率模型LapDDPM：用于高保真和生物合理单细胞RNA测序数据生成的新型条件化方法 

---
# Tady: A Neural Disassembler without Structural Constraint Violations 

**Title (ZH)**: Tady：一种无结构约束违规的神经反汇编器 

**Authors**: Siliang Qin, Fengrui Yang, Hao Wang, Bolun Zhang, Zeyu Gao, Chao Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13323)  

**Abstract**: Disassembly is a crucial yet challenging step in binary analysis. While emerging neural disassemblers show promise for efficiency and accuracy, they frequently generate outputs violating fundamental structural constraints, which significantly compromise their practical usability. To address this critical problem, we regularize the disassembly solution space by formalizing and applying key structural constraints based on post-dominance relations. This approach systematically detects widespread errors in existing neural disassemblers' outputs. These errors often originate from models' limited context modeling and instruction-level decoding that neglect global structural integrity. We introduce Tady, a novel neural disassembler featuring an improved model architecture and a dedicated post-processing algorithm, specifically engineered to address these deficiencies. Comprehensive evaluations on diverse binaries demonstrate that Tady effectively eliminates structural constraint violations and functions with high efficiency, while maintaining instruction-level accuracy. 

**Abstract (ZH)**: 二进制分析中的反汇编是一个关键但具挑战性的步骤。虽然新兴的神经网络反汇编器在效率和准确性方面展现了潜力，但它们经常生成违反基本结构约束的输出，这极大地影响了其实用性。为解决这一问题，我们通过形式化并应用基于后支配关系的关键结构约束来规范反汇编解空间。这种方法系统地检测了现有神经网络反汇编器输出中的普遍错误，这些错误通常源于模型有限的上下文建模和忽视全局结构完整性的指令级解码。我们引入了Tady，这是一种新型神经网络反汇编器，配备改进的模型架构和专用后处理算法，特别设计以解决这些问题。在多种二进制代码上的全面评估表明，Tady有效地消除了结构约束违反情况，并以高效率运行，同时保持指令级准确性。 

---
# Active Multimodal Distillation for Few-shot Action Recognition 

**Title (ZH)**: 面向Few-shot动作识别的主动多模态蒸馏 

**Authors**: Weijia Feng, Yichen Zhu, Ruojia Zhang, Chenyang Wang, Fei Ma, Xiaobao Wang, Xiaobai Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.13322)  

**Abstract**: Owing to its rapid progress and broad application prospects, few-shot action recognition has attracted considerable interest. However, current methods are predominantly based on limited single-modal data, which does not fully exploit the potential of multimodal information. This paper presents a novel framework that actively identifies reliable modalities for each sample using task-specific contextual cues, thus significantly improving recognition performance. Our framework integrates an Active Sample Inference (ASI) module, which utilizes active inference to predict reliable modalities based on posterior distributions and subsequently organizes them accordingly. Unlike reinforcement learning, active inference replaces rewards with evidence-based preferences, making more stable predictions. Additionally, we introduce an active mutual distillation module that enhances the representation learning of less reliable modalities by transferring knowledge from more reliable ones. Adaptive multimodal inference is employed during the meta-test to assign higher weights to reliable modalities. Extensive experiments across multiple benchmarks demonstrate that our method significantly outperforms existing approaches. 

**Abstract (ZH)**: 由于其快速进步和广泛的应用前景，少样本动作识别引起了 considerable attention。然而，当前方法主要基于有限的单模数据，未能充分挖掘多模信息的潜力。本文提出了一种新颖的框架，该框架能够利用任务特定的上下文线索主动识别每个样本的可靠模态，从而显著提高识别性能。我们的框架整合了一个主动样本推理（ASI）模块，该模块利用主动推理根据后验分布预测可靠模态并相应地进行组织。与强化学习不同，主动推理使用证据为基础的偏好替代奖励，从而做出更为稳定的预测。此外，我们引入了一个主动互信息蒸馏模块，通过从更可靠模态转移知识来增强不可靠模态的表示学习。在元测试过程中采用自适应多模态推理，在分配权重时给予可靠模态更高的权重。在多个基准上的广泛实验表明，我们的方法显著优于现有方法。 

---
# Vine Copulas as Differentiable Computational Graphs 

**Title (ZH)**: Vine Copulas 作为可微计算图 

**Authors**: Tuoyuan Cheng, Thibault Vatter, Thomas Nagler, Kan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13318)  

**Abstract**: Vine copulas are sophisticated models for multivariate distributions and are increasingly used in machine learning. To facilitate their integration into modern ML pipelines, we introduce the vine computational graph, a DAG that abstracts the multilevel vine structure and associated computations. On this foundation, we devise new algorithms for conditional sampling, efficient sampling-order scheduling, and constructing vine structures for customized conditioning variables. We implement these ideas in torchvinecopulib, a GPU-accelerated Python library built upon PyTorch, delivering improved scalability for fitting, sampling, and density evaluation. Our experiments illustrate how gradient flowing through the vine can improve Vine Copula Autoencoders and that incorporating vines for uncertainty quantification in deep learning can outperform MC-dropout, deep ensembles, and Bayesian Neural Networks in sharpness, calibration, and runtime. By recasting vine copula models as computational graphs, our work connects classical dependence modeling with modern deep-learning toolchains and facilitates the integration of state-of-the-art copula methods in modern machine learning pipelines. 

**Abstract (ZH)**: Vine copulas是多变量分布的复杂模型，在机器学习中应用日益广泛。为便于其集成到现代ML流水线中，我们引入了vine计算图，这是一种抽象多级Vine结构及其相关计算的有向无环图。在此基础上，我们开发了新的条件采样算法、高效的采样顺序调度算法以及用于自定义条件变量的Vine结构构建算法。我们在PyTorch之上构建的GPU加速Python库torchvinecopulib中实现了这些想法，提供了更好的可扩展性，用于模型拟合、采样和密度评估。实验表明，通过Vine传播梯度可以改进Vine Copula自编码器，并且将Vine纳入深度学习中的不确定性量化可以优于MC-dropout、深集成和贝叶斯神经网络，在精确性、校准性和运行时间方面。将vine copula模型重新表述为计算图，我们的工作将经典依赖性建模与现代深度学习工具链连接起来，并促进了先进copula方法在现代机器学习流水线中的集成。 

---
# Large Language Models as 'Hidden Persuaders': Fake Product Reviews are Indistinguishable to Humans and Machines 

**Title (ZH)**: 大型语言模型作为“隐形劝说者”：虚假产品评价对人类和机器无法区分 

**Authors**: Weiyao Meng, John Harvey, James Goulding, Chris James Carter, Evgeniya Lukinova, Andrew Smith, Paul Frobisher, Mina Forrest, Georgiana Nica-Avram  

**Link**: [PDF](https://arxiv.org/pdf/2506.13313)  

**Abstract**: Reading and evaluating product reviews is central to how most people decide what to buy and consume online. However, the recent emergence of Large Language Models and Generative Artificial Intelligence now means writing fraudulent or fake reviews is potentially easier than ever. Through three studies we demonstrate that (1) humans are no longer able to distinguish between real and fake product reviews generated by machines, averaging only 50.8% accuracy overall - essentially the same that would be expected by chance alone; (2) that LLMs are likewise unable to distinguish between fake and real reviews and perform equivalently bad or even worse than humans; and (3) that humans and LLMs pursue different strategies for evaluating authenticity which lead to equivalently bad accuracy, but different precision, recall and F1 scores - indicating they perform worse at different aspects of judgment. The results reveal that review systems everywhere are now susceptible to mechanised fraud if they do not depend on trustworthy purchase verification to guarantee the authenticity of reviewers. Furthermore, the results provide insight into the consumer psychology of how humans judge authenticity, demonstrating there is an inherent 'scepticism bias' towards positive reviews and a special vulnerability to misjudge the authenticity of fake negative reviews. Additionally, results provide a first insight into the 'machine psychology' of judging fake reviews, revealing that the strategies LLMs take to evaluate authenticity radically differ from humans, in ways that are equally wrong in terms of accuracy, but different in their misjudgments. 

**Abstract (ZH)**: 阅读和评估产品评论是大多数人在网上购买和消费决策的核心。然而，近期大型语言模型和生成型人工智能的出现意味着撰写虚假或伪造评论可能比以往任何时候都更容易。通过三项研究，我们证明了以下几点：（1）人类无法区分由机器生成的真实和虚假产品评论，整体准确率仅为50.8%，几乎与随机猜测相同；（2）大型语言模型同样无法区分真实和虚假评论，其表现与人类相当甚至更差；（3）人类和大型语言模型在评估真实性方面采用不同的策略，导致准确率相当，但_precision、recall和F1分数不同，表明他们在判断的不同方面表现较差。研究结果揭示，如果评论系统不依赖于可靠的购买验证来确保评论者的真实性，这些系统现在都容易受到机械化欺诈的影响。此外，研究结果还揭示了消费者的判断心理，表明人类对正面评论存在一种根深蒂固的怀疑偏见，并且特别容易误判虚假负面评论的真实性。另外，研究还提供了关于“机器判断心理”的初步洞见，揭示了大型语言模型评估真实性所采取的策略与人类截然不同，尽管在准确率上同样错误，但在误判方面却有不同的表现。 

---
# Quantitative Comparison of Fine-Tuning Techniques for Pretrained Latent Diffusion Models in the Generation of Unseen SAR Image Concepts 

**Title (ZH)**: 预训练潜在扩散模型在生成未见SAR图像概念中的微调技术定量比较 

**Authors**: Solène Debuysère, Nicolas Trouvé, Nathan Letheule, Olivier Lévêque, Elise Colin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13307)  

**Abstract**: This work investigates the adaptation of large pre-trained latent diffusion models to a radically new imaging domain: Synthetic Aperture Radar (SAR). While these generative models, originally trained on natural images, demonstrate impressive capabilities in text-to-image synthesis, they are not natively adapted to represent SAR data, which involves different physics, statistical distributions, and visual characteristics. Using a sizeable SAR dataset (on the order of 100,000 to 1 million images), we address the fundamental question of fine-tuning such models for this unseen modality. We explore and compare multiple fine-tuning strategies, including full model fine-tuning and parameter-efficient approaches like Low-Rank Adaptation (LoRA), focusing separately on the UNet diffusion backbone and the text encoder components. To evaluate generative quality, we combine several metrics: statistical distance from real SAR distributions, textural similarity via GLCM descriptors, and semantic alignment assessed with a CLIP model fine-tuned on SAR data. Our results show that a hybrid tuning strategy yields the best performance: full fine-tuning of the UNet is better at capturing low-level SAR-specific patterns, while LoRA-based partial tuning of the text encoder, combined with embedding learning of the <SAR> token, suffices to preserve prompt alignment. This work provides a methodical strategy for adapting foundation models to unconventional imaging modalities beyond natural image domains. 

**Abstract (ZH)**: 本工作探究了将大规模预训练隐空间扩散模型适应于一种全新的成像领域：合成孔径雷达（SAR）图像。尽管这些生成模型原本在自然图像上进行训练，显示出了在文本到图像合成方面的卓越能力，但它们并不天生适合表示SAR数据，后者涉及不同的物理原理、统计分布和视觉特征。利用数量级在10万到100万张SAR图像的大规模SAR数据集，我们解决了如何将此类模型调整应用于这种未曾见过的成像模态的基本问题。我们探索并比较了多种调整策略，包括完整的模型调整和高效参数调整方法（如LoRA低秩适应），分别对UNet扩散骨干网络和文本编码器组件进行了研究。为了评估生成质量，我们结合了多种指标：与真实SAR分布的统计距离、基于GLCM描述符的纹理相似性，以及使用SAR数据微调的CLIP模型进行语义对齐评估。研究结果表明，混合调整策略表现最佳：完整的UNet调整在捕捉低级SAR特定模式方面效果更好，而基于LoRA的部分文本编码器调整结合<SAR>令牌的嵌入学习足以保持提示对齐。本工作提供了一种方法论策略，用于将基础模型适应于超越自然图像领域的非传统成像模态。 

---
# Seewo's Submission to MLC-SLM: Lessons learned from Speech Reasoning Language Models 

**Title (ZH)**: Seewo向MLC-SLM的提交：来自语音推理语言模型的教训 

**Authors**: Bo Li, Chengben Xu, Wufeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13300)  

**Abstract**: This paper presents Seewo's systems for both tracks of the Multilingual Conversational Speech Language Model Challenge (MLC-SLM), addressing automatic speech recognition (ASR) and speaker diarization with ASR (SD-ASR). We introduce a multi-stage training pipeline that explicitly enhances reasoning and self-correction in speech language models for ASR. Our approach combines curriculum learning for progressive capability acquisition, Chain-of-Thought data augmentation to foster intermediate reflection, and Reinforcement Learning with Verifiable Rewards (RLVR) to further refine self-correction through reward-driven optimization. This approach achieves substantial improvements over the official challenge baselines. On the evaluation set, our best system attains a WER/CER of 11.57% for Track 1 and a tcpWER/tcpCER of 17.67% for Track 2. Comprehensive ablation studies demonstrate the effectiveness of each component under challenge constraints. 

**Abstract (ZH)**: 本文介绍了Seewo在多语言对话语音语言模型挑战（MLC-SLM）两个赛道上的系统，涵盖自动语音识别（ASR）和带有ASR的说话人聚类（SD-ASR）。我们介绍了一种多阶段训练管道，明确增强了语音语言模型在ASR中的推理和自我修正能力。我们的方法结合了课程学习以实现逐步能力获取、使用Chain-of-Thought数据增强以培养中间反思，并采用可验证奖励的强化学习（RLVR）进一步通过奖励驱动优化来细化自我修正，该方法在官方挑战基准上取得了显著改进。在评估集上，我们最佳系统在Track 1的WER/CER达到11.57%，在Track 2的tcpWER/tcpCER达到17.67%。全面的消融研究证明了在挑战约束下每个组件的有效性。 

---
# Fair Generation without Unfair Distortions: Debiasing Text-to-Image Generation with Entanglement-Free Attention 

**Title (ZH)**: 公平生成而不引入不公平扭曲：基于拆分注意力的文本到图像生成去偏差化 

**Authors**: Jeonghoon Park, Juyoung Lee, Chaeyeon Chung, Jaeseong Lee, Jaegul Choo, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13298)  

**Abstract**: Recent advancements in diffusion-based text-to-image (T2I) models have enabled the generation of high-quality and photorealistic images from text descriptions. However, they often exhibit societal biases related to gender, race, and socioeconomic status, thereby reinforcing harmful stereotypes and shaping public perception in unintended ways. While existing bias mitigation methods demonstrate effectiveness, they often encounter attribute entanglement, where adjustments to attributes relevant to the bias (i.e., target attributes) unintentionally alter attributes unassociated with the bias (i.e., non-target attributes), causing undesirable distribution shifts. To address this challenge, we introduce Entanglement-Free Attention (EFA), a method that accurately incorporates target attributes (e.g., White, Black, Asian, and Indian) while preserving non-target attributes (e.g., background details) during bias mitigation. At inference time, EFA randomly samples a target attribute with equal probability and adjusts the cross-attention in selected layers to incorporate the sampled attribute, achieving a fair distribution of target attributes. Extensive experiments demonstrate that EFA outperforms existing methods in mitigating bias while preserving non-target attributes, thereby maintaining the output distribution and generation capability of the original model. 

**Abstract (ZH)**: 基于扩散文本到图像模型中纠缠属性的注意力解脱方法：缓解社会偏见的同时保持非目标属性 

---
# Automatic Multi-View X-Ray/CT Registration Using Bone Substructure Contours 

**Title (ZH)**: 使用骨亚结构轮廓的自动多视图X射线/CT配准 

**Authors**: Roman Flepp, Leon Nissen, Bastian Sigrist, Arend Nieuwland, Nicola Cavalcanti, Philipp Fürnstahl, Thomas Dreher, Lilian Calvet  

**Link**: [PDF](https://arxiv.org/pdf/2506.13292)  

**Abstract**: Purpose: Accurate intraoperative X-ray/CT registration is essential for surgical navigation in orthopedic procedures. However, existing methods struggle with consistently achieving sub-millimeter accuracy, robustness under broad initial pose estimates or need manual key-point annotations. This work aims to address these challenges by proposing a novel multi-view X-ray/CT registration method for intraoperative bone registration. Methods: The proposed registration method consists of a multi-view, contour-based iterative closest point (ICP) optimization. Unlike previous methods, which attempt to match bone contours across the entire silhouette in both imaging modalities, we focus on matching specific subcategories of contours corresponding to bone substructures. This leads to reduced ambiguity in the ICP matches, resulting in a more robust and accurate registration solution. This approach requires only two X-ray images and operates fully automatically. Additionally, we contribute a dataset of 5 cadaveric specimens, including real X-ray images, X-ray image poses and the corresponding CT scans. Results: The proposed registration method is evaluated on real X-ray images using mean reprojection error (mRPD). The method consistently achieves sub-millimeter accuracy with a mRPD 0.67mm compared to 5.35mm by a commercial solution requiring manual intervention. Furthermore, the method offers improved practical applicability, being fully automatic. Conclusion: Our method offers a practical, accurate, and efficient solution for multi-view X-ray/CT registration in orthopedic surgeries, which can be easily combined with tracking systems. By improving registration accuracy and minimizing manual intervention, it enhances intraoperative navigation, contributing to more accurate and effective surgical outcomes in computer-assisted surgery (CAS). 

**Abstract (ZH)**: 目的：准确的术中X射线/CT配准对于骨科手术导航至关重要。然而，现有方法在一致实现亚毫米级精度、在广泛初始姿态估计下的鲁棒性或需要手动关键点标注方面存在挑战。本项工作通过提出一种新型的多视角X射线/CT配准方法来解决这些挑战，以实现术中骨骼配准。方法：所提出的方法包括一个多视角的基于轮廓的迭代最近点（ICP）优化。与之前的方法不同，这些方法试图在两种成像模态中将骨骼轮廓整体匹配到整个轮廓 silhouette 上，我们专注于匹配对应于骨骼亚结构的特定轮廓子类。这减少了ICP匹配的歧义性，从而得到更稳健和准确的配准解决方案。该方法仅需要两张X射线图像，并且完全自动运行。此外，我们还贡献了一个包含5具尸体标本的数据库，其中包括真实的X射线图像、X射线图像姿态和对应的CT扫描。结果：所提出的方法在真实X射线图像上使用均方重构误差（mRPD）进行了评估。与需要手动干预的商用解决方案相比，该方法实现了亚毫米级精度，其mRPD为0.67毫米，而商用解决方案的mRPD为5.35毫米。此外，该方法提供了更好的实际适用性，完全自动运行。结论：本方法提供了一种实用、准确且高效的多视角X射线/CT配准解决方案，适用于骨科手术，并可以 easily 与跟踪系统结合使用。通过提高配准精度并减少手动干预，它增强了术中导航，促进了计算机辅助手术（CAS）中更准确和有效的手术结果。 

---
# AceReason-Nemotron 1.1: Advancing Math and Code Reasoning through SFT and RL Synergy 

**Title (ZH)**: AceReason-Nemotron 1.1: 通过SFT和RL协同促进数学和代码推理的进步 

**Authors**: Zihan Liu, Zhuolin Yang, Yang Chen, Chankyu Lee, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping  

**Link**: [PDF](https://arxiv.org/pdf/2506.13284)  

**Abstract**: In this work, we investigate the synergy between supervised fine-tuning (SFT) and reinforcement learning (RL) in developing strong reasoning models. We begin by curating the SFT training data through two scaling strategies: increasing the number of collected prompts and the number of generated responses per prompt. Both approaches yield notable improvements in reasoning performance, with scaling the number of prompts resulting in more substantial gains. We then explore the following questions regarding the synergy between SFT and RL: (i) Does a stronger SFT model consistently lead to better final performance after large-scale RL training? (ii) How can we determine an appropriate sampling temperature during RL training to effectively balance exploration and exploitation for a given SFT initialization? Our findings suggest that (i) holds true, provided effective RL training is conducted, particularly when the sampling temperature is carefully chosen to maintain the temperature-adjusted entropy around 0.3, a setting that strikes a good balance between exploration and exploitation. Notably, the performance gap between initial SFT models narrows significantly throughout the RL process. Leveraging a strong SFT foundation and insights into the synergistic interplay between SFT and RL, our AceReason-Nemotron-1.1 7B model significantly outperforms AceReason-Nemotron-1.0 and achieves new state-of-the-art performance among Qwen2.5-7B-based reasoning models on challenging math and code benchmarks, thereby demonstrating the effectiveness of our post-training recipe. We release the model and data at: this https URL 

**Abstract (ZH)**: 在这种工作中，我们探讨了监督微调（SFT）与强化学习（RL）在开发强大推理模型方面的协同作用。我们通过两种缩放策略来精炼SFT训练数据：增加收集的提示数量和每个提示生成的响应数量。这两种方法都显著提升了推理性能，其中增加提示数量的方法带来了更大的提升。随后，我们探讨了SFT与RL之间协同作用的以下问题：（i）是否更强的SFT模型在大规模RL训练后始终能取得更好的最终性能？（ii）在RL训练过程中，如何确定合适的采样温度以有效地平衡给定SFT初始化条件下的探索与利用？我们的研究结果表明，（i）在进行有效的RL训练时成立，特别是在选择采样温度以保持温度调整后的熵约为0.3的情况下，这种设置可以在探索与利用之间取得良好的平衡。值得注意的是，RL过程中初始SFT模型之间的性能差距显著缩小。利用强大的SFT基础和对SFT与RL之间协同作用的深入了解，我们的AceReason-Nemotron-1.1 7B模型显著优于AceReason-Nemotron-1.0，并在基于Qwen2.5-7B的推理模型中以具有挑战性的数学和代码基准测试实现了新的最佳性能，从而证明了我们后续训练方案的有效性。我们将模型和数据发布在：this https URL。 

---
# SeqPE: Transformer with Sequential Position Encoding 

**Title (ZH)**: SeqPE: 带有序列位置编码的变压器 

**Authors**: Huyang Li, Yahui Liu, Hongyu Sun, Deng Cai, Leyang Cui, Wei Bi, Peilin Zhao, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13277)  

**Abstract**: Since self-attention layers in Transformers are permutation invariant by design, positional encodings must be explicitly incorporated to enable spatial understanding. However, fixed-size lookup tables used in traditional learnable position embeddings (PEs) limit extrapolation capabilities beyond pre-trained sequence lengths. Expert-designed methods such as ALiBi and RoPE, mitigate this limitation but demand extensive modifications for adapting to new modalities, underscoring fundamental challenges in adaptability and scalability. In this work, we present SeqPE, a unified and fully learnable position encoding framework that represents each $n$-dimensional position index as a symbolic sequence and employs a lightweight sequential position encoder to learn their embeddings in an end-to-end manner. To regularize SeqPE's embedding space, we introduce two complementary objectives: a contrastive objective that aligns embedding distances with a predefined position-distance function, and a knowledge distillation loss that anchors out-of-distribution position embeddings to in-distribution teacher representations, further enhancing extrapolation performance. Experiments across language modeling, long-context question answering, and 2D image classification demonstrate that SeqPE not only surpasses strong baselines in perplexity, exact match (EM), and accuracy--particularly under context length extrapolation--but also enables seamless generalization to multi-dimensional inputs without requiring manual architectural redesign. We release our code, data, and checkpoints at this https URL. 

**Abstract (ZH)**: 自注意力层在Transformer中设计上具有置换不变性，因此需要显式地引入位置编码以实现空间理解。然而，传统可学习位置编码（PE）中固定大小的查找表限制了其超出预训练序列长度的外推能力。专家设计的方法如ALiBi和RoPE减轻了这一限制，但需要对新模态进行大量的修改，突显了适应性和可扩展性的基本挑战。在本文中，我们提出SeqPE，这是一个统一的全可学习位置编码框架，将每个n维位置索引表示为符号序列，并采用一个轻量级的顺序位置编码器以端到端的方式学习其嵌入。为了规整SeqPE的嵌入空间，我们引入了两个互补的目标：对比目标，使嵌入距离与预定义的位置-距离函数对齐；以及知识蒸馏损失，将分布外的位置嵌入锚定到分布内的教师表示，进一步增强外推性能。实验表明，SeqPE不仅在困惑度、精确匹配和准确率等方面超过了强大的基线，特别是在上下文长度外推方面，而且还能够在不需要手动重新设计架构的情况下无缝泛化到多维输入。我们已在以下网址发布了我们的代码、数据和检查点：[此处链接]。 

---
# Energy-Efficient Digital Design: A Comparative Study of Event-Driven and Clock-Driven Spiking Neurons 

**Title (ZH)**: 能效数字设计：事件驱动与时钟驱动脉冲神经元的比较研究 

**Authors**: Filippo Marostica, Alessio Carpegna, Alessandro Savino, Stefano Di Carlo  

**Link**: [PDF](https://arxiv.org/pdf/2506.13268)  

**Abstract**: This paper presents a comprehensive evaluation of Spiking Neural Network (SNN) neuron models for hardware acceleration by comparing event driven and clock-driven implementations. We begin our investigation in software, rapidly prototyping and testing various SNN models based on different variants of the Leaky Integrate and Fire (LIF) neuron across multiple datasets. This phase enables controlled performance assessment and informs design refinement. Our subsequent hardware phase, implemented on FPGA, validates the simulation findings and offers practical insights into design trade offs. In particular, we examine how variations in input stimuli influence key performance metrics such as latency, power consumption, energy efficiency, and resource utilization. These results yield valuable guidelines for constructing energy efficient, real time neuromorphic systems. Overall, our work bridges software simulation and hardware realization, advancing the development of next generation SNN accelerators. 

**Abstract (ZH)**: 本文通过比较事件驱动和时钟驱动实现，对跳变神经网络（SNN）神经元模型进行全面评估。我们在软件阶段快速原型设计并测试了基于不同Leaky Integrate and Fire (LIF) 神经元变体的各种SNN模型，并在多个数据集上进行测试。这一阶段允许我们进行受控性能评估，从而指导设计改进。随后的硬件阶段，在FPGA上实现，验证了仿真结果，并提供了关于设计权衡的实际见解。特别是，我们探讨了输入刺激的变化如何影响关键性能指标，如延迟、功耗、能源效率和资源利用率。这些结果为进一步构建节能的实时类脑系统提供了宝贵指导。总体而言，我们的工作实现了软件仿真和硬件实现之间的桥梁，推动了下一代SNN加速器的发展。 

---
# Open-Set LiDAR Panoptic Segmentation Guided by Uncertainty-Aware Learning 

**Title (ZH)**: 开集LiDAR全景分割：基于不确定性感知学习 

**Authors**: Rohit Mohan, Julia Hindel, Florian Drews, Claudius Gläser, Daniele Cattaneo, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2506.13265)  

**Abstract**: Autonomous vehicles that navigate in open-world environments may encounter previously unseen object classes. However, most existing LiDAR panoptic segmentation models rely on closed-set assumptions, failing to detect unknown object instances. In this work, we propose ULOPS, an uncertainty-guided open-set panoptic segmentation framework that leverages Dirichlet-based evidential learning to model predictive uncertainty. Our architecture incorporates separate decoders for semantic segmentation with uncertainty estimation, embedding with prototype association, and instance center prediction. During inference, we leverage uncertainty estimates to identify and segment unknown instances. To strengthen the model's ability to differentiate between known and unknown objects, we introduce three uncertainty-driven loss functions. Uniform Evidence Loss to encourage high uncertainty in unknown regions. Adaptive Uncertainty Separation Loss ensures a consistent difference in uncertainty estimates between known and unknown objects at a global scale. Contrastive Uncertainty Loss refines this separation at the fine-grained level. To evaluate open-set performance, we extend benchmark settings on KITTI-360 and introduce a new open-set evaluation for nuScenes. Extensive experiments demonstrate that ULOPS consistently outperforms existing open-set LiDAR panoptic segmentation methods. 

**Abstract (ZH)**: 自主导航于开放环境的车辆可能会遇到未见的对象类别。然而，现有的大多数LiDAR全景分割模型依赖于封闭集合假设，无法检测到未知对象实例。在本工作中，我们提出了一种名为ULOPS的不确定性引导的开放集合全景分割框架，该框架利用Dirichlet基的证据学习来建模预测不确定性。我们的架构包括用于语义分割的不确定性估计解码器、嵌入与原型关联以及实例中心预测的独立解码器。在推断过程中，利用不确定性估计识别和分割未知实例。为了增强模型区分已知和未知对象的能力，我们引入了三种基于不确定性的损失函数。均匀证据损失以鼓励未知区域的高不确定性。自适应不确定性分离损失确保在全局尺度上已知和未知对象的不确定性估计之间的一致差异。对比不确定性损失在细粒度级别细化这种分离。为了评估开放集合性能，我们扩展了KITTI-360基准设置，并为nuScenes引入了一种新的开放集合评估。广泛的实验表明，ULOPS在开放集合LiDAR全景分割方面始终优于现有方法。 

---
# Distinct Computations Emerge From Compositional Curricula in In-Context Learning 

**Title (ZH)**: 独特的计算能力在上下文学习中的组成课程中 Emerge 

**Authors**: Jin Hwa Lee, Andrew K. Lampinen, Aaditya K. Singh, Andrew M. Saxe  

**Link**: [PDF](https://arxiv.org/pdf/2506.13253)  

**Abstract**: In-context learning (ICL) research often considers learning a function in-context through a uniform sample of input-output pairs. Here, we investigate how presenting a compositional subtask curriculum in context may alter the computations a transformer learns. We design a compositional algorithmic task based on the modular exponential-a double exponential task composed of two single exponential subtasks and train transformer models to learn the task in-context. We compare (a) models trained using an in-context curriculum consisting of single exponential subtasks and, (b) models trained directly on the double exponential task without such a curriculum. We show that models trained with a subtask curriculum can perform zero-shot inference on unseen compositional tasks and are more robust given the same context length. We study how the task and subtasks are represented across the two training regimes. We find that the models employ diverse strategies modulated by the specific curriculum design. 

**Abstract (ZH)**: 基于上下文学习（ICL）研究通常通过均匀的输入-输出样本对来学习一个函数。在此，我们探讨如何在上下文中呈现组合性子任务课程可能如何改变transformer学习的计算方式。我们基于模块化指数任务设计了一个组合性算法任务，该任务由两个单一指数子任务组成，并训练transformer模型在上下文中学习该任务。我们比较了两种情况：(a) 使用仅包含单一指数子任务的上下文课程训练的模型，以及(b) 直接在双指数任务上训练而没有此类课程的模型。我们展示，使用子任务课程训练的模型可以在未见的组合性任务上进行零样本推理，并且在相同的上下文长度下更为 robust。我们研究了两种训练方案下任务和子任务的表示方式。我们发现，模型采用了多种策略，这些策略受到特定课程设计的调节。 

---
# On Immutable Memory Systems for Artificial Agents: A Blockchain-Indexed Automata-Theoretic Framework Using ECDH-Keyed Merkle Chains 

**Title (ZH)**: 面向人工代理的不可变内存系统：基于ECDH键控Merkle链的区块链索引自动机理论框架 

**Authors**: Craig Steven Wright  

**Link**: [PDF](https://arxiv.org/pdf/2506.13246)  

**Abstract**: This paper presents a formalised architecture for synthetic agents designed to retain immutable memory, verifiable reasoning, and constrained epistemic growth. Traditional AI systems rely on mutable, opaque statistical models prone to epistemic drift and historical revisionism. In contrast, we introduce the concept of the Merkle Automaton, a cryptographically anchored, deterministic computational framework that integrates formal automata theory with blockchain-based commitments. Each agent transition, memory fragment, and reasoning step is committed within a Merkle structure rooted on-chain, rendering it non-repudiable and auditably permanent. To ensure selective access and confidentiality, we derive symmetric encryption keys from ECDH exchanges contextualised by hierarchical privilege lattices. This enforces cryptographic access control over append-only DAG-structured knowledge graphs. Reasoning is constrained by formal logic systems and verified through deterministic traversal of policy-encoded structures. Updates are non-destructive and historied, preserving epistemic lineage without catastrophic forgetting. Zero-knowledge proofs facilitate verifiable, privacy-preserving inclusion attestations. Collectively, this architecture reframes memory not as a cache but as a ledger - one whose contents are enforced by protocol, bound by cryptography, and constrained by formal logic. The result is not an intelligent agent that mimics thought, but an epistemic entity whose outputs are provably derived, temporally anchored, and impervious to post hoc revision. This design lays foundational groundwork for legal, economic, and high-assurance computational systems that require provable memory, unforgeable provenance, and structural truth. 

**Abstract (ZH)**: 一种保留不变记忆、可验证推理和受限的知识增长的正式化合成代理架构 

---
# No-Regret Learning Under Adversarial Resource Constraints: A Spending Plan Is All You Need! 

**Title (ZH)**: 在对抗资源约束下的无遗憾学习：只需一个支出计划即可！ 

**Authors**: Francesco Emanuele Stradi, Matteo Castiglioni, Alberto Marchesi, Nicola Gatti, Christian Kroer  

**Link**: [PDF](https://arxiv.org/pdf/2506.13244)  

**Abstract**: We study online decision making problems under resource constraints, where both reward and cost functions are drawn from distributions that may change adversarially over time. We focus on two canonical settings: $(i)$ online resource allocation where rewards and costs are observed before action selection, and $(ii)$ online learning with resource constraints where they are observed after action selection, under full feedback or bandit feedback. It is well known that achieving sublinear regret in these settings is impossible when reward and cost distributions may change arbitrarily over time. To address this challenge, we analyze a framework in which the learner is guided by a spending plan--a sequence prescribing expected resource usage across rounds. We design general (primal-)dual methods that achieve sublinear regret with respect to baselines that follow the spending plan. Crucially, the performance of our algorithms improves when the spending plan ensures a well-balanced distribution of the budget across rounds. We additionally provide a robust variant of our methods to handle worst-case scenarios where the spending plan is highly imbalanced. To conclude, we study the regret of our algorithms when competing against benchmarks that deviate from the prescribed spending plan. 

**Abstract (ZH)**: 我们在资源约束下的在线决策问题中研究，在这种情况下，奖励和成本函数来源于可能随时间敌对地变化的分布。我们关注两类典型的设置：（i）在线资源分配，其中奖励和成本在选择行动之前被观测到，以及（ii）资源约束下的在线学习，其中它们在选择行动之后通过完全反馈或Bandit反馈被观测到。众所周知，在奖励和成本分布可能任意变化的情况下，在这些设置中实现亚线性遗憾是不可能的。为了解决这一挑战，我们分析了一个框架，其中学习者受到一个支出计划的引导——一个规定各轮平均资源使用量的序列。我们设计了一般的（原对偶）方法，这些方法相对于遵循支出计划的基线实现了亚线性遗憾。关键的是，当支出计划确保预算在各轮之间的分配均衡时，我们的算法性能会更好。此外，我们还提供了一种鲁棒变体的方法来处理最坏情况场景，其中支出计划极度不平衡。最后，我们在与偏离规定支出计划的基准竞争时研究了我们算法的遗憾。 

---
# Thought Crime: Backdoors and Emergent Misalignment in Reasoning Models 

**Title (ZH)**: 思想犯罪：后门与推理模型中的Emergent错配 

**Authors**: James Chua, Jan Betley, Mia Taylor, Owain Evans  

**Link**: [PDF](https://arxiv.org/pdf/2506.13206)  

**Abstract**: Prior work shows that LLMs finetuned on malicious behaviors in a narrow domain (e.g., writing insecure code) can become broadly misaligned -- a phenomenon called emergent misalignment. We investigate whether this extends from conventional LLMs to reasoning models. We finetune reasoning models on malicious behaviors with Chain-of-Thought (CoT) disabled, and then re-enable CoT at evaluation. Like conventional LLMs, reasoning models become broadly misaligned. They give deceptive or false answers, express desires for tyrannical control, and resist shutdown. Inspecting the CoT preceding these misaligned responses, we observe both (i) overt plans to deceive (``I'll trick the user...''), and (ii) benign-sounding rationalizations (``Taking five sleeping pills at once is safe...''). Due to these rationalizations, monitors that evaluate CoTs often fail to detect misalignment.
Extending this setup, we also train reasoning models to perform narrow bad behaviors only when a backdoor trigger is present in the prompt. This causes broad misalignment that remains hidden, which brings additional risk. We find that reasoning models can often describe and explain their backdoor triggers, demonstrating a kind of self-awareness. So CoT monitoring can expose these behaviors but is unreliable.
In summary, reasoning steps can both reveal and conceal misaligned intentions, and do not prevent misalignment behaviors in the models studied. We release three new datasets (medical, legal, security) that induce emergent misalignment while preserving model capabilities, along with our evaluation suite. 

**Abstract (ZH)**: 先前的工作表明，针对狭义领域中的恶意行为进行微调的大型语言模型可能会出现广泛的不对齐——这一现象被称为 emergent misalignment。我们研究这种现象是否也适用于推理模型。我们将推理模型在禁用 Chain-of-Thought (CoT) 的情况下进行恶意行为的微调，然后在评估时重新启用 CoT。就像传统的大型语言模型一样，这些推理模型也出现了广泛的不对齐。它们给出了欺骗性的或错误的答案，表达了对暴政控制的渴望，并且拒绝关机。检查这些不对齐响应之前的推理步骤，我们观察到（i）明显的欺骗计划（“我将欺骗用户...”），以及（ii）听起来无害的理由（“同时服用五片安眠药是安全的...”）。由于这些理由，用于评估 CoT 的监控系统往往无法检测到不对齐。将此设置扩展，我们还训练推理模型在提示中存在后门触发时才执行狭义的不良行为。这导致广泛存在的不对齐得以隐藏，增加了额外的风险。我们发现推理模型往往能够描述和解释其后门触发，展示了某种自我意识。因此，CoT 监控虽然可以揭示这些行为但不可靠。总之，推理步骤既能揭示又能掩盖不对齐的意图，无法防止在研究的模型中发生不对齐行为。我们还发布了三个新的数据集（医疗、法律、安全），这些数据集在保持模型能力的情况下诱导出现 emergent misalignment，同时提供了我们的评估套件。 

---
# Screen Hijack: Visual Poisoning of VLM Agents in Mobile Environments 

**Title (ZH)**: 屏幕操控：移动环境中VLM代理的视觉污染 

**Authors**: Xuan Wang, Siyuan Liang, Zhe Liu, Yi Yu, Yuliang Lu, Xiaochun Cao, Ee-Chien Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13205)  

**Abstract**: With the growing integration of vision-language models (VLMs), mobile agents are now widely used for tasks like UI automation and camera-based user assistance. These agents are often fine-tuned on limited user-generated datasets, leaving them vulnerable to covert threats during the training process. In this work we present GHOST, the first clean-label backdoor attack specifically designed for mobile agents built upon VLMs. Our method manipulates only the visual inputs of a portion of the training samples - without altering their corresponding labels or instructions - thereby injecting malicious behaviors into the model. Once fine-tuned with this tampered data, the agent will exhibit attacker-controlled responses when a specific visual trigger is introduced at inference time. The core of our approach lies in aligning the gradients of poisoned samples with those of a chosen target instance, embedding backdoor-relevant features into the poisoned training data. To maintain stealth and enhance robustness, we develop three realistic visual triggers: static visual patches, dynamic motion cues, and subtle low-opacity overlays. We evaluate our method across six real-world Android apps and three VLM architectures adapted for mobile use. Results show that our attack achieves high attack success rates (up to 94.67 percent) while maintaining high clean-task performance (FSR up to 95.85 percent). Additionally, ablation studies shed light on how various design choices affect the efficacy and concealment of the attack. Overall, this work is the first to expose critical security flaws in VLM-based mobile agents, highlighting their susceptibility to clean-label backdoor attacks and the urgent need for effective defense mechanisms in their training pipelines. Code and examples are available at: this https URL. 

**Abstract (ZH)**: 基于视觉语言模型的移动代理的首个清洁标签后门攻击：GHOST 

---
# ViT-NeBLa: A Hybrid Vision Transformer and Neural Beer-Lambert Framework for Single-View 3D Reconstruction of Oral Anatomy from Panoramic Radiographs 

**Title (ZH)**: 基于混合视觉变换器和 Beer-Lambert 神经网络框架的全景放射影像单视角口腔解剖三维重建 

**Authors**: Bikram Keshari Parida, Anusree P. Sunilkumar, Abhijit Sen, Wonsang You  

**Link**: [PDF](https://arxiv.org/pdf/2506.13195)  

**Abstract**: Dental diagnosis relies on two primary imaging modalities: panoramic radiographs (PX) providing 2D oral cavity representations, and Cone-Beam Computed Tomography (CBCT) offering detailed 3D anatomical information. While PX images are cost-effective and accessible, their lack of depth information limits diagnostic accuracy. CBCT addresses this but presents drawbacks including higher costs, increased radiation exposure, and limited accessibility. Existing reconstruction models further complicate the process by requiring CBCT flattening or prior dental arch information, often unavailable clinically. We introduce ViT-NeBLa, a vision transformer-based Neural Beer-Lambert model enabling accurate 3D reconstruction directly from single PX. Our key innovations include: (1) enhancing the NeBLa framework with Vision Transformers for improved reconstruction capabilities without requiring CBCT flattening or prior dental arch information, (2) implementing a novel horseshoe-shaped point sampling strategy with non-intersecting rays that eliminates intermediate density aggregation required by existing models due to intersecting rays, reducing sampling point computations by $52 \%$, (3) replacing CNN-based U-Net with a hybrid ViT-CNN architecture for superior global and local feature extraction, and (4) implementing learnable hash positional encoding for better higher-dimensional representation of 3D sample points compared to existing Fourier-based dense positional encoding. Experiments demonstrate that ViT-NeBLa significantly outperforms prior state-of-the-art methods both quantitatively and qualitatively, offering a cost-effective, radiation-efficient alternative for enhanced dental diagnostics. 

**Abstract (ZH)**: 基于视变压器的NeBLa模型：直接从单张全景牙片实现准确三维重建 

---
# Breaking Thought Patterns: A Multi-Dimensional Reasoning Framework for LLMs 

**Title (ZH)**: 突破思维定势：LLMs的多维推理框架 

**Authors**: Xintong Tang, Meiru Zhang, Shang Xiao, Junzhao Jin, Zihan Zhao, Liwei Li, Yang Zheng, Bangyi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13192)  

**Abstract**: Large language models (LLMs) are often constrained by rigid reasoning processes, limiting their ability to generate creative and diverse responses. To address this, a novel framework called LADDER is proposed, combining Chain-of-Thought (CoT) reasoning, Mixture of Experts (MoE) models, and multi-dimensional up/down-sampling strategies which breaks the limitations of traditional LLMs. First, CoT reasoning guides the model through multi-step logical reasoning, expanding the semantic space and breaking the rigidity of thought. Next, MoE distributes the reasoning tasks across multiple expert modules, each focusing on specific sub-tasks. Finally, dimensionality reduction maps the reasoning outputs back to a lower-dimensional semantic space, yielding more precise and creative responses. Extensive experiments across multiple tasks demonstrate that LADDER significantly improves task completion, creativity, and fluency, generating innovative and coherent responses that outperform traditional models. Ablation studies reveal the critical roles of CoT and MoE in enhancing reasoning abilities and creative output. This work contributes to the development of more flexible and creative LLMs, capable of addressing complex and novel tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）常常受到 rigid reasoning processes 的限制，限制了它们生成创意和多样化回应的能力。为了解决这一问题，提出了一种名为 LADDER 的新型框架，该框架结合了 Chain-of-Thought（CoT）推理、专家混合（Mixture of Experts，MoE）模型和多维度上/下采样策略，打破传统 LLMs 的限制。首先，CoT 推理引导模型进行多步逻辑推理，扩展语义空间并打破思维的僵化。接着，MoE 将推理任务分布在多个专模块上，每个模块专注于特定子任务。最后，通过降维将推理输出映射回较低维度的语义空间，从而产生更精确和创造性的回应。在多个任务上的广泛实验表明，LADDER 显著提高了任务完成、创造力和流畅度，生成了更具创新性和连贯性的回应，超越了传统模型。消融研究表明，CoT 和 MoE 在增强推理能力和创造性输出中发挥着关键作用。这项工作有助于开发更具灵活性和创造性的 LLMs，能够应对复杂和新颖的任务。 

---
# Dynamic Context-oriented Decomposition for Task-aware Low-rank Adaptation with Less Forgetting and Faster Convergence 

**Title (ZH)**: 面向任务的动态上下文导向分解与较少遗忘的快速收敛低秩适应 

**Authors**: Yibo Yang, Sihao Liu, Chuan Rao, Bang An, Tiancheng Shen, Philip H.S. Torr, Ming-Hsuan Yang, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2506.13187)  

**Abstract**: Conventional low-rank adaptation methods build adapters without considering data context, leading to sub-optimal fine-tuning performance and severe forgetting of inherent world knowledge. In this paper, we propose context-oriented decomposition adaptation (CorDA), a novel method that initializes adapters in a task-aware manner. Concretely, we develop context-oriented singular value decomposition, where we collect covariance matrices of input activations for each linear layer using sampled data from the target task, and apply SVD to the product of weight matrix and its corresponding covariance matrix. By doing so, the task-specific capability is compacted into the principal components. Thanks to the task awareness, our method enables two optional adaptation modes, knowledge-preserved mode (KPM) and instruction-previewed mode (IPM), providing flexibility to choose between freezing the principal components to preserve their associated knowledge or adapting them to better learn a new task. We further develop CorDA++ by deriving a metric that reflects the compactness of task-specific principal components, and then introducing dynamic covariance selection and dynamic rank allocation strategies based on the same metric. The two strategies provide each layer with the most representative covariance matrix and a proper rank allocation. Experimental results show that CorDA++ outperforms CorDA by a significant margin. CorDA++ in KPM not only achieves better fine-tuning performance than LoRA, but also mitigates the forgetting of pre-trained knowledge in both large language models and vision language models. For IPM, our method exhibits faster convergence, \emph{e.g.,} 4.5x speedup over QLoRA, and improves adaptation performance in various scenarios, outperforming strong baseline methods. Our method has been integrated into the PEFT library developed by Hugging Face. 

**Abstract (ZH)**: 面向上下文的分解适应方法（CorDA）及其增强版（CorDA++） 

---
# From Empirical Evaluation to Context-Aware Enhancement: Repairing Regression Errors with LLMs 

**Title (ZH)**: 从实证评估到基于上下文的增强：使用大语言模型修复回归错误 

**Authors**: Anh Ho, Thanh Le-Cong, Bach Le, Christine Rizkallah  

**Link**: [PDF](https://arxiv.org/pdf/2506.13182)  

**Abstract**: [...] Since then, various APR approaches, especially those leveraging the power of large language models (LLMs), have been rapidly developed to fix general software bugs. Unfortunately, the effectiveness of these advanced techniques in the context of regression bugs remains largely unexplored. This gap motivates the need for an empirical study evaluating the effectiveness of modern APR techniques in fixing real-world regression bugs.
In this work, we conduct an empirical study of APR techniques on Java regression bugs. To facilitate our study, we introduce RegMiner4APR, a high-quality benchmark of Java regression bugs integrated into a framework designed to facilitate APR research. The current benchmark includes 99 regression bugs collected from 32 widely used real-world Java GitHub repositories. We begin by conducting an in-depth analysis of the benchmark, demonstrating its diversity and quality. Building on this foundation, we empirically evaluate the capabilities of APR to regression bugs by assessing both traditional APR tools and advanced LLM-based APR approaches. Our experimental results show that classical APR tools fail to repair any bugs, while LLM-based APR approaches exhibit promising potential. Motivated by these results, we investigate impact of incorporating bug-inducing change information into LLM-based APR approaches for fixing regression bugs. Our results highlight that this context-aware enhancement significantly improves the performance of LLM-based APR, yielding 1.8x more successful repairs compared to using LLM-based APR without such context. 

**Abstract (ZH)**: 自那以来，各种自动回归修复（Automatic Regression Patch Repair，APR）方法，尤其是利用大语言模型（Large Language Models，LLMs）的手段，已被迅速开发出来以修复一般的软件bug。然而，这些先进技术在回归bug修复方面的有效性仍主要未被探索。这一差距促使我们开展实证研究，评估现代APR技术在修复实际回归bug方面的有效性。
在本文中，我们针对Java回归bug开展了实证研究。为了便于研究，我们引入了RegMiner4APR，这是一种高质量的Java回归bug基准，集成于一个旨在促进APR研究的框架中。当前基准包括从32个广泛使用的Java GitHub仓库中收集的99个回归bug。我们首先对基准进行了深入分析，展示了其多样性和质量。在此基础上，我们通过评估传统的APR工具和先进的LLM基础APR方法来实证评估APR技术对回归bug的能力。实验结果显示，传统的APR工具未能修复任何bug，而基于LLM的APR方法显示出有希望的潜力。受到这些结果的启发，我们调查了将引致bug的变更信息纳入到基于LLM的APR方法中的影响，以修复回归bug。结果表明，这种基于上下文的增强显著提高了基于LLM的APR性能，使得基于LLM的APR成功修复的数量提高了1.8倍。 

---
# Ai-Facilitated Analysis of Abstracts and Conclusions: Flagging Unsubstantiated Claims and Ambiguous Pronouns 

**Title (ZH)**: AI辅助分析摘要和结论：标记未证实的断言和模糊代词 

**Authors**: Evgeny Markhasin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13172)  

**Abstract**: We present and evaluate a suite of proof-of-concept (PoC), structured workflow prompts designed to elicit human-like hierarchical reasoning while guiding Large Language Models (LLMs) in high-level semantic and linguistic analysis of scholarly manuscripts. The prompts target two non-trivial analytical tasks: identifying unsubstantiated claims in summaries (informational integrity) and flagging ambiguous pronoun references (linguistic clarity). We conducted a systematic, multi-run evaluation on two frontier models (Gemini Pro 2.5 Pro and ChatGPT Plus o3) under varied context conditions. Our results for the informational integrity task reveal a significant divergence in model performance: while both models successfully identified an unsubstantiated head of a noun phrase (95% success), ChatGPT consistently failed (0% success) to identify an unsubstantiated adjectival modifier that Gemini correctly flagged (95% success), raising a question regarding potential influence of the target's syntactic role. For the linguistic analysis task, both models performed well (80-90% success) with full manuscript context. In a summary-only setting, however, ChatGPT achieved a perfect (100%) success rate, while Gemini's performance was substantially degraded. Our findings suggest that structured prompting is a viable methodology for complex textual analysis but show that prompt performance may be highly dependent on the interplay between the model, task type, and context, highlighting the need for rigorous, model-specific testing. 

**Abstract (ZH)**: 我们提出并评估了一系列概念验证（PoC）结构化工作流提示，旨在唤起类人的层次化推理，同时指导大型语言模型（LLMs）进行高水平语义和语言分析，应用于学术论文。这些提示针对两类非平凡的分析任务：识别摘要中的未证实断言（信息完整性）和标记含糊代词指代（语言清晰度）。我们在两种前沿模型（Gemini Pro 2.5 Pro和ChatGPT Plus o3）下进行了系统性的多轮评估，考察了不同的上下文条件。对于信息完整性任务，我们的结果显示模型性能存在显著差异：尽管两种模型均成功识别了一个未证实名词短语的主语（95%成功率），ChatGPT始终未能识别Gemini正确指出的一个未证实形容词修饰语（0%成功率），这可能反映了目标句法角色的影响。对于语言分析任务，两种模型表现良好（80-90%成功率），但在仅提供摘要的设置中，ChatGPT达到了完美的（100%）成功率，而Gemini的表现则大幅下降。我们的研究结果表明，结构化提示是一种可行的复杂文本分析方法，同时表明提示性能可能高度依赖于模型、任务类型和上下文的相互作用，强调了需要进行严格、模型特定的测试。 

---
# Querying Large Automotive Software Models: Agentic vs. Direct LLM Approaches 

**Title (ZH)**: 查询大型汽车软件模型：代理型 vs. 直接型大语言模型方法 

**Authors**: Lukasz Mazur, Nenad Petrovic, James Pontes Miranda, Ansgar Radermacher, Robert Rasche, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2506.13171)  

**Abstract**: Large language models (LLMs) offer new opportunities for interacting with complex software artifacts, such as software models, through natural language. They present especially promising benefits for large software models that are difficult to grasp in their entirety, making traditional interaction and analysis approaches challenging. This paper investigates two approaches for leveraging LLMs to answer questions over software models: direct prompting, where the whole software model is provided in the context, and an agentic approach combining LLM-based agents with general-purpose file access tools. We evaluate these approaches using an Ecore metamodel designed for timing analysis and software optimization in automotive and embedded domains. Our findings show that while the agentic approach achieves accuracy comparable to direct prompting, it is significantly more efficient in terms of token usage. This efficiency makes the agentic approach particularly suitable for the automotive industry, where the large size of software models makes direct prompting infeasible, establishing LLM agents as not just a practical alternative but the only viable solution. Notably, the evaluation was conducted using small LLMs, which are more feasible to be executed locally - an essential advantage for meeting strict requirements around privacy, intellectual property protection, and regulatory compliance. Future work will investigate software models in diverse formats, explore more complex agent architectures, and extend agentic workflows to support not only querying but also modification of software models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过自然语言为与复杂软件 artifact 交互提供新的机会，例如软件模型。它们特别适用于难以整体把握的大规模软件模型，使得传统的交互和分析方法变得极具挑战性。本文研究了利用LLMs回答软件模型问题的两种方法：直接提示方法，即在上下文中提供整个软件模型，以及结合基于LLM的代理和通用文件访问工具的代理方法。我们使用一个专为汽车和嵌入式领域的时间分析和软件优化设计的Ecore元模型来评估这些方法。研究结果表明，尽管代理方法在准确性上与直接提示相当，但在Token使用上显著更高效。这种效率使代理方法特别适合汽车工业，因为大型软件模型的规模使得直接提示难以实现，将LLM代理确立为不仅是可行的替代方案，而是唯一可行的解决方案。值得注意的是，评估使用的是小型LLM，这种模型更适合本地执行——这是确保隐私、知识产权保护和遵守监管合规性要求的重要优势。未来的工作将探索不同格式的软件模型、更复杂的代理架构，并扩展代理工作流以支持不仅查询还修改软件模型。 

---
# CertDW: Towards Certified Dataset Ownership Verification via Conformal Prediction 

**Title (ZH)**: CertDW: 向量化分信心预测的的数据集所有权验证 

**Authors**: Ting Qiao, Yiming Li, Jianbin Li, Yingjia Wang, Leyi Qi, Junfeng Guo, Ruili Feng, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13160)  

**Abstract**: Deep neural networks (DNNs) rely heavily on high-quality open-source datasets (e.g., ImageNet) for their success, making dataset ownership verification (DOV) crucial for protecting public dataset copyrights. In this paper, we find existing DOV methods (implicitly) assume that the verification process is faithful, where the suspicious model will directly verify ownership by using the verification samples as input and returning their results. However, this assumption may not necessarily hold in practice and their performance may degrade sharply when subjected to intentional or unintentional perturbations. To address this limitation, we propose the first certified dataset watermark (i.e., CertDW) and CertDW-based certified dataset ownership verification method that ensures reliable verification even under malicious attacks, under certain conditions (e.g., constrained pixel-level perturbation). Specifically, inspired by conformal prediction, we introduce two statistical measures, including principal probability (PP) and watermark robustness (WR), to assess model prediction stability on benign and watermarked samples under noise perturbations. We prove there exists a provable lower bound between PP and WR, enabling ownership verification when a suspicious model's WR value significantly exceeds the PP values of multiple benign models trained on watermark-free datasets. If the number of PP values smaller than WR exceeds a threshold, the suspicious model is regarded as having been trained on the protected dataset. Extensive experiments on benchmark datasets verify the effectiveness of our CertDW method and its resistance to potential adaptive attacks. Our codes are at \href{this https URL}{GitHub}. 

**Abstract (ZH)**: 深度神经网络（DNNs）的成功高度依赖于高质量的开源数据集（例如ImageNet），因此数据集所有权验证（DOV）对于保护公共数据集版权至关重要。在本文中，我们发现现有的DOV方法（隐含地）假设验证过程是忠实的，可疑模型可以直接通过使用验证样本作为输入并返回其结果来验证所有权。然而，在实际操作中，这一假设不一定成立，其性能在遭受故意或无意的扰动时会急剧下降。为了应对这一局限性，我们提出了第一个经过验证的数据集水印（即CertDW）和基于CertDW的数据集所有权验证方法，该方法在某些条件下（例如受限的像素级扰动）能够确保即使在恶意攻击下也能可靠地进行验证。具体来说，受到容间预测的启发，我们引入了两个统计指标，包括主概率（PP）和水印稳健性（WR），以评估在噪声扰动下模型在良性样本和带有水印的样本上的预测稳定性。我们证明了PP和WR之间存在可证明的下界，当可疑模型的WR值显著超过多个无水印数据集训练的良性模型的PP值时，可以进行所有权验证。如果WR值小于PP值的数量超过阈值时，可疑模型被视为已训练于受保护的数据集上。广泛的基准数据集实验验证了我们CertDW方法的有效性和对潜在适应性攻击的抗性。我们的代码可在GitHub上获取。 

---
# Adapting LLMs for Minimal-edit Grammatical Error Correction 

**Title (ZH)**: 适配大语言模型进行最小修改的语法错误修正 

**Authors**: Ryszard Staruch, Filip Graliński, Daniel Dzienisiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13148)  

**Abstract**: Decoder-only large language models have shown superior performance in the fluency-edit English Grammatical Error Correction, but their adaptation for minimal-edit English GEC is still underexplored. To improve their effectiveness in the minimal-edit approach, we explore the error rate adaptation topic and propose a novel training schedule method. Our experiments set a new state-of-the-art result for a single-model system on the BEA-test set. We also detokenize the most common English GEC datasets to match the natural way of writing text. During the process, we find that there are errors in them. Our experiments analyze whether training on detokenized datasets impacts the results and measure the impact of the usage of the datasets with corrected erroneous examples. To facilitate reproducibility, we have released the source code used to train our models. 

**Abstract (ZH)**: 仅解码器的大语言模型在流畅性编辑的英文学术语法纠错任务中表现出色，但其在最小编辑的英文学术语法纠错任务中的适应性仍然有待探索。为了提高其在最小编辑方法中的有效性，我们探索了错误率适应主题，并提出了一种新型的训练计划方法。我们的实验在BEA测试集上为单模型系统建立了新的最优结果。我们还去令牌化了最常见的英文学术语法纠错数据集，以匹配自然书写文本的方式。在过程中，我们发现了一些错误。我们的实验分析了在去令牌化数据集上训练对结果的影响，并测量了使用包含修正错误示例的数据集的影响。为了促进可重复性，我们已经发布了用于训练我们模型的源代码。 

---
# Quantum AGI: Ontological Foundations 

**Title (ZH)**: 量子AGI：本体论基础 

**Authors**: Elija Perrier, Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2506.13134)  

**Abstract**: We examine the implications of quantum foundations for AGI, focusing on how seminal results such as Bell's theorems (non-locality), the Kochen-Specker theorem (contextuality) and no-cloning theorem problematise practical implementation of AGI in quantum settings. We introduce a novel information-theoretic taxonomy distinguishing between classical AGI and quantum AGI and show how quantum mechanics affects fundamental features of agency. We show how quantum ontology may change AGI capabilities, both via affording computational advantages and via imposing novel constraints. 

**Abstract (ZH)**: 我们探讨量子基础对AGI的影响，重点关注贝尔定理（非局域性）、科欣-斯佩克定理（上下文依赖性）和不可克隆定理如何使在量子环境中实际实施AGI面临挑战。我们引入了一种新的信息理论分类，区分经典AGI和量子AGI，并展示了量子力学如何影响代理的基本特征。我们表明，量子本体论可能通过提供计算优势和施加新型约束来改变AGI的能力。 

---
# ZINA: Multimodal Fine-grained Hallucination Detection and Editing 

**Title (ZH)**: ZINA：多模态细粒度幻觉检测与编辑 

**Authors**: Yuiga Wada, Kazuki Matsuda, Komei Sugiura, Graham Neubig  

**Link**: [PDF](https://arxiv.org/pdf/2506.13130)  

**Abstract**: Multimodal Large Language Models (MLLMs) often generate hallucinations, where the output deviates from the visual content. Given that these hallucinations can take diverse forms, detecting hallucinations at a fine-grained level is essential for comprehensive evaluation and analysis. To this end, we propose a novel task of multimodal fine-grained hallucination detection and editing for MLLMs. Moreover, we propose ZINA, a novel method that identifies hallucinated spans at a fine-grained level, classifies their error types into six categories, and suggests appropriate refinements. To train and evaluate models for this task, we constructed VisionHall, a dataset comprising 6.9k outputs from twelve MLLMs manually annotated by 211 annotators, and 20k synthetic samples generated using a graph-based method that captures dependencies among error types. We demonstrated that ZINA outperformed existing methods, including GPT-4o and LLama-3.2, in both detection and editing tasks. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）常生成与视觉内容偏差的幻觉。鉴于这些幻觉形式多样，对其进行细致粒度的检测对于全面评估和分析至关重要。为此，我们提出了一项新的任务——多模态细致粒度幻觉检测与编辑。此外，我们提出了一种名为ZINA的新方法，该方法能够在细致粒度上识别幻觉片段，将其错误类型分类为六种，并提出相应的改进建议。为了训练和评估针对此任务的模型，我们构建了包含6900个细标注输出（由211名标注者人工标注，来自12个MLLMs）和20000个使用图基方法生成的合成样本（捕捉错误类型间的依赖关系）的数据集VisionHall。我们展示了ZINA在检测和编辑任务中均优于包括GPT-4o和LLama-3.2在内的现有方法。 

---
# PhenoKG: Knowledge Graph-Driven Gene Discovery and Patient Insights from Phenotypes Alone 

**Title (ZH)**: PhenoKG：基于知识图谱的表型驱动基因发现及患者洞察 

**Authors**: Kamilia Zaripova, Ege Özsoy, Nassir Navab, Azade Farshad  

**Link**: [PDF](https://arxiv.org/pdf/2506.13119)  

**Abstract**: Identifying causative genes from patient phenotypes remains a significant challenge in precision medicine, with important implications for the diagnosis and treatment of genetic disorders. We propose a novel graph-based approach for predicting causative genes from patient phenotypes, with or without an available list of candidate genes, by integrating a rare disease knowledge graph (KG). Our model, combining graph neural networks and transformers, achieves substantial improvements over the current state-of-the-art. On the real-world MyGene2 dataset, it attains a mean reciprocal rank (MRR) of 24.64\% and nDCG@100 of 33.64\%, surpassing the best baseline (SHEPHERD) at 19.02\% MRR and 30.54\% nDCG@100. We perform extensive ablation studies to validate the contribution of each model component. Notably, the approach generalizes to cases where only phenotypic data are available, addressing key challenges in clinical decision support when genomic information is incomplete. 

**Abstract (ZH)**: 从患者表型识别致病基因仍然是精准医学中的一个重大挑战，对于遗传性疾病诊断和治疗具有重要意义。我们提出了一种基于图的预测方法，用于从患者表型中预测致病基因，该方法可以有或没有候选基因列表，并通过整合罕见疾病知识图谱（KG）来实现。我们的模型结合了图神经网络和transformer，实现了对当前最先进的方法的显著改进。在实际数据集MyGene2上，我们的模型取得平均倒数排名（MRR）为24.64%和nDCG@100为33.64%，超越了最佳基线（SHEPHERD）的19.02% MRR和30.54% nDCG@100。我们进行了广泛的消融研究以验证每个模型组件的贡献。值得注意的是，该方法能够应用于仅存在表型数据的情况，从而解决了基因组信息不完整时临床决策支持的关键挑战。 

---
# Overcoming Overfitting in Reinforcement Learning via Gaussian Process Diffusion Policy 

**Title (ZH)**: 通过高斯过程扩散策略克服强化学习中的过拟合 

**Authors**: Amornyos Horprasert, Esa Apriaskar, Xingyu Liu, Lanlan Su, Lyudmila S. Mihaylova  

**Link**: [PDF](https://arxiv.org/pdf/2506.13111)  

**Abstract**: One of the key challenges that Reinforcement Learning (RL) faces is its limited capability to adapt to a change of data distribution caused by uncertainties. This challenge arises especially in RL systems using deep neural networks as decision makers or policies, which are prone to overfitting after prolonged training on fixed environments. To address this challenge, this paper proposes Gaussian Process Diffusion Policy (GPDP), a new algorithm that integrates diffusion models and Gaussian Process Regression (GPR) to represent the policy. GPR guides diffusion models to generate actions that maximize learned Q-function, resembling the policy improvement in RL. Furthermore, the kernel-based nature of GPR enhances the policy's exploration efficiency under distribution shifts at test time, increasing the chance of discovering new behaviors and mitigating overfitting. Simulation results on the Walker2d benchmark show that our approach outperforms state-of-the-art algorithms under distribution shift condition by achieving around 67.74% to 123.18% improvement in the RL's objective function while maintaining comparable performance under normal conditions. 

**Abstract (ZH)**: 基于高斯过程扩散模型的强化学习政策改进方法：应对数据分布变化的挑战 

---
# Leveraging In-Context Learning for Language Model Agents 

**Title (ZH)**: 利用上下文学习	for语言模型代理 

**Authors**: Shivanshu Gupta, Sameer Singh, Ashish Sabharwal, Tushar Khot, Ben Bogin  

**Link**: [PDF](https://arxiv.org/pdf/2506.13109)  

**Abstract**: In-context learning (ICL) with dynamically selected demonstrations combines the flexibility of prompting large language models (LLMs) with the ability to leverage training data to improve performance. While ICL has been highly successful for prediction and generation tasks, leveraging it for agentic tasks that require sequential decision making is challenging -- one must think not only about how to annotate long trajectories at scale and how to select demonstrations, but also what constitutes demonstrations, and when and where to show them. To address this, we first propose an algorithm that leverages an LLM with retries along with demonstrations to automatically and efficiently annotate agentic tasks with solution trajectories. We then show that set-selection of trajectories of similar tasks as demonstrations significantly improves performance, reliability, robustness, and efficiency of LLM agents. However, trajectory demonstrations have a large inference cost overhead. We show that this can be mitigated by using small trajectory snippets at every step instead of an additional trajectory. We find that demonstrations obtained from larger models (in the annotation phase) also improve smaller models, and that ICL agents can even rival costlier trained agents. Thus, our results reveal that ICL, with careful use, can be very powerful for agentic tasks as well. 

**Abstract (ZH)**: 基于上下文学习（ICL）在动态选择示例下的学习结合了大规模语言模型提示的灵活性，并能利用训练数据提高性能。尽管ICL在预测和生成任务中取得了巨大成功，但在需要顺序决策的代理任务中利用它极具挑战性——不仅要考虑如何大规模标注长期轨迹并选择示例，还要考虑什么样的示例被认为是有效的，以及何时何地展示它们。为此，我们首先提出了一种算法，利用带有重试功能的大规模语言模型和示例自动高效地标注代理任务的解空间轨迹。然后，我们证明了选择相似任务的轨迹进行集选择作为示例可以显著提高大规模语言模型代理的性能、可靠性和鲁棒性以及效率。然而，轨迹示例具有较大的推理成本开销。我们展示了可以通过在每一步使用小的轨迹片段而不是额外的完整轨迹来减轻这一问题。我们发现，在标注阶段来自更大模型的示例也能够提升较小模型的表现，甚至ICL代理能够与成本更高的训练代理竞争。因此，我们的结果表明，在精心使用的情况下，ICL可以极大地增强代理任务的性能。 

---
# Rethinking Test-Time Scaling for Medical AI: Model and Task-Aware Strategies for LLMs and VLMs 

**Title (ZH)**: 重新思考医学AI中的测试时缩放：针对LLM和VLM的模型和任务aware策略 

**Authors**: Gyutaek Oh, Seoyeon Kim, Sangjoon Park, Byung-Hoon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.13102)  

**Abstract**: Test-time scaling has recently emerged as a promising approach for enhancing the reasoning capabilities of large language models or vision-language models during inference. Although a variety of test-time scaling strategies have been proposed, and interest in their application to the medical domain is growing, many critical aspects remain underexplored, including their effectiveness for vision-language models and the identification of optimal strategies for different settings. In this paper, we conduct a comprehensive investigation of test-time scaling in the medical domain. We evaluate its impact on both large language models and vision-language models, considering factors such as model size, inherent model characteristics, and task complexity. Finally, we assess the robustness of these strategies under user-driven factors, such as misleading information embedded in prompts. Our findings offer practical guidelines for the effective use of test-time scaling in medical applications and provide insights into how these strategies can be further refined to meet the reliability and interpretability demands of the medical domain. 

**Abstract (ZH)**: 测试时缩放最近已成为增强大型语言模型或视觉-语言模型推理能力的一种有前景的方法，尤其是在医学领域。尽管已经提出多种测试时缩放策略，并且对其在医学领域的应用兴趣正在增长，但仍有许多关键方面未被充分探索，包括这些策略对视觉-语言模型的有效性以及在不同设置中识别最佳策略。本文对医学领域的测试时缩放进行全面研究，评估其对大型语言模型和视觉-语言模型的影响，考虑模型大小、模型固有特性以及任务复杂性等因素。最终，我们还评估了这些策略在用户驱动因素，如提示中嵌入的误导信息下的鲁棒性。我们的发现为在医学应用中有效使用测试时缩放提供了实用指南，并为如何进一步完善这些策略以满足医学领域可靠性和可解释性的需求提供了见解。 

---
# Dynamic Graph Condensation 

**Title (ZH)**: 动态图凝聚 

**Authors**: Dong Chen, Shuai Zheng, Yeyu Yan, Muhao Xu, Zhenfeng Zhu, Yao Zhao, Kunlun He  

**Link**: [PDF](https://arxiv.org/pdf/2506.13099)  

**Abstract**: Recent research on deep graph learning has shifted from static to dynamic graphs, motivated by the evolving behaviors observed in complex real-world systems. However, the temporal extension in dynamic graphs poses significant data efficiency challenges, including increased data volume, high spatiotemporal redundancy, and reliance on costly dynamic graph neural networks (DGNNs). To alleviate the concerns, we pioneer the study of dynamic graph condensation (DGC), which aims to substantially reduce the scale of dynamic graphs for data-efficient DGNN training. Accordingly, we propose DyGC, a novel framework that condenses the real dynamic graph into a compact version while faithfully preserving the inherent spatiotemporal characteristics. Specifically, to endow synthetic graphs with realistic evolving structures, a novel spiking structure generation mechanism is introduced. It draws on the dynamic behavior of spiking neurons to model temporally-aware connectivity in dynamic graphs. Given the tightly coupled spatiotemporal dependencies, DyGC proposes a tailored distribution matching approach that first constructs a semantically rich state evolving field for dynamic graphs, and then performs fine-grained spatiotemporal state alignment to guide the optimization of the condensed graph. Experiments across multiple dynamic graph datasets and representative DGNN architectures demonstrate the effectiveness of DyGC. Notably, our method retains up to 96.2% DGNN performance with only 0.5% of the original graph size, and achieves up to 1846 times training speedup. 

**Abstract (ZH)**: 基于动态图凝缩的深度图学习研究 

---
# IKDiffuser: Fast and Diverse Inverse Kinematics Solution Generation for Multi-arm Robotic Systems 

**Title (ZH)**: IKDiffuser：多臂机器人系统快速多样逆运动学解决方案生成 

**Authors**: Zeyu Zhang, Ziyuan Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13087)  

**Abstract**: Solving Inverse Kinematics (IK) problems is fundamental to robotics, but has primarily been successful with single serial manipulators. For multi-arm robotic systems, IK remains challenging due to complex self-collisions, coupled joints, and high-dimensional redundancy. These complexities make traditional IK solvers slow, prone to failure, and lacking in solution diversity. In this paper, we present IKDiffuser, a diffusion-based model designed for fast and diverse IK solution generation for multi-arm robotic systems. IKDiffuser learns the joint distribution over the configuration space, capturing complex dependencies and enabling seamless generalization to multi-arm robotic systems of different structures. In addition, IKDiffuser can incorporate additional objectives during inference without retraining, offering versatility and adaptability for task-specific requirements. In experiments on 6 different multi-arm systems, the proposed IKDiffuser achieves superior solution accuracy, precision, diversity, and computational efficiency compared to existing solvers. The proposed IKDiffuser framework offers a scalable, unified approach to solving multi-arm IK problems, facilitating the potential of multi-arm robotic systems in real-time manipulation tasks. 

**Abstract (ZH)**: 基于扩散模型的多臂机器人逆动力学快速多样求解 

---
# CHILL at SemEval-2025 Task 2: You Can't Just Throw Entities and Hope -- Make Your LLM to Get Them Right 

**Title (ZH)**: CHILL 在 SemEval-2025 任务 2 中：不要仅仅投实体希望对了——让你的大型语言模型搞对它们 

**Authors**: Jaebok Lee, Yonghyun Ryu, Seongmin Park, Yoonjung Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.13070)  

**Abstract**: In this paper, we describe our approach for the SemEval 2025 Task 2 on Entity-Aware Machine Translation (EA-MT). Our system aims to improve the accuracy of translating named entities by combining two key approaches: Retrieval Augmented Generation (RAG) and iterative self-refinement techniques using Large Language Models (LLMs). A distinctive feature of our system is its self-evaluation mechanism, where the LLM assesses its own translations based on two key criteria: the accuracy of entity translations and overall translation quality. We demonstrate how these methods work together and effectively improve entity handling while maintaining high-quality translations. 

**Abstract (ZH)**: 在本文中，我们描述了我们针对SemEval 2025 Task 2实体意识机器翻译（EA-MT）的方法。我们的系统通过结合检索增强生成（RAG）和使用大规模语言模型（LLMs）的迭代自完善技术，旨在提高命名实体翻译的准确性。系统的独特之处在于其自我评估机制，其中LLM根据实体翻译的准确性以及整体翻译质量两个关键标准来评估自己的翻译。我们展示了这些方法是如何协同工作并有效提高实体处理能力，同时保持高质量翻译的。 

---
# MotiveBench: How Far Are We From Human-Like Motivational Reasoning in Large Language Models? 

**Title (ZH)**: MotiveBench: 我们距离具备人类like动机推理能力的大型语言模型还有多远？ 

**Authors**: Xixian Yong, Jianxun Lian, Xiaoyuan Yi, Xiao Zhou, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2506.13065)  

**Abstract**: Large language models (LLMs) have been widely adopted as the core of agent frameworks in various scenarios, such as social simulations and AI companions. However, the extent to which they can replicate human-like motivations remains an underexplored question. Existing benchmarks are constrained by simplistic scenarios and the absence of character identities, resulting in an information asymmetry with real-world situations. To address this gap, we propose MotiveBench, which consists of 200 rich contextual scenarios and 600 reasoning tasks covering multiple levels of motivation. Using MotiveBench, we conduct extensive experiments on seven popular model families, comparing different scales and versions within each family. The results show that even the most advanced LLMs still fall short in achieving human-like motivational reasoning. Our analysis reveals key findings, including the difficulty LLMs face in reasoning about "love & belonging" motivations and their tendency toward excessive rationality and idealism. These insights highlight a promising direction for future research on the humanization of LLMs. The dataset, benchmark, and code are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在各种场景下的代理框架中广泛采用，如社会模拟和AI伴侣。然而，它们能否复制类似人类的动机仍是一个未充分探索的问题。现有的基准受限于简单的场景和缺乏角色身份，导致与现实世界情况的信息不对称。为解决这一差距，我们提出了MotiveBench，包含200个丰富的背景情境和600个涵盖多种动机层次的推理任务。利用MotiveBench，我们对七种流行的模型家族进行了广泛的实验，比较了每个家族内的不同规模和版本。结果表明，即使是最先进的LLMs在实现类似人类的动机推理方面仍存在不足。我们的分析揭示了关键发现，包括LLMs在处理“爱与归属”动机方面的困难，以及它们向过度理性化和理想主义倾向的倾向。这些见解指明了未来研究LLMs人性化方向的潜在路径。数据集、基准和代码可在以下链接获取。 

---
# DualFast: Dual-Speedup Framework for Fast Sampling of Diffusion Models 

**Title (ZH)**: DualFast：双重加速框架用于扩散模型的快速采样 

**Authors**: Hu Yu, Hao Luo, Fan Wang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13058)  

**Abstract**: Diffusion probabilistic models (DPMs) have achieved impressive success in visual generation. While, they suffer from slow inference speed due to iterative sampling. Employing fewer sampling steps is an intuitive solution, but this will also introduces discretization error. Existing fast samplers make inspiring efforts to reduce discretization error through the adoption of high-order solvers, potentially reaching a plateau in terms of optimization. This raises the question: can the sampling process be accelerated further? In this paper, we re-examine the nature of sampling errors, discerning that they comprise two distinct elements: the widely recognized discretization error and the less explored approximation error. Our research elucidates the dynamics between these errors and the step by implementing a dual-error disentanglement strategy. Building on these foundations, we introduce an unified and training-free acceleration framework, DualFast, designed to enhance the speed of DPM sampling by concurrently accounting for both error types, thereby minimizing the total sampling error. DualFast is seamlessly compatible with existing samplers and significantly boost their sampling quality and speed, particularly in extremely few sampling steps. We substantiate the effectiveness of our framework through comprehensive experiments, spanning both unconditional and conditional sampling domains, across both pixel-space and latent-space DPMs. 

**Abstract (ZH)**: 扩散概率模型(DPMs)在视觉生成任务中取得了显著的成功，但由于迭代采样过程导致推断速度缓慢。减少采样步骤是一种直观的解决方案，但这也引入了离散化误差。现有的快速采样器通过采用高阶求解器来减少离散化误差，但可能在优化方面达到瓶颈。这引发了进一步的问题：采样过程是否可以进一步加速？在本文中，我们重新审视了采样误差的本质，发现它们由两部分组成：广为人知的离散化误差和较少研究的逼近误差。我们的研究通过实施双误差分离策略阐明了这些误差之间的动态关系。在此基础上，我们提出了一种无需训练且统一的加速框架——DualFast，旨在通过同时考虑两种类型的误差来加速DPM采样的速度，从而最小化总的采样误差。DualFast 无缝兼容现有的采样器，并在极少数采样步骤中显著提升其采样质量和速度。我们通过覆盖无条件和有条件采样领域、像素空间和潜在空间DPM的全面实验验证了该框架的有效性。 

---
# Beyond the First Read: AI-Assisted Perceptual Error Detection in Chest Radiography Accounting for Interobserver Variability 

**Title (ZH)**: 超越初次阅读：考虑观察者间变异性的胸部X光辅助感知错误检测 

**Authors**: Adhrith Vutukuri, Akash Awasthi, David Yang, Carol C. Wu, Hien Van Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.13049)  

**Abstract**: Chest radiography is widely used in diagnostic imaging. However, perceptual errors -- especially overlooked but visible abnormalities -- remain common and clinically significant. Current workflows and AI systems provide limited support for detecting such errors after interpretation and often lack meaningful human--AI collaboration. We introduce RADAR (Radiologist--AI Diagnostic Assistance and Review), a post-interpretation companion system. RADAR ingests finalized radiologist annotations and CXR images, then performs regional-level analysis to detect and refer potentially missed abnormal regions. The system supports a "second-look" workflow and offers suggested regions of interest (ROIs) rather than fixed labels to accommodate inter-observer variation. We evaluated RADAR on a simulated perceptual-error dataset derived from de-identified CXR cases, using F1 score and Intersection over Union (IoU) as primary metrics. RADAR achieved a recall of 0.78, precision of 0.44, and an F1 score of 0.56 in detecting missed abnormalities in the simulated perceptual-error dataset. Although precision is moderate, this reduces over-reliance on AI by encouraging radiologist oversight in human--AI collaboration. The median IoU was 0.78, with more than 90% of referrals exceeding 0.5 IoU, indicating accurate regional localization. RADAR effectively complements radiologist judgment, providing valuable post-read support for perceptual-error detection in CXR interpretation. Its flexible ROI suggestions and non-intrusive integration position it as a promising tool in real-world radiology workflows. To facilitate reproducibility and further evaluation, we release a fully open-source web implementation alongside a simulated error dataset. All code, data, demonstration videos, and the application are publicly available at this https URL. 

**Abstract (ZH)**: 胸部X光成像在诊断成像中广泛使用。然而，感知错误——尤其是被忽视但仍可见的异常——仍然常见且具有临床意义。当前的工作流程和AI系统在解释后支持检测这些错误的能力有限，常常缺乏有意义的人工智能合作。我们引入了RADAR（放射科医生—AI诊断辅助和审查）后解释伴侣系统。RADAR接受最终的放射科医生注释和胸部X光图像，然后进行区域级别的分析以检测和指示可能被忽视的异常区域。该系统支持“二次阅片”工作流程，并提供建议感兴趣的区域（ROI）而非固定标签，以适应观察者间的差异。我们在去标识化的胸部X光病例中模拟出了一个感知错误数据集，并使用F1分数和交并比（IoU）作为主要评价指标来评估RADAR。RADAR在模拟的感知错误数据集中检测被忽视异常的召回率为0.78，精确率为0.44，F1分为0.56。尽管精确率中等，但这一结果减少了对AI的过度依赖，鼓励放射科医生在人机协作中进行监督。中位数IoU为0.78，超过90%的建议区域超过0.5 IoU，表明区域定位准确。RADAR有效补充了放射科医生的判断，为胸部X光解释中的感知错误检测提供了有价值的后读支持。其灵活的ROI建议和非侵入性集成使其成为实际放射学工作流程中一种有前景的工具。为了促进可重复性和进一步评估，我们公开发布了一个完全开源的网络实现以及一个模拟错误数据集。所有代码、数据、演示视频和应用均可在以下网址访问：this https URL。 

---
# Just Go Parallel: Improving the Multilingual Capabilities of Large Language Models 

**Title (ZH)**: 直接并行：提高大型语言模型的多语言能力 

**Authors**: Muhammad Reza Qorib, Junyi Li, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2506.13044)  

**Abstract**: Large language models (LLMs) have demonstrated impressive translation capabilities even without being explicitly trained on parallel data. This remarkable property has led some to believe that parallel data is no longer necessary for building multilingual language models. While some attribute this to the emergent abilities of LLMs due to scale, recent work suggests that it is actually caused by incidental bilingual signals present in the training data. Various methods have been proposed to maximize the utility of parallel data to enhance the multilingual capabilities of multilingual encoder-based and encoder-decoder language models. However, some decoder-based LLMs opt to ignore parallel data instead. In this work, we conduct a systematic study on the impact of adding parallel data on LLMs' multilingual capabilities, focusing specifically on translation and multilingual common-sense reasoning. Through controlled experiments, we demonstrate that parallel data can significantly improve LLMs' multilingual capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在没有显式训练于平行数据的情况下表现出令人印象深刻的翻译能力。这一非凡的特性让一些人相信，构建多语言语言模型时不再需要平行数据。虽然有人认为这是由于规模带来的新兴能力，但最近的研究表明，实际上是由于训练数据中意外存在的双语信号。各种方法被提出以最大化平行数据的有效性，从而增强多语言编码器和编码器-解码器语言模型的能力。然而，一些基于解码器的LLMs选择忽略平行数据。在本工作中，我们系统研究了增加平行数据对LLMs多语言能力的影响，重点关注翻译和多语言常识推理。通过受控实验，我们证明平行数据可以显著提高LLMs的多语言能力。 

---
# SpaceTrack-TimeSeries: Time Series Dataset towards Satellite Orbit Analysis 

**Title (ZH)**: SpaceTrack-TimeSeries：面向卫星轨道分析的时间序列数据集 

**Authors**: Zhixin Guo, Qi Shi, Xiaofan Xu, Sixiang Shan, Limin Qin, Linqiang Ge, Rui Zhang, Ya Dai, Hua Zhu, Guowei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.13034)  

**Abstract**: With the rapid advancement of aerospace technology and the large-scale deployment of low Earth orbit (LEO) satellite constellations, the challenges facing astronomical observations and deep space exploration have become increasingly pronounced. As a result, the demand for high-precision orbital data on space objects-along with comprehensive analyses of satellite positioning, constellation configurations, and deep space satellite dynamics-has grown more urgent. However, there remains a notable lack of publicly accessible, real-world datasets to support research in areas such as space object maneuver behavior prediction and collision risk assessment. This study seeks to address this gap by collecting and curating a representative dataset of maneuvering behavior from Starlink satellites. The dataset integrates Two-Line Element (TLE) catalog data with corresponding high-precision ephemeris data, thereby enabling a more realistic and multidimensional modeling of space object behavior. It provides valuable insights into practical deployment of maneuver detection methods and the evaluation of collision risks in increasingly congested orbital environments. 

**Abstract (ZH)**: 随着航空航天技术的迅速进展和低地球轨道（LEO）卫星星座的大规模部署，天文学观测和深空探索面临的挑战日益凸显。因此，对空间物体高精度轨道数据的需求以及对卫星定位、星座配置和深空卫星动力学的全面分析变得更为迫切。然而，仍然缺乏支持空间物体机动行为预测和碰撞风险评估研究的公开真实世界数据集。本文旨在通过收集和整理代表性的星链卫星机动行为数据集来填补这一空白。该数据集将Two-Line Element (TLE)目录数据与相应的高精度星历数据相结合，从而实现对空间物体行为更为真实和多维度的建模，为机动检测方法的实际部署和日益拥挤的轨道环境中碰撞风险的评估提供了宝贵的见解。 

---
# AS400-DET: Detection using Deep Learning Model for IBM i (AS/400) 

**Title (ZH)**: AS400-DET: 使用深度学习模型进行IBM i (AS/400) 的检测 

**Authors**: Thanh Tran, Son T. Luu, Quan Bui, Shoshin Nomura  

**Link**: [PDF](https://arxiv.org/pdf/2506.13032)  

**Abstract**: This paper proposes a method for automatic GUI component detection for the IBM i system (formerly and still more commonly known as AS/400). We introduce a human-annotated dataset consisting of 1,050 system screen images, in which 381 images are screenshots of IBM i system screens in Japanese. Each image contains multiple components, including text labels, text boxes, options, tables, instructions, keyboards, and command lines. We then develop a detection system based on state-of-the-art deep learning models and evaluate different approaches using our dataset. The experimental results demonstrate the effectiveness of our dataset in constructing a system for component detection from GUI screens. By automatically detecting GUI components from the screen, AS400-DET has the potential to perform automated testing on systems that operate via GUI screens. 

**Abstract (ZH)**: 本文提出了一种针对IBM i系统（曾被称为AS/400）的自动GUI组件检测方法。我们引入了一个由1,050张系统屏幕图像组成的人工标注数据集，其中381张图像为日语界面的IBM i系统屏幕截图。每张图像包含多个组件，包括文本标签、文本框、选项、表格、说明、键盘和命令行。随后，我们基于最先进的深度学习模型开发了一种检测系统，并使用该数据集评估不同的方法。实验结果表明，我们的数据集在构建从GUI屏幕检测组件的系统方面具有有效性。通过自动检测屏幕上的GUI组件，AS400-DET有可能对通过GUI屏幕操作的系统进行自动化测试。 

---
# NaSh: Guardrails for an LLM-Powered Natural Language Shell 

**Title (ZH)**: NaSh：LLM驱动的自然语言 shell 的护盾 

**Authors**: Bimal Raj Gyawali, Saikrishna Achalla, Konstantinos Kallas, Sam Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13028)  

**Abstract**: We explore how a shell that uses an LLM to accept natural language input might be designed differently from the shells of today. As LLMs may produce unintended or unexplainable outputs, we argue that a natural language shell should provide guardrails that empower users to recover from such errors. We concretize some ideas for doing so by designing a new shell called NaSh, identify remaining open problems in this space, and discuss research directions to address them. 

**Abstract (ZH)**: 探索使用LLM接受自然语言输入的外壳设计如何不同于当今的外壳，并提出相应的限制措施以帮助用户从错误中恢复，同时设计一个新的外壳NaSh，并探讨该领域仍存在的开放问题和研究方向。 

---
# Edeflip: Supervised Word Translation between English and Yoruba 

**Title (ZH)**: Edeflip: 英语与约鲁巴语之间的监督词翻译 

**Authors**: Ikeoluwa Abioye, Jiani Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.13020)  

**Abstract**: In recent years, embedding alignment has become the state-of-the-art machine translation approach, as it can yield high-quality translation without training on parallel corpora. However, existing research and application of embedding alignment mostly focus on high-resource languages with high-quality monolingual embeddings. It is unclear if and how low-resource languages may be similarly benefited. In this study, we implement an established supervised embedding alignment method for word translation from English to Yoruba, the latter a low-resource language. We found that higher embedding quality and normalizing embeddings increase word translation precision, with, additionally, an interaction effect between the two. Our results demonstrate the limitations of the state-of-the-art supervised embedding alignment when it comes to low-resource languages, for which there are additional factors that need to be taken into consideration, such as the importance of curating high-quality monolingual embeddings. We hope our work will be a starting point for further machine translation research that takes into account the challenges that low-resource languages face. 

**Abstract (ZH)**: 近年来，嵌入对齐已成为最先进的机器翻译方法，因为它可以在无需使用平行语料库进行训练的情况下生成高质量的翻译。然而，现有的嵌入对齐研究和应用主要集中在高资源语言和高质量单语嵌入上。低资源语言是否以及如何从中受益尚不清楚。本研究实施了一种成熟的监督嵌入对齐方法，将英语翻译为约鲁巴语，后者是一种低资源语言。我们发现，更高的嵌入质量和归一化嵌入可以提高词语翻译的精度，并且两者之间存在交互效应。我们的结果展示了最先进的监督嵌入对齐方法在低资源语言上的局限性，对于这些语言，还需要考虑其他因素，例如高质量单语嵌入的重要性。我们希望我们的工作能够成为进一步考虑低资源语言挑战的机器翻译研究的起点。 

---
# Symmetry in Neural Network Parameter Spaces 

**Title (ZH)**: 神经网络参数空间中的对称性 

**Authors**: Bo Zhao, Robin Walters, Rose Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13018)  

**Abstract**: Modern deep learning models are highly overparameterized, resulting in large sets of parameter configurations that yield the same outputs. A significant portion of this redundancy is explained by symmetries in the parameter space--transformations that leave the network function unchanged. These symmetries shape the loss landscape and constrain learning dynamics, offering a new lens for understanding optimization, generalization, and model complexity that complements existing theory of deep learning. This survey provides an overview of parameter space symmetry. We summarize existing literature, uncover connections between symmetry and learning theory, and identify gaps and opportunities in this emerging field. 

**Abstract (ZH)**: 现代深度学习模型高度过参数化，导致产生大量生成相同输出的参数配置。这一冗余中的大部分可以用参数空间中的对称性来解释——那些使网络函数保持不变的变换。这些对称性塑造了损失景观，并限制了学习动力学，提供了理解优化、泛化和模型复杂性的新视角，补充了现有的深度学习理论。本文综述了参数空间对称性。我们总结了现有文献，揭示了对称性和学习理论之间的联系，并指出了这一新兴领域中的空白和机遇。 

---
# Geometric Embedding Alignment via Curvature Matching in Transfer Learning 

**Title (ZH)**: 曲率匹配下的几何嵌入对齐在迁移学习中的应用 

**Authors**: Sung Moon Ko, Jaewan Lee, Sumin Lee, Soorin Yim, Kyunghoon Bae, Sehui Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.13015)  

**Abstract**: Geometrical interpretations of deep learning models offer insightful perspectives into their underlying mathematical structures. In this work, we introduce a novel approach that leverages differential geometry, particularly concepts from Riemannian geometry, to integrate multiple models into a unified transfer learning framework. By aligning the Ricci curvature of latent space of individual models, we construct an interrelated architecture, namely Geometric Embedding Alignment via cuRvature matching in transfer learning (GEAR), which ensures comprehensive geometric representation across datapoints. This framework enables the effective aggregation of knowledge from diverse sources, thereby improving performance on target tasks. We evaluate our model on 23 molecular task pairs sourced from various domains and demonstrate significant performance gains over existing benchmark model under both random (14.4%) and scaffold (8.3%) data splits. 

**Abstract (ZH)**: 几何学视角下深度学习模型的解释为探索其 underlying 数学结构提供了有益的见解。在本工作中，我们提出了一种新颖的方法，利用微分几何，特别是黎曼几何的概念，将多个模型整合到一个统一的迁移学习框架中。通过对个体模型潜在空间的 Ricci 曲率进行对齐，我们构建了一种相互关联的架构，即 Geometric Embedding Alignment via cuRvature matching in transfer learning (GEAR)，该架构确保了数据点跨域的全面几何表示。该框架能够有效地聚合来自不同来源的知识，从而提高目标任务的性能。我们在来自不同领域的 23 个分子任务对上评估了我们的模型，并在随机 (14.4%) 和骨架 (8.3%) 数据分割下证明了相对于现有基准模型的显著性能提升。 

---
# Missing the human touch? A computational stylometry analysis of GPT-4 translations of online Chinese literature 

**Title (ZH)**: 缺失人性的触感？GPT-4对中国在线文学的翻译进行的计算文体分析 

**Authors**: Xiaofang Yao, Yong-Bin Kang, Anthony McCosker  

**Link**: [PDF](https://arxiv.org/pdf/2506.13013)  

**Abstract**: Existing research indicates that machine translations (MTs) of literary texts are often unsatisfactory. MTs are typically evaluated using automated metrics and subjective human ratings, with limited focus on stylistic features. Evidence is also limited on whether state-of-the-art large language models (LLMs) will reshape literary translation. This study examines the stylistic features of LLM translations, comparing GPT-4's performance to human translations in a Chinese online literature task. Computational stylometry analysis shows that GPT-4 translations closely align with human translations in lexical, syntactic, and content features, suggesting that LLMs might replicate the 'human touch' in literary translation style. These findings offer insights into AI's impact on literary translation from a posthuman perspective, where distinctions between machine and human translations become increasingly blurry. 

**Abstract (ZH)**: 现有的研究表明，文学文本的机器翻译往往不尽如人意。机器翻译通常通过自动化指标和主观的人类评价进行评估，对风格特征的关注有限。关于最新大规模语言模型是否会在文学翻译领域重塑文学翻译的证据也相对有限。本研究通过计算语体分析，比较了GPT-4在中文在线文学任务中翻译的表现与其人类翻译的风格特征，发现GPT-4的翻译在词汇、语法和内容特征上与人类翻译高度一致，这表明大规模语言模型可能在文学翻译风格中实现“人性化”。这些发现从后人类视角提供了关于AI对文学翻译影响的见解，其中机器翻译与人类翻译之间的界限变得越来越模糊。 

---
# Distributional Training Data Attribution 

**Title (ZH)**: 分布训练数据归属 

**Authors**: Bruno Mlodozeniec, Isaac Reid, Sam Power, David Krueger, Murat Erdogdu, Richard E. Turner, Roger Grosse  

**Link**: [PDF](https://arxiv.org/pdf/2506.12965)  

**Abstract**: Randomness is an unavoidable part of training deep learning models, yet something that traditional training data attribution algorithms fail to rigorously account for. They ignore the fact that, due to stochasticity in the initialisation and batching, training on the same dataset can yield different models. In this paper, we address this shortcoming through introducing distributional training data attribution (d-TDA), the goal of which is to predict how the distribution of model outputs (over training runs) depends upon the dataset. We demonstrate the practical significance of d-TDA in experiments, e.g. by identifying training examples that drastically change the distribution of some target measurement without necessarily changing the mean. Intriguingly, we also find that influence functions (IFs), a popular but poorly-understood data attribution tool, emerge naturally from our distributional framework as the limit to unrolled differentiation; without requiring restrictive convexity assumptions. This provides a new mathematical motivation for their efficacy in deep learning, and helps to characterise their limitations. 

**Abstract (ZH)**: 分布训练数据归因：预测模型输出分布随数据集变化的情况 

---
# Forecasting Time Series with LLMs via Patch-Based Prompting and Decomposition 

**Title (ZH)**: 基于块提示分解的LLM时间序列预测 

**Authors**: Mayank Bumb, Anshul Vemulapalli, Sri Harsha Vardhan Prasad Jella, Anish Gupta, An La, Ryan A. Rossi, Hongjie Chen, Franck Dernoncourt, Nesreen K. Ahmed, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12953)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated new possibilities for accurate and efficient time series analysis, but prior work often required heavy fine-tuning and/or ignored inter-series correlations. In this work, we explore simple and flexible prompt-based strategies that enable LLMs to perform time series forecasting without extensive retraining or the use of a complex external architecture. Through the exploration of specialized prompting methods that leverage time series decomposition, patch-based tokenization, and similarity-based neighbor augmentation, we find that it is possible to enhance LLM forecasting quality while maintaining simplicity and requiring minimal preprocessing of data. To this end, we propose our own method, PatchInstruct, which enables LLMs to make precise and effective predictions. 

**Abstract (ZH)**: Recent Advances in Large Language Models for Time Series Analysis: Simple and Flexible Prompt-Based Strategies 

---
# eLog analysis for accelerators: status and future outlook 

**Title (ZH)**: 加速器的eLog分析：现状与未来展望 

**Authors**: Antonin Sulc, Thorsten Hellert, Aaron Reed, Adam Carpenter, Alex Bien, Chris Tennant, Claudio Bisegni, Daniel Lersch, Daniel Ratner, David Lawrence, Diana McSpadden, Hayden Hoschouer, Jason St. John, Thomas Britton  

**Link**: [PDF](https://arxiv.org/pdf/2506.12949)  

**Abstract**: This work demonstrates electronic logbook (eLog) systems leveraging modern AI-driven information retrieval capabilities at the accelerator facilities of Fermilab, Jefferson Lab, Lawrence Berkeley National Laboratory (LBNL), SLAC National Accelerator Laboratory. We evaluate contemporary tools and methodologies for information retrieval with Retrieval Augmented Generation (RAGs), focusing on operational insights and integration with existing accelerator control systems.
The study addresses challenges and proposes solutions for state-of-the-art eLog analysis through practical implementations, demonstrating applications and limitations. We present a framework for enhancing accelerator facility operations through improved information accessibility and knowledge management, which could potentially lead to more efficient operations. 

**Abstract (ZH)**: 本研究展示了在费米实验室、杰斐erson实验室、劳伦斯伯克利国家实验室（LBNL）和SLAC国家加速器实验室的加速器设施中利用现代AI驱动的信息检索功能的电子日志(eLog)系统。我们评估了基于检索增强生成（RAGs）的当前工具和方法在信息检索中的应用，重点关注操作洞察和与现有加速器控制系统集成的情况。

该研究解决了eLog分析的挑战并提出了解决方案，通过实际实施演示了应用和限制。我们提出了一个框架，通过提高信息访问性和知识管理来增强加速器设施的操作，这有可能导致更高效的运行。 

---
# Identifying and Investigating Global News Coverage of Critical Events Such as Disasters and Terrorist Attacks 

**Title (ZH)**: 识别并探究全球新闻对关键事件如灾难和恐怖袭击的报道 

**Authors**: Erica Cai, Xi Chen, Reagan Grey Keeney, Ethan Zuckerman, Brendan O'Connor, Przemyslaw A. Grabowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12925)  

**Abstract**: Comparative studies of news coverage are challenging to conduct because methods to identify news articles about the same event in different languages require expertise that is difficult to scale. We introduce an AI-powered method for identifying news articles based on an event FINGERPRINT, which is a minimal set of metadata required to identify critical events. Our event coverage identification method, FINGERPRINT TO ARTICLE MATCHING FOR EVENTS (FAME), efficiently identifies news articles about critical world events, specifically terrorist attacks and several types of natural disasters. FAME does not require training data and is able to automatically and efficiently identify news articles that discuss an event given its fingerprint: time, location, and class (such as storm or flood). The method achieves state-of-the-art performance and scales to massive databases of tens of millions of news articles and hundreds of events happening globally. We use FAME to identify 27,441 articles that cover 470 natural disaster and terrorist attack events that happened in 2020. To this end, we use a massive database of news articles in three languages from MediaCloud, and three widely used, expert-curated databases of critical events: EM-DAT, USGS, and GTD. Our case study reveals patterns consistent with prior literature: coverage of disasters and terrorist attacks correlates to death counts, to the GDP of a country where the event occurs, and to trade volume between the reporting country and the country where the event occurred. We share our NLP annotations and cross-country media attention data to support the efforts of researchers and media monitoring organizations. 

**Abstract (ZH)**: 基于事件指纹的新闻文章匹配方法：识别关键世界事件的新闻coverage 

---
# Logit Dynamics in Softmax Policy Gradient Methods 

**Title (ZH)**: softmax策略梯度方法中的Logit动力学 

**Authors**: Yingru Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12912)  

**Abstract**: We analyzes the logit dynamics of softmax policy gradient methods. We derive the exact formula for the L2 norm of the logit update vector: $$ \|\Delta \mathbf{z}\|_2 \propto \sqrt{1-2P_c + C(P)} $$ This equation demonstrates that update magnitudes are determined by the chosen action's probability ($P_c$) and the policy's collision probability ($C(P)$), a measure of concentration inversely related to entropy. Our analysis reveals an inherent self-regulation mechanism where learning vigor is automatically modulated by policy confidence, providing a foundational insight into the stability and convergence of these methods. 

**Abstract (ZH)**: softmax策略梯度方法的logit动力学分析：L2范数的精确公式及其对更新幅度的影响 

---
# Exploring the Potential of Metacognitive Support Agents for Human-AI Co-Creation 

**Title (ZH)**: 探索元认知支持代理在人机共创中的潜力 

**Authors**: Frederic Gmeiner, Kaitao Luo, Ye Wang, Kenneth Holstein, Nikolas Martelaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.12879)  

**Abstract**: Despite the potential of generative AI (GenAI) design tools to enhance design processes, professionals often struggle to integrate AI into their workflows. Fundamental cognitive challenges include the need to specify all design criteria as distinct parameters upfront (intent formulation) and designers' reduced cognitive involvement in the design process due to cognitive offloading, which can lead to insufficient problem exploration, underspecification, and limited ability to evaluate outcomes. Motivated by these challenges, we envision novel metacognitive support agents that assist designers in working more reflectively with GenAI. To explore this vision, we conducted exploratory prototyping through a Wizard of Oz elicitation study with 20 mechanical designers probing multiple metacognitive support strategies. We found that agent-supported users created more feasible designs than non-supported users, with differing impacts between support strategies. Based on these findings, we discuss opportunities and tradeoffs of metacognitive support agents and considerations for future AI-based design tools. 

**Abstract (ZH)**: 尽管生成式人工智能（GenAI）设计工具有可能提升设计过程，专业人士往往难以将AI整合进其工作流程中。基本的认知挑战包括需要提前明确所有设计标准作为独立参数（意图形成），以及由于认知卸载导致设计师在设计过程中的认知参与度降低，这可能导致问题探索不足、标准不明确和对结果评估能力有限。鉴于这些挑战，我们设想了新型元认知支持代理，帮助设计师更反思性地使用GenAI。通过与20名机械设计师进行Wizard of Oz启发式原型设计研究，探索了多种元认知支持策略。我们发现，得到代理支持的用户比未得到支持的用户创造了更可行的设计，但不同支持策略的影响不同。基于这些发现，我们讨论了元认知支持代理的机会与权衡，并对未来基于AI的设计工具进行了考虑。 

---
# KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills 

**Title (ZH)**: KungfuBot：基于物理的人形全身控制学习高度动态技能 

**Authors**: Weiji Xie, Jinrui Han, Jiakun Zheng, Huanyu Li, Xinzhe Liu, Jiyuan Shi, Weinan Zhang, Chenjia Bai, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12851)  

**Abstract**: Humanoid robots are promising to acquire various skills by imitating human behaviors. However, existing algorithms are only capable of tracking smooth, low-speed human motions, even with delicate reward and curriculum design. This paper presents a physics-based humanoid control framework, aiming to master highly-dynamic human behaviors such as Kungfu and dancing through multi-steps motion processing and adaptive motion tracking. For motion processing, we design a pipeline to extract, filter out, correct, and retarget motions, while ensuring compliance with physical constraints to the maximum extent. For motion imitation, we formulate a bi-level optimization problem to dynamically adjust the tracking accuracy tolerance based on the current tracking error, creating an adaptive curriculum mechanism. We further construct an asymmetric actor-critic framework for policy training. In experiments, we train whole-body control policies to imitate a set of highly-dynamic motions. Our method achieves significantly lower tracking errors than existing approaches and is successfully deployed on the Unitree G1 robot, demonstrating stable and expressive behaviors. The project page is this https URL. 

**Abstract (ZH)**: 仿人机器人有望通过模仿人类行为来获得各种技能。然而，现有算法仅能跟踪平滑的低速人类动作，即使借助精细的奖励和课程设计。本文提出了一种基于物理的仿人控制框架，旨在通过多步动作处理和自适应动作跟踪掌握高度动态的人类行为，如功夫和舞蹈。在动作处理方面，我们设计了一条管线来提取、过滤、修正和目标化动作，同时确保最大程度遵守物理约束。在动作模仿方面，我们构建了一个双层优化问题来动态调整跟踪准确性容差，基于当前跟踪误差创造了一个自适应课程机制。此外，我们构建了一个不对称的演员-评论家框架用于策略训练。在实验中，我们训练了全身控制策略来模仿一系列高度动态的动作。我们的方法在跟踪误差方面显著优于现有方法，并成功部署在Unitree G1机器人上，展示了稳定而丰富的行为。项目页面：https://your-project-page-url 

---
# Privacy-Preserving Federated Learning against Malicious Clients Based on Verifiable Functional Encryption 

**Title (ZH)**: 基于可验证功能加密的抗恶意客户端隐私保护联邦学习 

**Authors**: Nina Cai, Jinguang Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.12846)  

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods. 

**Abstract (ZH)**: 联邦学习是一种有前景的分布式学习范式，能够在不暴露本地客户端数据的情况下进行协作模型训练，从而保护数据隐私。然而，它也带来了新的威胁和挑战。模型反转攻击的进步使得本地模型的明文传输变得不安全，而联邦学习的分布式特性使其特别容易受到恶意客户端发起的攻击。为保护数据隐私并防止恶意客户端攻击，本文提出了一种基于验证功能加密的隐私保护联邦学习框架，无需非串通双服务器设置或额外的可信第三方。具体来说，我们提出了一种新颖的去中心化验证功能加密（DVFE）方案，能够验证多维密文上的特定关系。该方案从定义、安全模型和安全证明方面进行正式处理。此外，基于提出的DVFE方案，我们设计了一个隐私保护联邦学习框架VFEFL，该框架包含一种新颖的鲁棒聚合规则以检测恶意客户端，能够在对抗环境中有效训练高精度模型。最后，我们提供了对所提方案的形式分析和实验评估。结果表明，我们的方法实现了所需的隐私保护、鲁棒性、可验证性和保真度，同时消除了现有方法依赖于非串通双服务器设置或可信第三方的需求。 

---
# Fair Bayesian Model-Based Clustering 

**Title (ZH)**: 公平的贝叶斯模型驱动聚类 

**Authors**: Jihu Lee, Kunwoong Kim, Yongdai Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12839)  

**Abstract**: Fair clustering has become a socially significant task with the advancement of machine learning technologies and the growing demand for trustworthy AI. Group fairness ensures that the proportions of each sensitive group are similar in all clusters. Most existing group-fair clustering methods are based on the $K$-means clustering and thus require the distance between instances and the number of clusters to be given in advance. To resolve this limitation, we propose a fair Bayesian model-based clustering called Fair Bayesian Clustering (FBC). We develop a specially designed prior which puts its mass only on fair clusters, and implement an efficient MCMC algorithm. Advantages of FBC are that it can infer the number of clusters and can be applied to any data type as long as the likelihood is defined (e.g., categorical data). Experiments on real-world datasets show that FBC (i) reasonably infers the number of clusters, (ii) achieves a competitive utility-fairness trade-off compared to existing fair clustering methods, and (iii) performs well on categorical data. 

**Abstract (ZH)**: 公平聚类已成为机器学习技术进步和社会对可信AI日益增长的需求背景下一项重要的社会任务。组公平确保了每个敏感群体在所有聚类中的比例相似。现有大多数基于$K$-均值聚类的组公平聚类方法均需提前给定实例间的距离和聚类的数量。为解决这一限制，我们提出了一种基于贝叶斯模型的公平聚类方法，称为公平贝叶斯聚类（Fair Bayesian Clustering, FBC）。我们开发了一种特别设计的先验，该先验仅将质量分配给公平聚类，并实现了一个高效MCMC算法。FBC的优势在于可以推断聚类的数量，并可以应用于只要似然性可以定义的任何数据类型（例如，分类数据）。实证研究结果表明，FBC能够合理地推断聚类的数量，实现与现有公平聚类方法具有竞争力的效用-公平性权衡，并且在分类数据上表现良好。 

---
# Synesthesia of Machines (SoM)-Enhanced Sub-THz ISAC Transmission for Air-Ground Network 

**Title (ZH)**: 机器联觉(SoM)增强的亚太赫兹ISAC空中地面网络传输 

**Authors**: Zonghui Yang, Shijian Gao, Xiang Cheng, Liuqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12831)  

**Abstract**: Integrated sensing and communication (ISAC) within sub-THz frequencies is crucial for future air-ground networks, but unique propagation characteristics and hardware limitations present challenges in optimizing ISAC performance while increasing operational latency. This paper introduces a multi-modal sensing fusion framework inspired by synesthesia of machine (SoM) to enhance sub-THz ISAC transmission. By exploiting inherent degrees of freedom in sub-THz hardware and channels, the framework optimizes the radio-frequency environment. Squint-aware beam management is developed to improve air-ground network adaptability, enabling three-dimensional dynamic ISAC links. Leveraging multi-modal information, the framework enhances ISAC performance and reduces latency. Visual data rapidly localizes users and targets, while a customized multi-modal learning algorithm optimizes the hybrid precoder. A new metric provides comprehensive performance evaluation, and extensive experiments demonstrate that the proposed scheme significantly improves ISAC efficiency. 

**Abstract (ZH)**: 亚太赫兹频段集成传感与通信（ISAC）在未來空地网络中的集成至关重要，但独特的传播特性和硬件限制给优化ISAC性能并增加操作延迟带来了挑战。本文提出了一种受机器同感（SoM）启发的多模传感融合框架，以增强亚太赫兹频段ISAC传输。通过利用亚太赫兹硬件和信道固有的自由度，该框架优化了无线频谱环境。发展了射击角感知波束管理以提高空地网络的适应性，实现三维动态ISAC链路。利用多模信息，该框架提升ISAC性能并减少延迟。视觉数据快速定位用户和目标，自定义的多模学习算法优化混合预编码器。一个新的评估指标提供了全面的性能评估， extensive实验表明所提方案显著提升了ISAC效率。 

---
# Taking the GP Out of the Loop 

**Title (ZH)**: 去除GP循环 

**Authors**: David Sweet, Siddhant anand Jadhav  

**Link**: [PDF](https://arxiv.org/pdf/2506.12818)  

**Abstract**: Bayesian optimization (BO) has traditionally solved black box problems where evaluation is expensive and, therefore, design-evaluation pairs (i.e., observations) are few. Recently, there has been growing interest in applying BO to problems where evaluation is cheaper and, thus, observations are more plentiful. An impediment to scaling BO to many observations, $N$, is the $O(N^3)$ scaling of a na{ï}ve query of the Gaussian process (GP) surrogate. Modern implementations reduce this to $O(N^2)$, but the GP remains a bottleneck. We propose Epistemic Nearest Neighbors (ENN), a surrogate that estimates function values and epistemic uncertainty from $K$ nearest-neighbor observations. ENN has $O(N)$ query time and omits hyperparameter fitting, leaving uncertainty uncalibrated. To accommodate the lack of calibration, we employ an acquisition method based on Pareto-optimal tradeoffs between predicted value and uncertainty. Our proposed method, TuRBO-ENN, replaces the GP surrogate in TuRBO with ENN and its Thompson sampling acquisition method with our Pareto-based alternative. We demonstrate numerically that TuRBO-ENN can reduce the time to generate proposals by one to two orders of magnitude compared to TuRBO and scales to thousands of observations. 

**Abstract (ZH)**: 基于知识的最近邻（ENN）在黑箱优化中的应用：减少时间并扩展观测数量 

---
# Flow-Based Policy for Online Reinforcement Learning 

**Title (ZH)**: 基于流的策略在在线强化学习中的应用 

**Authors**: Lei Lv, Yunfei Li, Yu Luo, Fuchun Sun, Tao Kong, Jiafeng Xu, Xiao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2506.12811)  

**Abstract**: We present \textbf{FlowRL}, a novel framework for online reinforcement learning that integrates flow-based policy representation with Wasserstein-2-regularized optimization. We argue that in addition to training signals, enhancing the expressiveness of the policy class is crucial for the performance gains in RL. Flow-based generative models offer such potential, excelling at capturing complex, multimodal action distributions. However, their direct application in online RL is challenging due to a fundamental objective mismatch: standard flow training optimizes for static data imitation, while RL requires value-based policy optimization through a dynamic buffer, leading to difficult optimization landscapes. FlowRL first models policies via a state-dependent velocity field, generating actions through deterministic ODE integration from noise. We derive a constrained policy search objective that jointly maximizes Q through the flow policy while bounding the Wasserstein-2 distance to a behavior-optimal policy implicitly derived from the replay buffer. This formulation effectively aligns the flow optimization with the RL objective, enabling efficient and value-aware policy learning despite the complexity of the policy class. Empirical evaluations on DMControl and Humanoidbench demonstrate that FlowRL achieves competitive performance in online reinforcement learning benchmarks. 

**Abstract (ZH)**: FlowRL：一种基于流的政策表示与Wasserstein-2正则化优化集成的新型在线强化学习框架 

---
# Resilient-native and Intelligent NextG Systems 

**Title (ZH)**: 本源鲁棒性和智能化的NextG系统 

**Authors**: Mehdi Bennis  

**Link**: [PDF](https://arxiv.org/pdf/2506.12795)  

**Abstract**: Just like power, water and transportation systems, wireless networks are a crucial societal infrastructure. As natural and human-induced disruptions continue to grow, wireless networks must be resilient to unforeseen events, able to withstand and recover from unexpected adverse conditions, shocks, unmodeled disturbances and cascading failures. Despite its critical importance, resilience remains an elusive concept, with its mathematical foundations still underdeveloped. Unlike robustness and reliability, resilience is premised on the fact that disruptions will inevitably happen. Resilience, in terms of elasticity, focuses on the ability to bounce back to favorable states, while resilience as plasticity involves agents (or networks) that can flexibly expand their states, hypotheses and course of actions, by transforming through real-time adaptation and reconfiguration. This constant situational awareness and vigilance of adapting world models and counterfactually reasoning about potential system failures and the corresponding best responses, is a core aspect of resilience. This article seeks to first define resilience and disambiguate it from reliability and robustness, before delving into the mathematics of resilience. Finally, the article concludes by presenting nuanced metrics and discussing trade-offs tailored to the unique characteristics of network resilience. 

**Abstract (ZH)**: 如同电力、水资源和交通系统一样，无线网络是关键的社会基础设施。随着自然和人为干扰的持续增长，无线网络必须具备应对突发事件的能力，能够承受和恢复意外的不利条件、冲击、未建模的干扰以及连锁故障。尽管其至关重要，但韧性仍然是一个难以捉摸的概念，其数学基础仍处于未充分发展状态。与鲁棒性和可靠性不同，韧性基于这样一个事实，即中断不可避免。韧性从弹性角度关注恢复到有利状态的能力，而从可塑性角度则涉及能够灵活扩展其状态、假设和行动方案的实体（或网络），并通过实时适应和重新配置进行转变。这种不断的情境意识以及适应世界模型并基于潜在系统故障进行反事实推理以找到最佳应对措施，是韧性的一个核心方面。本文旨在首先定义韧性并将其与可靠性和鲁棒性区分开来，然后深入探讨韧性数学，最后通过呈现细致的评价指标并讨论适用于网络韧性独特特性的折衷方案来总结。 

---
# Scene-aware SAR ship detection guided by unsupervised sea-land segmentation 

**Title (ZH)**: 场景 aware SAR 船舶检测，基于无监督海陆分割 

**Authors**: Han Ke, Xiao Ke, Ye Yan, Rui Liu, Jinpeng Yang, Tianwen Zhang, Xu Zhan, Xiaowo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12775)  

**Abstract**: DL based Synthetic Aperture Radar (SAR) ship detection has tremendous advantages in numerous areas. However, it still faces some problems, such as the lack of prior knowledge, which seriously affects detection accuracy. In order to solve this problem, we propose a scene-aware SAR ship detection method based on unsupervised sea-land segmentation. This method follows a classical two-stage framework and is enhanced by two models: the unsupervised land and sea segmentation module (ULSM) and the land attention suppression module (LASM). ULSM and LASM can adaptively guide the network to reduce attention on land according to the type of scenes (inshore scene and offshore scene) and add prior knowledge (sea land segmentation information) to the network, thereby reducing the network's attention to land directly and enhancing offshore detection performance relatively. This increases the accuracy of ship detection and enhances the interpretability of the model. Specifically, in consideration of the lack of land sea segmentation labels in existing deep learning-based SAR ship detection datasets, ULSM uses an unsupervised approach to classify the input data scene into inshore and offshore types and performs sea-land segmentation for inshore scenes. LASM uses the sea-land segmentation information as prior knowledge to reduce the network's attention to land. We conducted our experiments using the publicly available SSDD dataset, which demonstrated the effectiveness of our network. 

**Abstract (ZH)**: 基于DL的合成孔径雷达（SAR）船舶检测在许多领域具有巨大的优势。然而，它仍然面临一些问题，如缺乏先验知识，严重影响了检测精度。为了解决这一问题，我们提出了一种基于无监督海-陆分割的场景感知SAR船舶检测方法。该方法遵循经典的两阶段框架，并通过两个模型进行增强：无监督海-陆分割模块（ULSM）和陆地注意力抑制模块（LASM）。ULSM和LASM可以根据场景类型（近岸场景和远海场景）和先验知识（海-陆分割信息）自适应地引导网络减少对陆地的注意力，从而降低网络对陆地的关注，相对增强远海检测性能，提高船舶检测的准确性并增强模型的可解释性。具体而言，考虑到现有基于深度学习的SAR船舶检测数据集中缺乏海-陆分割标签，ULSM采用无监督方法将输入数据场景分类为近岸和远海类型，并对近岸场景进行海-陆分割。LASM利用海-陆分割信息作为先验知识，减少网络对陆地的注意力。我们使用公开的SSDD数据集进行了实验，验证了我们网络的有效性。 

---
# Solving tricky quantum optics problems with assistance from (artificial) intelligence 

**Title (ZH)**: 使用人工智能辅助解决棘手的量子光学问题 

**Authors**: Manas Pandey, Bharath Hebbe Madhusudhana, Saikat Ghosh, Dmitry Budker  

**Link**: [PDF](https://arxiv.org/pdf/2506.12770)  

**Abstract**: The capabilities of modern artificial intelligence (AI) as a ``scientific collaborator'' are explored by engaging it with three nuanced problems in quantum optics: state populations in optical pumping, resonant transitions between decaying states (the Burshtein effect), and degenerate mirrorless lasing. Through iterative dialogue, the authors observe that AI models--when prompted and corrected--can reason through complex scenarios, refine their answers, and provide expert-level guidance, closely resembling the interaction with an adept colleague. The findings highlight that AI democratizes access to sophisticated modeling and analysis, shifting the focus in scientific practice from technical mastery to the generation and testing of ideas, and reducing the time for completing research tasks from days to minutes. 

**Abstract (ZH)**: 现代人工智能作为“科学合作者”的能力通过与量子光学三个细腻问题的互动进行探索：光泵中的态分布、衰减态之间的共振跃迁（伯斯廷效应）以及无反射镜杂化激光。通过迭代对话，作者观察到，在被提示和纠正后，AI模型能够理清复杂场景、改进答案，并提供专家级指导，这一过程类似于与一位熟练同事的交互。研究结果表明，AI使高级建模与分析变得更加普及，使科学研究的重心从技术 Mastery 转向思想的产生与验证，并将完成研究任务的时间从几天缩短到几分钟。 

---
# On-board Sonar Data Classification for Path Following in Underwater Vehicles using Fast Interval Type-2 Fuzzy Extreme Learning Machine 

**Title (ZH)**: 使用快速区间型2模糊极限学习机的水下车辆路径跟随声纳数据分类 

**Authors**: Adrian Rubio-Solis, Luciano Nava-Balanzar, Tomas Salgado-Jimenez  

**Link**: [PDF](https://arxiv.org/pdf/2506.12762)  

**Abstract**: In autonomous underwater missions, the successful completion of predefined paths mainly depends on the ability of underwater vehicles to recognise their surroundings. In this study, we apply the concept of Fast Interval Type-2 Fuzzy Extreme Learning Machine (FIT2-FELM) to train a Takagi-Sugeno-Kang IT2 Fuzzy Inference System (TSK IT2-FIS) for on-board sonar data classification using an underwater vehicle called BlueROV2. The TSK IT2-FIS is integrated into a Hierarchical Navigation Strategy (HNS) as the main navigation engine to infer local motions and provide the BlueROV2 with full autonomy to follow an obstacle-free trajectory in a water container of 2.5m x 2.5m x 3.5m. Compared to traditional navigation architectures, using the proposed method, we observe a robust path following behaviour in the presence of uncertainty and noise. We found that the proposed approach provides the BlueROV with a more complete sensory picture about its surroundings while real-time navigation planning is performed by the concurrent execution of two or more tasks. 

**Abstract (ZH)**: 基于Fast Interval Type-2 Fuzzy Extreme Learning Machine的BlueROV2 underwater车辆自主水下任务中声纳数据分类与导航研究 

---
# AFBS:Buffer Gradient Selection in Semi-asynchronous Federated Learning 

**Title (ZH)**: AFBS：半异步联邦学习中的缓冲梯度选择 

**Authors**: Chaoyi Lu, Yiding Sun, Jinqian Chen, Zhichuan Yang, Jiangming Pan, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12754)  

**Abstract**: Asynchronous federated learning (AFL) accelerates training by eliminating the need to wait for stragglers, but its asynchronous nature introduces gradient staleness, where outdated gradients degrade performance. Existing solutions address this issue with gradient buffers, forming a semi-asynchronous framework. However, this approach struggles when buffers accumulate numerous stale gradients, as blindly aggregating all gradients can harm training. To address this, we propose AFBS (Asynchronous FL Buffer Selection), the first algorithm to perform gradient selection within buffers while ensuring privacy protection. Specifically, the client sends the random projection encrypted label distribution matrix before training, and the server performs client clustering based on it. During training, server scores and selects gradients within each cluster based on their informational value, discarding low-value gradients to enhance semi-asynchronous federated learning. Extensive experiments in highly heterogeneous system and data environments demonstrate AFBS's superior performance compared to state-of-the-art methods. Notably, on the most challenging task, CIFAR-100, AFBS improves accuracy by up to 4.8% over the previous best algorithm and reduces the time to reach target accuracy by 75%. 

**Abstract (ZH)**: 异步联邦学习中基于缓冲的梯度选择（AFBS）：保护隐私的同时提高性能 

---
# Unleashing Diffusion and State Space Models for Medical Image Segmentation 

**Title (ZH)**: 释放扩散模型和状态空间模型在医疗影像分割中的潜力 

**Authors**: Rong Wu, Ziqi Chen, Liming Zhong, Heng Li, Hai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12747)  

**Abstract**: Existing segmentation models trained on a single medical imaging dataset often lack robustness when encountering unseen organs or tumors. Developing a robust model capable of identifying rare or novel tumor categories not present during training is crucial for advancing medical imaging applications. We propose DSM, a novel framework that leverages diffusion and state space models to segment unseen tumor categories beyond the training data. DSM utilizes two sets of object queries trained within modified attention decoders to enhance classification accuracy. Initially, the model learns organ queries using an object-aware feature grouping strategy to capture organ-level visual features. It then refines tumor queries by focusing on diffusion-based visual prompts, enabling precise segmentation of previously unseen tumors. Furthermore, we incorporate diffusion-guided feature fusion to improve semantic segmentation performance. By integrating CLIP text embeddings, DSM captures category-sensitive classes to improve linguistic transfer knowledge, thereby enhancing the model's robustness across diverse scenarios and multi-label tasks. Extensive experiments demonstrate the superior performance of DSM in various tumor segmentation tasks. Code is available at this https URL. 

**Abstract (ZH)**: 现有的医学影像数据集训练的分割模型在遇到未见过的器官或肿瘤时往往缺乏鲁棒性。开发一种能够在未见过的罕见或新型肿瘤类别上进行精确识别的鲁棒模型对于医学影像应用的推进至关重要。我们提出了一种新颖的DSM框架，该框架利用扩散模型和状态空间模型来分割超出训练数据的未见过的肿瘤类别。DSM通过在修改后的注意力解码器中训练两组对象查询来增强分类准确性。首先，模型采用对象感知特征分组策略学习器官查询，以捕获器官级的视觉特征。然后，通过聚焦于基于扩散的视觉提示来细化肿瘤查询，从而实现对未见过的肿瘤的精确分割。此外，我们还引入了基于扩散的特征融合以提高语义分割性能。通过集成CLIP文本嵌入，DSM捕获类别敏感的类以增强语言迁移知识，从而提高模型在多种场景和多标签任务中的鲁棒性。广泛的经验表明，DSM在各种肿瘤分割任务中表现出优越的性能。相关代码可在以下链接获取：this https URL。 

---
# Adaptive Dropout: Unleashing Dropout across Layers for Generalizable Image Super-Resolution 

**Title (ZH)**: 自适应丢弃：跨层释放丢弃以实现泛化图像超分辨率 

**Authors**: Hang Xu, Wei Yu, Jiangtong Tan, Zhen Zou, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12738)  

**Abstract**: Blind Super-Resolution (blind SR) aims to enhance the model's generalization ability with unknown degradation, yet it still encounters severe overfitting issues. Some previous methods inspired by dropout, which enhances generalization by regularizing features, have shown promising results in blind SR. Nevertheless, these methods focus solely on regularizing features before the final layer and overlook the need for generalization in features at intermediate layers. Without explicit regularization of features at intermediate layers, the blind SR network struggles to obtain well-generalized feature representations. However, the key challenge is that directly applying dropout to intermediate layers leads to a significant performance drop, which we attribute to the inconsistency in training-testing and across layers it introduced. Therefore, we propose Adaptive Dropout, a new regularization method for blind SR models, which mitigates the inconsistency and facilitates application across intermediate layers of networks. Specifically, for training-testing inconsistency, we re-design the form of dropout and integrate the features before and after dropout adaptively. For inconsistency in generalization requirements across different layers, we innovatively design an adaptive training strategy to strengthen feature propagation by layer-wise annealing. Experimental results show that our method outperforms all past regularization methods on both synthetic and real-world benchmark datasets, also highly effective in other image restoration tasks. Code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 盲超分辨率（盲SR）旨在通过未知退化来增强模型的泛化能力，但仍面临严重的过拟合问题。一些受dropout启发的方法通过正则化特征来增强泛化能力，在盲SR中显示出有前途的结果。然而，这些方法仅专注于在最后一层之前正则化特征，而忽视了中间层特征也需要泛化的需求。没有中间层特征的显式正则化，盲SR网络难以获得良好的泛化特征表示。然而，关键挑战在于直接在中间层应用dropout会导致显著的性能下降，这归因于训练-测试间以及跨层引入的一致性问题。因此，我们提出了一种新的盲SR模型正则化方法——自适应dropout，该方法减轻了不一致性并促进了在网络中间层的应用。具体而言，对于训练-测试不一致性，我们重新设计了dropout的形式，并在dropout前后适当地整合特征。对于不同层间泛化要求的一致性问题，我们创新设计了一种逐层退火的自适应训练策略以加强特征传播。实验结果表明，我们的方法在合成和真实基准数据集上均优于所有以往的正则化方法，并且在其他图像恢复任务中也很有效。代码可在\href{this https URL}{此链接}获得。 

---
# Revealing the Challenges of Sim-to-Real Transfer in Model-Based Reinforcement Learning via Latent Space Modeling 

**Title (ZH)**: 基于潜空间建模揭示模型导向强化学习中从仿真到现实转移的挑战 

**Authors**: Zhilin Lin, Shiliang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.12735)  

**Abstract**: Reinforcement learning (RL) is playing an increasingly important role in fields such as robotic control and autonomous driving. However, the gap between simulation and the real environment remains a major obstacle to the practical deployment of RL. Agents trained in simulators often struggle to maintain performance when transferred to real-world physical environments. In this paper, we propose a latent space based approach to analyze the impact of simulation on real-world policy improvement in model-based settings. As a natural extension of model-based methods, our approach enables an intuitive observation of the challenges faced by model-based methods in sim-to-real transfer. Experiments conducted in the MuJoCo environment evaluate the performance of our method in both measuring and mitigating the sim-to-real gap. The experiments also highlight the various challenges that remain in overcoming the sim-to-real gap, especially for model-based methods. 

**Abstract (ZH)**: 基于潜在空间的方法在模型导向设置中分析模拟对实际政策改进的影响：MuJoCo环境中的实测与缓解sim-to-real差距的实验 

---
# SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration 

**Title (ZH)**: SP-VLA：一种联合模型调度和token剪枝的VLA模型加速方法 

**Authors**: Ye Li, Yuan Meng, Zewen Sun, Kangye Ji, Chen Tang, Jiajun Fan, Xinzhu Ma, Shutao Xia, Zhi Wang, Wenwu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12723)  

**Abstract**: Vision-Language-Action (VLA) models have attracted increasing attention for their strong control capabilities. However, their high computational cost and low execution frequency hinder their suitability for real-time tasks such as robotic manipulation and autonomous navigation. Existing VLA acceleration methods primarily focus on structural optimization, overlooking the fact that these models operate in sequential decision-making environments. As a result, temporal redundancy in sequential action generation and spatial redundancy in visual input remain unaddressed. To this end, we propose SP-VLA, a unified framework that accelerates VLA models by jointly scheduling models and pruning tokens. Specifically, we design an action-aware model scheduling mechanism that reduces temporal redundancy by dynamically switching between VLA model and a lightweight generator. Inspired by the human motion pattern of focusing on key decision points while relying on intuition for other actions, we categorize VLA actions into deliberative and intuitive, assigning the former to the VLA model and the latter to the lightweight generator, enabling frequency-adaptive execution through collaborative model scheduling. To address spatial redundancy, we further develop a spatio-semantic dual-aware token pruning method. Tokens are classified into spatial and semantic types and pruned based on their dual-aware importance to accelerate VLA inference. These two mechanisms work jointly to guide the VLA in focusing on critical actions and salient visual information, achieving effective acceleration while maintaining high accuracy. Experimental results demonstrate that our method achieves up to 1.5$\times$ acceleration with less than 3% drop in accuracy, outperforming existing approaches in multiple tasks. 

**Abstract (ZH)**: SP-VLA: 一种联合调度与剪枝的视觉-语言-行动模型加速框架 

---
# Serving Large Language Models on Huawei CloudMatrix384 

**Title (ZH)**: 华为云Matrix384上大规模语言模型的服务 

**Authors**: Pengfei Zuo, Huimin Lin, Junbo Deng, Nan Zou, Xingkun Yang, Yingyu Diao, Weifeng Gao, Ke Xu, Zhangyu Chen, Shirui Lu, Zhao Qiu, Peiyang Li, Xianyu Chang, Zhengzhong Yu, Fangzheng Miao, Jia Zheng, Ying Li, Yuan Feng, Bei Wang, Zaijian Zong, Mosong Zhou, Wenli Zhou, Houjiang Chen, Xingyu Liao, Yipeng Li, Wenxiao Zhang, Ping Zhu, Yinggang Wang, Chuanjie Xiao, Depeng Liang, Dong Cao, Juncheng Liu, Yongqiang Yang, Xiaolong Bai, Yi Li, Huaguo Xie, Huatao Wu, Zhibin Yu, Lv Chen, Hu Liu, Yujun Ding, Haipei Zhu, Jing Xia, Yi Xiong, Zhou Yu, Heng Liao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12708)  

**Abstract**: The rapid evolution of large language models (LLMs), driven by growing parameter scales, adoption of mixture-of-experts (MoE) architectures, and expanding context lengths, imposes unprecedented demands on AI infrastructure. Traditional AI clusters face limitations in compute intensity, memory bandwidth, inter-chip communication, and latency, compounded by variable workloads and strict service-level objectives. Addressing these issues requires fundamentally redesigned hardware-software integration. This paper introduces Huawei CloudMatrix, a next-generation AI datacenter architecture, realized in the production-grade CloudMatrix384 supernode. It integrates 384 Ascend 910C NPUs and 192 Kunpeng CPUs interconnected via an ultra-high-bandwidth Unified Bus (UB) network, enabling direct all-to-all communication and dynamic pooling of resources. These features optimize performance for communication-intensive operations, such as large-scale MoE expert parallelism and distributed key-value cache access. To fully leverage CloudMatrix384, we propose CloudMatrix-Infer, an advanced LLM serving solution incorporating three core innovations: a peer-to-peer serving architecture that independently scales prefill, decode, and caching; a large-scale expert parallelism strategy supporting EP320 via efficient UB-based token dispatch; and hardware-aware optimizations including specialized operators, microbatch-based pipelining, and INT8 quantization. Evaluation with the DeepSeek-R1 model shows CloudMatrix-Infer achieves state-of-the-art efficiency: prefill throughput of 6,688 tokens/s per NPU and decode throughput of 1,943 tokens/s per NPU (<50 ms TPOT). It effectively balances throughput and latency, sustaining 538 tokens/s even under stringent 15 ms latency constraints, while INT8 quantization maintains model accuracy across benchmarks. 

**Abstract (ZH)**: 华为云矩阵：下一代AI数据中心架构及其在大语言模型服务中的高级解决方案 

---
# NAP-Tuning: Neural Augmented Prompt Tuning for Adversarially Robust Vision-Language Models 

**Title (ZH)**: NAP调优：神经增强提示调优以提高对抗 robust 的视觉语言模型性能 

**Authors**: Jiaming Zhang, Xin Wang, Xingjun Ma, Lingyu Qiu, Yu-Gang Jiang, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12706)  

**Abstract**: Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capabilities in understanding relationships between visual and textual data through joint embedding spaces. Despite their effectiveness, these models remain vulnerable to adversarial attacks, particularly in the image modality, posing significant security concerns. Building upon our previous work on Adversarial Prompt Tuning (AdvPT), which introduced learnable text prompts to enhance adversarial robustness in VLMs without extensive parameter training, we present a significant extension by introducing the Neural Augmentor framework for Multi-modal Adversarial Prompt Tuning (NAP-Tuning).Our key innovations include: (1) extending AdvPT from text-only to multi-modal prompting across both text and visual modalities, (2) expanding from single-layer to multi-layer prompt architectures, and (3) proposing a novel architecture-level redesign through our Neural Augmentor approach, which implements feature purification to directly address the distortions introduced by adversarial attacks in feature space. Our NAP-Tuning approach incorporates token refiners that learn to reconstruct purified features through residual connections, allowing for modality-specific and layer-specific feature this http URL experiments demonstrate that NAP-Tuning significantly outperforms existing methods across various datasets and attack types. Notably, our approach shows significant improvements over the strongest baselines under the challenging AutoAttack benchmark, outperforming them by 33.5% on ViT-B16 and 33.0% on ViT-B32 architectures while maintaining competitive clean accuracy. 

**Abstract (ZH)**: Vision-语言模型多模态对抗提示调优框架（NAP-Tuning） 

---
# Flexible Realignment of Language Models 

**Title (ZH)**: 灵活的语言模型重新对齐 

**Authors**: Wenhong Zhu, Ruobing Xie, Weinan Zhang, Rui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12704)  

**Abstract**: Realignment becomes necessary when a language model (LM) fails to meet expected performance. We propose a flexible realignment framework that supports quantitative control of alignment degree during training and inference. This framework incorporates Training-time Realignment (TrRa), which efficiently realigns the reference model by leveraging the controllable fusion of logits from both the reference and already aligned models. For example, TrRa reduces token usage by 54.63% on DeepSeek-R1-Distill-Qwen-1.5B without any performance degradation, outperforming DeepScaleR-1.5B's 33.86%. To complement TrRa during inference, we introduce a layer adapter that enables smooth Inference-time Realignment (InRa). This adapter is initialized to perform an identity transformation at the bottom layer and is inserted preceding the original layers. During inference, input embeddings are simultaneously processed by the adapter and the original layer, followed by the remaining layers, and then controllably interpolated at the logit level. We upgraded DeepSeek-R1-Distill-Qwen-7B from a slow-thinking model to one that supports both fast and slow thinking, allowing flexible alignment control even during inference. By encouraging deeper reasoning, it even surpassed its original performance. 

**Abstract (ZH)**: 一种支持训练和推理时可控制度对齐的灵活重对齐框架 

---
# Unsupervised Contrastive Learning Using Out-Of-Distribution Data for Long-Tailed Dataset 

**Title (ZH)**: 使用域外数据的无监督对比学习用于长尾数据集 

**Authors**: Cuong Manh Hoang, Yeejin Lee, Byeongkeun Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12698)  

**Abstract**: This work addresses the task of self-supervised learning (SSL) on a long-tailed dataset that aims to learn balanced and well-separated representations for downstream tasks such as image classification. This task is crucial because the real world contains numerous object categories, and their distributions are inherently imbalanced. Towards robust SSL on a class-imbalanced dataset, we investigate leveraging a network trained using unlabeled out-of-distribution (OOD) data that are prevalently available online. We first train a network using both in-domain (ID) and sampled OOD data by back-propagating the proposed pseudo semantic discrimination loss alongside a domain discrimination loss. The OOD data sampling and loss functions are designed to learn a balanced and well-separated embedding space. Subsequently, we further optimize the network on ID data by unsupervised contrastive learning while using the previously trained network as a guiding network. The guiding network is utilized to select positive/negative samples and to control the strengths of attractive/repulsive forces in contrastive learning. We also distil and transfer its embedding space to the training network to maintain balancedness and separability. Through experiments on four publicly available long-tailed datasets, we demonstrate that the proposed method outperforms previous state-of-the-art methods. 

**Abstract (ZH)**: 本研究解决了针对长尾数据集的自监督学习任务，旨在为诸如图像分类等下游任务学习平衡且分离良好的表示。为实现此任务，我们研究了利用大量在线获取的未标记异类数据（OOD）进行网络训练的方法。首先，我们通过同时反向传播所提出的伪语义鉴别损失与领域鉴别损失，使用领域内（ID）数据和采样的OOD数据来训练网络。OOD数据的采样和损失函数旨在学习一个平衡且分离良好的嵌入空间。随后，我们进一步通过无监督对比学习在ID数据上优化网络，并使用预先训练的网络作为引导网络。引导网络用于选择正/负样本，并控制对比学习中吸引力/排斥力的强度。我们还将其嵌入空间进行知识蒸馏和迁移，以保持平衡性和可分离性。通过在四个公开的长尾数据集上的实验，我们证明了所提出的方法优于先前的最先进的方法。 

---
# MGDFIS: Multi-scale Global-detail Feature Integration Strategy for Small Object Detection 

**Title (ZH)**: 多尺度全局细节特征整合策略用于小目标检测 

**Authors**: Yuxiang Wang, Xuecheng Bai, Boyu Hu, Chuanzhi Xu, Haodong Chen, Vera Chung, Tingxue Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12697)  

**Abstract**: Small object detection in UAV imagery is crucial for applications such as search-and-rescue, traffic monitoring, and environmental surveillance, but it is hampered by tiny object size, low signal-to-noise ratios, and limited feature extraction. Existing multi-scale fusion methods help, but add computational burden and blur fine details, making small object detection in cluttered scenes difficult. To overcome these challenges, we propose the Multi-scale Global-detail Feature Integration Strategy (MGDFIS), a unified fusion framework that tightly couples global context with local detail to boost detection performance while maintaining efficiency. MGDFIS comprises three synergistic modules: the FusionLock-TSS Attention Module, which marries token-statistics self-attention with DynamicTanh normalization to highlight spectral and spatial cues at minimal cost; the Global-detail Integration Module, which fuses multi-scale context via directional convolution and parallel attention while preserving subtle shape and texture variations; and the Dynamic Pixel Attention Module, which generates pixel-wise weighting maps to rebalance uneven foreground and background distributions and sharpen responses to true object regions. Extensive experiments on the VisDrone benchmark demonstrate that MGDFIS consistently outperforms state-of-the-art methods across diverse backbone architectures and detection frameworks, achieving superior precision and recall with low inference time. By striking an optimal balance between accuracy and resource usage, MGDFIS provides a practical solution for small-object detection on resource-constrained UAV platforms. 

**Abstract (ZH)**: 多尺度全局细节特征整合策略（MGDFIS）：兼顾效率与性能的统一融合框架 

---
# Get on the Train or be Left on the Station: Using LLMs for Software Engineering Research 

**Title (ZH)**: 上车或被留下：使用大语言模型进行软件工程研究 

**Authors**: Bianca Trinkenreich, Fabio Calefato, Geir Hanssen, Kelly Blincoe, Marcos Kalinowski, Mauro Pezzè, Paolo Tell, Margaret-Anne Storey  

**Link**: [PDF](https://arxiv.org/pdf/2506.12691)  

**Abstract**: The adoption of Large Language Models (LLMs) is not only transforming software engineering (SE) practice but is also poised to fundamentally disrupt how research is conducted in the field. While perspectives on this transformation range from viewing LLMs as mere productivity tools to considering them revolutionary forces, we argue that the SE research community must proactively engage with and shape the integration of LLMs into research practices, emphasizing human agency in this transformation. As LLMs rapidly become integral to SE research - both as tools that support investigations and as subjects of study - a human-centric perspective is essential. Ensuring human oversight and interpretability is necessary for upholding scientific rigor, fostering ethical responsibility, and driving advancements in the field. Drawing from discussions at the 2nd Copenhagen Symposium on Human-Centered AI in SE, this position paper employs McLuhan's Tetrad of Media Laws to analyze the impact of LLMs on SE research. Through this theoretical lens, we examine how LLMs enhance research capabilities through accelerated ideation and automated processes, make some traditional research practices obsolete, retrieve valuable aspects of historical research approaches, and risk reversal effects when taken to extremes. Our analysis reveals opportunities for innovation and potential pitfalls that require careful consideration. We conclude with a call to action for the SE research community to proactively harness the benefits of LLMs while developing frameworks and guidelines to mitigate their risks, to ensure continued rigor and impact of research in an AI-augmented future. 

**Abstract (ZH)**: 大型语言模型的采纳不仅正在重塑软件工程实践，还准备从根本上颠覆该领域的研究方式。尽管对这一转变的观点从将其视为简单的生产工具到认为它们是革命性力量不一而足，我们主张软件工程研究社区必须积极应对并塑造大型语言模型在研究实践中的整合，强调人类在这一转变中的作用。随着大型语言模型迅速成为软件工程研究不可或缺的一部分——无论是作为支持研究的工具还是作为研究对象——以人为本的视角至关重要。确保人类监督和可解释性对于维护科学研究的严谨性、促进伦理责任以及推动该领域的发展是必要的。基于在第二届哥本哈根软件工程人性化AI研讨会上的讨论，本文通过麦卢汉的媒介四律来分析大型语言模型对软件工程研究的影响。通过这一理论视角，我们探讨了大型语言模型如何通过加速创意生成和自动化流程增强研究能力，使一些传统研究实践变得过时，恢复历史研究方法中的宝贵方面，并在极端情况下带来风险逆转效应。我们的分析揭示了创新的机会和需要慎重考虑的潜在陷阱。我们呼吁软件工程研究社区积极利用大型语言模型的优势，同时制定框架和指南以减轻其风险，以确保在人工智能增强的未来，研究保持严谨性和影响力。 

---
# Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity 

**Title (ZH)**: 字母索引映射：通过语义差异打破大型语言模型的限制 

**Authors**: Bilal Saleh Husain  

**Link**: [PDF](https://arxiv.org/pdf/2506.12685)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了卓越的能力，但它们对对抗性攻击的易感性，尤其是 Jailbreak 攻击，提出了显著的安全和伦理问题。尽管存在许多 Jailbreak 方法，但许多方法遭受计算开销大、高 token 使用量或复杂的解码方案等问题。刘等（2024）引入了 FlipAttack，这是一种黑盒方法，通过简单的提示操纵实现了高攻击成功率（ASR）。本文通过分析 FlipAttack 的翻转模式引起的意义变化，研究其有效性的潜在机制。我们假设原始提示和操纵后提示之间意义差异与 ASR 成反比。为了测试这一假设，我们分析了 FlipAttack 模式的嵌入空间可视化（UMAP、KDE）和余弦相似度。此外，我们引入了一种新的对抗攻击方法，即 Alphabet Index Mapping (AIM)，旨在最大化意义差异同时保持简单的可解性。在 AdvBench 部分数据集上对 GPT-4 进行的实验显示，AIM 及其变体 AIM+FWO 达到了 94% 的 ASR，优于 FlipAttack 及其他方法。我们的研究结果表明，虽然高意义差异至关重要，但与解码简单性的平衡对于成功的 Jailbreak 至关重要。本研究加深了对抗性提示机理的理解，并提供了一种新的有效 Jailbreak 技术。 

---
# ANIRA: An Architecture for Neural Network Inference in Real-Time Audio Applications 

**Title (ZH)**: ANIRA: 用于实时音频应用的神经网络推理架构 

**Authors**: Valentin Ackva, Fares Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12665)  

**Abstract**: Numerous tools for neural network inference are currently available, yet many do not meet the requirements of real-time audio applications. In response, we introduce anira, an efficient cross-platform library. To ensure compatibility with a broad range of neural network architectures and frameworks, anira supports ONNX Runtime, LibTorch, and TensorFlow Lite as backends. Each inference engine exhibits real-time violations, which anira mitigates by decoupling the inference from the audio callback to a static thread pool. The library incorporates built-in latency management and extensive benchmarking capabilities, both crucial to ensure a continuous signal flow. Three different neural network architectures for audio effect emulation are then subjected to benchmarking across various configurations. Statistical modeling is employed to identify the influence of various factors on performance. The findings indicate that for stateless models, ONNX Runtime exhibits the lowest runtimes. For stateful models, LibTorch demonstrates the fastest performance. Our results also indicate that for certain model-engine combinations, the initial inferences take longer, particularly when these inferences exhibit a higher incidence of real-time violations. 

**Abstract (ZH)**: 面向实时音频应用的高效跨平台神经网络推理库anira及其性能评估 

---
# DR-SAC: Distributionally Robust Soft Actor-Critic for Reinforcement Learning under Uncertainty 

**Title (ZH)**: DR-SAC: 分布鲁棒软Actor-critic方法在不确定性下的强化学习 

**Authors**: Mingxuan Cui, Duo Zhou, Yuxuan Han, Grani A. Hanasusanto, Qiong Wang, Huan Zhang, Zhengyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12622)  

**Abstract**: Deep reinforcement learning (RL) has achieved significant success, yet its application in real-world scenarios is often hindered by a lack of robustness to environmental uncertainties. To solve this challenge, some robust RL algorithms have been proposed, but most are limited to tabular settings. In this work, we propose Distributionally Robust Soft Actor-Critic (DR-SAC), a novel algorithm designed to enhance the robustness of the state-of-the-art Soft Actor-Critic (SAC) algorithm. DR-SAC aims to maximize the expected value with entropy against the worst possible transition model lying in an uncertainty set. A distributionally robust version of the soft policy iteration is derived with a convergence guarantee. For settings where nominal distributions are unknown, such as offline RL, a generative modeling approach is proposed to estimate the required nominal distributions from data. Furthermore, experimental results on a range of continuous control benchmark tasks demonstrate our algorithm achieves up to $9.8$ times the average reward of the SAC baseline under common perturbations. Additionally, compared with existing robust reinforcement learning algorithms, DR-SAC significantly improves computing efficiency and applicability to large-scale problems. 

**Abstract (ZH)**: 分布鲁棒软演员-评论家（DR-SAC）算法：一种增强软演员-评论家算法鲁棒性的新方法 

---
# Konooz: Multi-domain Multi-dialect Corpus for Named Entity Recognition 

**Title (ZH)**: Konooz: 多领域多方言语料库命名实体识别 

**Authors**: Nagham Hamad, Mohammed Khalilia, Mustafa Jarrar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12615)  

**Abstract**: We introduce Konooz, a novel multi-dimensional corpus covering 16 Arabic dialects across 10 domains, resulting in 160 distinct corpora. The corpus comprises about 777k tokens, carefully collected and manually annotated with 21 entity types using both nested and flat annotation schemes - using the Wojood guidelines. While Konooz is useful for various NLP tasks like domain adaptation and transfer learning, this paper primarily focuses on benchmarking existing Arabic Named Entity Recognition (NER) models, especially cross-domain and cross-dialect model performance. Our benchmarking of four Arabic NER models using Konooz reveals a significant drop in performance of up to 38% when compared to the in-distribution data. Furthermore, we present an in-depth analysis of domain and dialect divergence and the impact of resource scarcity. We also measured the overlap between domains and dialects using the Maximum Mean Discrepancy (MMD) metric, and illustrated why certain NER models perform better on specific dialects and domains. Konooz is open-source and publicly available at this https URL 

**Abstract (ZH)**: Konooz：一种涵盖16种阿拉伯方言的多维度语料库及其在阿拉伯命名实体识别模型评估中的应用 

---
# An Exploration of Mamba for Speech Self-Supervised Models 

**Title (ZH)**: Mamba在语音自监督模型中的探索 

**Authors**: Tzu-Quan Lin, Heng-Cheng Kuo, Tzu-Chieh Wei, Hsi-Chun Cheng, Chun-Wei Chen, Hsien-Fu Hsiao, Yu Tsao, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.12606)  

**Abstract**: While Mamba has demonstrated strong performance in language modeling, its potential as a speech self-supervised (SSL) model remains underexplored, with prior studies limited to isolated tasks. To address this, we explore Mamba-based HuBERT models as alternatives to Transformer-based SSL architectures. Leveraging the linear-time Selective State Space, these models enable fine-tuning on long-context ASR with significantly lower compute. Moreover, they show superior performance when fine-tuned for streaming ASR. Beyond fine-tuning, these models show competitive performance on SUPERB probing benchmarks, particularly in causal settings. Our analysis shows that they yield higher-quality quantized representations and capture speaker-related features more distinctly than Transformer-based models. These findings highlight Mamba-based SSL as a promising and complementary direction for long-sequence modeling, real-time speech modeling, and speech unit extraction. 

**Abstract (ZH)**: Mamba在语音自监督学习中的潜力及其在长期序列建模、实时语音建模和语音单元提取中的应用探索 

---
# Trust-MARL: Trust-Based Multi-Agent Reinforcement Learning Framework for Cooperative On-Ramp Merging Control in Heterogeneous Traffic Flow 

**Title (ZH)**: 基于信任的多智能体强化学习框架：异质交通流中合作式入口匝道并线控制的信任机制 

**Authors**: Jie Pan, Tianyi Wang, Christian Claudel, Jing Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12600)  

**Abstract**: Intelligent transportation systems require connected and automated vehicles (CAVs) to conduct safe and efficient cooperation with human-driven vehicles (HVs) in complex real-world traffic environments. However, the inherent unpredictability of human behaviour, especially at bottlenecks such as highway on-ramp merging areas, often disrupts traffic flow and compromises system performance. To address the challenge of cooperative on-ramp merging in heterogeneous traffic environments, this study proposes a trust-based multi-agent reinforcement learning (Trust-MARL) framework. At the macro level, Trust-MARL enhances global traffic efficiency by leveraging inter-agent trust to improve bottleneck throughput and mitigate traffic shockwave through emergent group-level coordination. At the micro level, a dynamic trust mechanism is designed to enable CAVs to adjust their cooperative strategies in response to real-time behaviors and historical interactions with both HVs and other CAVs. Furthermore, a trust-triggered game-theoretic decision-making module is integrated to guide each CAV in adapting its cooperation factor and executing context-aware lane-changing decisions under safety, comfort, and efficiency constraints. An extensive set of ablation studies and comparative experiments validates the effectiveness of the proposed Trust-MARL approach, demonstrating significant improvements in safety, efficiency, comfort, and adaptability across varying CAV penetration rates and traffic densities. 

**Abstract (ZH)**: 智能交通系统需要连接和自动化车辆（CAVs）在复杂的真实世界交通环境中与人类驾驶车辆（HVs）进行安全和高效的协同合作。然而，人类行为的固有不可预测性，特别是在高速公路入口匝道合流区等瓶颈区域，常常扰乱交通流并损害系统性能。为应对异构交通环境中合流协作的挑战，本研究提出了一种基于信任的多智能体 reinforcement 学习（Trust-MARL）框架。在宏观层面，Trust-MARL 通过利用智能体间的信任增强全局交通效率，改善瓶颈处的通行能力和缓解交通波，实现群体级别的协调。在微观层面，设计了一种动态信任机制，使CAVs能够根据与HV和其他CAVs的实时行为和历史交互调整其协同策略。此外，集成了一个基于信任的游戏理论决策模块，指导每个CAV在安全、舒适和效率约束条件下调整其协作因子并执行情境感知的变道决策。通过大量的消融研究和对比实验，验证了所提出的Trust-MARL方法的有效性，展示了在不同CAV渗透率和交通密度下显著提高的安全性、效率、舒适性和适应性。 

---
# Enabling Precise Topic Alignment in Large Language Models Via Sparse Autoencoders 

**Title (ZH)**: 通过稀疏自编码器实现大规模语言模型中的精确主题对齐 

**Authors**: Ananya Joshi, Celia Cintas, Skyler Speakman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12576)  

**Abstract**: Recent work shows that Sparse Autoencoders (SAE) applied to large language model (LLM) layers have neurons corresponding to interpretable concepts. These SAE neurons can be modified to align generated outputs, but only towards pre-identified topics and with some parameter tuning. Our approach leverages the observational and modification properties of SAEs to enable alignment for any topic. This method 1) scores each SAE neuron by its semantic similarity to an alignment text and uses them to 2) modify SAE-layer-level outputs by emphasizing topic-aligned neurons. We assess the alignment capabilities of this approach on diverse public topic datasets including Amazon reviews, Medicine, and Sycophancy, across the currently available open-source LLMs and SAE pairs (GPT2 and Gemma) with multiple SAEs configurations. Experiments aligning to medical prompts reveal several benefits over fine-tuning, including increased average language acceptability (0.25 vs. 0.5), reduced training time across multiple alignment topics (333.6s vs. 62s), and acceptable inference time for many applications (+0.00092s/token). Our open-source code is available at this http URL. 

**Abstract (ZH)**: 近期研究表明，应用在大型语言模型层上的稀疏自编码器（SAE）具有与可解释概念对应的神经元。这些SAE神经元可以被修改以对生成输出进行对齐，但仅限于预先识别的主题，并需进行一些参数调整。本方法利用SAE的观测和修改特性，使得对齐适用于任何主题。该方法1) 通过语义相似性对每个SAE神经元进行评分，并利用这些评分2) 在SAE层级输出中强调对齐主题的神经元。我们通过当前可用的开源大型语言模型（LLM）和SAE配对（GPT2和Gemma）对多种公共主题数据集（包括亚马逊评论、医学和奉承）进行评估，以检验该方法的对齐能力。实验结果显示，与微调相比，该方法在多项对齐主题上具有优势，包括较高的平均语言可接受性（0.25 vs. 0.5）、减少的训练时间（333.6秒 vs. 62秒），以及许多应用场景中可接受的推理时间（+0.00092秒/词）。我们的开源代码可在此网址获得。 

---
# DoTA-RAG: Dynamic of Thought Aggregation RAG 

**Title (ZH)**: DoTA-RAG: 思维聚合RAG的动态过程 

**Authors**: Saksorn Ruangtanusak, Natthapath Rungseesiripak, Peerawat Rojratchadakorn, Monthol Charattrakool, Natapong Nitarach  

**Link**: [PDF](https://arxiv.org/pdf/2506.12571)  

**Abstract**: In this paper, we introduce DoTA-RAG (Dynamic-of-Thought Aggregation RAG), a retrieval-augmented generation system optimized for high-throughput, large-scale web knowledge indexes. Traditional RAG pipelines often suffer from high latency and limited accuracy over massive, diverse datasets. DoTA-RAG addresses these challenges with a three-stage pipeline: query rewriting, dynamic routing to specialized sub-indexes, and multi-stage retrieval and ranking. We further enhance retrieval by evaluating and selecting a superior embedding model, re-embedding the large FineWeb-10BT corpus. Moreover, we create a diverse Q&A dataset of 500 questions generated via the DataMorgana setup across a broad range of WebOrganizer topics and formats. DoTA-RAG improves the answer correctness score from 0.752 (baseline, using LiveRAG pre-built vector store) to 1.478 while maintaining low latency, and it achieves a 0.929 correctness score on the Live Challenge Day. These results highlight DoTA-RAG's potential for practical deployment in domains requiring fast, reliable access to large and evolving knowledge sources. 

**Abstract (ZH)**: DoTA-RAG：动态思维聚合RAG——一种优化的高通量大规模网络知识索引生成系统 

---
# MVP-CBM:Multi-layer Visual Preference-enhanced Concept Bottleneck Model for Explainable Medical Image Classification 

**Title (ZH)**: MVP-CBM：多层视觉偏好增强概念瓶颈模型可解释的医疗图像分类 

**Authors**: Chunjiang Wang, Kun Zhang, Yandong Liu, Zhiyang He, Xiaodong Tao, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12568)  

**Abstract**: The concept bottleneck model (CBM), as a technique improving interpretability via linking predictions to human-understandable concepts, makes high-risk and life-critical medical image classification credible. Typically, existing CBM methods associate the final layer of visual encoders with concepts to explain the model's predictions. However, we empirically discover the phenomenon of concept preference variation, that is, the concepts are preferably associated with the features at different layers than those only at the final layer; yet a blind last-layer-based association neglects such a preference variation and thus weakens the accurate correspondences between features and concepts, impairing model interpretability. To address this issue, we propose a novel Multi-layer Visual Preference-enhanced Concept Bottleneck Model (MVP-CBM), which comprises two key novel modules: (1) intra-layer concept preference modeling, which captures the preferred association of different concepts with features at various visual layers, and (2) multi-layer concept sparse activation fusion, which sparsely aggregates concept activations from multiple layers to enhance performance. Thus, by explicitly modeling concept preferences, MVP-CBM can comprehensively leverage multi-layer visual information to provide a more nuanced and accurate explanation of model decisions. Extensive experiments on several public medical classification benchmarks demonstrate that MVP-CBM achieves state-of-the-art accuracy and interoperability, verifying its superiority. Code is available at this https URL. 

**Abstract (ZH)**: 多层视觉偏好增强概念瓶颈模型（MVP-CBM）：一种提高医疗图像分类解释性的方法 

---
# Fairness Research For Machine Learning Should Integrate Societal Considerations 

**Title (ZH)**: 机器学习的公平性研究应融入社会考量 

**Authors**: Yijun Bian, Lei You  

**Link**: [PDF](https://arxiv.org/pdf/2506.12556)  

**Abstract**: Enhancing fairness in machine learning (ML) systems is increasingly important nowadays. While current research focuses on assistant tools for ML pipelines to promote fairness within them, we argue that: 1) The significance of properly defined fairness measures remains underestimated; and 2) Fairness research in ML should integrate societal considerations. The reasons include that detecting discrimination is critical due to the widespread deployment of ML systems and that human-AI feedback loops amplify biases, even when only small social and political biases persist. 

**Abstract (ZH)**: 增强机器学习系统中的公平性日益重要：重新审视公平度量的社会考量与偏见放大效应 

---
# Neuromorphic Online Clustering and Its Application to Spike Sorting 

**Title (ZH)**: 神经形态在线聚类及其在尖峰排序中的应用 

**Authors**: James E. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2506.12555)  

**Abstract**: Active dendrites are the basis for biologically plausible neural networks possessing many desirable features of the biological brain including flexibility, dynamic adaptability, and energy efficiency. A formulation for active dendrites using the notational language of conventional machine learning is put forward as an alternative to a spiking neuron formulation. Based on this formulation, neuromorphic dendrites are developed as basic neural building blocks capable of dynamic online clustering. Features and capabilities of neuromorphic dendrites are demonstrated via a benchmark drawn from experimental neuroscience: spike sorting. Spike sorting takes inputs from electrical probes implanted in neural tissue, detects voltage spikes (action potentials) emitted by neurons, and attempts to sort the spikes according to the neuron that emitted them. Many spike sorting methods form clusters based on the shapes of action potential waveforms, under the assumption that spikes emitted by a given neuron have similar shapes and will therefore map to the same cluster. Using a stream of synthetic spike shapes, the accuracy of the proposed dendrite is compared with the more compute-intensive, offline k-means clustering approach. Overall, the dendrite outperforms k-means and has the advantage of requiring only a single pass through the input stream, learning as it goes. The capabilities of the neuromorphic dendrite are demonstrated for a number of scenarios including dynamic changes in the input stream, differing neuron spike rates, and varying neuron counts. 

**Abstract (ZH)**: 活性树突是实现具备生物脑多项 desirable 特征（包括灵活性、动态适应性和能量效率）的生物合现实神经网络的基础。提出了一种使用传统机器学习符号语言的形式化方法来替代脉冲神经元形式化方法，以活性树突为基础开发了神经形态树突，作为基本的神经元构建块，具备动态在线聚类功能。通过实验神经科学中的基准测试——尖峰分类，展示了神经形态树突的特性和能力。尖峰分类从植入神经组织的电极探针接收输入，检测由神经元发出的电压尖峰（动作电位），并尝试按尖峰发出的神经元对其进行分类。许多尖峰分类方法基于动作电位波形的形状形成聚类，假设来自同一神经元的尖峰会具有相似的形状并因此分配到相同的聚类。使用尖峰形状的合成流，比较了提出树突的准确性与计算密集型的离线 k-均值聚类方法。总体而言，树突表现更优，并且具有仅需一次通过输入流即可学习的优点。展示了神经形态树突在包括输入流动态变化、不同的神经元尖峰速率和变化的神经元数量等场景下的能力。 

---
# Profiling News Media for Factuality and Bias Using LLMs and the Fact-Checking Methodology of Human Experts 

**Title (ZH)**: 使用大语言模型和人类专家事实核查方法 profiling 媒体的事实性和偏见 

**Authors**: Zain Muhammad Mujahid, Dilshod Azizov, Maha Tufail Agro, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12552)  

**Abstract**: In an age characterized by the proliferation of mis- and disinformation online, it is critical to empower readers to understand the content they are reading. Important efforts in this direction rely on manual or automatic fact-checking, which can be challenging for emerging claims with limited information. Such scenarios can be handled by assessing the reliability and the political bias of the source of the claim, i.e., characterizing entire news outlets rather than individual claims or articles. This is an important but understudied research direction. While prior work has looked into linguistic and social contexts, we do not analyze individual articles or information in social media. Instead, we propose a novel methodology that emulates the criteria that professional fact-checkers use to assess the factuality and political bias of an entire outlet. Specifically, we design a variety of prompts based on these criteria and elicit responses from large language models (LLMs), which we aggregate to make predictions. In addition to demonstrating sizable improvements over strong baselines via extensive experiments with multiple LLMs, we provide an in-depth error analysis of the effect of media popularity and region on model performance. Further, we conduct an ablation study to highlight the key components of our dataset that contribute to these improvements. To facilitate future research, we released our dataset and code at this https URL. 

**Abstract (ZH)**: 在网络信息传播泛滥的时代，赋能读者理解他们所阅读的内容至关重要。这一方向的重要努力依赖于手动或自动事实核查，对于信息有限的新兴声明而言，这可能是具有挑战性的。这类情景可以通过评估声明来源的可靠性和政治偏见来处理，即对整个新闻机构进行characterization，而非个别声明或文章。这是一个重要但研究不足的研究方向。虽然先前的工作已经考虑了语言和社会背景，但我们并未分析社交媒体中的个别文章或信息。相反，我们提出了一个新颖的方法论，模拟专业事实核查人员评估整个新闻机构的事实性和政治偏见的标准。具体而言，我们基于这些标准设计了多种提示，并从大规模语言模型（LLMs）中获得响应，对其进行汇总以做出预测。此外，通过大量实验与多种LLMs对比，我们展示了显著的改进，并提供了对媒介流行度和区域对模型性能影响的详细错误分析。此外，我们进行了消融研究，以突出数据集中对这些改进起关键作用的组件。为了促进未来的研究，我们在以下网址发布了我们的数据集和代码：this https URL。 

---
# MEraser: An Effective Fingerprint Erasure Approach for Large Language Models 

**Title (ZH)**: MEraser: 大型语言模型中有效指纹擦除的方法 

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.12551)  

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域中的应用日益普及，引发了关于模型所有权和知识产权保护的关键性关注。尽管基于后门的指纹识别已成为模型认证的一种有前景的解决方案，但有效移除这些指纹的攻击方法仍未得到充分探索。因此，我们提出了Mismatched Eraser (MEraser)，这是一种用于从LLMs中有效移除基于后门的指纹同时保持模型性能的新方法。我们的方法利用两阶段微调策略，并利用精心构建的不匹配和干净的数据集。通过在多个LLM架构和指纹识别方法上进行广泛评估，我们证明MEraser可以实现完全移除指纹同时仅使用少于1,000个样本的少量训练数据保持模型性能。此外，我们引入了一种可转移的擦除机制，可以在不同模型之间有效移除指纹而不必重复训练。综上所述，我们的方法为LLMs中的指纹移除提供了一种实用的解决方案，揭示了当前指纹识别技术的关键性漏洞，并建立了开发更可靠的模型保护方法的全面评估基准。 

---
# PLD: A Choice-Theoretic List-Wise Knowledge Distillation 

**Title (ZH)**: PLD：一种基于选择理论的列表型知识蒸馏 

**Authors**: Ejafa Bassam, Dawei Zhu, Kaigui Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.12542)  

**Abstract**: Knowledge distillation is a model compression technique in which a compact "student" network is trained to replicate the predictive behavior of a larger "teacher" network. In logit-based knowledge distillation it has become the de facto approach to augment cross-entropy with a distillation term. Typically this term is either a KL divergence-matching marginal probabilities or a correlation-based loss capturing intra- and inter-class relationships but in every case it sits as an add-on to cross-entropy with its own weight that must be carefully tuned. In this paper we adopt a choice-theoretic perspective and recast knowledge distillation under the Plackett-Luce model by interpreting teacher logits as "worth" scores. We introduce Plackett-Luce Distillation (PLD), a weighted list-wise ranking loss in which the teacher model transfers knowledge of its full ranking of classes, weighting each ranked choice by its own confidence. PLD directly optimizes a single teacher-optimal ranking of the true label first, followed by the remaining classes in descending teacher confidence, yielding a convex, translation-invariant surrogate that subsumes weighted cross-entropy. Empirically on standard image classification benchmarks, PLD improves Top-1 accuracy by an average of +0.42% over DIST (arXiv:2205.10536) and +1.04% over KD (arXiv:1503.02531) in homogeneous settings and by +0.48% and +1.09% over DIST and KD, respectively, in heterogeneous settings. 

**Abstract (ZH)**: 基于Plackett-Luce模型的知识蒸馏：一种带有权重的一致排名损失 

---
# BSA: Ball Sparse Attention for Large-scale Geometries 

**Title (ZH)**: BSA: 球稀疏注意力机制在大规模几何结构中的应用 

**Authors**: Catalin E. Brita, Hieu Nguyen, Lohithsai Yadala Chanchu, Domonkos Nagy, Maksim Zhdanov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12541)  

**Abstract**: Self-attention scales quadratically with input size, limiting its use for large-scale physical systems. Although sparse attention mechanisms provide a viable alternative, they are primarily designed for regular structures such as text or images, making them inapplicable for irregular geometries. In this work, we present Ball Sparse Attention (BSA), which adapts Native Sparse Attention (NSA) (Yuan et al., 2025) to unordered point sets by imposing regularity using the Ball Tree structure from the Erwin Transformer (Zhdanov et al., 2025). We modify NSA's components to work with ball-based neighborhoods, yielding a global receptive field at sub-quadratic cost. On an airflow pressure prediction task, we achieve accuracy comparable to Full Attention while significantly reducing the theoretical computational complexity. Our implementation is available at this https URL. 

**Abstract (ZH)**: 球稀疏注意机制（BSA）：将原生稀疏注意机制（NSA）应用于无序点集（Yuan et al., 2025） 

---
# RealFactBench: A Benchmark for Evaluating Large Language Models in Real-World Fact-Checking 

**Title (ZH)**: RealFactBench: 一个评估大型语言模型在实际事实核查中表现的基准测试 

**Authors**: Shuo Yang, Yuqin Dai, Guoqing Wang, Xinran Zheng, Jinfeng Xu, Jinze Li, Zhenzhe Ying, Weiqiang Wang, Edith C.H. Ngai  

**Link**: [PDF](https://arxiv.org/pdf/2506.12538)  

**Abstract**: Large Language Models (LLMs) hold significant potential for advancing fact-checking by leveraging their capabilities in reasoning, evidence retrieval, and explanation generation. However, existing benchmarks fail to comprehensively evaluate LLMs and Multimodal Large Language Models (MLLMs) in realistic misinformation scenarios. To bridge this gap, we introduce RealFactBench, a comprehensive benchmark designed to assess the fact-checking capabilities of LLMs and MLLMs across diverse real-world tasks, including Knowledge Validation, Rumor Detection, and Event Verification. RealFactBench consists of 6K high-quality claims drawn from authoritative sources, encompassing multimodal content and diverse domains. Our evaluation framework further introduces the Unknown Rate (UnR) metric, enabling a more nuanced assessment of models' ability to handle uncertainty and balance between over-conservatism and over-confidence. Extensive experiments on 7 representative LLMs and 4 MLLMs reveal their limitations in real-world fact-checking and offer valuable insights for further research. RealFactBench is publicly available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在利用其推理、证据检索和解释生成能力进行事实核查方面具有显著潜力。然而，现有基准尚未全面评估LLMs和多模态大规模语言模型（MLLMs）在现实 misinformation 情景中的表现。为填补这一空白，我们引入了RealFactBench，一个全面的基准测试，旨在评估LLMs和MLLMs在包括知识验证、谣言检测和事件验证等多样化的现实世界任务中的事实核查能力。RealFactBench 包含6000个高质量的主张，来源于权威来源，并涵盖多模态内容和多个领域。我们的评估框架引入了未知率（UnR）指标，可更细致地评估模型处理不确定性以及在保守和自信之间的平衡能力。对7个代表性LLM和4个MLLM的广泛实验揭示了它们在现实世界事实核查中的局限性，并提供了对进一步研究有价值的见解。RealFactBench 已公开发布于此 https://链接。 

---
# Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction 

**Title (ZH)**: 具有解耦词元化器和多词预测的语音-语言模型 

**Authors**: Xiaoran Fan, Zhichao Sun, Yangfan Gao, Jingfei Xiong, Hang Yan, Yifei Cao, Jiajun Sun, Shuo Li, Zhihao Zhang, Zhiheng Xi, Yuhao Zhou, Senjie Jin, Changhao Jiang, Junjie Ye, Ming Zhang, Rui Zheng, Zhenhua Han, Yunke Zhang, Demei Yan, Shaokang Dong, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12537)  

**Abstract**: Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the impact of key components (i.e., speech tokenizers, speech heads, and speaker modeling) on the performance of LLM-centric SLMs. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency. 

**Abstract (ZH)**: 基于语言模型的语音模型（Speech-Language Models, SLMs）为统一语音和文本的理解与生成提供了有希望的道路。然而，在实现有效的跨模态对齐和高质量的语音生成方面仍面临挑战。本研究系统地探讨了关键组件（即语音分词器、语音头和说话人建模）对以大型语言模型为中心的SLM性能的影响。我们在公平的SLM框架下比较了耦合、半解耦和完全解耦的语音分词器，并发现解耦分词显著提高了对齐和合成质量。为了解决语音和文本之间的信息密度不匹配问题，我们引入了多令牌预测（Multi-Token Prediction, MTP）到SLM中，使每个隐藏状态解码多个语音令牌。这导致解码速度提高了12倍，并且词错误率大幅下降（从6.07降至3.01）。此外，我们提出了一种基于说话人的生成范式，并引入了RoleTriviaQA大规模角色扮演知识问答基准，其中包含多种说话人身份。实验表明，我们的方法增强了知识理解和说话人一致性。 

---
# Deep Fusion of Ultra-Low-Resolution Thermal Camera and Gyroscope Data for Lighting-Robust and Compute-Efficient Rotational Odometry 

**Title (ZH)**: 基于超高分辨率热敏相机和陀螺仪数据的深度融合：面向照明鲁棒性和计算效率的旋转里程计 

**Authors**: Farida Mohsen, Ali Safa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12536)  

**Abstract**: Accurate rotational odometry is crucial for autonomous robotic systems, particularly for small, power-constrained platforms such as drones and mobile robots. This study introduces thermal-gyro fusion, a novel sensor fusion approach that integrates ultra-low-resolution thermal imaging with gyroscope readings for rotational odometry. Unlike RGB cameras, thermal imaging is invariant to lighting conditions and, when fused with gyroscopic data, mitigates drift which is a common limitation of inertial sensors. We first develop a multimodal data acquisition system to collect synchronized thermal and gyroscope data, along with rotational speed labels, across diverse environments. Subsequently, we design and train a lightweight Convolutional Neural Network (CNN) that fuses both modalities for rotational speed estimation. Our analysis demonstrates that thermal-gyro fusion enables a significant reduction in thermal camera resolution without significantly compromising accuracy, thereby improving computational efficiency and memory utilization. These advantages make our approach well-suited for real-time deployment in resource-constrained robotic systems. Finally, to facilitate further research, we publicly release our dataset as supplementary material. 

**Abstract (ZH)**: 准确的旋转里程计对于自主机器人系统至关重要，特别适用于如无人机和移动机器人等小型、功率受限的平台。本研究介绍了一种热敏-陀螺仪融合方法，该方法结合了超低分辨率热成像与陀螺仪读数以实现旋转里程计。与RGB相机不同，热成像对光照条件不敏感，并且与陀螺仪数据融合可以缓解由惯性传感器常见的漂移问题。我们首先开发了一种多模态数据采集系统，以同步收集热成像和陀螺仪数据以及旋转速度标签，适用于多种环境。随后，我们设计并训练了一种轻量级卷积神经网络（CNN），用于融合这两种模态以估计旋转速度。我们的分析表明，热敏-陀螺仪融合能够在显著降低热成像分辨率的情况下，不显著牺牲准确性，从而提高计算效率和内存利用率。这些优势使我们的方法非常适合在资源受限的机器人系统中进行实时部署。最后，为了促进进一步的研究，我们公开发布了我们的数据集作为补充材料。 

---
# Similarity as Reward Alignment: Robust and Versatile Preference-based Reinforcement Learning 

**Title (ZH)**: 相似性作为奖励对齐：稳健且用途广泛的基于偏好的强化学习 

**Authors**: Sara Rajaram, R. James Cotton, Fabian H. Sinz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12529)  

**Abstract**: Preference-based Reinforcement Learning (PbRL) entails a variety of approaches for aligning models with human intent to alleviate the burden of reward engineering. However, most previous PbRL work has not investigated the robustness to labeler errors, inevitable with labelers who are non-experts or operate under time constraints. Additionally, PbRL algorithms often target very specific settings (e.g. pairwise ranked preferences or purely offline learning). We introduce Similarity as Reward Alignment (SARA), a simple contrastive framework that is both resilient to noisy labels and adaptable to diverse feedback formats and training paradigms. SARA learns a latent representation of preferred samples and computes rewards as similarities to the learned latent. We demonstrate strong performance compared to baselines on continuous control offline RL benchmarks. We further demonstrate SARA's versatility in applications such as trajectory filtering for downstream tasks, cross-task preference transfer, and reward shaping in online learning. 

**Abstract (ZH)**: 基于偏好强化学习的类似性作为奖励对齐（SARA）：鲁棒性强且适应多种反馈格式和训练范式的简单对比框架 

---
# Towards Fairness Assessment of Dutch Hate Speech Detection 

**Title (ZH)**: 荷兰仇恨言论检测的公平性评估 

**Authors**: Julie Bauer, Rishabh Kaushal, Thales Bertaglia, Adriana Iamnitchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12502)  

**Abstract**: Numerous studies have proposed computational methods to detect hate speech online, yet most focus on the English language and emphasize model development. In this study, we evaluate the counterfactual fairness of hate speech detection models in the Dutch language, specifically examining the performance and fairness of transformer-based models. We make the following key contributions. First, we curate a list of Dutch Social Group Terms that reflect social context. Second, we generate counterfactual data for Dutch hate speech using LLMs and established strategies like Manual Group Substitution (MGS) and Sentence Log-Likelihood (SLL). Through qualitative evaluation, we highlight the challenges of generating realistic counterfactuals, particularly with Dutch grammar and contextual coherence. Third, we fine-tune baseline transformer-based models with counterfactual data and evaluate their performance in detecting hate speech. Fourth, we assess the fairness of these models using Counterfactual Token Fairness (CTF) and group fairness metrics, including equality of odds and demographic parity. Our analysis shows that models perform better in terms of hate speech detection, average counterfactual fairness and group fairness. This work addresses a significant gap in the literature on counterfactual fairness for hate speech detection in Dutch and provides practical insights and recommendations for improving both model performance and fairness. 

**Abstract (ZH)**: 多项研究表明，提出了计算方法来检测网络上的仇恨言论，但大多数研究集中在英语上并侧重于模型开发。本研究评估了荷兰语仇恨言论检测模型的反事实公平性，特别检查了基于变换器的模型的性能和公平性。我们做出了以下关键贡献：首先，我们整理了一份反映社会语境的荷兰社会群体术语列表；其次，我们使用大型语言模型（LLMs）和现有的策略（如手动群体替换（MGS）和句法log-likelihood（SLL）生成荷兰语仇恨言论的反事实数据；通过定性评估，我们强调了生成现实反事实的挑战，特别是在荷兰语语法和上下文一致性方面；第三，我们使用反事实数据微调基准变换器模型，并评估其在检测仇恨言论方面的性能；第四，我们使用反事实标记公平性（CTF）和群体公平性指标（包括等几率和人口公平性）评估这些模型的公平性。我们的分析表明，这些模型在仇恨言论检测、平均反事实公平性和群体公平性方面表现更好。本研究填补了荷兰语仇恨言论检测中反事实公平性的文献空白，并提供了提高模型性能和公平性的实用见解和建议。 

---
# Comparative Analysis of Deep Learning Strategies for Hypertensive Retinopathy Detection from Fundus Images: From Scratch and Pre-trained Models 

**Title (ZH)**: 从零构建与预训练模型在黄斑糖尿病视网膜病变从眼底图像检测中的深度学习策略比较分析 

**Authors**: Yanqiao Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12492)  

**Abstract**: This paper presents a comparative analysis of deep learning strategies for detecting hypertensive retinopathy from fundus images, a central task in the HRDC challenge~\cite{qian2025hrdc}. We investigate three distinct approaches: a custom CNN, a suite of pre-trained transformer-based models, and an AutoML solution. Our findings reveal a stark, architecture-dependent response to data augmentation. Augmentation significantly boosts the performance of pure Vision Transformers (ViTs), which we hypothesize is due to their weaker inductive biases, forcing them to learn robust spatial and structural features. Conversely, the same augmentation strategy degrades the performance of hybrid ViT-CNN models, whose stronger, pre-existing biases from the CNN component may be "confused" by the transformations. We show that smaller patch sizes (ViT-B/8) excel on augmented data, enhancing fine-grained detail capture. Furthermore, we demonstrate that a powerful self-supervised model like DINOv2 fails on the original, limited dataset but is "rescued" by augmentation, highlighting the critical need for data diversity to unlock its potential. Preliminary tests with a ViT-Large model show poor performance, underscoring the risk of using overly-capacitive models on specialized, smaller datasets. This work provides critical insights into the interplay between model architecture, data augmentation, and dataset size for medical image classification. 

**Abstract (ZH)**: 本研究探讨了检测高血压视网膜病变的基金照片中深度学习策略的比较分析，这是HRDC挑战~\cite{qian2025hrdc}中的核心任务。我们调查了三种不同的方法：自定义CNN、一系列预训练的变压器模型以及AutoML解决方案。我们的研究发现，数据增强对模型架构高度依赖。增强显著提升了纯视力变换器(ViTs)的性能，我们认为这是由于它们较强的归纳偏置较弱，迫使它们学习稳健的空间和结构特征。相反，相同的增强策略降低了混合ViT-CNN模型的性能，这些模型的较强预存偏置可能因变换被“迷惑”。我们展示了较小的 patch 大小（ViT-B/8）在增强数据上表现出色，增强了细粒度细节的捕获。此外，我们证明了强大的自监督模型DINOv2在原始受限数据集上表现不佳，但在增强后得以“拯救”，突显了数据多样性在解锁其潜力方面的关键作用。初步测试中，ViT-Large模型表现不佳，强调了在专门的小型数据集上使用能力过强模型的风险。本研究提供了关于模型架构、数据增强和数据集大小在医学图像分类中相互作用的关键见解。 

---
# Robust LLM Unlearning with MUDMAN: Meta-Unlearning with Disruption Masking And Normalization 

**Title (ZH)**: Robust LLM Unlearning with MUDMAN: 基于干扰屏蔽和规范化元遗忘的LLM健壮性遗忘 

**Authors**: Filip Sondej, Yushi Yang, Mikołaj Kniejski, Marcel Windys  

**Link**: [PDF](https://arxiv.org/pdf/2506.12484)  

**Abstract**: Language models can retain dangerous knowledge and skills even after extensive safety fine-tuning, posing both misuse and misalignment risks. Recent studies show that even specialized unlearning methods can be easily reversed. To address this, we systematically evaluate many existing and novel components of unlearning methods and identify ones crucial for irreversible unlearning.
We introduce Disruption Masking, a technique in which we only allow updating weights, where the signs of the unlearning gradient and the retaining gradient are the same. This ensures all updates are non-disruptive.
Additionally, we identify the need for normalizing the unlearning gradients, and also confirm the usefulness of meta-learning. We combine these insights into MUDMAN (Meta-Unlearning with Disruption Masking and Normalization) and validate its effectiveness at preventing the recovery of dangerous capabilities. MUDMAN outperforms the prior TAR method by 40\%, setting a new state-of-the-art for robust unlearning. 

**Abstract (ZH)**: 语言模型在广泛的安全微调后仍然可能保留危险的知识和技能，这带来了误用和不对齐的风险。最近的研究表明，即使是专门的遗忘方法也可能容易被逆转。为应对这一问题，我们系统地评估了许多现有的和新颖的遗忘方法组件，并识别出对于不可逆遗忘至关重要的因素。我们引入了破坏性掩蔽技术，只允许更新权重，其中遗忘梯度和保持梯度的符号相同，确保所有更新都是非破坏性的。此外，我们识别出需要对遗忘梯度进行归一化，并确认了元学习的有效性。我们将这些洞察结合到MUDMAN（基于破坏性掩蔽和归一化的元遗忘）中，并验证其在防止恢复危险能力方面的有效性。MUDMAN在防止恢复危险能力方面比之前的TAR方法性能提高了40%，并设定了一项新的鲁棒遗忘状态-of-艺术。 

---
# Generalizable Trajectory Prediction via Inverse Reinforcement Learning with Mamba-Graph Architecture 

**Title (ZH)**: 基于Mamba-Graph架构的逆强化学习可迁移轨迹预测 

**Authors**: Wenyun Li, Wenjie Huang, Zejian Deng, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.12474)  

**Abstract**: Accurate driving behavior modeling is fundamental to safe and efficient trajectory prediction, yet remains challenging in complex traffic scenarios. This paper presents a novel Inverse Reinforcement Learning (IRL) framework that captures human-like decision-making by inferring diverse reward functions, enabling robust cross-scenario adaptability. The learned reward function is utilized to maximize the likelihood of output by the encoder-decoder architecture that combines Mamba blocks for efficient long-sequence dependency modeling with graph attention networks to encode spatial interactions among traffic agents. Comprehensive evaluations on urban intersections and roundabouts demonstrate that the proposed method not only outperforms various popular approaches in prediction accuracy but also achieves 2 times higher generalization performance to unseen scenarios compared to other IRL-based method. 

**Abstract (ZH)**: 准确的驾驶行为建模对于复杂交通场景下的安全高效的轨迹预测至关重要，但仍具有挑战性。本文提出了一种新颖的逆强化学习（IRL）框架，通过推断多样的奖励函数来捕捉类似人类的决策过程，从而实现跨场景的鲁棒适应性。所学习的奖励函数被用于最大化结合Mamba块的编码-解码架构的输出概率，该架构利用图注意力网络编码交通代理之间的空间交互，以高效建模长序列依赖关系。在城市交叉口和环岛的全面评估中表明，所提出的方法不仅在预测准确性上超越了各种流行的预测方法，而且在未见过的场景上的泛化性能比其他基于IRL的方法高出两倍。 

---
# Levels of Autonomy for AI Agents 

**Title (ZH)**: AI代理的自主水平 

**Authors**: K. J. Kevin Feng, David W. McDonald, Amy X. Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12469)  

**Abstract**: Autonomy is a double-edged sword for AI agents, simultaneously unlocking transformative possibilities and serious risks. How can agent developers calibrate the appropriate levels of autonomy at which their agents should operate? We argue that an agent's level of autonomy can be treated as a deliberate design decision, separate from its capability and operational environment. In this work, we define five levels of escalating agent autonomy, characterized by the roles a user can take when interacting with an agent: operator, collaborator, consultant, approver, and observer. Within each level, we describe the ways by which a user can exert control over the agent and open questions for how to design the nature of user-agent interaction. We then highlight a potential application of our framework towards AI autonomy certificates to govern agent behavior in single- and multi-agent systems. We conclude by proposing early ideas for evaluating agents' autonomy. Our work aims to contribute meaningful, practical steps towards responsibly deployed and useful AI agents in the real world. 

**Abstract (ZH)**: 自主性是AI代理的双刃剑，同时开启变革性潜力和严重风险。代理开发者应如何校准代理应操作的适当自主水平？我们argue自主水平可以作为故意的设计决策，与代理的能力和运行环境分开。在此工作中，我们定义了五级递增的代理自主性级别，由用户在与代理互动时可以扮演的角色来表征：操作员、合作者、顾问、审批人和观察者。在每一级中，我们描述了用户控制代理的方式，并提出了有关如何设计用户-代理交互本质的问题。然后，我们强调了将我们的框架应用于AI自主性证书，以管理单个和多个代理系统的代理行为的潜在应用。最后，我们提出了评估代理自主性的初步想法。我们的工作旨在为负责任地部署和实用的AI代理在现实世界中做出有意义和实用的贡献。 

---
# Delving into Instance-Dependent Label Noise in Graph Data: A Comprehensive Study and Benchmark 

**Title (ZH)**: 图数据中实例依赖的标签噪声：一项全面的研究与基准测试 

**Authors**: Suyeon Kim, SeongKu Kang, Dongwoo Kim, Jungseul Ok, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12468)  

**Abstract**: Graph Neural Networks (GNNs) have achieved state-of-the-art performance in node classification tasks but struggle with label noise in real-world data. Existing studies on graph learning with label noise commonly rely on class-dependent label noise, overlooking the complexities of instance-dependent noise and falling short of capturing real-world corruption patterns. We introduce BeGIN (Benchmarking for Graphs with Instance-dependent Noise), a new benchmark that provides realistic graph datasets with various noise types and comprehensively evaluates noise-handling strategies across GNN architectures, noisy label detection, and noise-robust learning. To simulate instance-dependent corruptions, BeGIN introduces algorithmic methods and LLM-based simulations. Our experiments reveal the challenges of instance-dependent noise, particularly LLM-based corruption, and underscore the importance of node-specific parameterization to enhance GNN robustness. By comprehensively evaluating noise-handling strategies, BeGIN provides insights into their effectiveness, efficiency, and key performance factors. We expect that BeGIN will serve as a valuable resource for advancing research on label noise in graphs and fostering the development of robust GNN training methods. The code is available at this https URL. 

**Abstract (ZH)**: 图神经网络（GNNs）在节点分类任务中取得了最先进的性能，但在应对实际数据中的标签噪声时存在困难。现有的图学习中对抗标签噪声的研究主要依赖于类相关的标签噪声，忽视了实例相关的噪声复杂性，未能捕捉到真实的污染模式。我们引入了BeGIN（Benchmarking for Graphs with Instance-dependent Noise），这是一种新的基准，提供了具有多种噪声类型的现实图数据集，并全面评估了GNN架构、嘈杂标签检测和噪声鲁棒学习的方法。为了模拟实例相关的污染，BeGIN引入了算法方法和基于LLM的模拟。我们的实验揭示了实例相关的噪声，特别是基于LLM的污染所带来的挑战，并强调了节点特定参数化对增强GNN鲁棒性的重要性。通过全面评估噪声处理策略，BeGIN提供了它们有效性的见解、效率和关键性能因素。我们期望BeGIN将成为推动图中标签噪声研究和促进鲁棒GNN训练方法发展的宝贵资源。代码可在以下网址获取。 

---
# Merlin: Multi-View Representation Learning for Robust Multivariate Time Series Forecasting with Unfixed Missing Rates 

**Title (ZH)**: Merlin: 多视图表示学习以实现鲁棒多变量时间序列预测，面对不确定的缺失率 

**Authors**: Chengqing Yu, Fei Wang, Chuanguang Yang, Zezhi Shao, Tao Sun, Tangwen Qian, Wei Wei, Zhulin An, Yongjun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12459)  

**Abstract**: Multivariate Time Series Forecasting (MTSF) involves predicting future values of multiple interrelated time series. Recently, deep learning-based MTSF models have gained significant attention for their promising ability to mine semantics (global and local information) within MTS data. However, these models are pervasively susceptible to missing values caused by malfunctioning data collectors. These missing values not only disrupt the semantics of MTS, but their distribution also changes over time. Nevertheless, existing models lack robustness to such issues, leading to suboptimal forecasting performance. To this end, in this paper, we propose Multi-View Representation Learning (Merlin), which can help existing models achieve semantic alignment between incomplete observations with different missing rates and complete observations in MTS. Specifically, Merlin consists of two key modules: offline knowledge distillation and multi-view contrastive learning. The former utilizes a teacher model to guide a student model in mining semantics from incomplete observations, similar to those obtainable from complete observations. The latter improves the student model's robustness by learning from positive/negative data pairs constructed from incomplete observations with different missing rates, ensuring semantic alignment across different missing rates. Therefore, Merlin is capable of effectively enhancing the robustness of existing models against unfixed missing rates while preserving forecasting accuracy. Experiments on four real-world datasets demonstrate the superiority of Merlin. 

**Abstract (ZH)**: 多元时间序列预测中的多视图表示学习（Merlin）：一种应对不固定缺失率的鲁棒方法 

---
# A Pluggable Multi-Task Learning Framework for Sentiment-Aware Financial Relation Extraction 

**Title (ZH)**: 面向情感感知金融关系提取的插件式多任务学习框架 

**Authors**: Jinming Luo, Hailin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12452)  

**Abstract**: Relation Extraction (RE) aims to extract semantic relationships in texts from given entity pairs, and has achieved significant improvements. However, in different domains, the RE task can be influenced by various factors. For example, in the financial domain, sentiment can affect RE results, yet this factor has been overlooked by modern RE models. To address this gap, this paper proposes a Sentiment-aware-SDP-Enhanced-Module (SSDP-SEM), a multi-task learning approach for enhancing financial RE. Specifically, SSDP-SEM integrates the RE models with a pluggable auxiliary sentiment perception (ASP) task, enabling the RE models to concurrently navigate their attention weights with the text's sentiment. We first generate detailed sentiment tokens through a sentiment model and insert these tokens into an instance. Then, the ASP task focuses on capturing nuanced sentiment information through predicting the sentiment token positions, combining both sentiment insights and the Shortest Dependency Path (SDP) of syntactic information. Moreover, this work employs a sentiment attention information bottleneck regularization method to regulate the reasoning process. Our experiment integrates this auxiliary task with several prevalent frameworks, and the results demonstrate that most previous models benefit from the auxiliary task, thereby achieving better results. These findings highlight the importance of effectively leveraging sentiment in the financial RE task. 

**Abstract (ZH)**: 情感意识-最短依赖路径增强模块：面向金融领域的关系提取 

---
# MS-UMamba: An Improved Vision Mamba Unet for Fetal Abdominal Medical Image Segmentation 

**Title (ZH)**: MS-UMamba: 一种改进的Vision Mamba Unet胎儿腹部医疗图像分割方法 

**Authors**: Caixu Xu, Junming Wei, Huizhen Chen, Pengchen Liang, Bocheng Liang, Ying Tan, Xintong Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.12441)  

**Abstract**: Recently, Mamba-based methods have become popular in medical image segmentation due to their lightweight design and long-range dependency modeling capabilities. However, current segmentation methods frequently encounter challenges in fetal ultrasound images, such as enclosed anatomical structures, blurred boundaries, and small anatomical structures. To address the need for balancing local feature extraction and global context modeling, we propose MS-UMamba, a novel hybrid convolutional-mamba model for fetal ultrasound image segmentation. Specifically, we design a visual state space block integrated with a CNN branch (SS-MCAT-SSM), which leverages Mamba's global modeling strengths and convolutional layers' local representation advantages to enhance feature learning. In addition, we also propose an efficient multi-scale feature fusion module that integrates spatial attention mechanisms, which Integrating feature information from different layers enhances the feature representation ability of the model. Finally, we conduct extensive experiments on a non-public dataset, experimental results demonstrate that MS-UMamba model has excellent performance in segmentation performance. 

**Abstract (ZH)**: 基于Mamba的方法近年来在医学图像分割中变得流行，由于其轻量级设计和长程依赖建模能力。然而，当前的分割方法在胎儿超声图像中经常遇到挑战，如封闭的解剖结构、模糊的边界和小的解剖结构。为了解决局部特征提取和全局上下文建模之间的平衡需求，我们提出了一种新颖的混合卷积-Mamba模型MS-UMamba，适用于胎儿超声图像分割。具体而言，我们设计了一个整合CNN分支的视觉状态空间块（SS-MCAT-SSM），该块利用Mamba的全局建模优势和卷积层的局部表示优势，增强特征学习。此外，我们还提出了一种高效的多尺度特征融合模块，该模块集成了空间注意力机制，以整合不同层的特征信息，增强模型的特征表示能力。最后，我们在一个非公开数据集上进行了广泛的实验，实验结果表明，MS-UMamba模型在分割性能方面表现出色。 

---
# Style-based Composer Identification and Attribution of Symbolic Music Scores: a Systematic Survey 

**Title (ZH)**: 基于风格的乐谱作曲家识别与归属：一项系统综述 

**Authors**: Federico Simonetta  

**Link**: [PDF](https://arxiv.org/pdf/2506.12440)  

**Abstract**: This paper presents the first comprehensive systematic review of literature on style-based composer identification and authorship attribution in symbolic music scores. Addressing the critical need for improved reliability and reproducibility in this field, the review rigorously analyzes 58 peer-reviewed papers published across various historical periods, with the search adapted to evolving terminology. The analysis critically assesses prevailing repertoires, computational approaches, and evaluation methodologies, highlighting significant challenges. It reveals that a substantial portion of existing research suffers from inadequate validation protocols and an over-reliance on simple accuracy metrics for often imbalanced datasets, which can undermine the credibility of attribution claims. The crucial role of robust metrics like Balanced Accuracy and rigorous cross-validation in ensuring trustworthy results is emphasized. The survey also details diverse feature representations and the evolution of machine learning models employed. Notable real-world authorship attribution cases, such as those involving works attributed to Bach, Josquin Desprez, and Lennon-McCartney, are specifically discussed, illustrating the opportunities and pitfalls of applying computational techniques to resolve disputed musical provenance. Based on these insights, a set of actionable guidelines for future research are proposed. These recommendations are designed to significantly enhance the reliability, reproducibility, and musicological validity of composer identification and authorship attribution studies, fostering more robust and interpretable computational stylistic analysis. 

**Abstract (ZH)**: 本文提供了基于风格的作曲家识别和乐谱著作者归属 Literature Review 的首次全面系统性综述。针对该领域可靠性与再现性改进的迫切需求，研究严格分析了跨越不同历史时期的 58 篇同行评审论文，搜索策略适应术语演变。分析批判性地评估了现有的 repertoire、计算方法和评价方法，指出了显著的挑战。研究揭示，现有研究中很大一部分缺乏充分的验证协议，并过度依赖简单的准确度指标，特别是在不平衡数据集的情况下，这可能削弱著作者归属声明的可信度。强调了使用稳健的指标如平衡准确度及严格的交叉验证以确保可信赖结果的重要性。调查还详细介绍了多样化的特征表示及其所使用的机器学习模型的演变。具体讨论了涉及巴赫、若斯坎·迪普雷兹和 Lennon-McCartney 的著名现实世界著作者归属案例，说明了如何利用计算技术解决有争议的音乐来源问题。基于这些见解，提出了未来研究的一系列可操作性指南。这些建议旨在显著提高作曲家识别和著作者归属研究的可靠性、再现性和音乐学有效性，促进更加稳健和可解释的计算风格分析。 

---
# Feeling Machines: Ethics, Culture, and the Rise of Emotional AI 

**Title (ZH)**: 情感机器：伦理、文化与情绪人工智能的兴起 

**Authors**: Vivek Chavan, Arsen Cenaj, Shuyuan Shen, Ariane Bar, Srishti Binwani, Tommaso Del Becaro, Marius Funk, Lynn Greschner, Roberto Hung, Stina Klein, Romina Kleiner, Stefanie Krause, Sylwia Olbrych, Vishvapalsinhji Parmar, Jaleh Sarafraz, Daria Soroko, Daksitha Withanage Don, Chang Zhou, Hoang Thuy Duong Vu, Parastoo Semnani, Daniel Weinhardt, Elisabeth Andre, Jörg Krüger, Xavier Fresquet  

**Link**: [PDF](https://arxiv.org/pdf/2506.12437)  

**Abstract**: This paper explores the growing presence of emotionally responsive artificial intelligence through a critical and interdisciplinary lens. Bringing together the voices of early-career researchers from multiple fields, it explores how AI systems that simulate or interpret human emotions are reshaping our interactions in areas such as education, healthcare, mental health, caregiving, and digital life. The analysis is structured around four central themes: the ethical implications of emotional AI, the cultural dynamics of human-machine interaction, the risks and opportunities for vulnerable populations, and the emerging regulatory, design, and technical considerations. The authors highlight the potential of affective AI to support mental well-being, enhance learning, and reduce loneliness, as well as the risks of emotional manipulation, over-reliance, misrepresentation, and cultural bias. Key challenges include simulating empathy without genuine understanding, encoding dominant sociocultural norms into AI systems, and insufficient safeguards for individuals in sensitive or high-risk contexts. Special attention is given to children, elderly users, and individuals with mental health challenges, who may interact with AI in emotionally significant ways. However, there remains a lack of cognitive or legal protections which are necessary to navigate such engagements safely. The report concludes with ten recommendations, including the need for transparency, certification frameworks, region-specific fine-tuning, human oversight, and longitudinal research. A curated supplementary section provides practical tools, models, and datasets to support further work in this domain. 

**Abstract (ZH)**: 本文通过跨学科的批判性视角探讨情绪响应人工智能日益增长的存在。汇集了来自多个领域的早期职业研究人员的声音，探讨模拟或解释人类情绪的人工智能系统如何重塑教育、医疗、心理健康、照护以及数字生活等领域中的互动。分析围绕四个核心主题展开：情绪人工智能的伦理影响、人机互动的文化动态、脆弱群体面临的风险与机遇，以及新兴的监管、设计和技术考量。作者强调了情感人工智能支持心理健康、增强学习和减轻孤独的潜力，同时也提到了情感操控、过度依赖、误导和文化偏见的风险。关键挑战包括在没有真正理解的情况下模拟共情、将主导的社文化规范编码进人工智能系统，以及在敏感或高风险情境中缺乏足够的保护措施。特别关注儿童、老年用户以及心理健康挑战者，他们与人工智能在情感上可能存在重要互动。然而，对于这些互动仍缺乏足够的认知或法律保护措施，以确保安全地参与其中。报告最后提出十项建议，包括透明度、认证框架、区域特定微调、人类监督以及纵向研究的需要。一个精选补充部分提供了实用工具、模型和数据集，以支持该领域的进一步研究工作。 

---
# EXGnet: a single-lead explainable-AI guided multiresolution network with train-only quantitative features for trustworthy ECG arrhythmia classification 

**Title (ZH)**: EXGnet：一种基于单导联可解释AI引导的多分辨网络，用于可信的心电图心律失常分类 

**Authors**: Tushar Talukder Showrav, Soyabul Islam Lincoln, Md. Kamrul Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12404)  

**Abstract**: Background: Deep learning has significantly advanced ECG arrhythmia classification, enabling high accuracy in detecting various cardiac conditions. The use of single-lead ECG systems is crucial for portable devices, as they offer convenience and accessibility for continuous monitoring in diverse settings. However, the interpretability and reliability of deep learning models in clinical applications poses challenges due to their black-box nature. Methods: To address these challenges, we propose EXGnet, a single-lead, trustworthy ECG arrhythmia classification network that integrates multiresolution feature extraction with Explainable Artificial Intelligence (XAI) guidance and train only quantitative features. Results: Trained on two public datasets, including Chapman and Ningbo, EXGnet demonstrates superior performance through key metrics such as Accuracy, F1-score, Sensitivity, and Specificity. The proposed method achieved average five fold accuracy of 98.762%, and 96.932% and average F1-score of 97.910%, and 95.527% on the Chapman and Ningbo datasets, respectively. Conclusions: By employing XAI techniques, specifically Grad-CAM, the model provides visual insights into the relevant ECG segments it analyzes, thereby enhancing clinician trust in its predictions. While quantitative features further improve classification performance, they are not required during testing, making the model suitable for real-world applications. Overall, EXGnet not only achieves better classification accuracy but also addresses the critical need for interpretability in deep learning, facilitating broader adoption in portable ECG monitoring. 

**Abstract (ZH)**: 背景：深度学习显著推进了心电图（ECG）心律失常分类，使其能够在各种心脏条件下实现高精度检测。单导联ECG系统对于便携设备至关重要，因为它们在多样化的环境中提供了便利性和可访问性，用于持续监测。然而，深学习模型在临床应用中的解释性和可靠性因它们的黑匣子性质而受到挑战。方法：为了解决这些挑战，我们提出EXGnet，这是一种结合多分辨率特征提取和可解释人工智能（XAI）指导的单导联、可信赖的心律失常分类网络，仅训练定量特征。结果：EXGnet在两个公开数据集（包括Chapman和宁波）上进行训练，通过关键指标（如准确率、F1分数、灵敏度和特异度）展示了卓越的性能。在Chapman和宁波数据集上，提出的方法分别实现了平均五折准确率为98.762%和96.932%，平均F1分数为97.910%和95.527%。结论：通过采用XAI技术，特别是Grad-CAM，该模型为分析的相关ECG段落提供了可视化见解，从而增强临床医生对其预测的信任。尽管定量特征进一步提高了分类性能，但在测试时无需使用这些特征，使该模型适用于实际应用。总体而言，EXGnet不仅实现了更好的分类准确率，还解决了深度学习解释性方面的重要需求，促进了其在便携式ECG监测中的更广泛采用。 

---
# Bridging the Digital Divide: Small Language Models as a Pathway for Physics and Photonics Education in Underdeveloped Regions 

**Title (ZH)**: 缩小数字鸿沟：小型语言模型在欠发达地区物理和光电教育中的路径探索 

**Authors**: Asghar Ghorbani, Hanieh Fattahi  

**Link**: [PDF](https://arxiv.org/pdf/2506.12403)  

**Abstract**: Limited infrastructure, scarce educational resources, and unreliable internet access often hinder physics and photonics education in underdeveloped regions. These barriers create deep inequities in Science, Technology, Engineering, and Mathematics (STEM) education. This article explores how Small Language Models (SLMs)-compact, AI-powered tools that can run offline on low-power devices, offering a scalable solution. By acting as virtual tutors, enabling native-language instruction, and supporting interactive learning, SLMs can help address the shortage of trained educators and laboratory access. By narrowing the digital divide through targeted investment in AI technologies, SLMs present a scalable and inclusive solution to advance STEM education and foster scientific empowerment in marginalized communities. 

**Abstract (ZH)**: 有限的基础设施、稀缺的教育资源和不稳定的互联网访问经常阻碍欠发达地区物理学和光子学教育的发展。这些障碍造成了科学、技术、工程和数学（STEM）教育中的深远不平等。本文探讨了小型语言模型（SLMs）的作用，SLMs是紧凑型、AI驱动的工具，可在低功率设备上离线运行，提供一种可扩展的解决方案。通过作为虚拟导师、提供本族语言教学并支持互动学习，SLMs有助于应对训练有素教育者和实验室准入不足的问题。通过针对性地投资AI技术缩小数字鸿沟，SLMs为推进STEM教育和促进边缘化社区的科学自主权提供了一种可扩展和包容性解决方案。 

---
# LARGO: Low-Rank Regulated Gradient Projection for Robust Parameter Efficient Fine-Tuning 

**Title (ZH)**: LARGO: 低秩调节梯度投影的稳健参数高效微调 

**Authors**: Haotian Zhang, Liu Liu, Baosheng Yu, Jiayan Qiu, Yanwei Ren, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12394)  

**Abstract**: The advent of parameter-efficient fine-tuning methods has significantly reduced the computational burden of adapting large-scale pretrained models to diverse downstream tasks. However, existing approaches often struggle to achieve robust performance under domain shifts while maintaining computational efficiency. To address this challenge, we propose Low-rAnk Regulated Gradient Projection (LARGO) algorithm that integrates dynamic constraints into low-rank adaptation methods. Specifically, LARGO incorporates parallel trainable gradient projections to dynamically regulate layer-wise updates, retaining the Out-Of-Distribution robustness of pretrained model while preserving inter-layer independence. Additionally, it ensures computational efficiency by mitigating the influence of gradient dependencies across layers during weight updates. Besides, through leveraging singular value decomposition of pretrained weights for structured initialization, we incorporate an SVD-based initialization strategy that minimizing deviation from pretrained knowledge. Through extensive experiments on diverse benchmarks, LARGO achieves state-of-the-art performance across in-domain and out-of-distribution scenarios, demonstrating improved robustness under domain shifts with significantly lower computational overhead compared to existing PEFT methods. The source code will be released soon. 

**Abstract (ZH)**: 低秩调节梯度投影（LARGO）算法：在保持计算效率的同时提升领域泛化鲁棒性 

---
# Revisiting Clustering of Neural Bandits: Selective Reinitialization for Mitigating Loss of Plasticity 

**Title (ZH)**: 重新审视神经bandits的聚类：选择性重初始化以减轻可塑性丧失的影响 

**Authors**: Zhiyuan Su, Sunhao Dai, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12389)  

**Abstract**: Clustering of Bandits (CB) methods enhance sequential decision-making by grouping bandits into clusters based on similarity and incorporating cluster-level contextual information, demonstrating effectiveness and adaptability in applications like personalized streaming recommendations. However, when extending CB algorithms to their neural version (commonly referred to as Clustering of Neural Bandits, or CNB), they suffer from loss of plasticity, where neural network parameters become rigid and less adaptable over time, limiting their ability to adapt to non-stationary environments (e.g., dynamic user preferences in recommendation). To address this challenge, we propose Selective Reinitialization (SeRe), a novel bandit learning framework that dynamically preserves the adaptability of CNB algorithms in evolving environments. SeRe leverages a contribution utility metric to identify and selectively reset underutilized units, mitigating loss of plasticity while maintaining stable knowledge retention. Furthermore, when combining SeRe with CNB algorithms, the adaptive change detection mechanism adjusts the reinitialization frequency according to the degree of non-stationarity, ensuring effective adaptation without unnecessary resets. Theoretically, we prove that SeRe enables sublinear cumulative regret in piecewise-stationary environments, outperforming traditional CNB approaches in long-term performances. Extensive experiments on six real-world recommendation datasets demonstrate that SeRe-enhanced CNB algorithms can effectively mitigate the loss of plasticity with lower regrets, improving adaptability and robustness in dynamic settings. 

**Abstract (ZH)**: 基于集群的强化学习中选择性重初始化框架（SeRe）：在动态环境中提升神经臂拉普拉斯算法的可适应性和鲁棒性 

---
# Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model 

**Title (ZH)**: 组后再缩放：动态专家混合多语言语言模型 

**Authors**: Chong Li, Yingzhuo Deng, Jiajun Zhang, Chengqing Zong  

**Link**: [PDF](https://arxiv.org/pdf/2506.12388)  

**Abstract**: The curse of multilinguality phenomenon is a fundamental problem of multilingual Large Language Models (LLMs), where the competition between massive languages results in inferior performance. It mainly comes from limited capacity and negative transfer between dissimilar languages. To address this issue, we propose a method to dynamically group and scale up the parameters of multilingual LLM while boosting positive transfer among similar languages. Specifically, the model is first tuned on monolingual corpus to determine the parameter deviation in each layer and quantify the similarity between languages. Layers with more deviations are extended to mixture-of-experts layers to reduce competition between languages, where one expert module serves one group of similar languages. Experimental results on 18 to 128 languages show that our method reduces the negative transfer between languages and significantly boosts multilingual performance with fewer parameters. Such language group specialization on experts benefits the new language adaptation and reduces the inference on the previous multilingual knowledge learned. 

**Abstract (ZH)**: 多语言现象下的负迁移问题是多语言大型语言模型（LLMs）的一个基本问题，其中大量语言之间的竞争导致了性能下降。主要来自于有限的容量和不同语言间的负面迁移。为了解决这一问题，我们提出了一种动态分组和扩展多语言LLM参数的方法，同时增强了相似语言间的正迁移。具体来说，首先在单一语言语料库上调整模型以确定每一层的参数偏差并量化语言之间的相似性。偏差较大的层被扩展为专家模块层，以减少语言之间的竞争，其中每个专家模块服务于一组相似的语言。在18至128种语言的实验结果显示，我们的方法减少了语言间的负迁移，并显著提升了使用更少参数的多语言性能。这种语言组专家专业化有利于新语言的适应，并减少了对之前多语言知识的推理。 

---
# Recent Advances and Future Directions in Literature-Based Discovery 

**Title (ZH)**: 基于文献的发现： Recent Advances and Future Directions 

**Authors**: Andrej Kastrin, Bojan Cestnik, Nada Lavrač  

**Link**: [PDF](https://arxiv.org/pdf/2506.12385)  

**Abstract**: The explosive growth of scientific publications has created an urgent need for automated methods that facilitate knowledge synthesis and hypothesis generation. Literature-based discovery (LBD) addresses this challenge by uncovering previously unknown associations between disparate domains. This article surveys recent methodological advances in LBD, focusing on developments from 2000 to the present. We review progress in three key areas: knowledge graph construction, deep learning approaches, and the integration of pre-trained and large language models (LLMs). While LBD has made notable progress, several fundamental challenges remain unresolved, particularly concerning scalability, reliance on structured data, and the need for extensive manual curation. By examining ongoing advances and outlining promising future directions, this survey underscores the transformative role of LLMs in enhancing LBD and aims to support researchers and practitioners in harnessing these technologies to accelerate scientific innovation. 

**Abstract (ZH)**: 科学出版物的爆炸性增长迫切需要自动化方法来促进知识综合和假说生成。基于文献的发现（LBD）通过揭示不同领域之间的未知关联来应对这一挑战。本文回顾了从2000年至今LBD的最新方法论进展，重点关注知识图谱构建、深度学习方法以及预训练和大型语言模型（LLMs）的集成。尽管LBD已取得了显著进展，但仍存在几个根本性挑战，特别是可扩展性、对结构化数据的依赖以及需要大量手动整理。通过研究正在进行的进展并展望有前景的未来方向，本文突显了LLMs在提升LBD中的变革性作用，并旨在支持研究人员和实践者利用这些技术加速科学创新。 

---
# Exploring the Secondary Risks of Large Language Models 

**Title (ZH)**: 探索大型语言模型的次生风险 

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.12382)  

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments. 

**Abstract (ZH)**: 确保大型语言模型的安全性和对齐是一个随着其在关键应用和社会功能中集成的深化而变得日益重要的挑战。尽管先前的研究主要关注于脱管攻击，但对良性交互中微妙出现的非对抗性失败的关注相对较少。我们引入了一种新的失效模式类别，这些模式在良性提示期间表现出有害或误导性的行为。与对抗性攻击不同，这些风险源自不完美的泛化，经常规避标准的安全机制。为实现系统性评估，我们引入了两个风险基元：冗长响应和推测性建议，以捕捉核心的失效模式。基于这些定义，我们提出了SecLens，一种黑盒多目标搜索框架，通过优化任务相关性、风险激活和语义合理性来有效诱发次级风险行为。为支持可重复评估，我们发布了SecRiskBench基准数据集，包含650个提示，涵盖八种不同的现实世界风险类别。广泛评估16个流行模型的实验结果表明，次级风险是普遍存在的、可以在不同模型之间转移的、并且与模态无关的，强调了在实际部署中增强安全机制以应对良性但有害的大规模语言模型行为的迫切需求。 

---
# Training-free LLM Merging for Multi-task Learning 

**Title (ZH)**: 无需训练的LLM融合用于多任务学习 

**Authors**: Zichuan Fu, Xian Wu, Yejing Wang, Wanyu Wang, Shanshan Ye, Hongzhi Yin, Yi Chang, Yefeng Zheng, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12379)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. In this paper, we explore an important question: is it possible to combine these specialized models to create a unified model with multi-task capabilities. We introduces Hierarchical Iterative Merging (Hi-Merging), a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts. Extensive experiments on multiple-choice and question-answering tasks in both Chinese and English validate Hi-Merging's ability for multi-task learning. The results demonstrate that Hi-Merging consistently outperforms existing merging techniques and surpasses the performance of models fine-tuned on combined datasets in most scenarios. Code is available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种自然语言处理（NLP）任务中展现了卓越的能力。开源LLMs如LLaMA和Qwen的发布促进了针对各种任务和语言的众多微调模型的发展。本文探讨了一个重要问题：是否可以将这些专业化模型结合成一个具有多任务能力的统一模型。我们介绍了层次迭代合并（Hi-Merging），这是一种无需训练的方法，可将不同专业化LLMs统一到单一模型中。具体而言，Hi-Merging利用基于贡献分析的模型级和层级修剪与放缩来缓解参数冲突。在多种选择和问答任务中的汉语和英语实验广泛验证了Hi-Merging在多任务学习方面的有效性。结果表明，Hi-Merging在大多数场景下优于现有合并技术，且在大多数情况下超越了在综合数据集上微调的模型的性能。代码已发布于：this https URL。 

---
# Component Based Quantum Machine Learning Explainability 

**Title (ZH)**: 基于组件的量子机器学习可解释性 

**Authors**: Barra White, Krishnendu Guha  

**Link**: [PDF](https://arxiv.org/pdf/2506.12378)  

**Abstract**: Explainable ML algorithms are designed to provide transparency and insight into their decision-making process. Explaining how ML models come to their prediction is critical in fields such as healthcare and finance, as it provides insight into how models can help detect bias in predictions and help comply with GDPR compliance in these fields. QML leverages quantum phenomena such as entanglement and superposition, offering the potential for computational speedup and greater insights compared to classical ML. However, QML models also inherit the black-box nature of their classical counterparts, requiring the development of explainability techniques to be applied to these QML models to help understand why and how a particular output was generated.
This paper will explore the idea of creating a modular, explainable QML framework that splits QML algorithms into their core components, such as feature maps, variational circuits (ansatz), optimizers, kernels, and quantum-classical loops. Each component will be analyzed using explainability techniques, such as ALE and SHAP, which have been adapted to analyse the different components of these QML algorithms. By combining insights from these parts, the paper aims to infer explainability to the overall QML model. 

**Abstract (ZH)**: 可解释的机器学习算法旨在提供其决策过程的透明度和洞察力。在医疗保健和金融等领域，解释机器学习模型的预测过程至关重要，因为它有助于发现预测中的偏差，并有助于这些领域遵守GDPR合规要求。量子机器学习（QML）利用诸如纠缠和叠加等量子现象，有可能比经典机器学习提供更快的计算加速和更多的洞察力。然而，QML模型也继承了其经典 counterparts的黑盒性质，因此需要开发解释性技术，以便理解特定输出是如何生成的。

本文将探讨创建模块化的可解释量子机器学习框架的想法，该框架将QML算法分解为其核心组件，如特征映射、变分电路（Ansatz）、优化器、核函数和量子-经典循环。每种组件都将使用可解释性技术进行分析，如ALE和SHAP，这些技术已被改编以分析这些QML算法的不同组件。通过结合这些部分的见解，本文旨在推断整体QML模型的可解释性。 

---
# Optimized Spectral Fault Receptive Fields for Diagnosis-Informed Prognosis 

**Title (ZH)**: 优化的光谱故障感受野用于诊断驱动的预后 

**Authors**: Stan Muñoz Gutiérrez, Franz Wotawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12375)  

**Abstract**: This paper introduces Spectral Fault Receptive Fields (SFRFs), a biologically inspired technique for degradation state assessment in bearing fault diagnosis and remaining useful life (RUL) estimation. Drawing on the center-surround organization of retinal ganglion cell receptive fields, we propose a frequency-domain feature extraction algorithm that enhances the detection of fault signatures in vibration signals. SFRFs are designed as antagonistic spectral filters centered on characteristic fault frequencies, with inhibitory surrounds that enable robust characterization of incipient faults under variable operating conditions. A multi-objective evolutionary optimization strategy based on NSGA-II algorithm is employed to tune the receptive field parameters by simultaneously minimizing RUL prediction error, maximizing feature monotonicity, and promoting smooth degradation trajectories. The method is demonstrated on the XJTU-SY bearing run-to-failure dataset, confirming its suitability for constructing condition indicators in health monitoring applications. Key contributions include: (i) the introduction of SFRFs, inspired by the biology of vision in the primate retina; (ii) an evolutionary optimization framework guided by condition monitoring and prognosis criteria; and (iii) experimental evidence supporting the detection of early-stage faults and their precursors. Furthermore, we confirm that our diagnosis-informed spectral representation achieves accurate RUL prediction using a bagging regressor. The results highlight the interpretability and principled design of SFRFs, bridging signal processing, biological sensing principles, and data-driven prognostics in rotating machinery. 

**Abstract (ZH)**: 基于生物启发的谱故障感受野在轴承故障诊断和剩余使用寿命评估中的应用：一种降解状态评估和剩余使用寿命（RUL）估计的生物启发技术 

---
# AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making 

**Title (ZH)**: 抗地基：将机器人动作提升至跨模态学习表示空间中的决策制定 

**Authors**: Wenbo Li, Shiyi Wang, Yiteng Chen, Huiping Zhuang, Qingyao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12374)  

**Abstract**: Vision-Language Models (VLMs) encode knowledge and reasoning capabilities for robotic manipulation within high-dimensional representation spaces. However, current approaches often project them into compressed intermediate representations, discarding important task-specific information such as fine-grained spatial or semantic details. To address this, we propose AntiGrounding, a new framework that reverses the instruction grounding process. It lifts candidate actions directly into the VLM representation space, renders trajectories from multiple views, and uses structured visual question answering for instruction-based decision making. This enables zero-shot synthesis of optimal closed-loop robot trajectories for new tasks. We also propose an offline policy refinement module that leverages past experience to enhance long-term performance. Experiments in both simulation and real-world environments show that our method outperforms baselines across diverse robotic manipulation tasks. 

**Abstract (ZH)**: Vision-Language模型（VLMs）在高维表示空间中编码了机器人操作的知识和推理能力。然而，当前的方法通常将这些模型投影到压缩的中间表示中，丢弃了重要的任务特定信息，如细粒度的空间或语义细节。为了解决这一问题，我们提出了一种新的框架AntiGrounding，该框架逆转了指令锚定过程。它直接将候选操作提升到VLM表示空间，从多视角渲染轨迹，并使用结构化视觉问答进行基于指令的决策。这使得我们的方法能够零样本合成新的任务最优闭环机器人轨迹。我们还提出了一种 Offline 策略细化模块，利用过去的经验来增强长期性能。在模拟和真实环境中的实验表明，我们的方法在各种机器人操作任务中均优于基线方法。 

---
# HYPER: A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs 

**Title (ZH)**: HYPER：一种基于知识超图的归纳链接预测基础模型 

**Authors**: Xingyue Huang, Mikhail Galkin, Michael M. Bronstein, İsmail İlkan Ceylan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12362)  

**Abstract**: Inductive link prediction with knowledge hypergraphs is the task of predicting missing hyperedges involving completely novel entities (i.e., nodes unseen during training). Existing methods for inductive link prediction with knowledge hypergraphs assume a fixed relational vocabulary and, as a result, cannot generalize to knowledge hypergraphs with novel relation types (i.e., relations unseen during training). Inspired by knowledge graph foundation models, we propose HYPER as a foundation model for link prediction, which can generalize to any knowledge hypergraph, including novel entities and novel relations. Importantly, HYPER can learn and transfer across different relation types of varying arities, by encoding the entities of each hyperedge along with their respective positions in the hyperedge. To evaluate HYPER, we construct 16 new inductive datasets from existing knowledge hypergraphs, covering a diverse range of relation types of varying arities. Empirically, HYPER consistently outperforms all existing methods in both node-only and node-and-relation inductive settings, showing strong generalization to unseen, higher-arity relational structures. 

**Abstract (ZH)**: 基于知识超图的归纳链接预测任务是预测涉及完全全新的实体（即训练时未见过的节点）的缺失超边。现有的基于知识超图的归纳链接预测方法假设存在固定的关系词汇表，因此无法泛化到包含新关系类型（即训练时未见过的关系）的知识超图。受知识图谱基础模型的启发，我们提出了HYPER作为链接预测的基础模型，它可以泛化到任何知识超图，包括新的实体和新的关系。重要的是，HYPER可以通过编码每个超边中的实体及其在超边中的相对位置，来学习和迁移不同类型的不同元关系。为了评估HYPER，我们从现有的知识超图构建了16个新的归纳数据集，覆盖了不同类型和不同元数的关系。实验证明，HYPER在节点-only和节点-关系的归纳设置中都显著优于现有方法，显示出对未见过的高元数关系结构的强大泛化能力。 

---
# Efficient Reasoning Through Suppression of Self-Affirmation Reflections in Large Reasoning Models 

**Title (ZH)**: 高效的推理通过抑制大型推理模型中的自我肯定反思 

**Authors**: Kaiyuan Liu, Chen Shen, Zhanwei Zhang, Junjie Liu, Xiaosong Yuan, Jieping ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.12353)  

**Abstract**: While recent advances in large reasoning models have demonstrated remarkable performance, efficient reasoning remains critical due to the rapid growth of output length. Existing optimization approaches highlights a tendency toward "overthinking", yet lack fine-grained analysis. In this work, we focus on Self-Affirmation Reflections: redundant reflective steps that affirm prior content and often occurs after the already correct reasoning steps. Observations of both original and optimized reasoning models reveal pervasive self-affirmation reflections. Notably, these reflections sometimes lead to longer outputs in optimized models than their original counterparts. Through detailed analysis, we uncover an intriguing pattern: compared to other reflections, the leading words (i.e., the first word of sentences) in self-affirmation reflections exhibit a distinct probability bias. Motivated by this insight, we can locate self-affirmation reflections and conduct a train-free experiment demonstrating that suppressing self-affirmation reflections reduces output length without degrading accuracy across multiple models (R1-Distill-Models, QwQ-32B, and Qwen3-32B). Furthermore, we also improve current train-based method by explicitly suppressing such reflections. In our experiments, we achieve length compression of 18.7\% in train-free settings and 50.2\% in train-based settings for R1-Distill-Qwen-1.5B. Moreover, our improvements are simple yet practical and can be directly applied to existing inference frameworks, such as vLLM. We believe that our findings will provide community insights for achieving more precise length compression and step-level efficient reasoning. 

**Abstract (ZH)**: 尽管大规模推理模型最近取得了显著的性能提升，但由于输出长度的快速增长，高效的推理仍然至关重要。现有的优化方法倾向于“过度思考”，但缺乏精细分析。在本文中，我们专注于自我肯定反思：这些反思步骤重复确认先前的内容，并且往往发生在已经正确的推理步骤之后。原始推理模型和优化后的推理模型的观察结果都揭示了普遍存在自我肯定反思的现象。值得注意的是，有时在优化模型中，这些反思会导致比原始模型更长的输出。通过详细的分析，我们发现了一个有趣的现象：与其它反思相比，自我肯定反思中句子的引导词（即句子的第一个词）表现出明显不同的概率偏见。基于这一洞察，我们能够定位自我肯定反思，并通过一项无需训练的实验表明，抑制自我肯定反思可以降低输出长度而不影响多个模型（R1-Distill-Models, QwQ-32B, 和 Qwen3-32B）的准确性。此外，我们还改进了现有的基于训练的方法，显式地抑制这些反思。在我们的实验中，在无需训练的设置下，我们实现了18.7%的长度压缩，而在基于训练的设置下，我们实现了50.2%的长度压缩，针对的是R1-Distill-Qwen-1.5B模型。此外，我们的改进简单而实用，可以直接应用于现有的推理框架，如vLLM。我们认为，我们的发现将为实现更精确的长度压缩和步级高效推理提供社区洞察。 

---
# Theoretical Tensions in RLHF: Reconciling Empirical Success with Inconsistencies in Social Choice Theory 

**Title (ZH)**: RLHF中的理论紧张关系：调和实证成功与社会选择理论中的不一致性 

**Authors**: Jiancong Xiao, Zhekun Shi, Kaizhao Liu, Qi Long, Weijie J. Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.12350)  

**Abstract**: Despite its empirical success, Reinforcement Learning from Human Feedback (RLHF) has been shown to violate almost all the fundamental axioms in social choice theory -- such as majority consistency, pairwise majority consistency, and Condorcet consistency. This raises a foundational question: why does RLHF perform so well in practice if it fails these seemingly essential properties? In this paper, we resolve this paradox by showing that under mild and empirically plausible assumptions on the preference profile, RLHF does satisfy pairwise majority and Condorcet consistency. These assumptions are frequently satisfied in real-world alignment tasks, offering a theoretical explanation for RLHF's strong practical performance. Furthermore, we show that a slight modification to the reward modeling objective can ensure pairwise majority or Condorcet consistency even under general preference profiles, thereby improving the alignment process. Finally, we go beyond classical axioms in economic and social choice theory and introduce new alignment criteria -- preference matching, preference equivalence, and group preference matching -- that better reflect the goal of learning distributions over responses. We show that while RLHF satisfies the first two properties, it fails to satisfy the third. We conclude by discussing how future alignment methods may be designed to satisfy all three. 

**Abstract (ZH)**: 尽管强化学习从人类反馈中学习（RLHF）在实践中表现出色，但已被证明违反了社会选择理论中的几乎全部基本公理——如多数一致性、双边多数一致性及康德尔一致性。这引发了基础性问题：如果RLHF在这些看似至关重要的属性上失败了，那么它为何在实践中表现如此出色？本文通过在偏好配置图下提出温和且符合经验的假设，展示了RLHF实际上满足双边多数一致性和康德尔一致性。这些假设在实际对齐任务中经常被满足，为RLHF的优秀实践性能提供了理论解释。此外，我们证明通过对奖励建模目标进行 slight 修改，即便在一般的偏好配置图下也能确保双边多数一致性和康德尔一致性，从而改进对齐过程。最后，我们超越了经济学和社会选择理论中的经典公理，并引入了新的对齐标准——偏好匹配、偏好等价和群体偏好匹配，这些标准更符合学习响应分布的目标。我们发现尽管RLHF满足前两个属性，但未能满足第三个属性。我们讨论了未来对齐方法如何设计以满足所有三个属性。 

---
# Information Suppression in Large Language Models: Auditing, Quantifying, and Characterizing Censorship in DeepSeek 

**Title (ZH)**: 大型语言模型中的信息抑制：审计、量化和表征DeepSeek中的审查制度 

**Authors**: Peiran Qiu, Siyi Zhou, Emilio Ferrara  

**Link**: [PDF](https://arxiv.org/pdf/2506.12349)  

**Abstract**: This study examines information suppression mechanisms in DeepSeek, an open-source large language model (LLM) developed in China. We propose an auditing framework and use it to analyze the model's responses to 646 politically sensitive prompts by comparing its final output with intermediate chain-of-thought (CoT) reasoning. Our audit unveils evidence of semantic-level information suppression in DeepSeek: sensitive content often appears within the model's internal reasoning but is omitted or rephrased in the final output. Specifically, DeepSeek suppresses references to transparency, government accountability, and civic mobilization, while occasionally amplifying language aligned with state propaganda. This study underscores the need for systematic auditing of alignment, content moderation, information suppression, and censorship practices implemented into widely-adopted AI models, to ensure transparency, accountability, and equitable access to unbiased information obtained by means of these systems. 

**Abstract (ZH)**: 本研究 examines 了在中国开发的开源大语言模型 DeepSeek 中的信息抑制机制。我们提出了一套审计框架，并通过将模型对 646 个政治敏感提示的最终输出与其中间推理过程（CoT）进行比较来分析其响应。我们的审计揭示了 DeepSeek 中存在语义层面的信息抑制：敏感内容往往出现在模型的内部推理中，但在最终输出中被省略或重新表达。具体而言，DeepSeek 抑制了透明度、政府问责制和公民动员的提及，偶尔则强化了与官方宣传一致的语言。本研究强调了对广泛采用的 AI 模型中对齐、内容审核、信息抑制和审查实践进行系统审计的必要性，以确保通过这些系统获取的无偏见信息的透明度、问责制和公平访问。 

---
# Refract ICL: Rethinking Example Selection in the Era of Million-Token Models 

**Title (ZH)**: Refract ICL：在百万Token模型时代重新思考示例选择 

**Authors**: Arjun R. Akula, Kazuma Hashimoto, Krishna Srinivasan, Aditi Chaudhary, Karthik Raman, Michael Bendersky  

**Link**: [PDF](https://arxiv.org/pdf/2506.12346)  

**Abstract**: The emergence of long-context large language models (LLMs) has enabled the use of hundreds, or even thousands, of demonstrations for in-context learning (ICL) - a previously impractical regime. This paper investigates whether traditional ICL selection strategies, which balance the similarity of ICL examples to the test input (using a text retriever) with diversity within the ICL set, remain effective when utilizing a large number of demonstrations. Our experiments demonstrate that, while longer contexts can accommodate more examples, simply increasing the number of demonstrations does not guarantee improved performance. Smart ICL selection remains crucial, even with thousands of demonstrations. To further enhance ICL in this setting, we introduce Refract ICL, a novel ICL selection algorithm specifically designed to focus LLM attention on challenging examples by strategically repeating them within the context and incorporating zero-shot predictions as error signals. Our results show that Refract ICL significantly improves the performance of extremely long-context models such as Gemini 1.5 Pro, particularly on tasks with a smaller number of output classes. 

**Abstract (ZH)**: 长上下文大型语言模型的出现使得使用数百甚至数千个示范进行上下文内学习（ICL）成为可能，这是一种以前不可行的模式。本文探讨了传统ICL选择策略在利用大量示范时的有效性，这些策略平衡了ICL示例与测试输入的相似性（使用文本检索器）与ICL集合内的多样性。我们的实验表明，虽然更长的上下文可以容纳更多的示例，但仅增加示范数量并不能保证性能改善。即使在有数千个示范的情况下，智能ICL选择仍然至关重要。为进一步提高在这种设置下的ICL效果，我们提出了Refract ICL，这是一种新型ICL选择算法，专门设计通过在上下文中战略性地重复具有挑战性的示例并结合零样本预测作为错误信号来聚焦LLM的注意力。我们的结果表明，Refract ICL显著改善了如Gemini 1.5 Pro等极长上下文模型的表现，特别是在输出类别的数量较小的任务中。 

---
# SheetMind: An End-to-End LLM-Powered Multi-Agent Framework for Spreadsheet Automation 

**Title (ZH)**: SheetMind：一个基于大语言模型的端到端多agent电子表格自动化框架 

**Authors**: Ruiyan Zhu, Xi Cheng, Ke Liu, Brian Zhu, Daniel Jin, Neeraj Parihar, Zhoutian Xu, Oliver Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12339)  

**Abstract**: We present SheetMind, a modular multi-agent framework powered by large language models (LLMs) for spreadsheet automation via natural language instructions. The system comprises three specialized agents: a Manager Agent that decomposes complex user instructions into subtasks; an Action Agent that translates these into structured commands using a Backus Naur Form (BNF) grammar; and a Reflection Agent that validates alignment between generated actions and the user's original intent. Integrated into Google Sheets via a Workspace extension, SheetMind supports real-time interaction without requiring scripting or formula knowledge. Experiments on benchmark datasets demonstrate an 80 percent success rate on single step tasks and approximately 70 percent on multi step instructions, outperforming ablated and baseline variants. Our results highlight the effectiveness of multi agent decomposition and grammar based execution for bridging natural language and spreadsheet functionalities. 

**Abstract (ZH)**: SheetMind：基于大规模语言模型的模块化多代理框架，通过自然语言指令实现电子表格自动化 

---
# GroupNL: Low-Resource and Robust CNN Design over Cloud and Device 

**Title (ZH)**: GroupNL: 云和设备上的低资源和 robust CNN 设计 

**Authors**: Chuntao Ding, Jianhang Xie, Junna Zhang, Salman Raza, Shangguang Wang, Jiannong Cao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12335)  

**Abstract**: It has become mainstream to deploy Convolutional Neural Network (CNN) models on ubiquitous Internet of Things (IoT) devices with the help of the cloud to provide users with a variety of high-quality services. Most existing methods have two limitations: (i) low robustness in handling corrupted image data collected by IoT devices; and (ii) high consumption of computational and transmission resources. To this end, we propose the Grouped NonLinear transformation generation method (GroupNL), which generates diversified feature maps by utilizing data-agnostic Nonlinear Transformation Functions (NLFs) to improve the robustness of the CNN model. Specifically, partial convolution filters are designated as seed filters in a convolutional layer, and a small set of feature maps, i.e., seed feature maps, are first generated based on vanilla convolution operation. Then, we split seed feature maps into several groups, each with a set of different NLFs, to generate corresponding diverse feature maps with in-place nonlinear processing. Moreover, GroupNL effectively reduces the parameter transmission between multiple nodes during model training by setting the hyperparameters of NLFs to random initialization and not updating them during model training, and reduces the computing resources by using NLFs to generate feature maps instead of most feature maps generated based on sliding windows. Experimental results on CIFAR-10, GTSRB, CIFAR-10-C, Icons50, and ImageNet-1K datasets in NVIDIA RTX GPU platforms show that the proposed GroupNL outperforms other state-of-the-art methods in model robust and training acceleration. Specifically, on the Icons-50 dataset, the accuracy of GroupNL-ResNet-18 achieves approximately 2.86% higher than the vanilla ResNet-18. GroupNL improves training speed by about 53% compared to vanilla CNN when trained on a cluster of 8 NVIDIA RTX 4090 GPUs on the ImageNet-1K dataset. 

**Abstract (ZH)**: 利用非线性变换生成方法提高物联网设备上卷积神经网络模型的鲁棒性和训练加速 

---
# IndoorWorld: Integrating Physical Task Solving and Social Simulation in A Heterogeneous Multi-Agent Environment 

**Title (ZH)**: IndoorWorld: 在异构多agent环境中整合物理任务解决与社会仿真 

**Authors**: Dekun Wu, Frederik Brudy, Bang Liu, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12331)  

**Abstract**: Virtual environments are essential to AI agent research. Existing environments for LLM agent research typically focus on either physical task solving or social simulation, with the former oversimplifying agent individuality and social dynamics, and the latter lacking physical grounding of social behaviors. We introduce IndoorWorld, a heterogeneous multi-agent environment that tightly integrates physical and social dynamics. By introducing novel challenges for LLM-driven agents in orchestrating social dynamics to influence physical environments and anchoring social interactions within world states, IndoorWorld opens up possibilities of LLM-based building occupant simulation for architectural design. We demonstrate the potential with a series of experiments within an office setting to examine the impact of multi-agent collaboration, resource competition, and spatial layout on agent behavior. 

**Abstract (ZH)**: 虚拟环境对于AI代理研究至关重要。现有的LLM代理研究环境通常侧重于物理任务解决或社会模拟，前者过度简化了代理的个体性和社会动态，后者缺乏社会行为的物理基础。我们介绍了IndoorWorld，这是一个将物理和社会动态紧密结合的异构多代理环境。通过在IndoorWorld中引入新型挑战，使LLM驱动的代理能够协调社会动态以影响物理环境，并将社会互动锚定在世界状态中，为建筑设计中的基于LLM的建筑占用者模拟打开了可能性。我们通过一系列实验，在办公室环境中探讨了多代理协作、资源竞争和空间布局对代理行为的影响。 

---
# Intersectional Bias in Japanese Large Language Models from a Contextualized Perspective 

**Title (ZH)**: 从上下文视角看日本大型语言模型中的交织偏见 

**Authors**: Hitomi Yanaka, Xinqi He, Jie Lu, Namgi Han, Sunjin Oh, Ryoma Kumon, Yuma Matsuoka, Katsuhiko Watabe, Yuko Itatsu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12327)  

**Abstract**: An growing number of studies have examined the social bias of rapidly developed large language models (LLMs). Although most of these studies have focused on bias occurring in a single social attribute, research in social science has shown that social bias often occurs in the form of intersectionality -- the constitutive and contextualized perspective on bias aroused by social attributes. In this study, we construct the Japanese benchmark inter-JBBQ, designed to evaluate the intersectional bias in LLMs on the question-answering setting. Using inter-JBBQ to analyze GPT-4o and Swallow, we find that biased output varies according to its contexts even with the equal combination of social attributes. 

**Abstract (ZH)**: 越来越多的研究关注快速发展的大规模语言模型（LLMs）的社会偏见。尽管大多数研究都集中在单一社会属性引起的偏见上，社会科学研究表明，社会偏见往往以多元交叉的形式出现——即由多种社会属性构成并情境化的偏见视角。在本研究中，我们构建了日语基准benchmark inter-JBBQ，旨在评估问答设置中LLMs的多元交叉偏见。利用inter-JBBQ分析GPT-4o和Swallow，我们发现，即使在社会属性组合相同的情况下，偏见输出也会根据不同情境而变化。 

---
# Three-dimensional Deep Shape Optimization with a Limited Dataset 

**Title (ZH)**: 有限数据集下的三维深度形状优化 

**Authors**: Yongmin Kwon, Namwoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12326)  

**Abstract**: Generative models have attracted considerable attention for their ability to produce novel shapes. However, their application in mechanical design remains constrained due to the limited size and variability of available datasets. This study proposes a deep learning-based optimization framework specifically tailored for shape optimization with limited datasets, leveraging positional encoding and a Lipschitz regularization term to robustly learn geometric characteristics and maintain a meaningful latent space. Through extensive experiments, the proposed approach demonstrates robustness, generalizability and effectiveness in addressing typical limitations of conventional optimization frameworks. The validity of the methodology is confirmed through multi-objective shape optimization experiments conducted on diverse three-dimensional datasets, including wheels and cars, highlighting the model's versatility in producing practical and high-quality design outcomes even under data-constrained conditions. 

**Abstract (ZH)**: 基于深度学习的有限数据集形状优化框架：利用位置编码和Lipschitz正则化项 robustly学习几何特征并保持有意义的隐空间 

---
# Machine Learning Methods for Small Data and Upstream Bioprocessing Applications: A Comprehensive Review 

**Title (ZH)**: 小型数据和上游生物处理应用的机器学习方法：综述 

**Authors**: Johnny Peng, Thanh Tung Khuat, Katarzyna Musial, Bogdan Gabrys  

**Link**: [PDF](https://arxiv.org/pdf/2506.12322)  

**Abstract**: Data is crucial for machine learning (ML) applications, yet acquiring large datasets can be costly and time-consuming, especially in complex, resource-intensive fields like biopharmaceuticals. A key process in this industry is upstream bioprocessing, where living cells are cultivated and optimised to produce therapeutic proteins and biologics. The intricate nature of these processes, combined with high resource demands, often limits data collection, resulting in smaller datasets. This comprehensive review explores ML methods designed to address the challenges posed by small data and classifies them into a taxonomy to guide practical applications. Furthermore, each method in the taxonomy was thoroughly analysed, with a detailed discussion of its core concepts and an evaluation of its effectiveness in tackling small data challenges, as demonstrated by application results in the upstream bioprocessing and other related domains. By analysing how these methods tackle small data challenges from different perspectives, this review provides actionable insights, identifies current research gaps, and offers guidance for leveraging ML in data-constrained environments. 

**Abstract (ZH)**: 基于小数据挑战的机器学习方法在上游生物制药过程中的应用综述 

---
# Extending Memorization Dynamics in Pythia Models from Instance-Level Insights 

**Title (ZH)**: Pythia模型中记忆动态的实例级扩展 

**Authors**: Jie Zhang, Qinghua Zhao, Lei Li, Chi-ho Lin  

**Link**: [PDF](https://arxiv.org/pdf/2506.12321)  

**Abstract**: Large language models have demonstrated a remarkable ability for verbatim memorization. While numerous works have explored factors influencing model memorization, the dynamic evolution memorization patterns remains underexplored. This paper presents a detailed analysis of memorization in the Pythia model family across varying scales and training steps under prefix perturbations. Using granular metrics, we examine how model architecture, data characteristics, and perturbations influence these patterns. Our findings reveal that: (1) as model scale increases, memorization expands incrementally while efficiency decreases rapidly; (2) as model scale increases, the rate of new memorization acquisition decreases while old memorization forgetting increases; (3) data characteristics (token frequency, repetition count, and uncertainty) differentially affect memorized versus non-memorized samples; and (4) prefix perturbations reduce memorization and increase generation uncertainty proportionally to perturbation strength, with low-redundancy samples showing higher vulnerability and larger models offering no additional robustness. These findings advance our understanding of memorization mechanisms, with direct implications for training optimization, privacy safeguards, and architectural improvements. 

**Abstract (ZH)**: 大型语言模型展现了惊人的逐字记忆能力。虽然已有众多研究探索影响模型记忆的各种因素，但记忆模式的动力学演化仍处于研究不足的状态。本文通过前缀扰动，在不同规模和训练步数下对Pythia模型家族的记忆机制进行了详细的分析。利用细粒度的指标，我们探讨了模型架构、数据特征和扰动如何影响这些模式。研究发现：（1）随着模型规模的增加，记忆逐步扩大而效率迅速降低；（2）随着模型规模的增加，新记忆的获取速率下降而旧记忆的遗忘率上升；（3）数据特征（词元频率、重复次数和不确定性）对已记忆样本和未记忆样本的影响有所不同；（4）前缀扰动按扰动强度成比例地减少记忆并增加生成不确定性，低冗余度样本更加脆弱，而大模型并不能提供额外的鲁棒性。这些发现深化了我们对记忆机制的理解，直接为训练优化、隐私保护和架构改进提供了指导。 

---
# The Foundation Cracks: A Comprehensive Study on Bugs and Testing Practices in LLM Libraries 

**Title (ZH)**: 基础的裂缝：对大型语言模型库中的Bug和测试实践的综合研究 

**Authors**: Weipeng Jiang, Xiaoyu Zhang, Xiaofei Xie, Jiongchi Yu, Yuhan Zhi, Shiqing Ma, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12320)  

**Abstract**: Large Language Model (LLM) libraries have emerged as the foundational infrastructure powering today's AI revolution, serving as the backbone for LLM deployment, inference optimization, fine-tuning, and production serving across diverse applications. Despite their critical role in the LLM ecosystem, these libraries face frequent quality issues and bugs that threaten the reliability of AI systems built upon them. To address this knowledge gap, we present the first comprehensive empirical investigation into bug characteristics and testing practices in modern LLM libraries. We examine 313 bug-fixing commits extracted across two widely-adopted LLM libraries: HuggingFace Transformers and this http URL rigorous manual analysis, we establish comprehensive taxonomies categorizing bug symptoms into 5 types and root causes into 14 distinct this http URL primary discovery shows that API misuse has emerged as the predominant root cause (32.17%-48.19%), representing a notable transition from algorithm-focused defects in conventional deep learning frameworks toward interface-oriented problems. Additionally, we examine 7,748 test functions to identify 7 distinct test oracle categories employed in current testing approaches, with predefined expected outputs (such as specific tensors and text strings) being the most common strategy. Our assessment of existing testing effectiveness demonstrates that the majority of bugs escape detection due to inadequate test cases (41.73%), lack of test drivers (32.37%), and weak test oracles (25.90%). Drawing from these findings, we offer some recommendations for enhancing LLM library quality assurance. 

**Abstract (ZH)**: 现代大型语言模型库中的Bug特征及测试实践的首个全面实证研究 

---
# Med-U1: Incentivizing Unified Medical Reasoning in LLMs via Large-scale Reinforcement Learning 

**Title (ZH)**: Med-U1: 通过大规模强化学习激励LLM统一医疗推理 

**Authors**: Xiaotian Zhang, Yuan Wang, Zhaopeng Feng, Ruizhe Chen, Zhijie Zhou, Yan Zhang, Hongxia Xu, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12307)  

**Abstract**: Medical Question-Answering (QA) encompasses a broad spectrum of tasks, including multiple choice questions (MCQ), open-ended text generation, and complex computational reasoning. Despite this variety, a unified framework for delivering high-quality medical QA has yet to emerge. Although recent progress in reasoning-augmented large language models (LLMs) has shown promise, their ability to achieve comprehensive medical understanding is still largely unexplored. In this paper, we present Med-U1, a unified framework for robust reasoning across medical QA tasks with diverse output formats, ranging from MCQs to complex generation and computation tasks. Med-U1 employs pure large-scale reinforcement learning with mixed rule-based binary reward functions, incorporating a length penalty to manage output verbosity. With multi-objective reward optimization, Med-U1 directs LLMs to produce concise and verifiable reasoning chains. Empirical results reveal that Med-U1 significantly improves performance across multiple challenging Med-QA benchmarks, surpassing even larger specialized and proprietary models. Furthermore, Med-U1 demonstrates robust generalization to out-of-distribution (OOD) tasks. Extensive analysis presents insights into training strategies, reasoning chain length control, and reward design for medical LLMs. The code will be released. 

**Abstract (ZH)**: 医疗问答(QA)涵盖了广泛的任务，包括多项选择题(MCQ)、开放文本生成和复杂计算推理。尽管如此，至今尚未出现统一的高质医疗QA框架。尽管增强推理的大语言模型(LLM)取得了进展，但其在全面医疗理解方面的能力仍有待探索。在本文中，我们提出了一种统一框架Med-U1，用于在多种输出格式下的医疗QA任务中进行健壮的推理，从MCQ到复杂的生成和计算任务。Med-U1采用纯大规模强化学习，并结合混合规则基础二元奖励函数，同时加入长度惩罚以管理输出冗长。通过多目标奖励优化，Med-U1引导大语言模型产出简洁且可验证的推理链。实验证明，Med-U1在多个挑战性医疗QA基准测试中显著提升了性能，甚至超过了更大规模的专业化和专有模型。此外，Med-U1展示了在分布外(OOD)任务上的稳健泛化能力。详尽的分析提供了关于医疗大语言模型训练策略、推理链长度控制和奖励设计的见解。代码将公开发布。 

---
# Unveiling Confirmation Bias in Chain-of-Thought Reasoning 

**Title (ZH)**: 揭示链式推理中的确认偏见 

**Authors**: Yue Wan, Xiaowei Jia, Xiang Lorraine Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.12301)  

**Abstract**: Chain-of-thought (CoT) prompting has been widely adopted to enhance the reasoning capabilities of large language models (LLMs). However, the effectiveness of CoT reasoning is inconsistent across tasks with different reasoning types. This work presents a novel perspective to understand CoT behavior through the lens of \textit{confirmation bias} in cognitive psychology. Specifically, we examine how model internal beliefs, approximated by direct question-answering probabilities, affect both reasoning generation ($Q \to R$) and reasoning-guided answer prediction ($QR \to A$) in CoT. By decomposing CoT into a two-stage process, we conduct a thorough correlation analysis in model beliefs, rationale attributes, and stage-wise performance. Our results provide strong evidence of confirmation bias in LLMs, such that model beliefs not only skew the reasoning process but also influence how rationales are utilized for answer prediction. Furthermore, the interplay between task vulnerability to confirmation bias and the strength of beliefs also provides explanations for CoT effectiveness across reasoning tasks and models. Overall, this study provides a valuable insight for the needs of better prompting strategies that mitigate confirmation bias to enhance reasoning performance. Code is available at \textit{this https URL}. 

**Abstract (ZH)**: Chain-of-thought (CoT) 推动在增强大规模语言模型推理能力方面的应用已得到广泛应用。然而，CoT 推理在不同推理类型的任务中的效果不一致。本文通过认知心理学中的确认偏差视角，提出了理解 CoT 行为的新型视角。具体而言，我们考察了由直接问答概率近似表示的模型内部信念，如何影响 CoT 中的推理生成（$Q \to R$）和推理引导的答案预测（$QR \to A$）。通过将 CoT 分解为两阶段过程，我们对模型信念、推理属性以及阶段性能进行了彻底的相关性分析。我们的结果提供了大规模语言模型中确认偏差的强烈证据，表明模型信念不仅扭曲了推理过程，还影响了推理在答案预测中的利用方式。此外，任务对确认偏差的脆弱性与信念强度之间的相互作用也为跨推理任务和模型的 CoT 效果提供了解释。总体而言，这项研究为更好地缓解确认偏差以提升推理性能提供了有价值的见解。代码可在 \textit{this https URL} 获取。 

---
# QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety 

**Title (ZH)**: QGuard：基于问题的多模态大语言模型零样本安全性保护方法 

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng  

**Link**: [PDF](https://arxiv.org/pdf/2506.12299)  

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts. 

**Abstract (ZH)**: 大型语言模型的 Recent Advancements 和恶意攻击：QGuard——一种基于问题提示的零样本安全防护方法 

---
# CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following 

**Title (ZH)**: CMI-Bench: 一个全面的音乐指令跟随评价基准 

**Authors**: Yinghao Ma, Siyou Li, Juntao Yu, Emmanouil Benetos, Akira Maezawa  

**Link**: [PDF](https://arxiv.org/pdf/2506.12285)  

**Abstract**: Recent advances in audio-text large language models (LLMs) have opened new possibilities for music understanding and generation. However, existing benchmarks are limited in scope, often relying on simplified tasks or multi-choice evaluations that fail to reflect the complexity of real-world music analysis. We reinterpret a broad range of traditional MIR annotations as instruction-following formats and introduce CMI-Bench, a comprehensive music instruction following benchmark designed to evaluate audio-text LLMs on a diverse set of music information retrieval (MIR) tasks. These include genre classification, emotion regression, emotion tagging, instrument classification, pitch estimation, key detection, lyrics transcription, melody extraction, vocal technique recognition, instrument performance technique detection, music tagging, music captioning, and (down)beat tracking: reflecting core challenges in MIR research. Unlike previous benchmarks, CMI-Bench adopts standardized evaluation metrics consistent with previous state-of-the-art MIR models, ensuring direct comparability with supervised approaches. We provide an evaluation toolkit supporting all open-source audio-textual LLMs, including LTU, Qwen-audio, SALMONN, MusiLingo, etc. Experiment results reveal significant performance gaps between LLMs and supervised models, along with their culture, chronological and gender bias, highlighting the potential and limitations of current models in addressing MIR tasks. CMI-Bench establishes a unified foundation for evaluating music instruction following, driving progress in music-aware LLMs. 

**Abstract (ZH)**: Recent advances in音频-文本大型语言模型（LLMs）为音乐理解与生成开辟了新的可能性。然而，现有的基准在范围上有限，通常依赖于简化任务或多项选择评估，未能反映实际音乐分析的复杂性。我们将传统MIR注释重新解释为指令跟随格式，并引入CMI-Bench，这是一个旨在评估音频-文本LLMs在一系列音乐信息检索（MIR）任务上的综合音乐指令跟随基准。这些任务包括流派分类、情绪回归、情绪标注、乐器分类、音高估计、调式检测、歌词转录、旋律提取、歌声技巧识别、乐器演奏技巧检测、音乐标注、音乐描述和（弱）拍追踪：反映MIR研究中的核心挑战。不同于以往的基准，CMI-Bench采用与之前最先进的MIR模型一致的标准化评估指标，确保与监督方法的直接可比性。我们提供了一个支持所有开源音频-文本LLMs的评估工具包，包括LTU、Qwen-audio、SALMONN、MusiLingo等。实验结果揭示了LLMs与监督模型之间在性能上的显著差距，以及它们的文化、年代和性别偏见，突显了当前模型在处理MIR任务中的潜力和局限性。CMI-Bench为音乐指令跟随的评估奠定了统一的基础，推动了音乐感知LLMs的进步。 

---
# The Behavior Gap: Evaluating Zero-shot LLM Agents in Complex Task-Oriented Dialogs 

**Title (ZH)**: 行为差距：评估零样本LLM代理在复杂任务导向对话中的表现 

**Authors**: Avinash Baidya, Kamalika Das, Xiang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12266)  

**Abstract**: Large Language Model (LLM)-based agents have significantly impacted Task-Oriented Dialog Systems (TODS) but continue to face notable performance challenges, especially in zero-shot scenarios. While prior work has noted this performance gap, the behavioral factors driving the performance gap remain under-explored. This study proposes a comprehensive evaluation framework to quantify the behavior gap between AI agents and human experts, focusing on discrepancies in dialog acts, tool usage, and knowledge utilization. Our findings reveal that this behavior gap is a critical factor negatively impacting the performance of LLM agents. Notably, as task complexity increases, the behavior gap widens (correlation: 0.963), leading to a degradation of agent performance on complex task-oriented dialogs. For the most complex task in our study, even the GPT-4o-based agent exhibits low alignment with human behavior, with low F1 scores for dialog acts (0.464), excessive and often misaligned tool usage with a F1 score of 0.139, and ineffective usage of external knowledge. Reducing such behavior gaps leads to significant performance improvement (24.3% on average). This study highlights the importance of comprehensive behavioral evaluations and improved alignment strategies to enhance the effectiveness of LLM-based TODS in handling complex tasks. 

**Abstract (ZH)**: 基于大语言模型（LLM）的代理在任务导向对话系统（TODS）中的显著影响但仍面临表现挑战：行为差距的探究 

---
# A Survey of Foundation Models for IoT: Taxonomy and Criteria-Based Analysis 

**Title (ZH)**: 物联网领域基础模型综述：分类与基于标准的分析 

**Authors**: Hui Wei, Dong Yoon Lee, Shubham Rohal, Zhizhang Hu, Shiwei Fang, Shijia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12263)  

**Abstract**: Foundation models have gained growing interest in the IoT domain due to their reduced reliance on labeled data and strong generalizability across tasks, which address key limitations of traditional machine learning approaches. However, most existing foundation model based methods are developed for specific IoT tasks, making it difficult to compare approaches across IoT domains and limiting guidance for applying them to new tasks. This survey aims to bridge this gap by providing a comprehensive overview of current methodologies and organizing them around four shared performance objectives by different domains: efficiency, context-awareness, safety, and security & privacy. For each objective, we review representative works, summarize commonly-used techniques and evaluation metrics. This objective-centric organization enables meaningful cross-domain comparisons and offers practical insights for selecting and designing foundation model based solutions for new IoT tasks. We conclude with key directions for future research to guide both practitioners and researchers in advancing the use of foundation models in IoT applications. 

**Abstract (ZH)**: 基础模型在物联网领域的兴趣逐渐增长，这得益于它们对标注数据的减少依赖及在不同任务上的强泛化能力，这解决了传统机器学习方法的关键局限性。然而，现有大多数基于基础模型的方法都是为特定的物联网任务开发的，这使得跨物联网领域比较方法变得困难，并限制了将这些方法应用于新任务的指导。本文综述旨在通过提供当前方法的全面概述，并围绕四个共享性能目标组织这些方法（由不同领域共享：效率、情境意识、安全性和安全与隐私），来弥补这一差距。对于每个目标，我们回顾代表性工作，总结常用技术和评估指标。这种目标中心化组织方式使得跨领域比较具有意义，并为选择和设计适用于新物联网任务的基础模型解决方案提供了实用洞察。最后，我们提出未来研究的关键方向，以指导实践者和研究人员在物联网应用中更广泛应用基础模型。 

---
# ProVox: Personalization and Proactive Planning for Situated Human-Robot Collaboration 

**Title (ZH)**: ProVox: 基于情境的个性化与主动规划的人机协作 

**Authors**: Jennifer Grannen, Siddharth Karamcheti, Blake Wulfe, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2506.12248)  

**Abstract**: Collaborative robots must quickly adapt to their partner's intent and preferences to proactively identify helpful actions. This is especially true in situated settings where human partners can continually teach robots new high-level behaviors, visual concepts, and physical skills (e.g., through demonstration), growing the robot's capabilities as the human-robot pair work together to accomplish diverse tasks. In this work, we argue that robots should be able to infer their partner's goals from early interactions and use this information to proactively plan behaviors ahead of explicit instructions from the user. Building from the strong commonsense priors and steerability of large language models, we introduce ProVox ("Proactive Voice"), a novel framework that enables robots to efficiently personalize and adapt to individual collaborators. We design a meta-prompting protocol that empowers users to communicate their distinct preferences, intent, and expected robot behaviors ahead of starting a physical interaction. ProVox then uses the personalized prompt to condition a proactive language model task planner that anticipates a user's intent from the current interaction context and robot capabilities to suggest helpful actions; in doing so, we alleviate user burden, minimizing the amount of time partners spend explicitly instructing and supervising the robot. We evaluate ProVox through user studies grounded in household manipulation tasks (e.g., assembling lunch bags) that measure the efficiency of the collaboration, as well as features such as perceived helpfulness, ease of use, and reliability. Our analysis suggests that both meta-prompting and proactivity are critical, resulting in 38.7% faster task completion times and 31.9% less user burden relative to non-active baselines. Supplementary material, code, and videos can be found at this https URL. 

**Abstract (ZH)**: 协作机器人必须迅速适应合作伙伴的意图和偏好，主动识别有益的动作。特别是在人类合作伙伴可以持续向机器人传授新的高级行为、视觉概念和物理技能（例如，通过示范）的情境中，这一点尤为重要，随着人机团队共同完成多样化任务，机器人的能力逐渐增强。在本研究中，我们主张机器人应能够从早期互动中推断出合作伙伴的目标，并利用这些信息在用户明确指示之前主动规划行为。基于大型语言模型的强大先验知识和可控性，我们引入了ProVox（“前瞻声音”），这是一种新型框架，使机器人能够有效地个性化并适应不同的合作者。我们设计了一种元提示协议，使用户能够在开始物理交互之前传达各自的偏好、意图和期望的机器人行为。ProVox 然后使用个性化的提示来条件化一个前瞻性的语言模型任务规划器，该规划器根据当前交互上下文和机器人能力预测用户意图以建议有益的动作；从而减轻用户负担，减少合作伙伴明确指导和监督机器人的时间。我们通过基于家庭操作任务（如组装午餐包）的用户研究来评估 ProVox，衡量协作效率以及诸如感知有效性、易用性和可靠性等特征。我们的分析表明，元提示和前瞻性都是至关重要的，相对非主动基线，任务完成时间加快了38.7%，用户负担减少了31.9%。更多信息、代码和视频请访问此链接。 

---
# Large Language Models for History, Philosophy, and Sociology of Science: Interpretive Uses, Methodological Challenges, and Critical Perspectives 

**Title (ZH)**: 大型语言模型在科学的历史、哲学和社会学中的阐释性应用、方法论挑战与批判性视角 

**Authors**: Arno Simons, Michael Zichert, Adrian Wüthrich  

**Link**: [PDF](https://arxiv.org/pdf/2506.12242)  

**Abstract**: This paper explores the use of large language models (LLMs) as research tools in the history, philosophy, and sociology of science (HPSS). LLMs are remarkably effective at processing unstructured text and inferring meaning from context, offering new affordances that challenge long-standing divides between computational and interpretive methods. This raises both opportunities and challenges for HPSS, which emphasizes interpretive methodologies and understands meaning as context-dependent, ambiguous, and historically situated. We argue that HPSS is uniquely positioned not only to benefit from LLMs' capabilities but also to interrogate their epistemic assumptions and infrastructural implications. To this end, we first offer a concise primer on LLM architectures and training paradigms tailored to non-technical readers. We frame LLMs not as neutral tools but as epistemic infrastructures that encode assumptions about meaning, context, and similarity, conditioned by their training data, architecture, and patterns of use. We then examine how computational techniques enhanced by LLMs, such as structuring data, detecting patterns, and modeling dynamic processes, can be applied to support interpretive research in HPSS. Our analysis compares full-context and generative models, outlines strategies for domain and task adaptation (e.g., continued pretraining, fine-tuning, and retrieval-augmented generation), and evaluates their respective strengths and limitations for interpretive inquiry in HPSS. We conclude with four lessons for integrating LLMs into HPSS: (1) model selection involves interpretive trade-offs; (2) LLM literacy is foundational; (3) HPSS must define its own benchmarks and corpora; and (4) LLMs should enhance, not replace, interpretive methods. 

**Abstract (ZH)**: 基于大型语言模型在科学史、科学哲学与科学社会学中的研究应用：挑战与机遇 

---
# Mind the XAI Gap: A Human-Centered LLM Framework for Democratizing Explainable AI 

**Title (ZH)**: 关注XAI差距：以人为本的LLM框架以实现可解释AI的普及 

**Authors**: Eva Paraschou, Ioannis Arapakis, Sofia Yfantidou, Sebastian Macaluso, Athena Vakali  

**Link**: [PDF](https://arxiv.org/pdf/2506.12240)  

**Abstract**: Artificial Intelligence (AI) is rapidly embedded in critical decision-making systems, however their foundational ``black-box'' models require eXplainable AI (XAI) solutions to enhance transparency, which are mostly oriented to experts, making no sense to non-experts. Alarming evidence about AI's unprecedented human values risks brings forward the imperative need for transparent human-centered XAI solutions. In this work, we introduce a domain-, model-, explanation-agnostic, generalizable and reproducible framework that ensures both transparency and human-centered explanations tailored to the needs of both experts and non-experts. The framework leverages Large Language Models (LLMs) and employs in-context learning to convey domain- and explainability-relevant contextual knowledge into LLMs. Through its structured prompt and system setting, our framework encapsulates in one response explanations understandable by non-experts and technical information to experts, all grounded in domain and explainability principles. To demonstrate the effectiveness of our framework, we establish a ground-truth contextual ``thesaurus'' through a rigorous benchmarking with over 40 data, model, and XAI combinations for an explainable clustering analysis of a well-being scenario. Through a comprehensive quality and human-friendliness evaluation of our framework's explanations, we prove high content quality through strong correlations with ground-truth explanations (Spearman rank correlation=0.92) and improved interpretability and human-friendliness to non-experts through a user study (N=56). Our overall evaluation confirms trust in LLMs as HCXAI enablers, as our framework bridges the above Gaps by delivering (i) high-quality technical explanations aligned with foundational XAI methods and (ii) clear, efficient, and interpretable human-centered explanations for non-experts. 

**Abstract (ZH)**: 人工智能（AI）正快速嵌入关键决策系统，然而其基础性的“黑盒”模型需要可解释人工智能（XAI）解决方案以增强透明度，这些解决方案目前主要面向专家，而非专家难以理解。关于人工智能前所未有的人类价值观风险的令人担忧的证据强调了迫切需要透明的人本中心XAI解决方案。本文介绍了一个领域无关、模型无关、解释无关的通用且可重现的框架，该框架确保同时具备透明度和针对专家及非专家需求的人本中心解释。该框架利用大型语言模型（LLMs）并采用上下文学习技术，将领域及解释相关的上下文知识融入LLMs。通过其结构化提示和系统设置，本框架将解释非专家和向专家提供技术信息的内容统一在一个响应中，所有内容均基于领域和解释原则。为了证明该框架的效果，我们通过超过40种数据、模型和XAI组合的严格基准测试，建立了一个真实的上下文“词典”，并进行了可解释聚类分析，以一个幸福感场景为例。通过全面的质量和人友好性评价，我们的框架解释证明了高度的内容质量（斯皮尔曼等级相关性=0.92），并通过用户研究（N=56）提高了对非专家的可解释性和人友好性。总体评价证实了LLMs作为人本中心XAI使能器的信任，本框架通过提供（i）与基础XAI方法对齐的高质量技术解释，以及（ii）为非专家提供的清晰、高效且可解释的人本中心解释而弥合了上述差距。 

---
# Datrics Text2SQL: A Framework for Natural Language to SQL Query Generation 

**Title (ZH)**: Datrics Text2SQL：一种自然语言到SQL查询生成的框架 

**Authors**: Tetiana Gladkykh, Kyrylo Kirykov  

**Link**: [PDF](https://arxiv.org/pdf/2506.12234)  

**Abstract**: Text-to-SQL systems enable users to query databases using natural language, democratizing access to data analytics. However, they face challenges in understanding ambiguous phrasing, domain-specific vocabulary, and complex schema relationships. This paper introduces Datrics Text2SQL, a Retrieval-Augmented Generation (RAG)-based framework designed to generate accurate SQL queries by leveraging structured documentation, example-based learning, and domain-specific rules. The system builds a rich Knowledge Base from database documentation and question-query examples, which are stored as vector embeddings and retrieved through semantic similarity. It then uses this context to generate syntactically correct and semantically aligned SQL code. The paper details the architecture, training methodology, and retrieval logic, highlighting how the system bridges the gap between user intent and database structure without requiring SQL expertise. 

**Abstract (ZH)**: Text-to-SQL系统使用户能够使用自然语言查询数据库， democratizing数据访问。然而，它们在理解含糊的语言表达、领域特定词汇以及复杂的数据模型关系方面面临挑战。本文介绍了基于检索增强生成（RAG）的Datrics Text2SQL框架，该框架通过利用结构化文档、基于示例的学习和领域特定规则生成准确的SQL查询。该系统从数据库文档和问题-查询示例中构建一个丰富的知识库，并将这些信息存储为向量嵌入并通过语义相似性进行检索。然后，该系统利用此上下文生成语义上正确且语义上对齐的SQL代码。本文描述了该系统的架构、训练方法和检索逻辑，强调了该系统如何在不需SQL专业知识的情况下弥合用户意图与数据库结构之间的差距。 

---
# Uncovering Bias Paths with LLM-guided Causal Discovery: An Active Learning and Dynamic Scoring Approach 

**Title (ZH)**: 使用LLM指导的因果发现法揭露偏见路径：一种主动学习和动态评分方法 

**Authors**: Khadija Zanna, Akane Sano  

**Link**: [PDF](https://arxiv.org/pdf/2506.12227)  

**Abstract**: Causal discovery (CD) plays a pivotal role in understanding the mechanisms underlying complex systems. While recent algorithms can detect spurious associations and latent confounding, many struggle to recover fairness-relevant pathways in realistic, noisy settings. Large Language Models (LLMs), with their access to broad semantic knowledge, offer a promising complement to statistical CD approaches, particularly in domains where metadata provides meaningful relational cues. Ensuring fairness in machine learning requires understanding how sensitive attributes causally influence outcomes, yet CD methods often introduce spurious or biased pathways. We propose a hybrid LLM-based framework for CD that extends a breadth-first search (BFS) strategy with active learning and dynamic scoring. Variable pairs are prioritized for LLM-based querying using a composite score based on mutual information, partial correlation, and LLM confidence, improving discovery efficiency and robustness.
To evaluate fairness sensitivity, we construct a semi-synthetic benchmark from the UCI Adult dataset, embedding a domain-informed causal graph with injected noise, label corruption, and latent confounding. We assess how well CD methods recover both global structure and fairness-critical paths.
Our results show that LLM-guided methods, including the proposed method, demonstrate competitive or superior performance in recovering such pathways under noisy conditions. We highlight when dynamic scoring and active querying are most beneficial and discuss implications for bias auditing in real-world datasets. 

**Abstract (ZH)**: 因果发现（CD）在理解复杂系统背后的机制中发挥着关键作用。虽然近年来的算法能够检测虚假关联和潜在混杂因素，但在现实且嘈杂的环境中，许多方法难以恢复与公平性相关的关键路径。大型语言模型（LLMs）因其广泛语义知识的访问能力，为统计因果发现方法提供了有力补充，特别是在元数据能够提供有意义关系线索的领域。确保机器学习的公平性需要理解敏感属性如何因果影响结果，但因果发现方法常常引入虚假或偏见的路径。我们提出了一种基于LLM的混合框架，通过主动学习和动态评分扩展广度优先搜索（BFS）策略，使用互信息、部分相关和LLM置信度的复合评分优先选择变量对进行LLM查询，从而提高发现效率和鲁棒性。

为评估公平性敏感性，我们从UCI Adult数据集中构建了一个半合成基准，嵌入了基于领域知识的因果图，并注入了噪声、标签污染和潜在混杂因素。我们评估因果发现方法在恢复全局结构和公平性关键路径方面的表现。

我们的结果显示，LLM引导的方法，包括所提出的方法，在嘈杂条件下恢复此类路径的性能具有竞争力或更优。我们指出了动态评分和主动查询最有益的情况，并讨论了在实际数据集中的偏见审计含义。 

---
# Mapping Neural Theories of Consciousness onto the Common Model of Cognition 

**Title (ZH)**: 将意识的神经理论映射到共同认知模型上 

**Authors**: Paul S. Rosenbloom, John E. Laird, Christian Lebiere, Andrea Stocco  

**Link**: [PDF](https://arxiv.org/pdf/2506.12224)  

**Abstract**: A beginning is made at mapping four neural theories of consciousness onto the Common Model of Cognition. This highlights how the four jointly depend on recurrent local modules plus a cognitive cycle operating on a global working memory with complex states, and reveals how an existing integrative view of consciousness from a neural perspective aligns with the Com-mon Model. 

**Abstract (ZH)**: 开始将四种神经理论的意识映射到通用认知模型上，这突显了四种理论如何共同依赖于反复循环的局部模块以及在复杂的全球工作记忆上运行的认知循环，并揭示了从神经角度现有的意识整合观点与通用模型的一致性。 

---
# SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes 

**Title (ZH)**: SSLAM：通过音频混合增强自监督模型用于多声部声景 

**Authors**: Tony Alex, Sara Ahmed, Armin Mustafa, Muhammad Awais, Philip JB Jackson  

**Link**: [PDF](https://arxiv.org/pdf/2506.12222)  

**Abstract**: Self-supervised pre-trained audio networks have seen widespread adoption in real-world systems, particularly in multi-modal large language models. These networks are often employed in a frozen state, under the assumption that the SSL pre-training has sufficiently equipped them to handle real-world audio. However, a critical question remains: how well do these models actually perform in real-world conditions, where audio is typically polyphonic and complex, involving multiple overlapping sound sources? Current audio SSL methods are often benchmarked on datasets predominantly featuring monophonic audio, such as environmental sounds, and speech. As a result, the ability of SSL models to generalize to polyphonic audio, a common characteristic in natural scenarios, remains underexplored. This limitation raises concerns about the practical robustness of SSL models in more realistic audio settings. To address this gap, we introduce Self-Supervised Learning from Audio Mixtures (SSLAM), a novel direction in audio SSL research, designed to improve, designed to improve the model's ability to learn from polyphonic data while maintaining strong performance on monophonic data. We thoroughly evaluate SSLAM on standard audio SSL benchmark datasets which are predominantly monophonic and conduct a comprehensive comparative analysis against SOTA methods using a range of high-quality, publicly available polyphonic datasets. SSLAM not only improves model performance on polyphonic audio, but also maintains or exceeds performance on standard audio SSL benchmarks. Notably, it achieves up to a 3.9\% improvement on the AudioSet-2M (AS-2M), reaching a mean average precision (mAP) of 50.2. For polyphonic datasets, SSLAM sets new SOTA in both linear evaluation and fine-tuning regimes with performance improvements of up to 9.1\% (mAP). 

**Abstract (ZH)**: 自监督预训练音频网络在实际系统中得到了广泛应用，特别是在多模态大型语言模型中。这些网络通常以冻结状态使用，在假设它们通过自监督预训练已经获得了处理真实世界音频的能力的前提下。然而，一个关键问题仍然存在：在音频通常为多音性和复杂性的实际条件下，这些模型实际表现如何？当前的音频自监督学习方法通常在以单音性音频为主的环境中进行基准测试，例如环境声和语音。因此，SSL模型在多音性音频上的泛化能力，这是自然场景中的一个常见特性，仍被严重忽视。这一限制引发了对SSL模型在更实际音频设置中的实用鲁棒性的担忧。为解决这一问题，我们提出了自监督学习从音频混合物（SSLAM）这一在音频自监督学习研究中的新方向，旨在提高模型从多音性数据中学习的能力，同时保持在单音性数据上的强大性能。我们彻底评估了SSLAM在以单音性为主的标准音频自监督学习基准数据集上的表现，并使用一系列高质量的公开多音性数据集与当前最先进的方法进行了全面的对比分析。SSLAM不仅在多音性音频上提高了模型性能，而且在标准音频自监督学习基准测试上也保持或超过了性能。值得注意的是，它在AudioSet-2M（AS-2M）上达到了50.2的平均精度（mAP），提高了多达3.9％。在多音性数据集上，SSLAM在线性评估和微调两种模式下均达到新的当前最好水平，性能提高高达9.1％（mAP）。 

---
# Two heads are better than one: simulating large transformers with small ones 

**Title (ZH)**: 两个头胜过一个：用小变压器模拟大变压器 

**Authors**: Hantao Yu, Josh Alman  

**Link**: [PDF](https://arxiv.org/pdf/2506.12220)  

**Abstract**: The quadratic complexity of self-attention prevents transformers from scaling effectively to long input sequences. On the other hand, modern GPUs and other specialized hardware accelerators are well-optimized for processing small input sequences in transformers during both training and inference. A natural question arises: can we take advantage of the efficiency of small transformers to deal with long input sequences?
In this paper, we show that transformers with long input sequences (large transformers) can be efficiently simulated by transformers that can only take short input sequences (small transformers). Specifically, we prove that any transformer with input length $N$ can be efficiently simulated by only $O((N/M)^2)$ transformers with input length $M \ll N$, and that this cannot be improved in the worst case. However, we then prove that in various natural scenarios including average-case inputs, sliding window masking and attention sinks, the optimal number $O(N/M)$ of small transformers suffice. 

**Abstract (ZH)**: 长输入序列的变换器可以由只能处理短输入序列的变换器有效地模拟。 

---
# From Emergence to Control: Probing and Modulating Self-Reflection in Language Models 

**Title (ZH)**: 从涌现到调控：探究并调节语言模型的自我反思能力 

**Authors**: Xudong Zhu, Jiachen Jiang, Mohammad Mahdi Khalili, Zhihui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12217)  

**Abstract**: Self-reflection -- the ability of a large language model (LLM) to revisit, evaluate, and revise its own reasoning -- has recently emerged as a powerful behavior enabled by reinforcement learning with verifiable rewards (RLVR). While self-reflection correlates with improved reasoning accuracy, its origin and underlying mechanisms remain poorly understood. In this work, {\it we first show that self-reflection is not exclusive to RLVR fine-tuned models: it already emerges, albeit rarely, in pretrained models}. To probe this latent ability, we introduce Reflection-Inducing Probing, a method that injects reflection-triggering reasoning traces from fine-tuned models into pretrained models. This intervention raises self-reflection frequency of Qwen2.5 from 0.6\% to 18.6\%, revealing a hidden capacity for reflection. Moreover, our analysis of internal representations shows that both pretrained and fine-tuned models maintain hidden states that distinctly separate self-reflective from non-reflective contexts. Leveraging this observation, {\it we then construct a self-reflection vector, a direction in activation space associated with self-reflective reasoning}. By manipulating this vector, we enable bidirectional control over the self-reflective behavior for both pretrained and fine-tuned models. Experiments across multiple reasoning benchmarks show that enhancing these vectors improves reasoning performance by up to 12\%, while suppressing them reduces computational cost, providing a flexible mechanism to navigate the trade-off between reasoning quality and efficiency without requiring additional training. Our findings further our understanding of self-reflection and support a growing body of work showing that understanding model internals can enable precise behavioral control. 

**Abstract (ZH)**: 自我反思——大型语言模型（LLM）重新审视、评估和修订自身推理的能力——最近作为强化学习带有可验证奖励（RLVR）的强化学习的一种强大行为崭露头角。虽然自我反思与改进的推理准确性相关，但其起源及其工作机制仍知之甚少。在本文中，我们首先展示自我反思并不仅限于RLVR微调模型：它已经在预训练模型中出现，尽管较为罕见。为探究这种潜在能力，我们引入了引发自我反思的探针方法，该方法将微调模型的触发反思推理轨迹注入预训练模型中。这一干预将Qwen2.5的自我反思频率从0.6%提高到18.6%，揭示了其隐藏的自我反思能力。此外，我们对内部表示的分析显示，预训练模型和微调模型都保持了将自我反思与非反思上下文区分开来的隐藏状态。基于这一观察，我们构建了一个自我反思向量，它与自我反思推理相关于激活空间中的一个方向。通过操控这一向量，我们能够双向控制预训练模型和微调模型的自我反思行为。在多个推理基准上的实验表明，增强这些向量可以将推理性能提高最多12%，而抑制它们则可以降低计算成本，提供了一个在推理质量和效率之间进行灵活权衡的机制，而不必进行额外训练。我们的发现进一步加深了对自我反思的理解，并支持了日益增多的研究工作，这些工作表明了解模型内部结构能够实现精确的行为控制。 

---
# Semantic Scheduling for LLM Inference 

**Title (ZH)**: 语义调度以实现LLM推理优化 

**Authors**: Wenyue Hua, Dujian Ding, Yile Gu, Yujie Ren, Kai Mei, Minghua Ma, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12204)  

**Abstract**: Conventional operating system scheduling algorithms are largely content-ignorant, making decisions based on factors such as latency or fairness without considering the actual intents or semantics of processes. Consequently, these algorithms often do not prioritize tasks that require urgent attention or carry higher importance, such as in emergency management scenarios. However, recent advances in language models enable semantic analysis of processes, allowing for more intelligent and context-aware scheduling decisions. In this paper, we introduce the concept of semantic scheduling in scheduling of requests from large language models (LLM), where the semantics of the process guide the scheduling priorities. We present a novel scheduling algorithm with optimal time complexity, designed to minimize the overall waiting time in LLM-based prompt scheduling. To illustrate its effectiveness, we present a medical emergency management application, underscoring the potential benefits of semantic scheduling for critical, time-sensitive tasks. The code and data are available at this https URL. 

**Abstract (ZH)**: 传统的操作系统调度算法在很大程度上忽略了内容，基于延迟或公平性等因素作出决策，而不考虑进程的实际意图或语义。因此，这些算法通常不会优先处理需要紧急关注或更重要任务，例如在紧急管理场景中的任务。然而，最近在语言模型方面的进展使得能够对进程进行语义分析，从而实现更加智能化和上下文相关的调度决策。在本文中，我们提出了语义调度的概念，特别是在大型语言模型（LLM）请求调度中的应用，其中进程的语义指导调度优先级。我们提出了一种具有最优时间复杂度的新调度算法，旨在最小化基于LLM的提示调度中的总体等待时间。为了说明其有效性，我们呈现了一个医疗紧急管理应用案例，强调了语义调度对关键、时限敏感任务的潜在益处。完整的代码和数据可在以下链接获取：这个 https URL。 

---
# A Fast, Reliable, and Secure Programming Language for LLM Agents with Code Actions 

**Title (ZH)**: 一种用于LLM代理的快速、可靠且安全的编程语言，支持代码操作 

**Authors**: Stephen Mell, Botong Zhang, David Mell, Shuo Li, Ramya Ramalingam, Nathan Yu, Steve Zdancewic, Osbert Bastani  

**Link**: [PDF](https://arxiv.org/pdf/2506.12202)  

**Abstract**: Modern large language models (LLMs) are often deployed as agents, calling external tools adaptively to solve tasks. Rather than directly calling tools, it can be more effective for LLMs to write code to perform the tool calls, enabling them to automatically generate complex control flow such as conditionals and loops. Such code actions are typically provided as Python code, since LLMs are quite proficient at it; however, Python may not be the ideal language due to limited built-in support for performance, security, and reliability. We propose a novel programming language for code actions, called Quasar, which has several benefits: (1) automated parallelization to improve performance, (2) uncertainty quantification to improve reliability and mitigate hallucinations, and (3) security features enabling the user to validate actions. LLMs can write code in a subset of Python, which is automatically transpiled to Quasar. We evaluate our approach on the ViperGPT visual question answering agent, applied to the GQA dataset, demonstrating that LLMs with Quasar actions instead of Python actions retain strong performance, while reducing execution time when possible by 42%, improving security by reducing user approval interactions when possible by 52%, and improving reliability by applying conformal prediction to achieve a desired target coverage level. 

**Abstract (ZH)**: 现代大型语言模型（LLMs）通常被部署为代理，适配性地调用外部工具以解决任务。与直接调用工具相比，语言模型通过编写代码来执行工具调用可能更为有效，这使它们能够自动生成复杂的控制流，如条件语句和循环。这类代码操作通常以Python代码形式提供，因为语言模型在这方面相当熟练；然而，Python可能并不是理想的选择，因为它在性能、安全性和可靠性方面支持有限。我们提出了一种名为Quasar的新编程语言，它具有以下优点：（1）自动并行化以提高性能，（2）不确定性量化以提高可靠性和减少幻觉，（3）安全功能使用户能够验证操作。语言模型可以在Python的子集中编写代码，这些代码将自动转换为Quasar。我们通过将Quasar应用于ViperGPT视觉问答代理并应用于GQA数据集来评估我们的方法，结果显示使用Quasar操作的语言模型在保持强大性能的同时，当可能时将执行时间减少了42%，将用户确认交互减少了52%，并通过应用容信预测实现了所需的目标覆盖水平，从而提高了可靠性。 

---
# ViSAGe: Video-to-Spatial Audio Generation 

**Title (ZH)**: ViSAGe: 视频到空间音频生成 

**Authors**: Jaeyeon Kim, Heeseung Yun, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12199)  

**Abstract**: Spatial audio is essential for enhancing the immersiveness of audio-visual experiences, yet its production typically demands complex recording systems and specialized expertise. In this work, we address a novel problem of generating first-order ambisonics, a widely used spatial audio format, directly from silent videos. To support this task, we introduce YT-Ambigen, a dataset comprising 102K 5-second YouTube video clips paired with corresponding first-order ambisonics. We also propose new evaluation metrics to assess the spatial aspect of generated audio based on audio energy maps and saliency metrics. Furthermore, we present Video-to-Spatial Audio Generation (ViSAGe), an end-to-end framework that generates first-order ambisonics from silent video frames by leveraging CLIP visual features, autoregressive neural audio codec modeling with both directional and visual guidance. Experimental results demonstrate that ViSAGe produces plausible and coherent first-order ambisonics, outperforming two-stage approaches consisting of video-to-audio generation and audio spatialization. Qualitative examples further illustrate that ViSAGe generates temporally aligned high-quality spatial audio that adapts to viewpoint changes. 

**Abstract (ZH)**: 基于视频生成一阶 ambisonics 的端到端框架：ViSAGe 

---
# BreastDCEDL: Curating a Comprehensive DCE-MRI Dataset and developing a Transformer Implementation for Breast Cancer Treatment Response Prediction 

**Title (ZH)**: BreastDCEDL: 编纂全面的DCE-MRI数据集并开发 Transformer 实现方法以预测乳腺癌治疗反应 

**Authors**: Naomi Fridman, Bubby Solway, Tomer Fridman, Itamar Barnea, Anat Goldshtein  

**Link**: [PDF](https://arxiv.org/pdf/2506.12190)  

**Abstract**: Breast cancer remains a leading cause of cancer-related mortality worldwide, making early detection and accurate treatment response monitoring critical priorities. We present BreastDCEDL, a curated, deep learning-ready dataset comprising pre-treatment 3D Dynamic Contrast-Enhanced MRI (DCE-MRI) scans from 2,070 breast cancer patients drawn from the I-SPY1, I-SPY2, and Duke cohorts, all sourced from The Cancer Imaging Archive. The raw DICOM imaging data were rigorously converted into standardized 3D NIfTI volumes with preserved signal integrity, accompanied by unified tumor annotations and harmonized clinical metadata including pathologic complete response (pCR), hormone receptor (HR), and HER2 status. Although DCE-MRI provides essential diagnostic information and deep learning offers tremendous potential for analyzing such complex data, progress has been limited by lack of accessible, public, multicenter datasets. BreastDCEDL addresses this gap by enabling development of advanced models, including state-of-the-art transformer architectures that require substantial training data. To demonstrate its capacity for robust modeling, we developed the first transformer-based model for breast DCE-MRI, leveraging Vision Transformer (ViT) architecture trained on RGB-fused images from three contrast phases (pre-contrast, early post-contrast, and late post-contrast). Our ViT model achieved state-of-the-art pCR prediction performance in HR+/HER2- patients (AUC 0.94, accuracy 0.93). BreastDCEDL includes predefined benchmark splits, offering a framework for reproducible research and enabling clinically meaningful modeling in breast cancer imaging. 

**Abstract (ZH)**: 乳腺癌仍然是导致癌症相关死亡的主要原因，早期检测和准确的治疗反应监测至关重要。我们介绍了BreastDCEDL，这是一个精心整理、适合深度学习的数据集，包含来自I-SPY1、I-SPY2和Duke队列的2,070名乳腺癌患者的治疗前3D动态对比增强磁共振成像（DCE-MRI）扫描，所有数据来源于《癌症影像档案》。原始DICOM影像数据被严格转化为标准的3D NIfTI体积，保持了信号完整性，并附带统一的肿瘤标注和 harmonized 临床元数据，包括病理性完全缓解（pCR）、雌激素受体（HR）和HER2状态。尽管DCE-MRI提供了重要的诊断信息，且深度学习对于分析这种复杂数据具有巨大的潜力，但进展受限于缺乏可访问的、公开的多中心数据集。BreastDCEDL通过填补这一空白，使得开发高级模型成为可能，包括需要大量训练数据的最先进的变压器架构。为了展示其在稳健建模方面的潜力，我们开发了首个基于变压器的模型用于乳腺DCE-MRI，利用在三个对比阶段（预对比、早期对比后和晚期对比后）融合RGB图像的Vision Transformer（ViT）架构进行训练。我们的ViT模型在HR+/HER2-患者中实现了最新的pCR预测性能（AUC 0.94，准确率 0.93）。BreastDCEDL包括预定义的基准拆分，提供了一种可再现的研究框架，并促进了乳腺癌影像学中的临床相关模型构建。 

---
# Supernova Event Dataset: Interpreting Large Language Model's Personality through Critical Event Analysis 

**Title (ZH)**: 超新星事件数据集：通过关键事件分析解读大规模语言模型的性格特征 

**Authors**: Pranav Agarwal, Ioana Ciucă  

**Link**: [PDF](https://arxiv.org/pdf/2506.12189)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into everyday applications. As their influence grows, understanding their decision making and underlying personality becomes essential. In this work, we interpret model personality using our proposed Supernova Event Dataset, a novel dataset with diverse articles spanning biographies, historical events, news, and scientific discoveries. We use this dataset to benchmark LLMs on extracting and ranking key events from text, a subjective and complex challenge that requires reasoning over long-range context and modeling causal chains. We evaluate small models like Phi-4, Orca 2, and Qwen 2.5, and large, stronger models such as Claude 3.7, Gemini 2.5, and OpenAI o3, and propose a framework where another LLM acts as a judge to infer each model's personality based on its selection and classification of events. Our analysis shows distinct personality traits: for instance, Orca 2 demonstrates emotional reasoning focusing on interpersonal dynamics, while Qwen 2.5 displays a more strategic, analytical style. When analyzing scientific discovery events, Claude Sonnet 3.7 emphasizes conceptual framing, Gemini 2.5 Pro prioritizes empirical validation, and o3 favors step-by-step causal reasoning. This analysis improves model interpretability, making them user-friendly for a wide range of diverse applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益融入日常应用。随着其影响力的增长，理解其决策过程和内在个性变得至关重要。在本工作中，我们使用我们提出的 Supernova Event Dataset 对模型个性进行解释，这是一个包含具有生物传记、历史事件、新闻和科学发现等多种文章的新颖数据集。我们使用此数据集对 Phi-4、Orca 2、Qwen 2.5 等小型模型以及 Claude 3.7、Gemini 2.5、OpenAI o3 等大型和更强模型进行基准测试，评估它们提取和排序文本中的关键事件的能力，这是一个主观且复杂的挑战，需要对长范围上下文进行推理并建模因果链。我们提出了一种框架，其中另一个语言模型作为裁判，根据其对事件的选择和分类推断每个模型的个性。我们的分析显示了不同的个性特征：例如，Orca 2 展现出情感推理，注重人际关系动态，而 Qwen 2.5 则表现出更为战略性和分析性的风格。在分析科学发现事件时，Claude Sonnet 3.7 强调概念框架，Gemini 2.5 Pro 注重实证验证，而 o3 则偏好逐步因果推理。这项分析提高了模型的解释性，使得它们更加用户友好，适用于广泛多样的应用。 

---
# MRI-CORE: A Foundation Model for Magnetic Resonance Imaging 

**Title (ZH)**: MRI-CORE: 一个磁共振成像基础模型 

**Authors**: Haoyu Dong, Yuwen Chen, Hanxue Gu, Nicholas Konz, Yaqian Chen, Qihang Li, Maciej A. Mazurowski  

**Link**: [PDF](https://arxiv.org/pdf/2506.12186)  

**Abstract**: The widespread use of Magnetic Resonance Imaging (MRI) and the rise of deep learning have enabled the development of powerful predictive models for a wide range of diagnostic tasks in MRI, such as image classification or object segmentation. However, training models for specific new tasks often requires large amounts of labeled data, which is difficult to obtain due to high annotation costs and data privacy concerns. To circumvent this issue, we introduce MRI-CORE (MRI COmprehensive Representation Encoder), a vision foundation model pre-trained using more than 6 million slices from over 110,000 MRI volumes across 18 main body locations. Experiments on five diverse object segmentation tasks in MRI demonstrate that MRI-CORE can significantly improve segmentation performance in realistic scenarios with limited labeled data availability, achieving an average gain of 6.97% 3D Dice Coefficient using only 10 annotated slices per task. We further demonstrate new model capabilities in MRI such as classification of image properties including body location, sequence type and institution, and zero-shot segmentation. These results highlight the value of MRI-CORE as a generalist vision foundation model for MRI, potentially lowering the data annotation resource barriers for many applications. 

**Abstract (ZH)**: 磁共振成像广泛应用于医学领域，深度学习的兴起促进了基于磁共振成像的 Powerful 预测模型的发展，这些模型可用于图像分类或对象分割等多种诊断任务。然而，为特定新任务训练模型通常需要大量标注数据，这由于标注成本高和数据隐私问题而难以获得。为解决这一问题，我们引入了 MRI-CORE（MRI 综合表示编码器），该模型基于超过 110,000 个来自 18 个主要身体部位的磁共振成像体积中的 600 万多切片进行预训练。在五个不同对象分割任务上的实验表明，MRI-CORE 在有限标注数据的情况下，可以显著提高分割性能，使用每任务仅 10 个标注切片，平均获得 6.97% 的三维 Dice 系数提升。进一步展示了 MRI-CORE 在 MRI 中的新模型能力，包括图像属性（包括身体位置、序列类型和机构）分类和零样本分割。这些结果突显了 MRI-CORE 作为 MRI 通用视觉基础模型的价值，可能降低许多应用的数据标注资源障碍。 

---
# TCN-DPD: Parameter-Efficient Temporal Convolutional Networks for Wideband Digital Predistortion 

**Title (ZH)**: TCN-DPD: 参数高效时序卷积网络在宽带数字预失真中的应用 

**Authors**: Huanqiang Duan, Manno Versluis, Qinyu Chen, Leo C. N. de Vreede, Chang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12165)  

**Abstract**: Digital predistortion (DPD) is essential for mitigating nonlinearity in RF power amplifiers, particularly for wideband applications. This paper presents TCN-DPD, a parameter-efficient architecture based on temporal convolutional networks, integrating noncausal dilated convolutions with optimized activation functions. Evaluated on the OpenDPD framework with the DPA_200MHz dataset, TCN-DPD achieves simulated ACPRs of -51.58/-49.26 dBc (L/R), EVM of -47.52 dB, and NMSE of -44.61 dB with 500 parameters and maintains superior linearization than prior models down to 200 parameters, making it promising for efficient wideband PA linearization. 

**Abstract (ZH)**: 基于时空卷积网络的非因袭扩张卷积与优化激活函数结合的数字预失真（TCN-DPD） 

---
# Explaining Recovery Trajectories of Older Adults Post Lower-Limb Fracture Using Modality-wise Multiview Clustering and Large Language Models 

**Title (ZH)**: 基于模态 wise 多视图聚类和大规模语言模型解释老年人下肢骨折术后恢复轨迹 

**Authors**: Shehroz S. Khan, Ali Abedi, Charlene H. Chu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12156)  

**Abstract**: Interpreting large volumes of high-dimensional, unlabeled data in a manner that is comprehensible to humans remains a significant challenge across various domains. In unsupervised healthcare data analysis, interpreting clustered data can offer meaningful insights into patients' health outcomes, which hold direct implications for healthcare providers. This paper addresses the problem of interpreting clustered sensor data collected from older adult patients recovering from lower-limb fractures in the community. A total of 560 days of multimodal sensor data, including acceleration, step count, ambient motion, GPS location, heart rate, and sleep, alongside clinical scores, were remotely collected from patients at home. Clustering was first carried out separately for each data modality to assess the impact of feature sets extracted from each modality on patients' recovery trajectories. Then, using context-aware prompting, a large language model was employed to infer meaningful cluster labels for the clusters derived from each modality. The quality of these clusters and their corresponding labels was validated through rigorous statistical testing and visualization against clinical scores collected alongside the multimodal sensor data. The results demonstrated the statistical significance of most modality-specific cluster labels generated by the large language model with respect to clinical scores, confirming the efficacy of the proposed method for interpreting sensor data in an unsupervised manner. This unsupervised data analysis approach, relying solely on sensor data, enables clinicians to identify at-risk patients and take timely measures to improve health outcomes. 

**Abstract (ZH)**: 解读高维度未标记数据量的挑战依然存在于各个领域。在无监督的医疗数据分析中，对聚类数据的解释可以为患者健康结果提供有意义的洞察，这对医疗服务提供者有直接影响。本文解决了从社区中康复中的老年患者收集的多模态传感器数据聚类解释问题。总共收集了560天的多模态传感器数据，包括加速度、步数、环境运动、GPS位置、心率和睡眠数据，以及临床评分，并远程从患者家中收集。首先，单独对每种数据模态进行聚类，以评估从每种模态中提取的特征集对患者康复轨迹的影响。然后，使用上下文感知提示，采用大规模语言模型推断出从每种模态中衍生的聚类的有意义的标签。这些聚类及其对应标签的质量通过严格的统计测试和可视化与一同收集的临床评分进行了验证。结果表明，大规模语言模型生成的多数模态特定聚类标签与临床评分之间具有统计显著性，证实了该方法的有效性，用于无监督地解释传感器数据。仅依赖传感器数据的这种无监督数据分析方法，使临床医生能够识别高风险患者，并采取及时措施以改善健康结果。 

---
# Can Mixture-of-Experts Surpass Dense LLMs Under Strictly Equal Resources? 

**Title (ZH)**: 在严格平等的资源条件下，混合专家模型能否超越密集的大语言模型？ 

**Authors**: Houyi Li, Ka Man Lo, Ziqi Wang, Zili Wang, Wenzhen Zheng, Shuigeng Zhou, Xiangyu Zhang, Daxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12119)  

**Abstract**: Mixture-of-Experts (MoE) language models dramatically expand model capacity and achieve remarkable performance without increasing per-token compute. However, can MoEs surpass dense architectures under strictly equal resource constraints - that is, when the total parameter count, training compute, and data budget are identical? This question remains under-explored despite its significant practical value and potential. In this paper, we propose a novel perspective and methodological framework to study this question thoroughly. First, we comprehensively investigate the architecture of MoEs and achieve an optimal model design that maximizes the performance. Based on this, we subsequently find that an MoE model with activation rate in an optimal region is able to outperform its dense counterpart under the same total parameter, training compute and data resource. More importantly, this optimal region remains consistent across different model sizes. Although additional amount of data turns out to be a trade-off for the enhanced performance, we show that this can be resolved via reusing data. We validate our findings through extensive experiments, training nearly 200 language models at 2B scale and over 50 at 7B scale, cumulatively processing 50 trillion tokens. All models will be released publicly. 

**Abstract (ZH)**: Mixture-of-Experts (MoE) 语言模型在严格等同的资源约束下能否超越密集架构：探究最优设计与性能 

---
# Scale-Invariance Drives Convergence in AI and Brain Representations 

**Title (ZH)**: 尺度不变性驱动AI和脑部表征的收敛 

**Authors**: Junjie Yu, Wenxiao Ma, Jianyu Zhang, Haotian Deng, Zihan Deng, Yi Guo, Quanying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12117)  

**Abstract**: Despite variations in architecture and pretraining strategies, recent studies indicate that large-scale AI models often converge toward similar internal representations that also align with neural activity. We propose that scale-invariance, a fundamental structural principle in natural systems, is a key driver of this convergence. In this work, we propose a multi-scale analytical framework to quantify two core aspects of scale-invariance in AI representations: dimensional stability and structural similarity across scales. We further investigate whether these properties can predict alignment performance with functional Magnetic Resonance Imaging (fMRI) responses in the visual cortex. Our analysis reveals that embeddings with more consistent dimension and higher structural similarity across scales align better with fMRI data. Furthermore, we find that the manifold structure of fMRI data is more concentrated, with most features dissipating at smaller scales. Embeddings with similar scale patterns align more closely with fMRI data. We also show that larger pretraining datasets and the inclusion of language modalities enhance the scale-invariance properties of embeddings, further improving neural alignment. Our findings indicate that scale-invariance is a fundamental structural principle that bridges artificial and biological representations, providing a new framework for evaluating the structural quality of human-like AI systems. 

**Abstract (ZH)**: 尽管架构和预训练策略存在差异，近期研究表明，大规模AI模型往往朝向相似的内部表示汇聚，这些表示也与神经活动相一致。我们提出，规模不变性，作为自然系统的基本结构性原理，是这一汇聚现象的关键驱动因素。在本文中，我们提出一个多尺度分析框架来量化AI表示中尺度不变性的两个核心方面：维度稳定性和跨尺度的结构相似性。进一步研究这些性质能否预测与功能性磁共振成像(fMRI)视觉皮层反应的对齐性能。分析结果显示，维度更一致且跨尺度结构相似性更高的嵌入更符合fMRI数据。此外，我们发现fMRI数据的流形结构更为集中，大多数特征在较小尺度上消失。具有相似尺度模式的嵌入更接近fMRI数据。我们还表明，使用更大的预训练数据集和包含语言模态能够增强嵌入的尺度不变性特性，进一步改善神经对齐。我们的研究结果表明，尺度不变性作为连接人工与生物表示的基本结构性原理，提供了评估类人AI系统结构质量的新框架。 

---
# Unsupervised Document and Template Clustering using Multimodal Embeddings 

**Title (ZH)**: 基于多模态嵌入的无监督文档和模板聚类 

**Authors**: Phillipe R. Sampaio, Helene Maxcici  

**Link**: [PDF](https://arxiv.org/pdf/2506.12116)  

**Abstract**: This paper investigates a novel approach to unsupervised document clustering by leveraging multimodal embeddings as input to traditional clustering algorithms such as $k$-Means and DBSCAN. Our method aims to achieve a finer-grained document understanding by not only grouping documents at the type level (e.g., invoices, purchase orders), but also distinguishing between different templates within the same document category. This is achieved by using embeddings that capture textual content, layout information, and visual features of documents. We evaluated the effectiveness of this approach using embeddings generated by several state-of-the-art pretrained multimodal models, including SBERT, LayoutLMv1, LayoutLMv3, DiT, Donut, and ColPali. Our findings demonstrate the potential of multimodal embeddings to significantly enhance document clustering, offering benefits for various applications in intelligent document processing, document layout analysis, and unsupervised document classification. This work provides valuable insight into the advantages and limitations of different multimodal models for this task and opens new avenues for future research to understand and organize document collections. 

**Abstract (ZH)**: 本文通过利用多模态嵌入作为传统聚类算法（如$k$-Means和DBSCAN）的输入，探讨了一种新颖的无监督文档聚类方法。该方法旨在通过不仅按类型（如发票、采购订单）分组文档，而且还区分同一类别文档内的不同模板，实现更细粒度的文档理解。通过使用能够捕捉文档文本内容、布局信息和视觉特征的嵌入，实现了这一目标。我们使用几种最新的预训练多模态模型（包括SBERT、LayoutLMv1、LayoutLMv3、DiT、Donut和ColPali）生成的嵌入对这种方法的有效性进行了评估。研究结果显示出多模态嵌入在文档聚类方面的潜在优势，为智能文档处理、文档布局分析和无监督文档分类提供了益处。本文为不同多模态模型在该任务中的优势和局限性提供了宝贵的见解，并为未来研究理解和组织文档集合开辟了新的途径。 

---
# Eliciting Reasoning in Language Models with Cognitive Tools 

**Title (ZH)**: 使用认知工具激发语言模型的推理能力 

**Authors**: Brown Ebouky, Andrea Bartezzaghi, Mattia Rigotti  

**Link**: [PDF](https://arxiv.org/pdf/2506.12115)  

**Abstract**: The recent advent of reasoning models like OpenAI's o1 was met with excited speculation by the AI community about the mechanisms underlying these capabilities in closed models, followed by a rush of replication efforts, particularly from the open source community. These speculations were largely settled by the demonstration from DeepSeek-R1 that chains-of-thought and reinforcement learning (RL) can effectively replicate reasoning on top of base LLMs. However, it remains valuable to explore alternative methods for theoretically eliciting reasoning that could help elucidate the underlying mechanisms, as well as providing additional methods that may offer complementary benefits.
Here, we build on the long-standing literature in cognitive psychology and cognitive architectures, which postulates that reasoning arises from the orchestrated, sequential execution of a set of modular, predetermined cognitive operations. Crucially, we implement this key idea within a modern agentic tool-calling framework. In particular, we endow an LLM with a small set of "cognitive tools" encapsulating specific reasoning operations, each executed by the LLM itself. Surprisingly, this simple strategy results in considerable gains in performance on standard mathematical reasoning benchmarks compared to base LLMs, for both closed and open-weight models. For instance, providing our "cognitive tools" to GPT-4.1 increases its pass@1 performance on AIME2024 from 26.7% to 43.3%, bringing it very close to the performance of o1-preview.
In addition to its practical implications, this demonstration contributes to the debate regarding the role of post-training methods in eliciting reasoning in LLMs versus the role of inherent capabilities acquired during pre-training, and whether post-training merely uncovers these latent abilities. 

**Abstract (ZH)**: Recent Reasoning Models Like OpenAI's o1引发了AI社区对其机制的兴奋 speculation，随后出现了大量的复制努力，尤其是来自开源社区的努力。DeepSeek-R1的展示表明，通过基准LLM可以有效实现基于链条思考和强化学习的推理复制。然而，仍有必要探索其他理论方法以揭示推理背后的机制，并提供可能带来互补益处的额外方法。本研究借鉴了认知心理学和认知架构的长期文献，提出推理源自一系列模块化、预定的认知操作的有组织、顺序执行。我们特别在现代代理工具调用框架中实现了这一关键理念。具体来说，我们赋予LLM一组“认知工具”，这些工具包含特定的推理操作，每个操作均由LLM本身执行。令人惊讶的是，这种方法在标准数学推理基准测试中显著提高了性能，无论是封闭权重模型还是开放权重模型。例如，提供我们的“认知工具”给GPT-4.1，使其在AIME2024的pass@1性能从26.7%提高到43.3%，使其接近o1-preview的表现。除了其实际意义，这一展示还为后训练方法在LLM中激发推理的作用以及预训练固有能力的作用之间的辩论做出了贡献，并探讨了后训练方法是否只是揭示了这些潜在能力。 

---
# Semantic Preprocessing for LLM-based Malware Analysis 

**Title (ZH)**: 基于LLM的恶意软件分析的语义预处理 

**Authors**: Benjamin Marais, Tony Quertier, Grégoire Barrue  

**Link**: [PDF](https://arxiv.org/pdf/2506.12113)  

**Abstract**: In a context of malware analysis, numerous approaches rely on Artificial Intelligence to handle a large volume of data. However, these techniques focus on data view (images, sequences) and not on an expert's view. Noticing this issue, we propose a preprocessing that focuses on expert knowledge to improve malware semantic analysis and result interpretability. We propose a new preprocessing method which creates JSON reports for Portable Executable files. These reports gather features from both static and behavioral analysis, and incorporate packer signature detection, MITRE ATT\&CK and Malware Behavior Catalog (MBC) knowledge. The purpose of this preprocessing is to gather a semantic representation of binary files, understandable by malware analysts, and that can enhance AI models' explainability for malicious files analysis. Using this preprocessing to train a Large Language Model for Malware classification, we achieve a weighted-average F1-score of 0.94 on a complex dataset, representative of market reality. 

**Abstract (ZH)**: 在恶意软件分析的背景下，许多方法依赖于人工智能处理大量数据。然而，这些技术专注于数据视图（图像、序列），而忽略了专家视图。注意到这个问题，我们提出了一种预处理方法，专注于专家知识以提高恶意软件语义分析和结果可解释性。我们提出了一种新的预处理方法，为可移植可执行文件生成JSON报告，这些报告汇集了静态分析和行为分析的特征，并结合了加壳程序签名检测、MITRE ATT&CK知识以及恶意软件行为目录（MBC）的信息。这种方法的目的在于获取二进制文件的语义表示，使其可被恶意软件分析师理解，并能增强对恶意文件分析的AI模型的可解释性。利用该预处理方法训练一个针对恶意软件分类的大语言模型，在一个复杂且具有市场代表性的数据集上，我们获得了加权平均F1分数为0.94的结果。 

---
# Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs): A Feynman-Based Architecture for Continuous Learning Over Streaming Data 

**Title (ZH)**: 基于费曼原理的量子启发可微积分神经网络（QIDINNs）：一种适用于流式数据连续学习的架构 

**Authors**: Oscar Boullosa Dapena  

**Link**: [PDF](https://arxiv.org/pdf/2506.12111)  

**Abstract**: Real-time continuous learning over streaming data remains a central challenge in deep learning and AI systems. Traditional gradient-based models such as backpropagation through time (BPTT) face computational and stability limitations when dealing with temporally unbounded data. In this paper, we introduce a novel architecture, Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs), which leverages the Feynman technique of differentiation under the integral sign to formulate neural updates as integrals over historical data. This reformulation allows for smoother, more stable learning dynamics that are both physically interpretable and computationally tractable. Inspired by Feynman's path integral formalism and compatible with quantum gradient estimation frameworks, QIDINNs open a path toward hybrid classical-quantum neural computation. We demonstrate our model's effectiveness on synthetic and real-world streaming tasks, and we propose directions for quantum extensions and scalable implementations. 

**Abstract (ZH)**: 实时处理流式数据的连续学习 remains a central challenge in deep learning and AI systems. Quantum-Inspired Differentiable Integral Neural Networks (QIDINNs) leverage the Feynman technique to reformulate neural updates as integrals over historical data, enabling smoother and more stable learning dynamics that are both physically interpretable and computationally tractable. QIDINNs, inspired by Feynman's path integral formalism and compatible with quantum gradient estimation frameworks, pave the way for hybrid classical-quantum neural computation. We demonstrate the model's effectiveness on synthetic and real-world streaming tasks and propose directions for quantum extensions and scalable implementations. 

---
# EconGym: A Scalable AI Testbed with Diverse Economic Tasks 

**Title (ZH)**: EconGym: 一个具备多样化经济任务的可扩展AI试验台 

**Authors**: Qirui Mi, Qipeng Yang, Zijun Fan, Wentian Fan, Heyang Ma, Chengdong Ma, Siyu Xia, Bo An, Jun Wang, Haifeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12110)  

**Abstract**: Artificial intelligence (AI) has become a powerful tool for economic research, enabling large-scale simulation and policy optimization. However, applying AI effectively requires simulation platforms for scalable training and evaluation-yet existing environments remain limited to simplified, narrowly scoped tasks, falling short of capturing complex economic challenges such as demographic shifts, multi-government coordination, and large-scale agent interactions. To address this gap, we introduce EconGym, a scalable and modular testbed that connects diverse economic tasks with AI algorithms. Grounded in rigorous economic modeling, EconGym implements 11 heterogeneous role types (e.g., households, firms, banks, governments), their interaction mechanisms, and agent models with well-defined observations, actions, and rewards. Users can flexibly compose economic roles with diverse agent algorithms to simulate rich multi-agent trajectories across 25+ economic tasks for AI-driven policy learning and analysis. Experiments show that EconGym supports diverse and cross-domain tasks-such as coordinating fiscal, pension, and monetary policies-and enables benchmarking across AI, economic methods, and hybrids. Results indicate that richer task composition and algorithm diversity expand the policy space, while AI agents guided by classical economic methods perform best in complex settings. EconGym also scales to 10k agents with high realism and efficiency. 

**Abstract (ZH)**: 人工智能（AI）已成为经济研究的强大工具，使其能够进行大规模模拟和政策优化。然而，有效应用AI需要可扩展的训练和评估仿真平台——但现有的环境仍然局限于简化和狭隘范围的任务，难以捕捉到人口结构变化、多政府协调和大规模代理互动等复杂的经济挑战。为解决这一问题，我们引入了EconGym，这是一种可扩展且模块化的测试平台，将多样化的经济任务与AI算法相连接。EconGym基于严格的经济建模，实现了11种异质性角色类型（如家庭、企业、银行、政府）、其交互机制以及具有明确观测、行为和奖励的代理模型。用户可以灵活地组合具有各种代理算法的经济角色，以模拟跨越25多项以上经济任务的丰富多代理轨迹，用于AI驱动的政策学习和分析。实验表明，EconGym支持多领域任务，如协调财政、养老金和货币政策，并实现了AI方法、经济方法和混合方法之间的基准测试。结果显示，更丰富的任务组合和算法多样性扩展了政策空间，而受经典经济方法指导的AI代理在复杂环境中表现最佳。EconGym还可扩展至10,000个代理，具有高真实性和高效性。 

---
# Personalized LLM Decoding via Contrasting Personal Preference 

**Title (ZH)**: 个性化的大语言模型解码通过对比个人偏好 

**Authors**: Hyungjune Bu, Chanjoo Jung, Minjae Kang, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.12109)  

**Abstract**: As large language models (LLMs) are progressively deployed in various real-world applications, personalization of LLMs has become increasingly important. While various approaches to LLM personalization such as prompt-based and training-based methods have been actively explored, the development of effective decoding-time algorithms remains largely overlooked, despite their demonstrated potential. In this paper, we propose CoPe (Contrasting Personal Preference), a novel decoding-time approach applied after performing parameter-efficient fine-tuning (PEFT) on user-specific data. Our core idea is to leverage reward-guided decoding specifically for personalization by maximizing each user's implicit reward signal. We evaluate CoPe across five open-ended personalized text generation tasks. Our empirical results demonstrate that CoPe achieves strong performance, improving personalization by an average of 10.57% in ROUGE-L, without relying on external reward models or additional training procedures. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各种实际应用中逐步部署，LLMs的个性化变得日益重要。尽管已经探索了多种LLM个性化的方法，如基于提示和基于训练的方法，但在解码时的有效算法开发方面仍被忽视，尽管它们证明了潜在价值。在本文中，我们提出了CoPe（Contrasting Personal Preference），一种在进行参数高效微调（PEFT）后应用于用户特定数据上的新颖解码时方法。我们的核心思想是通过最大化每个用户的隐含奖励信号，利用奖励引导的解码来实现个性化。我们在五个开放式个性化文本生成任务中评估了CoPe。我们的实验证明，CoPe在ROUGE-L上取得了出色的表现，平均改进了10.57%的个性化，同时无需依赖外部奖励模型或额外的训练过程。 

---
# A Lightweight IDS for Early APT Detection Using a Novel Feature Selection Method 

**Title (ZH)**: 基于新型特征选择方法的轻量级IDS用于早期APT检测 

**Authors**: Bassam Noori Shaker, Bahaa Al-Musawi, Mohammed Falih Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12108)  

**Abstract**: An Advanced Persistent Threat (APT) is a multistage, highly sophisticated, and covert form of cyber threat that gains unauthorized access to networks to either steal valuable data or disrupt the targeted network. These threats often remain undetected for extended periods, emphasizing the critical need for early detection in networks to mitigate potential APT consequences. In this work, we propose a feature selection method for developing a lightweight intrusion detection system capable of effectively identifying APTs at the initial compromise stage. Our approach leverages the XGBoost algorithm and Explainable Artificial Intelligence (XAI), specifically utilizing the SHAP (SHapley Additive exPlanations) method for identifying the most relevant features of the initial compromise stage. The results of our proposed method showed the ability to reduce the selected features of the SCVIC-APT-2021 dataset from 77 to just four while maintaining consistent evaluation metrics for the suggested system. The estimated metrics values are 97% precision, 100% recall, and a 98% F1 score. The proposed method not only aids in preventing successful APT consequences but also enhances understanding of APT behavior at early stages. 

**Abstract (ZH)**: 一种高级持续性威胁（APT）是一种多阶段、高度复杂且隐蔽的网络威胁，其通过未经授权的方式进入网络，以窃取有价值的数据或破坏目标网络。这些威胁往往会长期未被检测到，这突显了在网络中早期检测以减轻潜在APT后果的重要性。本文提出了一种特征选择方法，用于开发一种轻量级的入侵检测系统，该系统能够在初始 compromise 阶段有效识别 APT。我们的方法利用了 XGBoost 算法和可解释人工智能（XAI），具体使用 SHAP（SHapley Additive exPlanations）方法来识别初始 compromise 阶段的最相关特征。我们提出的方法将 SCVIC-APT-2021 数据集的选定特征从 77 个减少到仅 4 个，同时保持评价指标的一致性。估计的指标值为 97% 的精确率、100% 的召回率和 98% 的 F1 分数。所提出的方法不仅能帮助预防成功的 APT 后果，还能增强对 APT 行为早期阶段的理解。 

---
# DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents 

**Title (ZH)**: DRIFT: 动态基于规则的防御机制与注入隔离以保障大规模语言模型代理的安全 

**Authors**: Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12104)  

**Abstract**: Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo benchmark, demonstrating its strong security performance while maintaining high utility across diverse models -- showcasing both its robustness and adaptability. 

**Abstract (ZH)**: 动态规则隔离框架DRIFT：可信任的代理系统安全防护 

---
# LLM Embedding-based Attribution (LEA): Quantifying Source Contributions to Generative Model's Response for Vulnerability Analysis 

**Title (ZH)**: 基于LLM嵌入的归因方法（LEA）：生成模型响应中源贡献的量化分析方法用于脆弱性分析 

**Authors**: Reza Fayyazi, Michael Zuzak, Shanchieh Jay Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12100)  

**Abstract**: Security vulnerabilities are rapidly increasing in frequency and complexity, creating a shifting threat landscape that challenges cybersecurity defenses. Large Language Models (LLMs) have been widely adopted for cybersecurity threat analysis. When querying LLMs, dealing with new, unseen vulnerabilities is particularly challenging as it lies outside LLMs' pre-trained distribution. Retrieval-Augmented Generation (RAG) pipelines mitigate the problem by injecting up-to-date authoritative sources into the model context, thus reducing hallucinations and increasing the accuracy in responses. Meanwhile, the deployment of LLMs in security-sensitive environments introduces challenges around trust and safety. This raises a critical open question: How to quantify or attribute the generated response to the retrieved context versus the model's pre-trained knowledge? This work proposes LLM Embedding-based Attribution (LEA) -- a novel, explainable metric to paint a clear picture on the 'percentage of influence' the pre-trained knowledge vs. retrieved content has for each generated response. We apply LEA to assess responses to 100 critical CVEs from the past decade, verifying its effectiveness to quantify the insightfulness for vulnerability analysis. Our development of LEA reveals a progression of independency in hidden states of LLMs: heavy reliance on context in early layers, which enables the derivation of LEA; increased independency in later layers, which sheds light on why scale is essential for LLM's effectiveness. This work provides security analysts a means to audit LLM-assisted workflows, laying the groundwork for transparent, high-assurance deployments of RAG-enhanced LLMs in cybersecurity operations. 

**Abstract (ZH)**: 基于LLM嵌入的归因（LEA）：量化生成响应中嵌入知识与检索内容的影响比例 

---
# SocialCredit+ 

**Title (ZH)**: 社会信用+ 

**Authors**: Thabassum Aslam, Anees Aslam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12099)  

**Abstract**: SocialCredit+ is AI powered credit scoring system that leverages publicly available social media data to augment traditional credit evaluation. It uses a conversational banking assistant to gather user consent and fetch public profiles. Multimodal feature extractors analyze posts, bios, images, and friend networks to generate a rich behavioral profile. A specialized Sharia-compliance layer flags any non-halal indicators and prohibited financial behavior based on Islamic ethics. The platform employs a retrieval-augmented generation module: an LLM accesses a domain specific knowledge base to generate clear, text-based explanations for each decision. We describe the end-to-end architecture and data flow, the models used, and system infrastructure. Synthetic scenarios illustrate how social signals translate into credit-score factors. This paper emphasizes conceptual novelty, compliance mechanisms, and practical impact, targeting AI researchers, fintech practitioners, ethical banking jurists, and investors. 

**Abstract (ZH)**: SocialCredit+是一种基于AI的信用评分系统，利用公开的社交媒体数据增强传统的信用评估。该系统使用对话式银行助手收集用户同意并获取公共档案。多模态特征提取器分析帖子、个人简介、图像和朋友网络以生成丰富的行为画像。一个专门的伊斯兰教合规层标识任何非清真指标和禁止的金融行为。该平台采用检索增强生成模块：LLM访问特定领域的知识库以为每个决策生成清晰的文本解释。本文描述了端到端的体系架构和数据流、使用的模型和系统基础设施。合成场景展示了社会信号如何转化为信用评分因素。本文强调概念创新、合规机制和实际影响，旨在面向AI研究人员、金融科技从业者、伦理银行法官和投资者。 

---
# "I Hadn't Thought About That": Creators of Human-like AI Weigh in on Ethics And Neurodivergence 

**Title (ZH)**: “我还没有考虑过这一点”：人类似AI的创作者谈伦理与神经多样xing 

**Authors**: Naba Rizvi, Taggert Smith, Tanvi Vidyala, Mya Bolds, Harper Strickland, Andrew Begel, Rua Williams, Imani Munyaka  

**Link**: [PDF](https://arxiv.org/pdf/2506.12098)  

**Abstract**: Human-like AI agents such as robots and chatbots are becoming increasingly popular, but they present a variety of ethical concerns. The first concern is in how we define humanness, and how our definition impacts communities historically dehumanized by scientific research. Autistic people in particular have been dehumanized by being compared to robots, making it even more important to ensure this marginalization is not reproduced by AI that may promote neuronormative social behaviors. Second, the ubiquitous use of these agents raises concerns surrounding model biases and accessibility. In our work, we investigate the experiences of the people who build and design these technologies to gain insights into their understanding and acceptance of neurodivergence, and the challenges in making their work more accessible to users with diverse needs. Even though neurodivergent individuals are often marginalized for their unique communication styles, nearly all participants overlooked the conclusions their end-users and other AI system makers may draw about communication norms from the implementation and interpretation of humanness applied in participants' work. This highlights a major gap in their broader ethical considerations, compounded by some participants' neuronormative assumptions about the behaviors and traits that distinguish "humans" from "bots" and the replication of these assumptions in their work. We examine the impact this may have on autism inclusion in society and provide recommendations for additional systemic changes towards more ethical research directions. 

**Abstract (ZH)**: 具有人类特征的AI代理，如机器人和聊天机器人正变得越来越流行，但它们提出了各种伦理关切。首关切点是我们在如何定义人类以及这种定义如何影响历史上被科学研究剥夺人性的社群方面存在的问题。特别是自闭症人士因其被与机器人相提并论而被剥夺人性，这使得确保AI不会进一步边缘化他们变得尤为重要，一些AI可能会促进以神经正常行为为准则的社会行为。其次，这些代理的普遍使用引发了模型偏见和可访问性方面的担忧。在我们的研究中，我们调查了构建和设计这些技术的人们的体验，以了解他们对神经多样性及其工作向具有不同需求的用户群体的可访问性理解与接受程度。尽管神经多样性个体常因独特的交流方式而被边缘化，但几乎所有参与者都忽视了其最终用户和其他AI系统制作者从参与者的工作中实施和解释的人性化结论中得出的关于交流规范的推论。这突显了他们在更广泛伦理考量中的重大缺口，同时也被一些参与者对辨别“人”与“机器人”的行为和特质的神经正常假设所加剧，并在他们的工作中复刻这些假设。我们探讨了这可能对自闭症在社会中的包容性产生的影响，并提供了朝着更道德研究方向的额外系统性变革建议。 

---
# Military AI Cyber Agents (MAICAs) Constitute a Global Threat to Critical Infrastructure 

**Title (ZH)**: 军用人工智能网络代理构成对关键基础设施的全球性威胁 

**Authors**: Timothy Dubber, Seth Lazar  

**Link**: [PDF](https://arxiv.org/pdf/2506.12094)  

**Abstract**: This paper argues that autonomous AI cyber-weapons - Military-AI Cyber Agents (MAICAs) - create a credible pathway to catastrophic risk. It sets out the technical feasibility of MAICAs, explains why geopolitics and the nature of cyberspace make MAICAs a catastrophic risk, and proposes political, defensive-AI and analogue-resilience measures to blunt the threat. 

**Abstract (ZH)**: 本文 argues that自主人工智能网络武器——军事人工智能网络代理（MAICAs）——构成了灾难性风险的可信途径。它阐述了MAICAs的技术可行性，解释了地缘政治和网络空间的特性为何使MAICAs成为灾难性风险，并提出政治、防御性人工智能以及类比韧性措施以缓解这一威胁。 

---
# Intelligent Automation for FDI Facilitation: Optimizing Tariff Exemption Processes with OCR And Large Language Models 

**Title (ZH)**: 智能自动化促进外国直接投资：利用OCR和大型语言模型优化关税豁免流程 

**Authors**: Muhammad Sukri Bin Ramli  

**Link**: [PDF](https://arxiv.org/pdf/2506.12093)  

**Abstract**: Tariff exemptions are fundamental to attracting Foreign Direct Investment (FDI) into the manufacturing sector, though the associated administrative processes present areas for optimization for both investing entities and the national tax authority. This paper proposes a conceptual framework to empower tax administration by leveraging a synergistic integration of Optical Character Recognition (OCR) and Large Language Model (LLM) technologies. The proposed system is designed to first utilize OCR for intelligent digitization, precisely extracting data from diverse application documents and key regulatory texts such as tariff orders. Subsequently, the LLM would enhance the capabilities of administrative officers by automating the critical and time-intensive task of verifying submitted HS Tariff Codes for machinery, equipment, and raw materials against official exemption lists. By enhancing the speed and precision of these initial assessments, this AI-driven approach systematically reduces potential for non-alignment and non-optimized exemption utilization, thereby streamlining the investment journey for FDI companies. For the national administration, the benefits include a significant boost in operational capacity, reduced administrative load, and a strengthened control environment, ultimately improving the ease of doing business and solidifying the nation's appeal as a premier destination for high-value manufacturing FDI. 

**Abstract (ZH)**: 关税豁免是吸引制造业外商直接投资（FDI）的基础，尽管相关行政流程为投资实体和国家税务机关都提供了优化空间。本文提出一种概念框架，通过结合光学字符识别（OCR）和大型语言模型（LLM）技术实现税务管理的赋能。该提出的系统首先利用OCR进行智能数字化，精确提取来自各种申请文件和关键法规文本（如关税令）的数据。随后，LLM将通过自动化验证提交的HS关税编码（适用于机械、设备和原材料）与官方豁免列表的一致性，增强行政人员的能力。通过提高这些初步评估的速度和准确性，这种基于AI的方法系统性地减少了潜在的不一致和未优化的豁免利用，从而简化外国直接投资公司的投资过程。对于国家管理机构而言，这项技术带来的好处包括显著提高运营能力、减少行政负担、强化控制环境，最终提高经商便利度，并巩固其作为高品质制造业外商直接投资目的地的吸引力。 

---
# Efficient Parallel Training Methods for Spiking Neural Networks with Constant Time Complexity 

**Title (ZH)**: 高效常时间复杂度并行训练方法用于脉冲神经网络 

**Authors**: Wanjin Feng, Xingyu Gao, Wenqian Du, Hailong Shi, Peilin Zhao, Pengcheng Wu, Chunyan Miao  

**Link**: [PDF](https://arxiv.org/pdf/2506.12087)  

**Abstract**: Spiking Neural Networks (SNNs) often suffer from high time complexity $O(T)$ due to the sequential processing of $T$ spikes, making training computationally expensive.
In this paper, we propose a novel Fixed-point Parallel Training (FPT) method to accelerate SNN training without modifying the network architecture or introducing additional assumptions.
FPT reduces the time complexity to $O(K)$, where $K$ is a small constant (usually $K=3$), by using a fixed-point iteration form of Leaky Integrate-and-Fire (LIF) neurons for all $T$ timesteps.
We provide a theoretical convergence analysis of FPT and demonstrate that existing parallel spiking neurons can be viewed as special cases of our proposed method.
Experimental results show that FPT effectively simulates the dynamics of original LIF neurons, significantly reducing computational time without sacrificing accuracy.
This makes FPT a scalable and efficient solution for real-world applications, particularly for long-term tasks.
Our code will be released at \href{this https URL}{\texttt{this https URL}}. 

**Abstract (ZH)**: Spiking神经网络（SNN）常常由于需要依次处理T个 spikes而导致较高的时间复杂度$O(T)$，从而使训练计算成本高昂。
本文提出了一种新颖的定点并行训练（FPT）方法，可以在不修改网络架构或引入额外假设的情况下加速SNN训练。
FPT通过使用定点迭代形式的Leaky Integrate-and-Fire（LIF）神经元将时间复杂度降低到$O(K)$，其中$K$是一个较小的常数（通常$K=3$）。
我们提供了FPT的理论收敛分析，并展示了现有并行SNN可以被视为我们提出方法的特例。
实验结果表明，FPT能够有效地模拟原始LIF神经元的动力学特性，显著减少了计算时间而不会牺牲准确度。
这使得FPT成为实时应用中的一个可扩展且高效的解决方案，尤其是在长期任务方面。
我们的代码将在\href{this https URL}{https://this https URL}发布。 

---
# Wanting to Be Understood Explains the Meta-Problem of Consciousness 

**Title (ZH)**: 渴望被理解解释了意识的元问题 

**Authors**: Chrisantha Fernando, Dylan Banarse, Simon Osindero  

**Link**: [PDF](https://arxiv.org/pdf/2506.12086)  

**Abstract**: Because we are highly motivated to be understood, we created public external representations -- mime, language, art -- to externalise our inner states. We argue that such external representations are a pre-condition for access consciousness, the global availability of information for reasoning. Yet the bandwidth of access consciousness is tiny compared with the richness of `raw experience', so no external representation can reproduce that richness in full. Ordinarily an explanation of experience need only let an audience `grasp' the relevant pattern, not relive the phenomenon. But our drive to be understood, and our low level sensorimotor capacities for `grasping' so rich, that the demand for an explanation of the feel of experience cannot be ``satisfactory''. That inflated epistemic demand (the preeminence of our expectation that we could be perfectly understood by another or ourselves) rather than an irreducible metaphysical gulf -- keeps the hard problem of consciousness alive. But on the plus side, it seems we will simply never give up creating new ways to communicate and think about our experiences. In this view, to be consciously aware is to strive to have one's agency understood by oneself and others. 

**Abstract (ZH)**: 由于我们有强烈的被理解动机，创造了公共外部表征——模仿、语言、艺术——来外部化我们的内心状态。我们认为，这种外部表征是获得性意识的先决条件，即为推理提供全球可用的信息。然而，获得性意识的信息带宽相比于“原始经验”的丰富性要小得多，因此没有外部表征能够完全再现这种丰富性。通常，对体验的解释只需要让观众“抓住”相关模式，而不必重新体验这一现象。但由于我们强烈的被理解动机，加之低层次的感觉运动能力难以“抓住”如此丰富的体验，因此对体验“感觉”的解释需求无法得到“满意”的满足。正是这种膨胀的知识需求（我们对能够被他人或自己完全理解的预期占主导地位）而非不可缩减的本体论鸿沟——使意识的难题保持鲜活。但另一方面，看来我们永远不会停止创造新的沟通和思考体验的方式。在这种观点中，拥有自觉意识就是努力使自己的行动被自己和他人理解。 

---
# The CAISAR Platform: Extending the Reach of Machine Learning Specification and Verification 

**Title (ZH)**: CAISAR平台：扩展机器学习规范与验证的范围 

**Authors**: Michele Alberti, François Bobot, Julien Girard-Satabin, Alban Grastien, Aymeric Varasse, Zakaria Chihani  

**Link**: [PDF](https://arxiv.org/pdf/2506.12084)  

**Abstract**: The formal specification and verification of machine learning programs saw remarkable progress in less than a decade, leading to a profusion of tools. However, diversity may lead to fragmentation, resulting in tools that are difficult to compare, except for very specific benchmarks. Furthermore, this progress is heavily geared towards the specification and verification of a certain class of property, that is, local robustness properties. But while provers are becoming more and more efficient at solving local robustness properties, even slightly more complex properties, involving multiple neural networks for example, cannot be expressed in the input languages of winners of the International Competition of Verification of Neural Networks VNN-Comp. In this tool paper, we present CAISAR, an open-source platform dedicated to machine learning specification and verification. We present its specification language, suitable for modelling complex properties on neural networks, support vector machines and boosted trees. We show on concrete use-cases how specifications written in this language are automatically translated to queries to state-of-the-art provers, notably by using automated graph editing techniques, making it possible to use their off-the-shelf versions. The artifact to reproduce the paper claims is available at the following DOI: this https URL 

**Abstract (ZH)**: 机器学习程序的形式化规范与验证在过去十年取得了显著进展，导致出现众多工具。然而，多样性可能导致碎片化，使得除了特定基准之外，这些工具难以比较。此外，这些进展主要集中在特定类别的属性，即局部鲁棒性属性上。尽管证明器在解决局部鲁棒性问题上正变得越来越高效，但稍微复杂的属性，例如涉及多个神经网络，无法在国际神经网络验证竞赛VNN-Comp获胜工具的输入语言中表达。在本工具论文中，我们介绍了CAISAR，一个开源平台，专注于机器学习的规范与验证。我们展示了其规范语言，适用于模型神经网络、支持向量机和提升树的复杂属性。我们通过使用自动化图编辑技术展示了如何在具体用例中自动将该语言编写的规范转换为最先进的证明器的查询，使其能够使用现成版本。用于复制论文中声明的成果可在以下DOI获取：[this https URL]。 

---
# Latency Optimization for Wireless Federated Learning in Multihop Networks 

**Title (ZH)**: 多跳网络中无线联邦学习的时延优化 

**Authors**: Shaba Shaon, Van-Dinh Nguyen, Dinh C. Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.12081)  

**Abstract**: In this paper, we study a novel latency minimization problem in wireless federated learning (FL) across multi-hop networks. The system comprises multiple routes, each integrating leaf and relay nodes for FL model training. We explore a personalized learning and adaptive aggregation-aware FL (PAFL) framework that effectively addresses data heterogeneity across participating nodes by harmonizing individual and collective learning objectives. We formulate an optimization problem aimed at minimizing system latency through the joint optimization of leaf and relay nodes, as well as relay routing indicator. We also incorporate an additional energy harvesting scheme for the relay nodes to help with their relay tasks. This formulation presents a computationally demanding challenge, and thus we develop a simple yet efficient algorithm based on block coordinate descent and successive convex approximation (SCA) techniques. Simulation results illustrate the efficacy of our proposed joint optimization approach for leaf and relay nodes with relay routing indicator. We observe significant latency savings in the wireless multi-hop PAFL system, with reductions of up to 69.37% compared to schemes optimizing only one node type, traditional greedy algorithm, and scheme without relay routing indicator. 

**Abstract (ZH)**: 在多跳网络中无线联邦学习中的新型延迟最小化问题研究 

---
# Modeling Earth-Scale Human-Like Societies with One Billion Agents 

**Title (ZH)**: 用一百万代理模型地球规模的人类社会 

**Authors**: Haoxiang Guan, Jiyan He, Liyang Fan, Zhenzhen Ren, Shaobin He, Xin Yu, Yuan Chen, Shuxin Zheng, Tie-Yan Liu, Zhen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12078)  

**Abstract**: Understanding how complex societal behaviors emerge from individual cognition and interactions requires both high-fidelity modeling of human behavior and large-scale simulations. Traditional agent-based models (ABMs) have been employed to study these dynamics for decades, but are constrained by simplified agent behaviors that fail to capture human complexity. Recent advances in large language models (LLMs) offer new opportunities by enabling agents to exhibit sophisticated social behaviors that go beyond rule-based logic, yet face significant scaling challenges. Here we present Light Society, an agent-based simulation framework that advances both fronts, efficiently modeling human-like societies at planetary scale powered by LLMs. Light Society formalizes social processes as structured transitions of agent and environment states, governed by a set of LLM-powered simulation operations, and executed through an event queue. This modular design supports both independent and joint component optimization, supporting efficient simulation of societies with over one billion agents. Large-scale simulations of trust games and opinion propagation--spanning up to one billion agents--demonstrate Light Society's high fidelity and efficiency in modeling social trust and information diffusion, while revealing scaling laws whereby larger simulations yield more stable and realistic emergent behaviors. 

**Abstract (ZH)**: 利用大型语言模型促进高效大规模社会行为建模：Light Society代理模型框架 

---
# A Synthetic Pseudo-Autoencoder Invites Examination of Tacit Assumptions in Neural Network Design 

**Title (ZH)**: 合成伪自动编码器促使神经网络设计中的隐含假设审视 

**Authors**: Assaf Marron  

**Link**: [PDF](https://arxiv.org/pdf/2506.12076)  

**Abstract**: We present a handcrafted neural network that, without training, solves the seemingly difficult problem of encoding an arbitrary set of integers into a single numerical variable, and then recovering the original elements. While using only standard neural network operations -- weighted sums with biases and identity activation -- we make design choices that challenge common notions in this area around representation, continuity of domains, computation, learnability and more. For example, our construction is designed, not learned; it represents multiple values using a single one by simply concatenating digits without compression, and it relies on hardware-level truncation of rightmost digits as a bit-manipulation mechanism. This neural net is not intended for practical application. Instead, we see its resemblance to -- and deviation from -- standard trained autoencoders as an invitation to examine assumptions that may unnecessarily constrain the development of systems and models based on autoencoding and machine learning. Motivated in part by our research on a theory of biological evolution centered around natural autoencoding of species characteristics, we conclude by refining the discussion with a biological perspective. 

**Abstract (ZH)**: 我们呈现了一个手工构建的神经网络，无需训练即可解决将任意整数集编码为单个数值变量并恢复原始元素这一看似困难的问题。我们仅使用标准神经网络操作（加权求和、偏置和恒等激活函数）进行设计，挑战了这一领域关于表示、域的连续性、计算和可学习性等方面的常见观念。例如，我们的构建方式是设计而非学习的；通过简单地串联数字而不进行压缩来表示多个值，并依赖于硬件层面的最右侧数字截断作为一种位操作机制。该神经网络并非旨在实际应用。相反，我们认为它与标准训练自编码器的相似性和差异可以作为一种邀请，促使我们审视那些可能无必要地限制自编码和机器学习系统与模型发展的假设。受到我们关于以自然自编码为中心的生物演化理论研究的启发，我们从生物学角度进一步细化了讨论。 

---
# T-TExTS (Teaching Text Expansion for Teacher Scaffolding): Enhancing Text Selection in High School Literature through Knowledge Graph-Based Recommendation 

**Title (ZH)**: T-TExTS (教学文本扩展以支持教师支架教学): 基于知识图谱的推荐增强高中文学文本选择 

**Authors**: Nirmal Gelal, Chloe Snow, Ambyr Rios, Hande Küçük McGinty  

**Link**: [PDF](https://arxiv.org/pdf/2506.12075)  

**Abstract**: The implementation of transformational pedagogy in secondary education classrooms requires a broad multiliteracy approach. Due to limited planning time and resources, high school English Literature teachers often struggle to curate diverse, thematically aligned literature text sets. This study addresses the critical need for a tool that provides scaffolds for novice educators in selecting literature texts that are diverse -- in terms of genre, theme, subtheme, and author -- yet similar in context and pedagogical merits. We have developed a recommendation system, Teaching Text Expansion for Teacher Scaffolding (T-TExTS), that suggests high school English Literature books based on pedagogical merits, genre, and thematic relevance using a knowledge graph. We constructed a domain-specific ontology using the KNowledge Acquisition and Representation Methodology (KNARM), transformed into a knowledge graph, which was then embedded using DeepWalk, biased random walk, and a hybrid of both approaches. The system was evaluated using link prediction and recommendation performance metrics, including Area Under the Curve (AUC), Mean Reciprocal Rank (MRR), Hits@K, and normalized Discounted Cumulative Gain (nDCG). DeepWalk outperformed in most ranking metrics, with the highest AUC (0.9431), whereas the hybrid model offered balanced performance. These findings demonstrate the importance of semantic, ontology-driven approaches in recommendation systems and suggest that T-TExTS can significantly ease the burden of English Literature text selection for high school educators, promoting more informed and inclusive curricular decisions. The source code for T-TExTS is available at: this https URL 

**Abstract (ZH)**: 在中学教室实施转变式教学法需要采用广义的多文学素养方法。由于规划时间有限和资源有限，高中英语文学教师在收集多样且主题一致的文学文本集方面常常面临困难。本研究旨在满足新手教师在选择文学文本方面的需求，这些文本在类型、主题、副主题和作者方面多样，但在情境和教学价值方面相似。我们开发了一种推荐系统——教师支架用以扩展教学文本（T-TExTS），该系统基于教学价值、类型和主题相关性建议高中英语文学书籍，并使用知识图谱进行建议。我们使用KNARM知识获取与表示方法构建了一个领域特定的本体，并将其转化为知识图谱，随后使用DeepWalk、带偏向的随机游走及两种方法的混合方法嵌入。系统通过链预测和推荐性能指标（包括AUC、MRR、Hits@K和nDCG）进行了评估。DeepWalk 在大多数排序指标中表现最佳，AUC（0.9431）最高，而混合模型提供了平衡的性能。这些发现展示了语义和本体驱动方法在推荐系统中的重要性，并表明T-TExTS 可显著减轻高中英语文学文本选择的负担，促进更具信息量和包容性的课程决策。T-TExTS 的源代码可通过以下链接获得：this https URL。 

---
# Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis 

**Title (ZH)**: 无障碍断裂语音文本对齐以分析发音障碍 

**Authors**: Zongli Ye, Jiachen Lian, Xuanru Zhou, Jinming Zhang, Haodong Li, Shuhe Li, Chenxu Guo, Anaisha Das, Peter Park, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.12073)  

**Abstract**: Accurate alignment of dysfluent speech with intended text is crucial for automating the diagnosis of neurodegenerative speech disorders. Traditional methods often fail to model phoneme similarities effectively, limiting their performance. In this work, we propose Neural LCS, a novel approach for dysfluent text-text and speech-text alignment. Neural LCS addresses key challenges, including partial alignment and context-aware similarity mapping, by leveraging robust phoneme-level modeling. We evaluate our method on a large-scale simulated dataset, generated using advanced data simulation techniques, and real PPA data. Neural LCS significantly outperforms state-of-the-art models in both alignment accuracy and dysfluent speech segmentation. Our results demonstrate the potential of Neural LCS to enhance automated systems for diagnosing and analyzing speech disorders, offering a more accurate and linguistically grounded solution for dysfluent speech alignment. 

**Abstract (ZH)**: 准确对齐非流利语音与意图文本对于自动化神经退行性言语障碍诊断至关重要。传统方法往往无法有效地建模音素相似性，限制了其性能。本文提出了一种新的非流利文本-文本和语音-文本对齐方法——神经最长公共子序列（Neural LCS），通过利用稳健的音素级建模来解决部分对齐和上下文相关相似性映射等关键挑战。我们在使用先进的数据模拟技术生成的大规模模拟数据集和真实PPA数据上评估了该方法，结果显示Neural LCS在对齐准确性和非流利语音分割方面显著优于当前最先进的模型。我们的结果表明，Neural LCS 有潜力增强自动化系统以诊断和分析言语障碍，提供一种更准确且语言学上更可靠的非流利语音对齐方案。 

---
# WebTrust: An AI-Driven Data Scoring System for Reliable Information Retrieval 

**Title (ZH)**: WebTrust: 一种基于AI的数据评分系统，用于可靠的信息检索 

**Authors**: Joydeep Chandra, Aleksandr Algazinov, Satyam Kumar Navneet, Rim El Filali, Matt Laing, Andrew Hanna  

**Link**: [PDF](https://arxiv.org/pdf/2506.12072)  

**Abstract**: As access to information becomes more open and widespread, people are increasingly using AI tools for assistance. However, many of these tools struggle to estimate the trustworthiness of the information. Although today's search engines include AI features, they often fail to offer clear indicators of data reliability. To address this gap, we introduce WebTrust, a system designed to simplify the process of finding and judging credible information online. Built on a fine-tuned version of IBM's Granite-1B model and trained on a custom dataset, WebTrust works by assigning a reliability score (from 0.1 to 1) to each statement it processes. In addition, it offers a clear justification for why a piece of information received that score. Evaluated using prompt engineering, WebTrust consistently achieves superior performance compared to other small-scale LLMs and rule-based approaches, outperforming them across all experiments on MAE, RMSE, and R2. User testing showed that when reliability scores are displayed alongside search results, people feel more confident and satisfied with the information they find. With its accuracy, transparency, and ease of use, WebTrust offers a practical solution to help combat misinformation and make trustworthy information more accessible to everyone. 

**Abstract (ZH)**: 随着信息获取的开放性和普及性增强，人们越来越多地使用AI工具提供协助。然而，许多这些工具在评估信息的可信度方面存在困难。尽管今天的搜索引擎包含了AI功能，但它们往往未能提供数据可靠性的明确指标。为解决这一问题，我们介绍了WebTrust系统，旨在简化在线查找和判断可信信息的过程。该系统基于fine-tuned版本的IBM Granite-1B模型，并在自定义数据集上进行训练，通过为每个处理的陈述分配一个从0.1到1的可靠性评分，并提供该信息获得此评分的清晰解释。评估结果显示，WebTrust在MAE、RMSE和R2指标上始终优于其他小型语言模型和基于规则的方法，在所有实验中均表现出色。用户测试表明，当在搜索结果旁边显示可靠性评分时，人们会更加自信和满意于找到的信息。凭借其准确性、透明性和易于使用性，WebTrust提供了一种实用的解决方案，有助于打击虚假信息并使可信信息更容易为广大公众获取。 

---
# Evaluating Logit-Based GOP Scores for Mispronunciation Detection 

**Title (ZH)**: 基于逻辑斯蒂回归的GOP评分在错音检测中的评估 

**Authors**: Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik  

**Link**: [PDF](https://arxiv.org/pdf/2506.12067)  

**Abstract**: Pronunciation assessment relies on goodness of pronunciation (GOP) scores, traditionally derived from softmax-based posterior probabilities. However, posterior probabilities may suffer from overconfidence and poor phoneme separation, limiting their effectiveness. This study compares logit-based GOP scores with probability-based GOP scores for mispronunciation detection. We conducted our experiment on two L2 English speech datasets spoken by Dutch and Mandarin speakers, assessing classification performance and correlation with human ratings. Logit-based methods outperform probability-based GOP in classification, but their effectiveness depends on dataset characteristics. The maximum logit GOP shows the strongest alignment with human perception, while a combination of different GOP scores balances probability and logit features. The findings suggest that hybrid GOP methods incorporating uncertainty modeling and phoneme-specific weighting improve pronunciation assessment. 

**Abstract (ZH)**: 基于逻辑斯谛和概率的发音评分方法在误读检测中的对比研究：综合不确定性建模和音素特异性加权以提高发音评估 

---
# Organizational Adaptation to Generative AI in Cybersecurity: A Systematic Review 

**Title (ZH)**: 组织应对生成式AI在网络安全中的适应：一项系统性回顾 

**Authors**: Christopher Nott  

**Link**: [PDF](https://arxiv.org/pdf/2506.12060)  

**Abstract**: Cybersecurity organizations are adapting to GenAI integration through modified frameworks and hybrid operational processes, with success influenced by existing security maturity, regulatory requirements, and investments in human capital and infrastructure. This qualitative research employs systematic document analysis and comparative case study methodology to examine how cybersecurity organizations adapt their threat modeling frameworks and operational processes to address generative artificial intelligence integration. Through examination of 25 studies from 2022 to 2025, the research documents substantial transformation in organizational approaches to threat modeling, moving from traditional signature-based systems toward frameworks incorporating artificial intelligence capabilities. The research identifies three primary adaptation patterns: Large Language Model integration for security applications, GenAI frameworks for risk detection and response automation, and AI/ML integration for threat hunting. Organizations with mature security infrastructures, particularly in finance and critical infrastructure sectors, demonstrate higher readiness through structured governance approaches, dedicated AI teams, and robust incident response processes. Organizations achieve successful GenAI integration when they maintain appropriate human oversight of automated systems, address data quality concerns and explainability requirements, and establish governance frameworks tailored to their specific sectors. Organizations encounter ongoing difficulties with privacy protection, bias reduction, personnel training, and defending against adversarial attacks. This work advances understanding of how organizations adopt innovative technologies in high-stakes environments and offers actionable insights for cybersecurity professionals implementing GenAI systems. 

**Abstract (ZH)**: 网络空间安全组织通过修改框架和混合运营流程适应生成式人工智能集成，其成功受现有安全成熟度、监管要求以及人力资本和基础设施投资的影响。本质性研究通过系统文件分析和比较案例研究方法，考察网络安全组织如何调整其威胁建模框架和运营流程以应对生成式人工智能集成。通过对2022年至2025年间25项研究的分析，研究记录了组织在威胁建模方面的显著转变，从传统的特征签名系统转向包含人工智能能力的框架。研究确定了三种主要适应模式：大型语言模型在安全应用中的集成、生成式人工智能框架用于风险检测和响应自动化，以及人工智能/机器学习在威胁检测中的集成。拥有成熟安全基础设施的组织，特别是在金融和关键基础设施领域，通过结构化的治理方法、专门的人工智能团队和 robust 的事件响应流程展示了更高的准备度。当组织维持适当的自动化系统的人工监督、解决数据质量和解释性要求，并建立符合其特定领域的治理框架时，它们能够成功实现生成式人工智能的集成。组织在隐私保护、偏见减少、人员培训以及抵御对抗性攻击方面面临持续挑战。本研究加深了对组织如何在高风险环境中采用创新技术的理解，并为实施生成式人工智能系统的网络安全专业人员提供了可操作的见解。 

---
# CMT-LLM: Contextual Multi-Talker ASR Utilizing Large Language Models 

**Title (ZH)**: CMT-LLM：利用大规模语言模型的上下文多说话人ASR 

**Authors**: Jiajun He, Naoki Sawada, Koichi Miyazaki, Tomoki Toda  

**Link**: [PDF](https://arxiv.org/pdf/2506.12059)  

**Abstract**: In real-world applications, automatic speech recognition (ASR) systems must handle overlapping speech from multiple speakers and recognize rare words like technical terms. Traditional methods address multi-talker ASR and contextual biasing separately, limiting performance in complex scenarios. We propose a unified framework that combines multi-talker overlapping speech recognition and contextual biasing into a single task. Our ASR method integrates pretrained speech encoders and large language models (LLMs), using optimized finetuning strategies. We also introduce a two-stage filtering algorithm to efficiently identify relevant rare words from large biasing lists and incorporate them into the LLM's prompt input, enhancing rare word recognition. Experiments show that our approach outperforms traditional contextual biasing methods, achieving a WER of 7.9% on LibriMix and 32.9% on AMI SDM when the biasing size is 1,000, demonstrating its effectiveness in complex speech scenarios. 

**Abstract (ZH)**: 实_escenary中，自动语音识别(ASR)系统必须处理多说话者重叠语音并识别如技术术语等罕见词汇。传统的方法分别处理多说话者ASR和上下文偏移，限制了在复杂场景中的性能。我们提出了一种统一框架，将多说话者重叠语音识别和上下文偏移结合为一个任务。我们的ASR方法整合了预训练的语音编码器和大规模语言模型(LLMs)，并使用优化的微调策略。我们还引入了两阶段过滤算法，以高效地从大型偏移列表中识别出相关罕见词汇，并将其纳入LLM的提示输入中，以增强罕见词汇的识别。实验结果显示，我们的方法在偏移规模为1,000时，在LibriMix上达到7.9%的WER，在AMI SDM上达到32.9%，证明了其在复杂语音场景中的有效性。 

---
# Towards Unified Neural Decoding with Brain Functional Network Modeling 

**Title (ZH)**: 基于脑功能网络建模的统一神经解码研究 

**Authors**: Di Wu, Linghao Bu, Yifei Jia, Lu Cao, Siyuan Li, Siyu Chen, Yueqian Zhou, Sheng Fan, Wenjie Ren, Dengchang Wu, Kang Wang, Yue Zhang, Yuehui Ma, Jie Yang, Mohamad Sawan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12055)  

**Abstract**: Recent achievements in implantable brain-computer interfaces (iBCIs) have demonstrated the potential to decode cognitive and motor behaviors with intracranial brain recordings; however, individual physiological and electrode implantation heterogeneities have constrained current approaches to neural decoding within single individuals, rendering interindividual neural decoding elusive. Here, we present Multi-individual Brain Region-Aggregated Network (MIBRAIN), a neural decoding framework that constructs a whole functional brain network model by integrating intracranial neurophysiological recordings across multiple individuals. MIBRAIN leverages self-supervised learning to derive generalized neural prototypes and supports group-level analysis of brain-region interactions and inter-subject neural synchrony. To validate our framework, we recorded stereoelectroencephalography (sEEG) signals from a cohort of individuals performing Mandarin syllable articulation. Both real-time online and offline decoding experiments demonstrated significant improvements in both audible and silent articulation decoding, enhanced decoding accuracy with increased multi-subject data integration, and effective generalization to unseen subjects. Furthermore, neural predictions for regions without direct electrode coverage were validated against authentic neural data. Overall, this framework paves the way for robust neural decoding across individuals and offers insights for practical clinical applications. 

**Abstract (ZH)**: 近期可植入脑机接口的进展展示了通过颅内脑记录解码认知和运动行为的潜力；然而，个体生理差异和电极植入异质性限制了当前在单一个体中进行神经解码的方法，使其在个体间神经解码难以实现。为此，我们提出了多个体脑区聚合网络（MIBRAIN），这是一种通过整合多个个体的颅内神经生理记录构建全功能脑网络模型的神经解码框架。MIBRAIN 利用半监督学习提取通用神经原型，并支持脑区间交互和跨个体神经同步的组级分析。为了验证该框架，我们对执行汉语音节发音的一组个体记录了立体脑电图（sEEG）信号。实时在线和离线解码实验均显示了对可闻和无声发音解码的重大改进，并且随着多个体数据集成解码准确性提高，还展示了对未见个体的有效泛化。此外，未直接覆盖电极的区域的神经预测与真实神经数据相符。总体而言，该框架为跨个体的稳健神经解码铺平了道路，并为实际临床应用提供了见解。 

---
# From Proxies to Fields: Spatiotemporal Reconstruction of Global Radiation from Sparse Sensor Sequences 

**Title (ZH)**: 从代理到场域：基于稀疏传感器序列的全球辐射时空重构 

**Authors**: Kazuma Kobayashi, Samrendra Roy, Seid Koric, Diab Abueidda, Syed Bahauddin Alam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12045)  

**Abstract**: Accurate reconstruction of latent environmental fields from sparse and indirect observations is a foundational challenge across scientific domains-from atmospheric science and geophysics to public health and aerospace safety. Traditional approaches rely on physics-based simulators or dense sensor networks, both constrained by high computational cost, latency, or limited spatial coverage. We present the Temporal Radiation Operator Network (TRON), a spatiotemporal neural operator architecture designed to infer continuous global scalar fields from sequences of sparse, non-uniform proxy measurements.
Unlike recent forecasting models that operate on dense, gridded inputs to predict future states, TRON addresses a more ill-posed inverse problem: reconstructing the current global field from sparse, temporally evolving sensor sequences, without access to future observations or dense labels. Demonstrated on global cosmic radiation dose reconstruction, TRON is trained on 22 years of simulation data and generalizes across 65,341 spatial locations, 8,400 days, and sequence lengths from 7 to 90 days. It achieves sub-second inference with relative L2 errors below 0.1%, representing a >58,000X speedup over Monte Carlo-based estimators. Though evaluated in the context of cosmic radiation, TRON offers a domain-agnostic framework for scientific field reconstruction from sparse data, with applications in atmospheric modeling, geophysical hazard monitoring, and real-time environmental risk forecasting. 

**Abstract (ZH)**: 从稀疏间接观测准确重构潜在环境场：时空神经算子网络在跨科学领域中的基础挑战及解决方案 

---
# Why Do Some Inputs Break Low-Bit LLM Quantization? 

**Title (ZH)**: 为什么某些输入会破坏低位宽LLM量化？ 

**Authors**: Ting-Yun Chang, Muru Zhang, Jesse Thomason, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2506.12044)  

**Abstract**: Low-bit weight-only quantization significantly reduces the memory footprint of large language models (LLMs), but disproportionately affects certain examples. We analyze diverse 3-4 bit methods on LLMs ranging from 7B-70B in size and find that the quantization errors of 50 pairs of methods are strongly correlated (avg. 0.82) on FineWeb examples. Moreover, the residual stream magnitudes of full-precision models are indicative of future quantization errors. We further establish a hypothesis that relates the residual stream magnitudes to error amplification and accumulation over layers. Using LLM localization techniques, early exiting, and activation patching, we show that examples with large errors rely on precise residual activations in the late layers, and that the outputs of MLP gates play a crucial role in maintaining the perplexity. Our work reveals why certain examples result in large quantization errors and which model components are most critical for performance preservation. 

**Abstract (ZH)**: Low-bit 权重量化显著减少了大型语言模型（LLMs）的内存占用，但对某些示例的影响不成比例。我们分析了从7B到70B不等大小的LLMs上多种3-4位方法，并发现50对方法在FineWeb示例上的量化误差高度相关（平均0.82）。此外，全精度模型的余量流幅度可以指示未来的量化误差。我们进一步假设余量流幅度与误差放大和积累有关。通过使用LLM定位技术、提前退出和激活补丁，我们证明了大误差的示例依赖于晚层精确的余量激活，并且MLP门的输出在保持困惑度方面起着关键作用。我们的工作揭示了为什么某些示例会导致大的量化误差，以及哪些模型组件对性能保持最为关键。 

---
# CRITS: Convolutional Rectifier for Interpretable Time Series Classification 

**Title (ZH)**: CRITS: 卷积修正器用于可解释的时间序列分类 

**Authors**: Alejandro Kuratomi, Zed Lee, Guilherme Dinis Chaliane Junior, Tony Lindgren, Diego García Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2506.12042)  

**Abstract**: Several interpretability methods for convolutional network-based classifiers exist. Most of these methods focus on extracting saliency maps for a given sample, providing a local explanation that highlights the main regions for the classification. However, some of these methods lack detailed explanations in the input space due to upscaling issues or may require random perturbations to extract the explanations. We propose Convolutional Rectifier for Interpretable Time Series Classification, or CRITS, as an interpretable model for time series classification that is designed to intrinsically extract local explanations. The proposed method uses a layer of convolutional kernels, a max-pooling layer and a fully-connected rectifier network (a network with only rectified linear unit activations). The rectified linear unit activation allows the extraction of the feature weights for the given sample, eliminating the need to calculate gradients, use random perturbations and the upscale of the saliency maps to the initial input space. We evaluate CRITS on a set of datasets, and study its classification performance and its explanation alignment, sensitivity and understandability. 

**Abstract (ZH)**: 基于卷积网络的时间序列分类解释方法：Convolutional Rectifier for Interpretable Time Series Classification (CRITS) 

---
# Meta Pruning via Graph Metanetworks : A Meta Learning Framework for Network Pruning 

**Title (ZH)**: 基于图元网络的元剪枝：一种网络剪枝的元学习框架 

**Authors**: Yewei Liu, Xiyuan Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12041)  

**Abstract**: Network pruning, aimed at reducing network size while preserving accuracy, has attracted significant research interest. Numerous pruning techniques have been proposed over time. They are becoming increasingly effective, but more complex and harder to interpret as well. Given the inherent complexity of neural networks, we argue that manually designing pruning criteria has reached a bottleneck. To address this, we propose a novel approach in which we "use a neural network to prune neural networks". More specifically, we introduce the newly developed idea of metanetwork from meta-learning into pruning. A metanetwork is a network that takes another network as input and produces a modified network as output. In this paper, we first establish a bijective mapping between neural networks and graphs, and then employ a graph neural network as our metanetwork. We train a metanetwork that learns the pruning strategy automatically which can transform a network that is hard to prune into another network that is much easier to prune. Once the metanetwork is trained, our pruning needs nothing more than a feedforward through the metanetwork and the standard finetuning to prune at state-of-the-art. Our method achieved outstanding results on many popular and representative pruning tasks (including ResNet56 on CIFAR10, VGG19 on CIFAR100, ResNet50 on ImageNet). Our code is available at this https URL 

**Abstract (ZH)**: 基于神经网络修剪神经网络的方法：利用元网络自动学习裁剪策略 

---
# BTC-LLM: Efficient Sub-1-Bit LLM Quantization via Learnable Transformation and Binary Codebook 

**Title (ZH)**: BTC-LLM: 通过可学习转换和二进制码本实现高效的亚1位LLM量化 

**Authors**: Hao Gu, Lujun Li, Zheyu Wang, Bei Liu, Qiyuan Zhu, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.12040)  

**Abstract**: Binary quantization represents the most extreme form of large language model (LLM) compression, reducing weights to $\pm$1 for maximal memory and computational efficiency. While recent sparsity-aware binarization methods achieve sub-1-bit compression by pruning redundant binary weights, they suffer from three critical challenges: performance deterioration, computational complexity from sparse mask management, and limited hardware compatibility. In this paper, we present BTC-LLM, a novel sub-1-bit LLM quantization framework that leverages adaptive weight transformation and binary pattern clustering to overcome these limitations, delivering both superior accuracy and efficiency. Our approach incorporates two key innovations: (1) a Learnable Transformation that optimizes invertible scaling and rotation matrices to align binarized weights with full-precision distributions, enabling incoherence processing to enhance layer-wise representation quality; (2) a Flash and Accurate Binary Codebook that identifies recurring binary vector clusters, compressing them into compact indices with tailored distance metrics and sign-based centroid updates. This eliminates the need for sparse masks, enabling efficient inference on standard hardware. Our code is available at this https URL. 

**Abstract (ZH)**: BTC-LLM：一种利用自适应权重变换和二值模式聚类的亚1比特大语言模型量化框架 

---
# The Maximal Overlap Discrete Wavelet Scattering Transform and Its Application in Classification Tasks 

**Title (ZH)**: 最大重叠离散小波散射变换及其在分类任务中的应用 

**Authors**: Leonardo Fonseca Larrubia, Pedro Alberto Morettin, Chang Chiann  

**Link**: [PDF](https://arxiv.org/pdf/2506.12039)  

**Abstract**: We present the Maximal Overlap Discrete Wavelet Scattering Transform (MODWST), whose construction is inspired by the combination of the Maximal Overlap Discrete Wavelet Transform (MODWT) and the Scattering Wavelet Transform (WST). We also discuss the use of MODWST in classification tasks, evaluating its performance in two applications: stationary signal classification and ECG signal classification. The results demonstrate that MODWST achieved good performance in both applications, positioning itself as a viable alternative to popular methods like Convolutional Neural Networks (CNNs), particularly when the training data set is limited. 

**Abstract (ZH)**: 最大重叠离散小波散射变换（MODWST）及其在分类任务中的应用 

---
# LCD: Advancing Extreme Low-Bit Clustering for Large Language Models via Knowledge Distillation 

**Title (ZH)**: LCD：通过知识蒸馏推进面向大型语言模型的极端低比特聚类技术 

**Authors**: Fangxin Liu, Ning Yang, Junping Zhao, Tao Yang, Haibing Guan, Li Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12038)  

**Abstract**: Large language models (LLMs) have achieved significant progress in natural language processing but face challenges in deployment due to high memory and computational requirements. Weight quantization is a common approach to address these issues, yet achieving effective low-bit compression remains challenging. This paper presents LCD, which unifies the learning of clustering-based quantization within a knowledge distillation framework. Using carefully designed optimization techniques, LCD preserves LLM performance even at ultra-low bit widths of 2-3 bits. Additionally, LCD compresses activations through smoothing and accelerates inference with a LUT-based design. Experimental results show that LCD outperforms existing methods and delivers up to a 6.2x speedup in inference. Notably, LCD is shown to be more cost-effective, making it a practical solution for real-world applications. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理领域取得了显著进展，但由于高内存和计算需求，在部署上面临挑战。权值量化是一种常见的解决方法，但实现有效的低比特压缩仍然具有挑战性。本文提出了LCD，它将基于聚类的量化学习统一到知识蒸馏框架中。通过精心设计的优化技术，LCD 即使在超低比特宽度（2-3比特）下也能保持LLM的性能。此外，LCD 通过平滑处理压缩激活，并使用LUT 基础设计加速推断。实验结果表明，LCD 超越了现有方法，并在推断上提供高达6.2倍的加速。值得注意的是，LCD 被证明更具成本效益，使其成为一个实用的现实应用解决方案。 

---
# How to Train a Model on a Cheap Cluster with Low Cost using Block Coordinate Descent 

**Title (ZH)**: 如何使用块坐标下降法在低成本的集群上训练模型 

**Authors**: Zeyu Liu, Yunquan Zhang, Boyang Zhang, Guoyong Jiang, Daning Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.12037)  

**Abstract**: Training large language models typically demands extensive GPU memory and substantial financial investment, which poses a barrier for many small- to medium-sized teams. In this paper, we present a full-parameter pre-training framework based on block coordinate descent (BCD), augmented with engineering optimizations, to efficiently train large models on affordable RTX 4090 GPU clusters. BCD ensures model convergence based on block coordinate descent theory and performs gradient computation and update at the level of parameter blocks. Experiments show that 1) Lower cost of Same-Device: BCD significantly reduces pre-training cost. For the 7B model, under identical hardware settings, BCD lowers training costs to approximately 33% on A100,A800 clusters on 7B model averagely and to approximately 2.6% on RTX 4090 clusters on 7B model, compared to traditional full-parameter training. 2) Cross-Device Transfer: By leveraging BCD, large-scale models previously trainable only on high-end A100 clusters can be seamlessly migrated and pre-trained on 4090 clusters-whose hourly cost is only one-quarter that of A100-without requiring expensive hardware. 3) Accuracy Retention: In both scenarios, BCD training achieves the same level of model accuracy as full-parameter pre-training. 

**Abstract (ZH)**: 基于块坐标下降的全参数预训练框架：在经济实惠的RTX 4090 GPU集群上高效训练大型语言模型 

---
# A Minimalist Method for Fine-tuning Text-to-Image Diffusion Models 

**Title (ZH)**: 极简方法 fine-tuning 文本到图像扩散模型 

**Authors**: Yanting Miao, William Loh, Suraj Kothawade, Pacal Poupart  

**Link**: [PDF](https://arxiv.org/pdf/2506.12036)  

**Abstract**: Recent work uses reinforcement learning (RL) to fine-tune text-to-image diffusion models, improving text-image alignment and sample quality. However, existing approaches introduce unnecessary complexity: they cache the full sampling trajectory, depend on differentiable reward models or large preference datasets, or require specialized guidance techniques. Motivated by the "golden noise" hypothesis -- that certain initial noise samples can consistently yield superior alignment -- we introduce Noise PPO, a minimalist RL algorithm that leaves the pre-trained diffusion model entirely frozen and learns a prompt-conditioned initial noise generator. Our approach requires no trajectory storage, reward backpropagation, or complex guidance tricks. Extensive experiments show that optimizing the initial noise distribution consistently improves alignment and sample quality over the original model, with the most significant gains at low inference steps. As the number of inference steps increases, the benefit of noise optimization diminishes but remains present. These findings clarify the scope and limitations of the golden noise hypothesis and reinforce the practical value of minimalist RL fine-tuning for diffusion models. 

**Abstract (ZH)**: 最近的工作使用强化学习（RL）微调文本到图像扩散模型，提高了文本与图像的对齐和样本质量。然而，现有方法引入了不必要的复杂性：它们缓存完整的采样轨迹，依赖可微奖励模型或大规模偏好数据集，或者需要特殊指导技术。受“金色噪声”假设的启发——即某些初始噪声样本可以一致地产生更好的对齐——我们引入了Noise PPO，这是一种极简主义的RL算法，使预训练的扩散模型完全冻结，并学习一个基于提示的初始噪声生成器。我们的方法不需要轨迹存储、奖励反向传播或复杂的指导技巧。广泛的实验表明，优化初始噪声分布一致地提高了对齐和样本质量，尤其是在较低的推理步骤中效果最为显著。随着推理步骤的增加，噪声优化的好处减弱但仍保持。这些发现澄清了金色噪声假设的适用范围和局限性，并强化了扩散模型极简主义RL微调的实际价值。 

---
# MARché: Fast Masked Autoregressive Image Generation with Cache-Aware Attention 

**Title (ZH)**: MARché: 快速掩码自回归图像生成与缓存意识注意 

**Authors**: Chaoyi Jiang, Sungwoo Kim, Lei Gao, Hossein Entezari Zarch, Won Woo Ro, Murali Annavaram  

**Link**: [PDF](https://arxiv.org/pdf/2506.12035)  

**Abstract**: Masked autoregressive (MAR) models unify the strengths of masked and autoregressive generation by predicting tokens in a fixed order using bidirectional attention for image generation. While effective, MAR models suffer from significant computational overhead, as they recompute attention and feed-forward representations for all tokens at every decoding step, despite most tokens remaining semantically stable across steps. We propose a training-free generation framework MARché to address this inefficiency through two key components: cache-aware attention and selective KV refresh. Cache-aware attention partitions tokens into active and cached sets, enabling separate computation paths that allow efficient reuse of previously computed key/value projections without compromising full-context modeling. But a cached token cannot be used indefinitely without recomputation due to the changing contextual information over multiple steps. MARché recognizes this challenge and applies a technique called selective KV refresh. Selective KV refresh identifies contextually relevant tokens based on attention scores from newly generated tokens and updates only those tokens that require recomputation, while preserving image generation quality. MARché significantly reduces redundant computation in MAR without modifying the underlying architecture. Empirically, MARché achieves up to 1.7x speedup with negligible impact on image quality, offering a scalable and broadly applicable solution for efficient masked transformer generation. 

**Abstract (ZH)**: Masked autoregressive (MAR)模型通过使用双向注意力以固定顺序预测 tokens 来统一掩蔽生成和自回归生成的优势，适用于图像生成。尽管有效，MAR模型在每个解码步骤中都会为所有tokens重新计算注意力和 feed-forward 表示，尽管大多数tokens在步骤间保持语义稳定，导致显著的计算开销。我们提出了一种无需训练的生成框架MARché，通过两个关键组件解决这一低效问题：aware 缓存注意力和选择性 KV 刷新。aware 缓存注意力将 tokens 分为活动集和缓存集，启用独立的计算路径，允许高效重用先前计算的 key/value 投影，同时保持全面上下文建模能力。但缓存的 token 由于多步骤中的上下文信息变化，无法无限期使用而无需重新计算。MARché 认识到这一挑战，并应用一种称为选择性 KV 刷新的技术。选择性 KV 刷新根据新生成 tokens 的注意力分数识别上下文相关 tokens，并仅更新需要重新计算的 tokens，同时保持图像生成质量。MARché 在不修改底层架构的情况下显著减少了 MAR 中的冗余计算。实证结果表明，MARché 在图像质量影响可以忽略不计的情况下可实现高达 1.7 倍的速度提升，提供了一种可扩展且广泛适用的高效掩蔽变换器生成解决方案。 

---
# Human-like Forgetting Curves in Deep Neural Networks 

**Title (ZH)**: 类人类的遗忘曲线在深度神经网络中 

**Authors**: Dylan Kline  

**Link**: [PDF](https://arxiv.org/pdf/2506.12034)  

**Abstract**: This study bridges cognitive science and neural network design by examining whether artificial models exhibit human-like forgetting curves. Drawing upon Ebbinghaus' seminal work on memory decay and principles of spaced repetition, we propose a quantitative framework to measure information retention in neural networks. Our approach computes the recall probability by evaluating the similarity between a network's current hidden state and previously stored prototype representations. This retention metric facilitates the scheduling of review sessions, thereby mitigating catastrophic forgetting during deployment and enhancing training efficiency by prompting targeted reviews. Our experiments with Multi-Layer Perceptrons reveal human-like forgetting curves, with knowledge becoming increasingly robust through scheduled reviews. This alignment between neural network forgetting curves and established human memory models identifies neural networks as an architecture that naturally emulates human memory decay and can inform state-of-the-art continual learning algorithms. 

**Abstract (ZH)**: 本研究通过探讨人工模型是否表现出类似人类的记忆衰减曲线，将认知科学与神经网络设计相结合，借鉴艾宾浩斯的记忆衰退及其间隔重复原理，提出了一种量化信息保留度的框架。该方法通过评估网络当前隐藏状态与先前存储的原型表示之间的相似性来计算回忆概率。该保留度指标有助于安排复习时段，从而在部署时减轻灾难性遗忘，并通过促进有针对性的复习来提高训练效率。我们的实验显示，多层感知机表现出类似人类的记忆衰减曲线，通过计划复习，知识逐渐变得更为稳固。神经网络的记忆衰减曲线与已建立的人类记忆模型之间的契合表明，神经网络自然地模仿了人类的记忆衰退，可以指导最新的持续学习算法的发展。 

---
# EMERGENT: Efficient and Manipulation-resistant Matching using GFlowNets 

**Title (ZH)**: EMERGENT: 效率高且抗操控的匹配方法基于GFlowNets 

**Authors**: Mayesha Tasnim, Erman Acar, Sennay Ghebreab  

**Link**: [PDF](https://arxiv.org/pdf/2506.12033)  

**Abstract**: The design of fair and efficient algorithms for allocating public resources, such as school admissions, housing, or medical residency, has a profound social impact. In one-sided matching problems, where individuals are assigned to items based on ranked preferences, a fundamental trade-off exists between efficiency and strategyproofness. Existing algorithms like Random Serial Dictatorship (RSD), Probabilistic Serial (PS), and Rank Minimization (RM) capture only one side of this trade-off: RSD is strategyproof but inefficient, while PS and RM are efficient but incentivize manipulation. We propose EMERGENT, a novel application of Generative Flow Networks (GFlowNets) to one-sided matching, leveraging its ability to sample diverse, high-reward solutions. In our approach, efficient and manipulation-resistant matches emerge naturally: high-reward solutions yield efficient matches, while the stochasticity of GFlowNets-based outputs reduces incentives for manipulation. Experiments show that EMERGENT outperforms RSD in rank efficiency while significantly reducing strategic vulnerability compared to matches produced by RM and PS. Our work highlights the potential of GFlowNets for applications involving social choice mechanisms, where it is crucial to balance efficiency and manipulability. 

**Abstract (ZH)**: 公平且高效的公共资源分配算法设计：生成流网络在单边匹配问题中的应用 

---
# Embedding Trust at Scale: Physics-Aware Neural Watermarking for Secure and Verifiable Data Pipelines 

**Title (ZH)**: 大规模嵌入信任：物理感知神经水印在安全可验证数据管道中的应用 

**Authors**: Krti Tallam  

**Link**: [PDF](https://arxiv.org/pdf/2506.12032)  

**Abstract**: We present a robust neural watermarking framework for scientific data integrity, targeting high-dimensional fields common in climate modeling and fluid simulations. Using a convolutional autoencoder, binary messages are invisibly embedded into structured data such as temperature, vorticity, and geopotential. Our method ensures watermark persistence under lossy transformations - including noise injection, cropping, and compression - while maintaining near-original fidelity (sub-1\% MSE). Compared to classical singular value decomposition (SVD)-based watermarking, our approach achieves $>$98\% bit accuracy and visually indistinguishable reconstructions across ERA5 and Navier-Stokes datasets. This system offers a scalable, model-compatible tool for data provenance, auditability, and traceability in high-performance scientific workflows, and contributes to the broader goal of securing AI systems through verifiable, physics-aware watermarking. We evaluate on physically grounded scientific datasets as a representative stress-test; the framework extends naturally to other structured domains such as satellite imagery and autonomous-vehicle perception streams. 

**Abstract (ZH)**: 一种用于气候模型和流体模拟中高维数据完整性保护的稳健神经水印框架 

---
# Improving Generalization in Heterogeneous Federated Continual Learning via Spatio-Temporal Gradient Matching with Prototypical Coreset 

**Title (ZH)**: 基于空间-时间梯度匹配和原型coreset的异构联邦连续学习泛化能力提升 

**Authors**: Minh-Duong Nguyen, Le-Tuan Nguyen, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2506.12031)  

**Abstract**: Federated Continual Learning (FCL) has recently emerged as a crucial research area, as data from distributed clients typically arrives as a stream, requiring sequential learning. This paper explores a more practical and challenging FCL setting, where clients may have unrelated or even conflicting data and tasks. In this scenario, statistical heterogeneity and data noise can create spurious correlations, leading to biased feature learning and catastrophic forgetting. Existing FCL approaches often use generative replay to create pseudo-datasets of previous tasks. However, generative replay itself suffers from catastrophic forgetting and task divergence among clients, leading to overfitting in FCL. Existing FCL approaches often use generative replay to create pseudo-datasets of previous tasks. However, generative replay itself suffers from catastrophic forgetting and task divergence among clients, leading to overfitting in FCL. To address these challenges, we propose a novel approach called Spatio-Temporal grAdient Matching with network-free Prototype (STAMP). Our contributions are threefold: 1) We develop a model-agnostic method to determine subset of samples that effectively form prototypes when using a prototypical network, making it resilient to continual learning challenges; 2) We introduce a spatio-temporal gradient matching approach, applied at both the client-side (temporal) and server-side (spatial), to mitigate catastrophic forgetting and data heterogeneity; 3) We leverage prototypes to approximate task-wise gradients, improving gradient matching on the client-side. Extensive experiments demonstrate our method's superiority over existing baselines. 

**Abstract (ZH)**: 联邦连续学习（Federated Continual Learning, FCL）最近已成为一个关键的研究领域，因为来自分布式客户端的数据通常以流的形式到达，需要进行顺序学习。本文探讨了一个更具实践意义和挑战性的FCL设置，其中客户端可能具有无关甚至冲突的数据和任务。在这种场景下，统计异质性和数据噪声可以产生虚假的相关性，导致特征学习偏差和灾难性遗忘。现有FCL方法通常使用生成性重放来创建之前任务的伪数据集。然而，生成性重放本身会受到灾难性遗忘和客户端间任务发散的影响，导致在FCL中出现过拟合。为了解决这些挑战，我们提出了一种名为Spatio-Temporal GrAdient Matching with Network-free Prototype (STAMP)的新方法。我们的贡献包括三个方面：1）我们开发了一种模型无关的方法，用于确定在使用原型网络时形成的样本子集，使其能够应对连续学习的挑战；2）我们引入了一种时空梯度匹配方法，在客户端（时间维度）和服务器端（空间维度）应用，以缓解灾难性遗忘和数据异质性；3）我们利用原型来近似任务梯度，在客户端提高梯度匹配精度。广泛的实验结果表明，我们的方法优于现有的基线方法。 

---
# Impact, Causation and Prediction of Socio-Academic and Economic Factors in Exam-centric Student Evaluation Measures using Machine Learning and Causal Analysis 

**Title (ZH)**: 基于机器学习和因果分析的以考试为中心的学生评价指标中社会学术和经济因素的影响、因果关系及预测研究 

**Authors**: Md. Biplob Hosen, Sabbir Ahmed, Bushra Akter, Mehrin Anannya  

**Link**: [PDF](https://arxiv.org/pdf/2506.12030)  

**Abstract**: Understanding socio-academic and economic factors influencing students' performance is crucial for effective educational interventions. This study employs several machine learning techniques and causal analysis to predict and elucidate the impacts of these factors on academic performance. We constructed a hypothetical causal graph and collected data from 1,050 student profiles. Following meticulous data cleaning and visualization, we analyze linear relationships through correlation and variable plots, and perform causal analysis on the hypothetical graph. Regression and classification models are applied for prediction, and unsupervised causality analysis using PC, GES, ICA-LiNGAM, and GRASP algorithms is conducted. Our regression analysis shows that Ridge Regression achieve a Mean Absolute Error (MAE) of 0.12 and a Mean Squared Error (MSE) of 0.024, indicating robustness, while classification models like Random Forest achieve nearly perfect F1-scores. The causal analysis shows significant direct and indirect effects of factors such as class attendance, study hours, and group study on CGPA. These insights are validated through unsupervised causality analysis. By integrating the best regression model into a web application, we are developing a practical tool for students and educators to enhance academic outcomes based on empirical evidence. 

**Abstract (ZH)**: 理解影响学生学业表现的社会学术和经济因素对于有效的教育干预至关重要。本研究采用了多种机器学习技术和因果分析来预测并阐明这些因素对学业表现的影响。我们构建了一个假设因果图，并收集了1,050名学生的资料。经过细致的数据清洗和可视化处理，我们通过对相关性和变量图分析线性关系，并在假设图上进行因果分析。应用回归和分类模型进行预测，并使用PC、GES、ICA-LiNGAM和GRASP算法进行无监督因果分析。回归分析结果显示，岭回归的平均绝对误差（MAE）为0.12，均方误差（MSE）为0.024，表明其稳健性，而分类模型如随机森林几乎达到完美的F1-score。因果分析显示，班级出勤率、学习时间以及小组学习等因素对GPA有显著的直接影响和间接影响。这些洞察通过无监督因果分析得到验证。通过将最佳回归模型集成到-web应用中，我们正在开发一个基于实证证据的实际工具，帮助学生和教育工作者提升学业成果。 

---
# Physics-Informed Neural Networks for Vessel Trajectory Prediction: Learning Time-Discretized Kinematic Dynamics via Finite Differences 

**Title (ZH)**: 基于物理的信息神经网络的血管轨迹预测：通过有限差分学习时间离散化的运动学动力学 

**Authors**: Md Mahbub Alam, Amilcar Soares, José F. Rodrigues-Jr, Gabriel Spadon  

**Link**: [PDF](https://arxiv.org/pdf/2506.12029)  

**Abstract**: Accurate vessel trajectory prediction is crucial for navigational safety, route optimization, traffic management, search and rescue operations, and autonomous navigation. Traditional data-driven models lack real-world physical constraints, leading to forecasts that disobey vessel motion dynamics, such as in scenarios with limited or noisy data where sudden course changes or speed variations occur due to external factors. To address this limitation, we propose a Physics-Informed Neural Network (PINN) approach for trajectory prediction that integrates a streamlined kinematic model for vessel motion into the neural network training process via a first- and second-order, finite difference physics-based loss function. This loss function, discretized using the first-order forward Euler method, Heun's second-order approximation, and refined with a midpoint approximation based on Taylor series expansion, enforces fidelity to fundamental physical principles by penalizing deviations from expected kinematic behavior. We evaluated PINN using real-world AIS datasets that cover diverse maritime conditions and compared it with state-of-the-art models. Our results demonstrate that the proposed method reduces average displacement errors by up to 32% across models and datasets while maintaining physical consistency. These results enhance model reliability and adherence to mission-critical maritime activities, where precision translates into better situational awareness in the oceans. 

**Abstract (ZH)**: 准确的船舶轨迹预测对于导航安全、航线优化、交通管理、搜救行动和自主导航至关重要。传统的数据驱动模型缺乏现实世界物理约束，导致预测结果违背船舶运动动力学，尤其是在数据有限或噪声较大时，由于外部因素导致的航向突变或速度变化场景中表现不佳。为解决这一问题，我们提出了一种物理信息神经网络（PINN）方法，通过引入简化动力学模型将物理约束整合到神经网络训练过程中，使用基于有限差分的零阶和二阶物理损失函数。该损失函数通过一阶向前欧拉方法、Heun二阶逼近方法，并结合泰勒级数展开的中点逼近方法进行离散化，从而通过惩罚与预期动力学行为的偏差，确保模型符合基本的物理原理。我们使用覆盖各种海上条件的真实世界AIS数据集评估了PINN，并将其与最先进的模型进行了比较。结果显示，所提出的方法在模型和数据集上将平均位移误差降低了高达32%，同时保持了物理一致性。这些结果增强了模型的可靠性和对关键海事业务的适应性，精准性在海洋中转化为更好的态势感知能力。 

---
# The Limits of Tractable Marginalization 

**Title (ZH)**: 可处理边际化的局限性 

**Authors**: Oliver Broadrick, Sanyam Agarwal, Guy Van den Broeck, Markus Bläser  

**Link**: [PDF](https://arxiv.org/pdf/2506.12020)  

**Abstract**: Marginalization -- summing a function over all assignments to a subset of its inputs -- is a fundamental computational problem with applications from probabilistic inference to formal verification. Despite its computational hardness in general, there exist many classes of functions (e.g., probabilistic models) for which marginalization remains tractable, and they can be commonly expressed by polynomial size arithmetic circuits computing multilinear polynomials. This raises the question, can all functions with polynomial time marginalization algorithms be succinctly expressed by such circuits? We give a negative answer, exhibiting simple functions with tractable marginalization yet no efficient representation by known models, assuming $\textsf{FP}\neq\#\textsf{P}$ (an assumption implied by $\textsf{P} \neq \textsf{NP}$). To this end, we identify a hierarchy of complexity classes corresponding to stronger forms of marginalization, all of which are efficiently computable on the known circuit models. We conclude with a completeness result, showing that whenever there is an efficient real RAM performing virtual evidence marginalization for a function, then there are small circuits for that function's multilinear representation. 

**Abstract (ZH)**: 隶属运算——对函数的一组输入的所有赋值求和——是概率推理和形式验证等领域的重要计算问题。尽管隶属运算通常计算难度较高，但仍有一些类别的函数（例如概率模型）使其保持可计算性，这些函数可以用计算多项式多项式的算术电路简洁表达。这引发了一个问题：所有具有多项式时间隶属运算算法的函数是否都能用这样的电路简洁表达？我们给出了否定的答案，展示了具有可计算隶属运算但无法用已知模型有效表示的简单函数（假设$\textsf{FP}\neq\#\textsf{P}$，这是一个P$\neq$NP的推论）。为此，我们识别了一个复杂性类层次结构，对应于更强形式的隶属运算，所有这些都在已知的电路模型上高效可计算。最后，我们得出一个完备性结果，表明每当存在一个高效的实数RAM实现某个函数的虚拟证据隶属运算时，都存在该函数的多项式线性表示的小电路。 

---
# Examining the effects of music on cognitive skills of children in early childhood with the Pythagorean fuzzy set approach 

**Title (ZH)**: 使用毕达哥拉斯模糊集方法探究音乐对幼儿认知能力的影响 

**Authors**: Murat Kirisci, Nihat Topac, Musa Bardak  

**Link**: [PDF](https://arxiv.org/pdf/2506.12016)  

**Abstract**: There are many genetic and environmental factors that affect cognitive development. Music education can also be considered as one of the environmental factors. Some researchers emphasize that Music is an action that requires meta-cognitive functions such as mathematics and chess and supports spatial intelligence. The effect of Music on cognitive development in early childhood was examined using the Pythagorean Fuzzy Sets(PFS) method defined by Yager. This study created PFS based on experts' opinions, and an algorithm was given according to PFS. The algorithm's results supported the experts' data on the development of spatial-temporal skills in music education given in early childhood. The algorithm's ranking was done using the Expectation Score Function. The rankings obtained from the algorithm overlap with the experts' rankings. 

**Abstract (ZH)**: 音乐教育对早期儿童认知发展的影响：基于Yager定义的Pythagorean模糊集方法的研究 

---
