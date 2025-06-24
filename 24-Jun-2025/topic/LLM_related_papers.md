# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective 

**Title (ZH)**: 演化内省提示：一种开放视角的自复制观点 

**Authors**: Jianyu Wang, Zhiqiang Hu, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2506.17930)  

**Abstract**: We propose a novel prompt design paradigm that challenges conventional wisdom in large language model (LLM) prompting. While conventional wisdom prioritizes well-crafted instructions and demonstrations for in-context learning (ICL), we show that pruning random demonstrations into seemingly incoherent "gibberish" can remarkably improve performance across diverse tasks. Notably, the "gibberish" always matches or surpasses state-of-the-art automatic prompt optimization techniques, achieving substantial gains regardless of LLM alignment. Nevertheless, discovering an effective pruning strategy is non-trivial, as existing attribution methods and prompt compression algorithms fail to deliver robust results, let alone human intuition. In terms of this, we propose a self-discover prompt optimization framework, PromptQuine, an evolutionary search framework that automatically searches for the pruning strategy by itself using only low-data regimes. Much like the emergent complexity in nature--such as symbiosis and self-organization--arising in response to resource constraints, our framework evolves and refines unconventional yet highly effective prompts by leveraging only the tokens present within the context. We demonstrate its effectiveness across classification, multi-choice question answering, generation and math reasoning tasks across LLMs, while achieving decent runtime efficiency. We hope our findings can guide mechanistic studies on in-context learning, and provide a call to action, to pave the way for more open-ended search algorithms for more effective LLM prompting. 

**Abstract (ZH)**: 我们提出了一种新颖的提示设计范式，挑战了大型语言模型（LLM）提示领域的传统智慧。虽然传统观点强调精心设计的指令和示范以进行上下文内学习（ICL），我们证明将随机示范精简为看似不连贯的“胡言乱语”可以显著提升跨多种任务的表现。值得注意的是，“胡言乱语”总是能够匹敌或超越最先进的自动提示优化技术，在不同LLM对齐的情况下也能取得显著的进步。然而，发现有效的精简策略是不简单的，现有的归因方法和提示压缩算法无法提供稳健的结果，更不用说依靠人类直觉。为此，我们提出了一种自我发现提示优化框架——PromptQuine，这是一种进化搜索框架，仅通过低数据环境自动搜索精简策略。我们的框架通过利用上下文中的词元，进化并精炼出不寻常但极为有效的提示。我们展示了其在分类、多项选择题回答、生成和数学推理任务中的有效性，同时具有不错的运行时效率。我们希望我们的发现能够指导有关上下文内学习的机制性研究，并提供一种行动号召，以铺平更多开放搜索算法的道路，使LLM提示更加高效。 

---
# Steering Conceptual Bias via Transformer Latent-Subspace Activation 

**Title (ZH)**: 通过变换器潜空间激活引导概念偏见 

**Authors**: Vansh Sharma, Venkat Raman  

**Link**: [PDF](https://arxiv.org/pdf/2506.18887)  

**Abstract**: This work examines whether activating latent subspaces in language models (LLMs) can steer scientific code generation toward a specific programming language. Five causal LLMs were first evaluated on scientific coding prompts to quantify their baseline bias among four programming languages. A static neuron-attribution method, perturbing the highest activated MLP weight for a C++ or CPP token, proved brittle and exhibited limited generalization across prompt styles and model scales. To address these limitations, a gradient-refined adaptive activation steering framework (G-ACT) was developed: per-prompt activation differences are clustered into a small set of steering directions, and lightweight per-layer probes are trained and refined online to select the appropriate steering vector. In LLaMA-3.2 3B, this approach reliably biases generation towards the CPP language by increasing the average probe classification accuracy by 15% and the early layers (0-6) improving the probe classification accuracy by 61.5% compared to the standard ACT framework. For LLaMA-3.3 70B, where attention-head signals become more diffuse, targeted injections at key layers still improve language selection. Although per-layer probing introduces a modest inference overhead, it remains practical by steering only a subset of layers and enables reproducible model behavior. These results demonstrate a scalable, interpretable and efficient mechanism for concept-level control for practical agentic systems. 

**Abstract (ZH)**: 本研究探讨激活语言模型（LLMs）中的潜在子空间是否可以引导科学研究代码生成向特定编程语言发展。首先评估了五种因果LLMs在科学编程提示上的表现，以量化它们在四种编程语言中的基线偏差。静态神经元归因方法通过扰动C++或CPP标记激活的最高MLP权重，证明了其脆弱性，并且在提示风格和模型规模方面的泛化能力有限。为了解决这些问题，开发了一种梯度细化自适应激活引导框架（G-ACT）：将每种提示的激活差异聚类成一组引导方向，并在线训练和细化轻量级的逐层探针以选择合适的引导向量。在LLaMA-3.2 3B中，通过将探针分类准确率平均提高15%以及早期层（0-6）提高61.5%，该方法可靠地偏置了生成方向偏向CPP语言。对于LLaMA-3.3 70B，其中注意力头信号变得更加弥散，对关键层的针对性注入仍然可以改善语言选择。尽管逐层探针引入了轻微的推理开销，但在仅引导部分层的情况下仍然可行，并且能够使模型行为可重复。这些结果展示了可扩展、可解释和高效的机制，用于现实世界代理系统的概念级控制。 

---
# TRIZ Agents: A Multi-Agent LLM Approach for TRIZ-Based Innovation 

**Title (ZH)**: TRIZ智能体：一种基于TRIZ创新的多智能体大语言模型方法 

**Authors**: Kamil Szczepanik, Jarosław A. Chudziak  

**Link**: [PDF](https://arxiv.org/pdf/2506.18783)  

**Abstract**: TRIZ, the Theory of Inventive Problem Solving, is a structured, knowledge-based framework for innovation and abstracting problems to find inventive solutions. However, its application is often limited by the complexity and deep interdisciplinary knowledge required. Advancements in Large Language Models (LLMs) have revealed new possibilities for automating parts of this process. While previous studies have explored single LLMs in TRIZ applications, this paper introduces a multi-agent approach. We propose an LLM-based multi-agent system, called TRIZ agents, each with specialized capabilities and tool access, collaboratively solving inventive problems based on the TRIZ methodology. This multi-agent system leverages agents with various domain expertise to efficiently navigate TRIZ steps. The aim is to model and simulate an inventive process with language agents. We assess the effectiveness of this team of agents in addressing complex innovation challenges based on a selected case study in engineering. We demonstrate the potential of agent collaboration to produce diverse, inventive solutions. This research contributes to the future of AI-driven innovation, showcasing the advantages of decentralized problem-solving in complex ideation tasks. 

**Abstract (ZH)**: TRIZ基多代理系统：基于大型语言模型的发明问题解决方法 

---
# Programming by Backprop: LLMs Acquire Reusable Algorithmic Abstractions During Code Training 

**Title (ZH)**: 基于反向传播的程序构建：大规模语言模型在代码训练过程中习得可重用的算法抽象 

**Authors**: Jonathan Cook, Silvia Sapora, Arash Ahmadian, Akbir Khan, Tim Rocktaschel, Jakob Foerster, Laura Ruis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18777)  

**Abstract**: Training large language models (LLMs) on source code significantly enhances their general-purpose reasoning abilities, but the mechanisms underlying this generalisation are poorly understood. In this paper, we propose Programming by Backprop (PBB) as a potential driver of this effect - teaching a model to evaluate a program for inputs by training on its source code alone, without ever seeing I/O examples. To explore this idea, we finetune LLMs on two sets of programs representing simple maths problems and algorithms: one with source code and I/O examples (w/ IO), the other with source code only (w/o IO). We find evidence that LLMs have some ability to evaluate w/o IO programs for inputs in a range of experimental settings, and make several observations. Firstly, PBB works significantly better when programs are provided as code rather than semantically equivalent language descriptions. Secondly, LLMs can produce outputs for w/o IO programs directly, by implicitly evaluating the program within the forward pass, and more reliably when stepping through the program in-context via chain-of-thought. We further show that PBB leads to more robust evaluation of programs across inputs than training on I/O pairs drawn from a distribution that mirrors naturally occurring data. Our findings suggest a mechanism for enhanced reasoning through code training: it allows LLMs to internalise reusable algorithmic abstractions. Significant scope remains for future work to enable LLMs to more effectively learn from symbolic procedures, and progress in this direction opens other avenues like model alignment by training on formal constitutional principles. 

**Abstract (ZH)**: 训练大规模语言模型（LLMs）在源代码上显著增强了其通用推理能力，但其背后的机制尚不完全清楚。在这篇文章中，我们提出了通过反向传播进行编程（PBB）作为这种效果的潜在驱动因素——通过仅使用源代码训练模型来评估程序，而不曾见过输入/输出示例。为了探索这一想法，我们在两类程序上微调LLMs：一类包括源代码和输入/输出示例（带IO），另一类仅包括源代码（不带IO），代表简单的数学问题和算法。我们发现证据表明LLMs在一系列实验设置中具有评估不带IO程序的能力，并作出了几个观察。首先，当程序以代码形式提供时，PBB的效果显著优于以语义等价的语言描述提供。其次，LLMs可以直接通过隐式在前向传播过程中评估程序来生成不带IO程序的输出，并且通过逐步推理的方式，在上下文中的确更可靠。我们还展示出，与来自具有自然分布的数据的输入/输出配对训练相比，PBB导致在不同输入下的程序评估更具稳健性。我们的发现表明了一种通过代码训练增强推理的机制：它允许LLMs内化可重用的算法抽象。未来工作的显著空间在于使LLMs更有效地从符号程序中学习，并且在此方向上的进展打开了其他途径，如通过在形式宪法原则上训练来进行模型对齐。 

---
# AggTruth: Contextual Hallucination Detection using Aggregated Attention Scores in LLMs 

**Title (ZH)**: AggTruth: 使用聚合注意力得分检测上下文错觉在大语言模型中的应用 

**Authors**: Piotr Matys, Jan Eliasz, Konrad Kiełczyński, Mikołaj Langner, Teddy Ferdinan, Jan Kocoń, Przemysław Kazienko  

**Link**: [PDF](https://arxiv.org/pdf/2506.18628)  

**Abstract**: In real-world applications, Large Language Models (LLMs) often hallucinate, even in Retrieval-Augmented Generation (RAG) settings, which poses a significant challenge to their deployment. In this paper, we introduce AggTruth, a method for online detection of contextual hallucinations by analyzing the distribution of internal attention scores in the provided context (passage). Specifically, we propose four different variants of the method, each varying in the aggregation technique used to calculate attention scores. Across all LLMs examined, AggTruth demonstrated stable performance in both same-task and cross-task setups, outperforming the current SOTA in multiple scenarios. Furthermore, we conducted an in-depth analysis of feature selection techniques and examined how the number of selected attention heads impacts detection performance, demonstrating that careful selection of heads is essential to achieve optimal results. 

**Abstract (ZH)**: 在实际应用中，大型语言模型（LLMs）经常产生幻觉，即使在检索增强生成（RAG）设置中也是如此，这对其部署构成了重大挑战。在本文中，我们介绍了AggTruth方法，该方法通过分析提供上下文（段落）中的内部注意力分数分布来在线检测上下文幻觉。具体地，我们提出了四种不同版本的方法，每种方法使用的聚合技术均有所不同。在所有检查的LLM中，AggTruth在同任务和跨任务设置中均表现出稳定的性能，并在多个场景中优于当前SOTA。此外，我们深入分析了特征选择技术，并研究了选择的注意力头数量对检测性能的影响，表明谨慎选择注意力头对于实现最佳结果至关重要。 

---
# How Robust is Model Editing after Fine-Tuning? An Empirical Study on Text-to-Image Diffusion Models 

**Title (ZH)**: 微调后模型编辑的鲁棒性：Text-to-Image扩散模型的实证研究 

**Authors**: Feng He, Zhenyang Liu, Marco Valentino, Zhixue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18428)  

**Abstract**: Model editing offers a low-cost technique to inject or correct a particular behavior in a pre-trained model without extensive retraining, supporting applications such as factual correction and bias mitigation. Despite this common practice, it remains unknown whether edits persist after fine-tuning or whether they are inadvertently reversed. This question has fundamental practical implications. For example, if fine-tuning removes prior edits, it could serve as a defence mechanism against hidden malicious edits. Vice versa, the unintended removal of edits related to bias mitigation could pose serious safety concerns. We systematically investigate the interaction between model editing and fine-tuning in the context of T2I diffusion models, which are known to exhibit biases and generate inappropriate content. Our study spans two T2I model families (Stable Diffusion and FLUX), two sota editing techniques, and three fine-tuning methods (DreamBooth, LoRA, and DoRA). Through an extensive empirical analysis across diverse editing tasks and evaluation metrics, our findings reveal a trend: edits generally fail to persist through fine-tuning, even when fine-tuning is tangential or unrelated to the edits. Notably, we observe that DoRA exhibits the strongest edit reversal effect. At the same time, among editing methods, UCE demonstrates greater robustness, retaining significantly higher efficacy post-fine-tuning compared to ReFACT. These findings highlight a crucial limitation in current editing methodologies, emphasizing the need for more robust techniques to ensure reliable long-term control and alignment of deployed AI systems. These findings have dual implications for AI safety: they suggest that fine-tuning could serve as a remediation mechanism for malicious edits while simultaneously highlighting the need for re-editing after fine-tuning to maintain beneficial safety and alignment properties. 

**Abstract (ZH)**: 模型编辑提供了一种低成本技术，可以在预训练模型中注入或纠正特定行为，而不需要广泛的重新训练，支持例如事实纠正和偏见缓解的应用。尽管这一做法非常常见，但仍不清楚编辑在微调后是否会持续存在，或者是否会被无意中逆转。这个问题具有根本性的实践意义。例如，如果微调消除了先前的编辑，则可以作为一种防御机制，抵御隐藏的恶意编辑。反之，微调无意中消除了与偏见缓解相关的编辑，则可能会带来严重的安全风险。我们系统地研究了T2I扩散模型中模型编辑与微调之间的相互作用，这些模型已知存在偏见并生成不适当的内容。我们的研究涵盖了两种T2I模型家族（Stable Diffusion和FLUX）、两种最先进的编辑技术以及三种微调方法（DreamBooth、LoRA和DoRA）。通过在多种编辑任务和评估指标上的广泛实验分析，我们的发现显示了一种趋势：即使在微调与编辑相关性较弱或无关情况下，编辑通常也难以在微调后持久存在。值得注意的是，我们观察到DoRA表现出最强烈的编辑逆转效果。同时，在编辑方法中，UCE显示出较高的鲁棒性，微调后的效果显著优于ReFACT。这些发现揭示了当前编辑方法的一个关键局限性，强调了需要更稳健的技术，以确保已部署AI系统的可靠长期控制和对齐。这些发现对AI安全具有双重含义：它们表明微调可以作为一种修复机制，以应对恶意编辑，同时强调了在微调后重新编辑的必要性，以保持有益的安全和对齐特性。 

---
# A Large Language Model-based Multi-Agent Framework for Analog Circuits' Sizing Relationships Extraction 

**Title (ZH)**: 基于大型语言模型的多代理框架用于模拟电路尺寸关系提取 

**Authors**: Chengjie Liu, Weiyu Chen, Huiyao Xu, Yuan Du, Jun Yang, Li Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.18424)  

**Abstract**: In the design process of the analog circuit pre-layout phase, device sizing is an important step in determining whether an analog circuit can meet the required performance metrics. Many existing techniques extract the circuit sizing task as a mathematical optimization problem to solve and continuously improve the optimization efficiency from a mathematical perspective. But they ignore the automatic introduction of prior knowledge, fail to achieve effective pruning of the search space, which thereby leads to a considerable compression margin remaining in the search space. To alleviate this problem, we propose a large language model (LLM)-based multi-agent framework for analog circuits' sizing relationships extraction from academic papers. The search space in the sizing process can be effectively pruned based on the sizing relationship extracted by this framework. Eventually, we conducted tests on 3 types of circuits, and the optimization efficiency was improved by $2.32 \sim 26.6 \times$. This work demonstrates that the LLM can effectively prune the search space for analog circuit sizing, providing a new solution for the combination of LLMs and conventional analog circuit design automation methods. 

**Abstract (ZH)**: 基于大语言模型的多agent框架在模拟电路尺寸关系提取中的应用：有效 prune 布局前阶段尺寸优化搜索空间并提升优化效率 

---
# Dynamic Knowledge Exchange and Dual-diversity Review: Concisely Unleashing the Potential of a Multi-Agent Research Team 

**Title (ZH)**: 动态知识交流与双多样性审查：简明释放多剂型研究团队的潜力 

**Authors**: Weilun Yu, Shixiang Tang, Yonggui Huang, Nanqing Dong, Li Fan, Honggang Qi, Wei Liu, Xiaoli Diao, Xi Chen, Wanli Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18348)  

**Abstract**: Scientific progress increasingly relies on effective collaboration among researchers, a dynamic that large language models (LLMs) have only begun to emulate. While recent LLM-based scientist agents show promise in autonomous scientific discovery, they often lack the interactive reasoning and evaluation mechanisms essential to real-world research. We propose IDVSCI (Internal Discussion and Vote SCIentists), a multi-agent framework built on LLMs that incorporates two key innovations: a Dynamic Knowledge Exchange mechanism enabling iterative feedback among agents, and a Dual-Diversity Review paradigm that simulates heterogeneous expert evaluation. These components jointly promote deeper reasoning and the generation of more creative and impactful scientific ideas. To evaluate the effectiveness and generalizability of our approach, we conduct experiments on two datasets: a widely used benchmark in computer science and a new dataset we introduce in the health sciences domain. Results show that IDVSCI consistently achieves the best performance across both datasets, outperforming existing systems such as AI Scientist and VIRSCI. These findings highlight the value of modeling interaction and peer review dynamics in LLM-based autonomous research. 

**Abstract (ZH)**: 基于大语言模型的内部讨论与投票科学家：促进深度推理和创新科学理念生成的多agent框架 

---
# Advanced For-Loop for QML algorithm search 

**Title (ZH)**: Advanced For-Loop for QML Algorithm Search 

**Authors**: FuTe Wong  

**Link**: [PDF](https://arxiv.org/pdf/2506.18260)  

**Abstract**: This paper introduces an advanced framework leveraging Large Language Model-based Multi-Agent Systems (LLMMA) for the automated search and optimization of Quantum Machine Learning (QML) algorithms. Inspired by Google DeepMind's FunSearch, the proposed system works on abstract level to iteratively generates and refines quantum transformations of classical machine learning algorithms (concepts), such as the Multi-Layer Perceptron, forward-forward and backpropagation algorithms. As a proof of concept, this work highlights the potential of agentic frameworks to systematically explore classical machine learning concepts and adapt them for quantum computing, paving the way for efficient and automated development of QML algorithms. Future directions include incorporating planning mechanisms and optimizing strategy in the search space for broader applications in quantum-enhanced machine learning. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统（LLMMA）的量子机器学习算法自动搜索与优化框架：从经典机器学习概念出发探索量子计算潜力 

---
# The 4th Dimension for Scaling Model Size 

**Title (ZH)**: 第四维：扩展模型规模的新维度 

**Authors**: Ruike Zhu, Hanwen Zhang, Tianyu Shi, Chi Wang, Tianyi Zhou, Zengyi Qin  

**Link**: [PDF](https://arxiv.org/pdf/2506.18233)  

**Abstract**: Scaling the size of large language models typically involves three dimensions: depth, width, and the number of parameters. In this work, we explore a fourth dimension, virtual logical depth (VLD), which increases the effective algorithmic depth without changing the overall parameter count by reusing parameters within the model. Although parameter reuse is not a new concept, its potential and characteristics in model scaling have not been thoroughly studied. Through carefully designed controlled experiments, we make the following key discoveries regarding VLD scaling:
VLD scaling forces the knowledge capacity of the model to remain almost constant, with only minor variations.
VLD scaling enables a significant improvement in reasoning capability, provided the scaling method is properly implemented.
The number of parameters correlates with knowledge capacity, but not with reasoning capability. Under certain conditions, it is not necessary to increase the parameter count to enhance reasoning.
These findings are consistent across various model configurations and are likely to be generally valid within the scope of our experiments. 

**Abstract (ZH)**: 增大大型语言模型的规模通常涉及三个维度：深度、宽度和参数量。在本工作中，我们探索了一个新的维度，即虚拟逻辑深度（VLD），通过在模型内部重用参数，增加有效的算法深度而不会改变整体的参数数量。尽管参数重用不是一个新的概念，但其在模型规模扩展中的潜力和特性尚未进行全面研究。通过精心设计的受控实验，我们对于VLD规模扩展做出了以下关键发现：
VLD规模扩展迫使模型的知识容量几乎保持不变，仅有轻微的变化。
当规模扩展方法正确实施时，VLD规模扩展能够显著提高推理能力。
参数量与知识容量相关，但与推理能力无关。在某些条件下，为了增强推理能力，并不需要增加参数数量。
上述发现适用于各种模型配置，并且在我们实验的范围内很可能具有普遍适用性。 

---
# AI Through the Human Lens: Investigating Cognitive Theories in Machine Psychology 

**Title (ZH)**: AI 透过人类视角：探究机器心理学中的认知理论 

**Authors**: Akash Kundu, Rishika Goswami  

**Link**: [PDF](https://arxiv.org/pdf/2506.18156)  

**Abstract**: We investigate whether Large Language Models (LLMs) exhibit human-like cognitive patterns under four established frameworks from psychology: Thematic Apperception Test (TAT), Framing Bias, Moral Foundations Theory (MFT), and Cognitive Dissonance. We evaluated several proprietary and open-source models using structured prompts and automated scoring. Our findings reveal that these models often produce coherent narratives, show susceptibility to positive framing, exhibit moral judgments aligned with Liberty/Oppression concerns, and demonstrate self-contradictions tempered by extensive rationalization. Such behaviors mirror human cognitive tendencies yet are shaped by their training data and alignment methods. We discuss the implications for AI transparency, ethical deployment, and future work that bridges cognitive psychology and AI safety 

**Abstract (ZH)**: 我们通过心理学中建立的四种框架（主题投射测试TAT、框构偏差、道德基础理论MFT和认知失调）探究大型语言模型（LLMs）是否表现出人类似的心智模式。我们使用结构化提示和自动评分评估了几种 proprietary 和开源模型。研究发现，这些模型经常生成连贯的故事线，容易受到正面框构的影响，展现出与自由/压迫关切相一致的道德判断，并表现出通过广泛理性化来减轻自我矛盾的行为。这些行为反映出人类的心智倾向，但同时也受到其训练数据和对齐方法的影响。我们讨论了这些发现对人工智能透明度、伦理部署以及认知心理学与人工智能安全交叉领域未来工作的含义。 

---
# CoachGPT: A Scaffolding-based Academic Writing Assistant 

**Title (ZH)**: CoachGPT：基于支架式的学术写作辅助工具 

**Authors**: Fumian Chen, Sotheara Veng, Joshua Wilson, Xiaoming Li, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18149)  

**Abstract**: Academic writing skills are crucial for students' success, but can feel overwhelming without proper guidance and practice, particularly when writing in a second language. Traditionally, students ask instructors or search dictionaries, which are not universally accessible. Early writing assistants emerged as rule-based systems that focused on detecting misspellings, subject-verb disagreements, and basic punctuation errors; however, they are inaccurate and lack contextual understanding. Machine learning-based assistants demonstrate a strong ability for language understanding but are expensive to train. Large language models (LLMs) have shown remarkable capabilities in generating responses in natural languages based on given prompts. Still, they have a fundamental limitation in education: they generate essays without teaching, which can have detrimental effects on learning when misused. To address this limitation, we develop CoachGPT, which leverages large language models (LLMs) to assist individuals with limited educational resources and those who prefer self-paced learning in academic writing. CoachGPT is an AI agent-based web application that (1) takes instructions from experienced educators, (2) converts instructions into sub-tasks, and (3) provides real-time feedback and suggestions using large language models. This unique scaffolding structure makes CoachGPT unique among existing writing assistants. Compared to existing writing assistants, CoachGPT provides a more immersive writing experience with personalized feedback and guidance. Our user studies prove the usefulness of CoachGPT and the potential of large language models for academic writing. 

**Abstract (ZH)**: 学术写作技能对于学生的成功至关重要，但缺乏恰当的指导和练习时会感觉令人望而却步，尤其是在使用第二语言写作时。传统上，学生会向教师请教或查阅字典，但这并非普遍可行。早期的写作助手是基于规则的系统，主要侧重于检测拼写错误、主谓一致问题和基本标点错误；然而，这些系统不够准确且缺乏上下文理解。基于机器学习的助手在语言理解方面表现出色，但它们的训练成本高昂。大规模语言模型（LLMs）展示了根据给定提示生成自然语言响应的非凡能力。然而，在教育方面，它们存在根本局限：它们生成论文而不进行教学，这在不当使用时会损害学习效果。为解决这一局限，我们开发了CoachGPT，利用大规模语言模型（LLMs）辅助资源有限的个人及偏好自主学习的学生进行学术写作。CoachGPT 是基于AI代理的网络应用程序，(1) 从经验丰富的教育者那里获取指令，(2) 将指令转换为子任务，并(3) 使用大规模语言模型提供实时反馈和建议。这种独特的支架结构使CoachGPT 在现有的写作助手中独具特色。与现有的写作助手相比，CoachGPT 提供了更加沉浸式的写作体验，并提供个性化反馈和指导。我们的用户研究证明了CoachGPT 的实用性以及大规模语言模型在学术写作中的潜力。 

---
# Leveraging Large Language Model for Intelligent Log Processing and Autonomous Debugging in Cloud AI Platforms 

**Title (ZH)**: 利用大规模语言模型进行云AI平台中的智能日志处理与自主调试 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17900)  

**Abstract**: With the increasing complexity and rapid expansion of the scale of AI systems in cloud platforms, the log data generated during system operation is massive, unstructured, and semantically ambiguous, which brings great challenges to fault location and system self-repair. In order to solve this problem, this paper proposes an intelligent log processing and automatic debugging framework based on Large Language Model (LLM), named Intelligent Debugger (LLM-ID). This method is extended on the basis of the existing pre-trained Transformer model, and integrates a multi-stage semantic inference mechanism to realize the context understanding of system logs and the automatic reconstruction of fault chains. Firstly, the system log is dynamically structured, and the unsupervised clustering and embedding mechanism is used to extract the event template and semantic schema. Subsequently, the fine-tuned LLM combined with the multi-round attention mechanism to perform contextual reasoning on the log sequence to generate potential fault assumptions and root cause paths. Furthermore, this paper introduces a reinforcement learning-based policy-guided recovery planner, which is driven by the remediation strategy generated by LLM to support dynamic decision-making and adaptive debugging in the cloud environment. Compared with the existing rule engine or traditional log analysis system, the proposed model has stronger semantic understanding ability, continuous learning ability and heterogeneous environment adaptability. Experiments on the cloud platform log dataset show that LLM-ID improves the fault location accuracy by 16.2%, which is significantly better than the current mainstream methods 

**Abstract (ZH)**: 基于大型语言模型的智能日志处理与自动调试框架：LLM-ID 

---
# Towards Robust Fact-Checking: A Multi-Agent System with Advanced Evidence Retrieval 

**Title (ZH)**: 面向鲁棒事实核查：一种先进的证据检索多Agent系统 

**Authors**: Tam Trinh, Manh Nguyen, Truong-Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2506.17878)  

**Abstract**: The rapid spread of misinformation in the digital era poses significant challenges to public discourse, necessitating robust and scalable fact-checking solutions. Traditional human-led fact-checking methods, while credible, struggle with the volume and velocity of online content, prompting the integration of automated systems powered by Large Language Models (LLMs). However, existing automated approaches often face limitations, such as handling complex claims, ensuring source credibility, and maintaining transparency. This paper proposes a novel multi-agent system for automated fact-checking that enhances accuracy, efficiency, and explainability. The system comprises four specialized agents: an Input Ingestion Agent for claim decomposition, a Query Generation Agent for formulating targeted subqueries, an Evidence Retrieval Agent for sourcing credible evidence, and a Verdict Prediction Agent for synthesizing veracity judgments with human-interpretable explanations. Evaluated on benchmark datasets (FEVEROUS, HOVER, SciFact), the proposed system achieves a 12.3% improvement in Macro F1-score over baseline methods. The system effectively decomposes complex claims, retrieves reliable evidence from trusted sources, and generates transparent explanations for verification decisions. Our approach contributes to the growing field of automated fact-checking by providing a more accurate, efficient, and transparent verification methodology that aligns with human fact-checking practices while maintaining scalability for real-world applications. Our source code is available at this https URL 

**Abstract (ZH)**: 数字时代错误信息的快速传播对公共话语构成了重大挑战，需要 robust 和可扩展的事实核查解决方案。传统的以人类为主导的事实核查方法虽然可靠，但在处理大量和快速变化的在线内容方面存在困难，因此需要结合基于大型语言模型（LLMs）的自动化系统。然而，现有的自动化方法常常面临处理复杂声明、保证来源可信度和保持透明度等方面的限制。本文提出了一种新型多智能体系统，以提高事实核查的准确性、效率和可解释性。该系统包括四个专门化的智能体：输入摄取智能体负责声明分解、查询生成智能体负责形成针对性的子查询、证据检索智能体负责获取可靠的证据，以及判决预测智能体负责综合真实性的判断并生成可由人类理解的解释。该系统在基准数据集（FEVEROUS、HOVER、SciFact）上的评价中，相对于基线方法取得了12.3%的宏F1分数改善。该系统能够有效分解复杂声明、从可信来源检索可靠证据，并为验证决策生成透明的解释。通过提供一种更准确、更高效、更透明的验证方法，我们的方法在自动化事实核查领域取得了进展，该方法与人类事实核查实践相一致，同时保持了对实际应用的可扩展性。源代码可在以下网址获取。 

---
# Out of Control -- Why Alignment Needs Formal Control Theory (and an Alignment Control Stack) 

**Title (ZH)**: 失控——为什么对齐需要形式化的控制理论（以及一个对齐控制栈） 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2506.17846)  

**Abstract**: This position paper argues that formal optimal control theory should be central to AI alignment research, offering a distinct perspective from prevailing AI safety and security approaches. While recent work in AI safety and mechanistic interpretability has advanced formal methods for alignment, they often fall short of the generalisation required of control frameworks for other technologies. There is also a lack of research into how to render different alignment/control protocols interoperable. We argue that by recasting alignment through principles of formal optimal control and framing alignment in terms of hierarchical stack from physical to socio-technical layers according to which controls may be applied we can develop a better understanding of the potential and limitations for controlling frontier models and agentic AI systems. To this end, we introduce an Alignment Control Stack which sets out a hierarchical layered alignment stack, identifying measurement and control characteristics at each layer and how different layers are formally interoperable. We argue that such analysis is also key to the assurances that will be needed by governments and regulators in order to see AI technologies sustainably benefit the community. Our position is that doing so will bridge the well-established and empirically validated methods of optimal control with practical deployment considerations to create a more comprehensive alignment framework, enhancing how we approach safety and reliability for advanced AI systems. 

**Abstract (ZH)**: 形式化最优控制理论在AI对齐研究中的核心地位：从控制框架视角探讨AI安全与 interoperability 的新途径 

---
# Bayesian Social Deduction with Graph-Informed Language Models 

**Title (ZH)**: 基于图 informant 语言模型的贝叶斯社会推理 

**Authors**: Shahab Rahimirad, Guven Gergerli, Lucia Romero, Angela Qian, Matthew Lyle Olson, Simon Stepputtis, Joseph Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2506.17788)  

**Abstract**: Social reasoning - inferring unobservable beliefs and intentions from partial observations of other agents - remains a challenging task for large language models (LLMs). We evaluate the limits of current reasoning language models in the social deduction game Avalon and find that while the largest models demonstrate strong performance, they require extensive test-time inference and degrade sharply when distilled to smaller, real-time-capable variants. To address this, we introduce a hybrid reasoning framework that externalizes belief inference to a structured probabilistic model, while using an LLM for language understanding and interaction. Our approach achieves competitive performance with much larger models in Agent-Agent play and, notably, is the first language agent to defeat human players in a controlled study - achieving a 67% win rate and receiving higher qualitative ratings than both reasoning baselines and human teammates. We release code, models, and a dataset to support future work on social reasoning in LLM agents, which can be found at this https URL 

**Abstract (ZH)**: 社会推理——从部分观察到的其他代理的信念和意图中推断不可观测的信念和意图——仍然是大型语言模型（LLMs）面临的一项具有挑战性的任务。我们评估了当前推理语言模型在社会推理游戏Avalon中的限制，并发现虽然最大的模型表现出色，但在被精简为更小、支持实时处理的变体时，推理性能会急剧下降。为了解决这个问题，我们引入了一种混合推理框架，该框架将信念推断外部化到结构化的概率模型中，同时使用LLM进行语言理解和交互。我们的方法在代理-代理对战中实现了与更大模型相当的性能，并且值得注意的是，这是我们第一次在控制研究中让语言代理击败人类玩家——胜率为67%，并且在定性评分方面高于两种推理基线和人类队友。我们发布了支持未来LLM代理社会推理研究的代码、模型和数据集，可在以下链接获取：this https URL。 

---
# AnyMAC: Cascading Flexible Multi-Agent Collaboration via Next-Agent Prediction 

**Title (ZH)**: AnyMAC：基于下一代理预测的级联灵活多代理协作 

**Authors**: Song Wang, Zhen Tan, Zihan Chen, Shuang Zhou, Tianlong Chen, Jundong Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17784)  

**Abstract**: Recent progress in large language model (LLM)-based multi-agent collaboration highlights the power of structured communication in enabling collective intelligence. However, existing methods largely rely on static or graph-based inter-agent topologies, lacking the potential adaptability and flexibility in communication. In this work, we propose a new framework that rethinks multi-agent coordination through a sequential structure rather than a graph structure, offering a significantly larger topology space for multi-agent communication. Our method focuses on two key directions: (1) Next-Agent Prediction, which selects the most suitable agent role at each step, and (2) Next-Context Selection (NCS), which enables each agent to selectively access relevant information from any previous step. Together, these components construct task-adaptive communication pipelines that support both role flexibility and global information flow. Extensive evaluations across multiple benchmarks demonstrate that our approach achieves superior performance while substantially reducing communication overhead. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的多智能体协作 recent progress 强调结构化通信在促进集体智能方面的能力。然而，现有方法主要依赖静态或图结构的智能体间拓扑，缺乏在通信中适应性和灵活性的潜力。在本工作中，我们提出了一种新的框架，通过顺序结构而不是图结构重新思考多智能体协调，为多智能体通信提供了更大的拓扑空间。我们的方法集中在两个关键方向：（1）下一智能体预测，即在每一步选择最适合的智能体角色，以及（2）下一上下文选择（NCS），使每个智能体能够有选择地访问任一步骤的相关信息。这些组件共同构建了支持角色灵活性和全局信息流动的任务自适应通信管道。在多个基准上的广泛评估表明，我们的方法在显著减少通信开销的同时实现了卓越的性能。 

---
# Beyond Syntax: Action Semantics Learning for App Agents 

**Title (ZH)**: 超越句法：应用代理的动作语义学习 

**Authors**: Bohan Tang, Dezhao Luo, Jingxuan Chen, Shaogang Gong, Jianye Hao, Jun Wang, Kun Shao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17697)  

**Abstract**: The advent of Large Language Models (LLMs) enables the rise of App agents that interpret user intent and operate smartphone Apps through actions such as clicking and scrolling. While prompt-based solutions with closed LLM APIs show promising ability, they incur heavy compute costs and external API dependency. Fine-tuning smaller open-source LLMs solves these limitations. However, current fine-tuning methods use a syntax learning paradigm that forces agents to reproduce exactly the ground truth action strings, leading to out-of-distribution (OOD) vulnerability. To fill this gap, we propose Action Semantics Learning (ASL), a novel learning framework, where the learning objective is capturing the semantics of the ground truth actions. Specifically, inspired by the programming language theory, we define the action semantics for App agents as the state transition induced by the action in the user interface. With this insight, ASL employs a novel SEmantic Estimator (SEE) to compute a semantic reward to train the App agents in generating actions aligned with the semantics of ground truth actions, even when the syntactic forms differ. To support the effectiveness of ASL, we theoretically demonstrate the superior robustness of ASL for the OOD problem compared with the existing syntax learning paradigm. Extensive experiments on offline and online smartphone App operation benchmarks show that ASL significantly improves the accuracy and generalisation of App agents over existing methods. 

**Abstract (ZH)**: 大型语言模型的出现使应用代理得以兴起，这些代理通过点击和滚动等操作来解释用户意图并操作智能手机应用。虽然基于提示的解决方案展示了令人信服的能力，但它们产生了巨大的计算成本并依赖外部API。微调较小的开源语言模型解决了这些问题。然而，当前的微调方法使用了语法学习范式，要求代理完全复制地面真实动作字符串，导致了分布外（OOD）的脆弱性。为了解决这一问题，我们提出了动作语义学习（Action Semantics Learning，ASL），这是一种新的学习框架，其学习目标是捕获地面真实动作的语义。具体来说，受到编程语言理论的启发，我们将应用代理的动作语义定义为由动作在用户界面中引发的状态转换。基于这一洞察，ASL 使用了一个新颖的语义估计器（Semantic Estimator，SEE）来计算语义奖励，以训练应用代理生成与地面真实动作语义相匹配的动作，即使两者在语法形式上有所不同。为了支持ASL的有效性，我们从理论上证明了与现有语法学习范式相比，ASL 在分布外问题上具有更强的鲁棒性。在离线和在线智能手机应用操作基准测试中的广泛实验显示，ASL 显著提升了应用代理比现有方法的准确性和泛化能力。 

---
# Measuring and Augmenting Large Language Models for Solving Capture-the-Flag Challenges 

**Title (ZH)**: 测量与增强大型语言模型以解决Capture-the-Flag挑战 

**Authors**: Zimo Ji, Daoyuan Wu, Wenyuan Jiang, Pingchuan Ma, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17644)  

**Abstract**: Capture-the-Flag (CTF) competitions are crucial for cybersecurity education and training. As large language models (LLMs) evolve, there is increasing interest in their ability to automate CTF challenge solving. For example, DARPA has organized the AIxCC competition since 2023 to advance AI-powered automated offense and defense. However, this demands a combination of multiple abilities, from knowledge to reasoning and further to actions. In this paper, we highlight the importance of technical knowledge in solving CTF problems and deliberately construct a focused benchmark, CTFKnow, with 3,992 questions to measure LLMs' performance in this core aspect. Our study offers a focused and innovative measurement of LLMs' capability in understanding CTF knowledge and applying it to solve CTF challenges. Our key findings reveal that while LLMs possess substantial technical knowledge, they falter in accurately applying this knowledge to specific scenarios and adapting their strategies based on feedback from the CTF environment.
Based on insights derived from this measurement study, we propose CTFAgent, a novel LLM-driven framework for advancing CTF problem-solving. CTFAgent introduces two new modules: two-stage Retrieval Augmented Generation (RAG) and interactive Environmental Augmentation, which enhance LLMs' technical knowledge and vulnerability exploitation on CTF, respectively. Our experimental results show that, on two popular CTF datasets, CTFAgent both achieves over 80% performance improvement. Moreover, in the recent picoCTF2024 hosted by CMU, CTFAgent ranked in the top 23.6% of nearly 7,000 participating teams. This reflects the benefit of our measurement study and the potential of our framework in advancing LLMs' capabilities in CTF problem-solving. 

**Abstract (ZH)**: 大型语言模型在Capture-the-Flag (CTF) 挑战自动化中的关键作用：CTFKnow基准与CTFAgent框架研究 

---
# Taming the Untamed: Graph-Based Knowledge Retrieval and Reasoning for MLLMs to Conquer the Unknown 

**Title (ZH)**: 驯服未知：基于图的知识检索与推理以使大模型克服未知领域 

**Authors**: Bowen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17589)  

**Abstract**: The real value of knowledge lies not just in its accumulation, but in its potential to be harnessed effectively to conquer the unknown. Although recent multimodal large language models (MLLMs) exhibit impressing multimodal capabilities, they often fail in rarely encountered domain-specific tasks due to limited relevant knowledge. To explore this, we adopt visual game cognition as a testbed and select Monster Hunter: World as the target to construct a multimodal knowledge graph (MH-MMKG), which incorporates multi-modalities and intricate entity relations. We also design a series of challenging queries based on MH-MMKG to evaluate the models' ability for complex knowledge retrieval and reasoning. Furthermore, we propose a multi-agent retriever that enables a model to autonomously search relevant knowledge without additional training. Experimental results show that our approach significantly enhances the performance of MLLMs, providing a new perspective on multimodal knowledge-augmented reasoning and laying a solid foundation for future research. 

**Abstract (ZH)**: 知识的实际价值不仅在于积累，更在于有效利用以攻克未知。尽管近期的多模态大型语言模型（MLLMs）展现出令人印象深刻的多模态能力，但在罕见遭遇的领域特定任务中往往由于缺乏相关知识而失败。为此，我们以视觉游戏认知作为测试平台，选择《怪物猎人：世界》为目标，构建了一个多模态知识图谱（MH-MMKG），该图谱融合了多模态信息和复杂的实体关系。我们还基于MH-MMKG设计了一系列复杂的查询，以评估模型进行复杂知识检索和推理的能力。此外，我们提出了一种多智能体检索器，使模型能够自主搜索相关知识而无需额外训练。实验结果表明，我们的方法显著提升了MLLMs的表现，为多模态知识增强推理提供了新的视角，并为未来研究奠定了坚实基础。 

---
# Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models 

**Title (ZH)**: Cite Pretrain: 无需检索的知识归属对于大型语言模型 

**Authors**: Yukun Huang, Sanxing Chen, Jian Pei, Manzil Zaheer, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2506.17585)  

**Abstract**: Trustworthy language models should provide both correct and verifiable answers. While language models can sometimes attribute their outputs to pretraining data, their citations are often unreliable due to hallucination. As a result, current systems insert citations by querying an external retriever at inference time, introducing latency, infrastructure dependence, and vulnerability to retrieval noise. We explore whether LLMs can be made to reliably attribute to the documents seen during (continual) pretraining--without test-time retrieval--by revising the training process. To evaluate this, we release CitePretrainBench, a benchmark that mixes real-world corpora (Wikipedia, Common Crawl, arXiv) with novel, unseen documents and probes both short-form (single fact) and long-form (multi-fact) citation tasks. Our approach follows a two-stage process: (1) continual pretraining to bind facts to persistent document identifiers, and (2) instruction tuning to elicit citation behavior. We find that simple Passive Indexing, which appends an identifier to each document, helps memorize verbatim text but fails on paraphrased or compositional facts. Instead, we propose Active Indexing, which continually pretrains on synthetic QA pairs that (1) restate each fact in diverse compositional forms, and (2) require bidirectional source-to-fact and fact-to-source generation, jointly teaching the model to generate content from a cited source and to attribute its own answers. Experiments with Qwen2.5-7B and 3B show that Active Indexing consistently outperforms Passive Indexing across all tasks and models, with citation precision gains up to 30.2 percent. Our ablation studies reveal that performance continues to improve as we scale the amount of augmented data, showing a clear upward trend even at 16 times the original token count. 

**Abstract (ZH)**: 可信的语言模型应该提供准确且可验证的答案。虽然语言模型有时可以将其输出归因于预训练数据，但由于幻觉，其引用往往是不可靠的。当前系统在推理时通过查询外部检索器插入引用，这引入了延迟、基础设施依赖性和检索噪声的脆弱性。我们探讨是否可以通过修订训练过程使大规模语言模型在（持续）预训练期间可靠地归因于其看到的文档——而不依赖于测试时的检索。为此，我们发布了CitePretrainBench基准，该基准混合了真实世界语料库（维基百科、通用爬虫、arXiv）和新颖的未见文档，并测试了短格式（单一事实）和长格式（多事实）的引用任务。我们的方法遵循两阶段过程：（1）持续预训练将事实绑定到持久化的文档标识符，（2）指令微调以引发引用行为。我们发现简单被动索引，即在每个文档后追加一个标识符，有助于记忆直引文本，但在多义或组成性的事实方面失败。相反，我们提出了主动索引，即持续预训练合成问答对，（1）以多种组成形式重述每个事实，（2）需要双向源到事实和事实到源生成，共同教导模型从引用的源生成内容并归因自己的答案。使用Qwen2.5-7B和3B的实验显示，主动索引在所有任务和模型中均优于被动索引，引文献精度提高多达30.2%。我们的消融研究显示，随着增强数据量的增加，性能持续提升，即使在原始词元数的16倍时，也表现出明显的上升趋势。 

---
# From Unstructured Communication to Intelligent RAG: Multi-Agent Automation for Supply Chain Knowledge Bases 

**Title (ZH)**: 从无结构通信到智能RAG：供应链知识库的多代理自动化 

**Authors**: Yao Zhang, Zaixi Shang, Silpan Patel, Mikel Zuniga  

**Link**: [PDF](https://arxiv.org/pdf/2506.17484)  

**Abstract**: Supply chain operations generate vast amounts of operational data; however, critical knowledge such as system usage practices, troubleshooting workflows, and resolution techniques often remains buried within unstructured communications like support tickets, emails, and chat logs. While RAG systems aim to leverage such communications as a knowledge base, their effectiveness is limited by raw data challenges: support tickets are typically noisy, inconsistent, and incomplete, making direct retrieval suboptimal. Unlike existing RAG approaches that focus on runtime optimization, we introduce a novel offline-first methodology that transforms these communications into a structured knowledge base. Our key innovation is a LLMs-based multi-agent system orchestrating three specialized agents: Category Discovery for taxonomy creation, Categorization for ticket grouping, and Knowledge Synthesis for article generation. Applying our methodology to real-world support tickets with resolution notes and comments, our system creates a compact knowledge base - reducing total volume to just 3.4% of original ticket data while improving quality. Experiments demonstrate that our prebuilt knowledge base in RAG systems significantly outperforms traditional RAG implementations (48.74% vs. 38.60% helpful answers) and achieves a 77.4% reduction in unhelpful responses. By automating institutional knowledge capture that typically remains siloed in experts' heads, our solution translates to substantial operational efficiency: reducing support workload, accelerating resolution times, and creating self-improving systems that automatically resolve approximately 50% of future supply chain tickets. Our approach addresses a key gap in knowledge management by transforming transient communications into structured, reusable knowledge through intelligent offline processing rather than latency-inducing runtime architectures. 

**Abstract (ZH)**: 供应链运营生成大量操作数据；然而，诸如系统使用实践、故障排除工作流程和解决方案技术等关键知识往往埋藏在支持工单、电子邮件和聊天日志等非结构化通信中。虽然RAG系统旨在利用此类通信作为知识库，但它们的有效性受限于原始数据挑战：支持工单通常嘈杂、不一致且不完整，直接检索效果不佳。不同于现有RAG方法侧重于运行时优化，我们提出了一种新型的离线优先方法，将这些通信转换为结构化知识库。我们的关键创新是一种基于LLMs的多代理系统，协调三个专门代理：分类发现用于分类目录创建，分类用于票务分组，知识综合用于文章生成。将我们的方法应用于包含解决注释和支持评论的真实世界支持工单，我们的系统创建了一个紧凑的知识库——数据总量仅占原始工单数据的3.4%，同时提高了质量。实验表明，我们的预构建知识库在RAG系统中的表现显著优于传统的RAG实现（48.74%对比38.60%有帮助的回答），并且减少了77.4%的无用回答。通过自动化机构知识捕获，通常局限于专家头脑中，我们的解决方案提高了运营效率：减少了支持工作量，加速了解决时间，并创建了能够自动解决未来约50%供应链工单的自改进系统。我们的方法通过智能离线处理将瞬态通信转化为可重复使用的结构化知识，填补了知识管理中的关键空白，而不是依赖于引入延迟的运行时架构。 

---
# OmniReflect: Discovering Transferable Constitutions for LLM agents via Neuro-Symbolic Reflections 

**Title (ZH)**: OmniReflect: 通过神经符号反思发现适用于LLM代理的可转移构成要素 

**Authors**: Manasa Bharadwaj, Nikhil Verma, Kevin Ferreira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17449)  

**Abstract**: Efforts to improve Large Language Model (LLM) agent performance on complex tasks have largely focused on fine-tuning and iterative self-correction. However, these approaches often lack generalizable mechanisms for longterm learning and remain inefficient in dynamic environments. We introduce OmniReflect, a hierarchical, reflection-driven framework that constructs a constitution, a compact set of guiding principles distilled from task experiences, to enhance the effectiveness and efficiency of an LLM agent. OmniReflect operates in two modes: Self-sustaining, where a single agent periodically curates its own reflections during task execution, and Co-operative, where a Meta-advisor derives a constitution from a small calibration set to guide another agent. To construct these constitutional principles, we employ Neural, Symbolic, and NeuroSymbolic techniques, offering a balance between contextual adaptability and computational efficiency. Empirical results averaged across models show major improvements in task success, with absolute gains of +10.3% on ALFWorld, +23.8% on BabyAI, and +8.3% on PDDL in the Self-sustaining mode. Similar gains are seen in the Co-operative mode, where a lightweight Qwen3-4B ReAct agent outperforms all Reflexion baselines on BabyAI. These findings highlight the robustness and effectiveness of OmniReflect across environments and backbones. 

**Abstract (ZH)**: 面向复杂任务的大型语言模型代理性能改进努力主要集中在微调和迭代自我修正上。然而，这些方法往往缺乏可泛化的长期学习机制，并且在动态环境中效率低下。我们 introduces OmniReflect，一种层次化、基于反思的框架，通过从任务经验中提炼出一套紧凑的指导原则来构建宪法，以提高大型语言模型代理的有效性和效率。OmniReflect 运行在两种模式下：自我维持模式，其中单个代理在执行任务期间定期整理自己的反思；合作模式，其中元顾问从校准集中提取宪法以指导另一个代理。为了构建这些宪法原则，我们采用了神经、符号和神经符号技术，提供了上下文适应性和计算效率之间的平衡。在模型上的实验证明了显著的任务成功率改进，在自我维持模式下，ALFWorld 增加了 10.3%，BabyAI 增加了 23.8%，PDDL 增加了 8.3%。在合作模式下，一个轻量级的 Qwen3-4B ReAct 代理在 BabyAI 上的表现优于所有反射基线。这些发现突显了 OmniReflect 在不同环境和基础架构中的稳健性和有效性。 

---
# Evaluating Generalization and Representation Stability in Small LMs via Prompting 

**Title (ZH)**: 通过提示评估小型语言模型的泛化能力和表示稳定性 

**Authors**: Rahul Raja, Arpita Vats  

**Link**: [PDF](https://arxiv.org/pdf/2506.17289)  

**Abstract**: We investigate the generalization capabilities of small language models under two popular adaptation paradigms: few-shot prompting and supervised fine-tuning. While prompting is often favored for its parameter efficiency and flexibility, it remains unclear how robust this approach is in low-resource settings and under distributional shifts. This paper presents a comparative study of prompting and fine-tuning across task formats, prompt styles, and model scales, with a focus on their behavior in both in-distribution and out-of-distribution (OOD) settings.
Beyond accuracy, we analyze the internal representations learned by each approach to assess the stability and abstraction of task-specific features. Our findings highlight critical differences in how small models internalize and generalize knowledge under different adaptation strategies. This work offers practical guidance for model selection in low-data regimes and contributes empirical insight into the ongoing debate over prompting versus fine-tuning. Code for the experiments is available at the following 

**Abstract (ZH)**: 我们研究了小型语言模型在两种流行的适应 paradigm：少样本提示和监督微调下的泛化能力。虽然提示因参数效率和灵活性而常被青睐，但在资源稀缺环境下以及分布变化时，这种做法的鲁棒性仍不清楚。本文在任务格式、提示风格和模型规模上对提示和微调进行了比较研究，重点关注它们在既定分布和出分布（OOD）设置下的行为。

除了准确性之外，我们还分析了每种方法学习到的内部表示，以评估任务特定特征的稳定性和抽象程度。我们的研究结果突显了在不同适应策略下小模型内部化和泛化知识的关键差异。这项工作为低数据环境下的模型选择提供了实践指导，并为提示与微调之间的持续争论提供了实证见解。实验代码可在以下地址获取。 

---
# OMEGA: Can LLMs Reason Outside the Box in Math? Evaluating Exploratory, Compositional, and Transformative Generalization 

**Title (ZH)**: OMEGA：大型语言模型能在数学中进行框外推理吗？探究性、组合性和转换性泛化的评估 

**Authors**: Yiyou Sun, Shawn Hu, Georgia Zhou, Ken Zheng, Hannaneh Hajishirzi, Nouha Dziri, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.18880)  

**Abstract**: Recent large-scale language models (LLMs) with long Chain-of-Thought reasoning-such as DeepSeek-R1-have achieved impressive results on Olympiad-level mathematics benchmarks. However, they often rely on a narrow set of strategies and struggle with problems that require a novel way of thinking. To systematically investigate these limitations, we introduce OMEGA-Out-of-distribution Math Problems Evaluation with 3 Generalization Axes-a controlled yet diverse benchmark designed to evaluate three axes of out-of-distribution generalization, inspired by Boden's typology of creativity: (1) Exploratory-applying known problem solving skills to more complex instances within the same problem domain; (2) Compositional-combining distinct reasoning skills, previously learned in isolation, to solve novel problems that require integrating these skills in new and coherent ways; and (3) Transformative-adopting novel, often unconventional strategies by moving beyond familiar approaches to solve problems more effectively. OMEGA consists of programmatically generated training-test pairs derived from templated problem generators across geometry, number theory, algebra, combinatorics, logic, and puzzles, with solutions verified using symbolic, numerical, or graphical methods. We evaluate frontier (or top-tier) LLMs and observe sharp performance degradation as problem complexity increases. Moreover, we fine-tune the Qwen-series models across all generalization settings and observe notable improvements in exploratory generalization, while compositional generalization remains limited and transformative reasoning shows little to no improvement. By isolating and quantifying these fine-grained failures, OMEGA lays the groundwork for advancing LLMs toward genuine mathematical creativity beyond mechanical proficiency. 

**Abstract (ZH)**: Recent大规模语言模型（LLMs）具备长链推理能力——如DeepSeek-R1——在奥林匹克级数学基准测试中取得了显著成果。然而，它们往往依赖于狭窄的战略集并在需要新型思维方式的问题上挣扎。为系统地探讨这些局限性，我们引入了OMEGA：异域数学问题评估——一个控制多样但全面的基准，旨在评估由布登创造力类型学启发的三个异域泛化轴：（1）探索性：将已知问题解决技能应用于同一问题域内的更复杂实例；（2）组合性：结合此前孤立学习的不同推理技能，以解决需要以新颖且连贯的方式整合这些技能的新问题；（3）变革性：采用新颖且往往是非传统的策略，超越熟悉的解决方法以更有效地解决问题。OMEGA包含从几何、数论、代数、组合数学、逻辑和谜题等领域的模板问题生成器生成的训练-测试对，并通过符号、数值或图形方法验证解决方案。我们评估了前沿（或顶级）LLM并在问题复杂度增加时观察到性能急剧下降。此外，我们针对所有泛化设置微调Qwen系列模型，并观察到在探索性泛化方面取得了显著改进，而组合性泛化仍然受限且变革性推理几乎没有改善。通过分离和量化这些细微的失败，OMEGA为推动LLM朝着超越机械技能的真实数学创造力奠定了基础。 

---
# CommVQ: Commutative Vector Quantization for KV Cache Compression 

**Title (ZH)**: CommVQ: 交换式向量量化在KV缓存压缩中的应用 

**Authors**: Junyan Li, Yang Zhang, Muhammad Yusuf Hassan, Talha Chafekar, Tianle Cai, Zhile Ren, Pengsheng Guo, Foroozan Karimzadeh, Colorado Reed, Chong Wang, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18879)  

**Abstract**: Large Language Models (LLMs) are increasingly used in applications requiring long context lengths, but the key-value (KV) cache often becomes a memory bottleneck on GPUs as context grows. To address this, we propose Commutative Vector Quantization (CommVQ) to significantly reduce memory usage for long-context LLM inference. We first introduce additive quantization with a lightweight encoder and codebook to compress the KV cache, which can be decoded via simple matrix multiplication. To further reduce computational costs during decoding, we design the codebook to be commutative with Rotary Position Embedding (RoPE) and train it using an Expectation-Maximization (EM) algorithm. This enables efficient integration of decoding into the self-attention mechanism. Our approach achieves high accuracy with additive quantization and low overhead via the RoPE-commutative codebook. Experiments on long-context benchmarks and GSM8K show that our method reduces FP16 KV cache size by 87.5% with 2-bit quantization, while outperforming state-of-the-art KV cache quantization methods. Notably, it enables 1-bit KV cache quantization with minimal accuracy loss, allowing a LLaMA-3.1 8B model to run with a 128K context length on a single RTX 4090 GPU. The source code is available at: this https URL. 

**Abstract (ZH)**: 长上下文长度Large语言模型（LLMs）中的可交换向量量化（CommVQ）显著减少了GPU内存使用，以进行长期上下文LLM推理。 

---
# LongWriter-Zero: Mastering Ultra-Long Text Generation via Reinforcement Learning 

**Title (ZH)**: LongWriter-Zero: 通过强化学习掌握超长文本生成 

**Authors**: Yuhao Wu, Yushi Bai, Zhiqiang Hu, Roy Ka-Wei Lee, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.18841)  

**Abstract**: Ultra-long generation by large language models (LLMs) is a widely demanded scenario, yet it remains a significant challenge due to their maximum generation length limit and overall quality degradation as sequence length increases. Previous approaches, exemplified by LongWriter, typically rely on ''teaching'', which involves supervised fine-tuning (SFT) on synthetic long-form outputs. However, this strategy heavily depends on synthetic SFT data, which is difficult and costly to construct, often lacks coherence and consistency, and tends to be overly artificial and structurally monotonous. In this work, we propose an incentivization-based approach that, starting entirely from scratch and without relying on any annotated or synthetic data, leverages reinforcement learning (RL) to foster the emergence of ultra-long, high-quality text generation capabilities in LLMs. We perform RL training starting from a base model, similar to R1-Zero, guiding it to engage in reasoning that facilitates planning and refinement during the writing process. To support this, we employ specialized reward models that steer the LLM towards improved length control, writing quality, and structural formatting. Experimental evaluations show that our LongWriter-Zero model, trained from Qwen2.5-32B, consistently outperforms traditional SFT methods on long-form writing tasks, achieving state-of-the-art results across all metrics on WritingBench and Arena-Write, and even surpassing 100B+ models such as DeepSeek R1 and Qwen3-235B. We open-source our data and model checkpoints under this https URL 

**Abstract (ZH)**: 超长生成由大型语言模型（LLMs）实现：一种基于激励的方法 

---
# Understanding Software Engineering Agents: A Study of Thought-Action-Result Trajectories 

**Title (ZH)**: 理解软件工程代理：一种思考-行动-结果轨迹研究 

**Authors**: Islem Bouzenia, Michael Pradel  

**Link**: [PDF](https://arxiv.org/pdf/2506.18824)  

**Abstract**: Large Language Model (LLM)-based agents are increasingly employed to automate complex software engineering tasks such as program repair and issue resolution. These agents operate by autonomously generating natural language thoughts, invoking external tools, and iteratively refining their solutions. Despite their widespread adoption, the internal decision-making processes of these agents remain largely unexplored, limiting our understanding of their operational dynamics and failure modes. In this paper, we present a large-scale empirical study of the thought-action-result trajectories of three state-of-the-art LLM-based agents: \textsc{RepairAgent}, \textsc{AutoCodeRover}, and \textsc{OpenHands}. We unify their interaction logs into a common format, capturing 120 trajectories and 2822 LLM interactions focused on program repair and issue resolution. Our study combines quantitative analyses of structural properties, action patterns, and token usage with qualitative assessments of reasoning coherence and feedback integration. We identify key trajectory characteristics such as iteration counts and token consumption, recurring action sequences, and the semantic coherence linking thoughts, actions, and their results. Our findings reveal behavioral motifs and anti-patterns that distinguish successful from failed executions, providing actionable insights for improving agent design, including prompting strategies, failure diagnosis, and anti-pattern detection. We release our dataset and annotation framework to support further research on transparent and robust autonomous software engineering agents. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理越来越多地被用来自动化复杂的软件工程任务，如程序修复和问题解决。这些代理通过自主生成自然语言思维、调用外部工具并迭代优化其解决方案来运作。尽管它们已经在广泛采用中，但这些代理内部决策过程的研究仍然不足，限制了我们对其运作动态和失败模式的理解。在本文中，我们提出了对三款先进的基于LLM的代理——\textsc{RepairAgent}、\textsc{AutoCodeRover} 和 \textsc{OpenHands}——思维-行动-结果轨迹的大规模实证研究。我们将它们的交互日志统一到一个共同格式中，记录了120条轨迹和2822次LLM交互，专注于程序修复和问题解决。我们的研究结合了结构属性的定量分析、行动模式和词汇使用量的分析，以及对推理连贯性和反馈整合的定性评估。我们识别出了关键的轨迹特征，如迭代次数和词汇消耗量、反复出现的动作序列以及思维、行动和结果之间的语义连贯性。我们的发现揭示了区分成功和失败执行的行为模式和反模式，为改进代理设计提供了实际指导，包括提示策略、故障诊断和反模式检测。我们发布我们的数据集和注释框架以支持对透明和稳健的自主软件工程代理的进一步研究。 

---
# RWESummary: A Framework and Test for Choosing Large Language Models to Summarize Real-World Evidence (RWE) Studies 

**Title (ZH)**: RWESummary：选择用于总结真实世界证据研究的大语言模型的框架与测试方法 

**Authors**: Arjun Mukerji, Michael L. Jackson, Jason Jones, Neil Sanghavi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18819)  

**Abstract**: Large Language Models (LLMs) have been extensively evaluated for general summarization tasks as well as medical research assistance, but they have not been specifically evaluated for the task of summarizing real-world evidence (RWE) from structured output of RWE studies. We introduce RWESummary, a proposed addition to the MedHELM framework (Bedi, Cui, Fuentes, Unell et al., 2025) to enable benchmarking of LLMs for this task. RWESummary includes one scenario and three evaluations covering major types of errors observed in summarization of medical research studies and was developed using Atropos Health proprietary data. Additionally, we use RWESummary to compare the performance of different LLMs in our internal RWE summarization tool. At the time of publication, with 13 distinct RWE studies, we found the Gemini 2.5 models performed best overall (both Flash and Pro). We suggest RWESummary as a novel and useful foundation model benchmark for real-world evidence study summarization. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用摘要任务和医疗研究辅助领域得到了广泛评估，但尚未专门评估其在从结构化RWE研究输出中总结实际世界证据（RWE）的任务上的性能。我们提出了RWESummary，作为MedHELM框架（Bedi, Cui, Fuentes, Unell et al., 2025）的一个新增内容，以便对LLMs进行此类任务的基准测试。RWESummary包含一个场景和三项评估，涵盖了在总结医疗研究文章中观察到的主要类型错误，并使用Atropos Health专有数据开发。此外，我们使用RWESummary比较了不同LLMs在内部RWE总结工具中的性能。截至出版时，使用13篇不同的RWE研究，我们发现Gemini 2.5模型整体表现最佳（包括Flash和Pro版本）。我们建议RWESummary作为实际世界证据研究总结的新颖且有用的基准模型。 

---
# Benchmarking the Pedagogical Knowledge of Large Language Models 

**Title (ZH)**: 大规模语言模型的教 学知识基准研究 

**Authors**: Maxime Lelièvre, Amy Waldock, Meng Liu, Natalia Valdés Aspillaga, Alasdair Mackintosh, María José Ogando Portelo, Jared Lee, Paul Atherton, Robin A. A. Ince, Oliver G. B. Garrod  

**Link**: [PDF](https://arxiv.org/pdf/2506.18710)  

**Abstract**: Benchmarks like Massive Multitask Language Understanding (MMLU) have played a pivotal role in evaluating AI's knowledge and abilities across diverse domains. However, existing benchmarks predominantly focus on content knowledge, leaving a critical gap in assessing models' understanding of pedagogy - the method and practice of teaching. This paper introduces The Pedagogy Benchmark, a novel dataset designed to evaluate large language models on their Cross-Domain Pedagogical Knowledge (CDPK) and Special Education Needs and Disability (SEND) pedagogical knowledge. These benchmarks are built on a carefully curated set of questions sourced from professional development exams for teachers, which cover a range of pedagogical subdomains such as teaching strategies and assessment methods. Here we outline the methodology and development of these benchmarks. We report results for 97 models, with accuracies spanning a range from 28% to 89% on the pedagogical knowledge questions. We consider the relationship between cost and accuracy and chart the progression of the Pareto value frontier over time. We provide online leaderboards at this https URL which are updated with new models and allow interactive exploration and filtering based on various model properties, such as cost per token and open-vs-closed weights, as well as looking at performance in different subjects. LLMs and generative AI have tremendous potential to influence education and help to address the global learning crisis. Education-focused benchmarks are crucial to measure models' capacities to understand pedagogical concepts, respond appropriately to learners' needs, and support effective teaching practices across diverse contexts. They are needed for informing the responsible and evidence-based deployment of LLMs and LLM-based tools in educational settings, and for guiding both development and policy decisions. 

**Abstract (ZH)**: 大规模多任务语言理解基准（MMLU）等基准在评估AI的知识和能力方面发挥了关键作用，但现有的基准主要集中在内容知识上，忽略了对模型教学法理解的评估——教学方法和实践。本文介绍了教学基准，这是一个新型数据集，旨在评估大型语言模型在跨领域教学知识（CDPK）和特殊教育需求与残疾（SEND）教学知识方面的表现。这些基准建立在精心选择的问题集上，这些问题源自教师专业发展考试，涵盖了教学策略和评估方法等多种教学亚领域。本文概述了这些基准的方法学和开发过程。我们报告了97个模型在教学知识问题上的准确率，范围从28%到89%。我们考虑了成本与准确率之间的关系，并追踪了帕累托价值前沿随时间的变化。我们在以下网址提供了在线排行榜，会定期更新新的模型，并允许基于各种模型属性（如每个标记的成本和开放权重）进行互动探索和筛选，还涵盖了不同学科的表现。大规模语言模型和生成式AI在教育领域具有巨大潜力，可以影响教育并有助于解决全球学习危机。以教育为目标的基准对于衡量模型理解教学概念、适当回应学习者的需求以及支持不同背景下有效教学实践的能力至关重要。这些基准对于在教育环境中负责任地部署大规模语言模型及其工具以及指导开发和政策决策是必要的。 

---
# Is There a Case for Conversation Optimized Tokenizers in Large Language Models? 

**Title (ZH)**: 大型语言模型中对话优化分词器是否有必要？ 

**Authors**: Raquel Ferrando, Javier Conde, Gonzalo Martínez, Pedro Reviriego  

**Link**: [PDF](https://arxiv.org/pdf/2506.18674)  

**Abstract**: The computational and energy costs of Large Language Models (LLMs) have increased exponentially driven by the growing model sizes and the massive adoption of LLMs by hundreds of millions of users. The unit cost of an LLM is the computation of a token. Therefore, the tokenizer plays an important role in the efficiency of a model, and they are carefully optimized to minimize the number of tokens for the text in their training corpus. One of the most popular applications of LLMs are chatbots that interact with users. A key observation is that, for those chatbots, what is important is the performance of the tokenizer in the user text input and the chatbot responses. Those are most likely different from the text in the training corpus. So, a question that immediately arises is whether there is a potential benefit in optimizing tokenizers for chatbot conversations. In this paper, this idea is explored for different tokenizers by using a publicly available corpus of chatbot conversations to redesign their vocabularies and evaluate their performance in this domain. The results show that conversation-optimized tokenizers consistently reduce the number of tokens in chatbot dialogues, which can lead to meaningful energy savings, in the range of 5% to 10% while having minimal or even slightly positive impact on tokenization efficiency for the original training corpus. 

**Abstract (ZH)**: 大型语言模型（LLMs）的计算和能量成本因模型规模的扩大和数百亿用户的大规模采用而呈指数增长。LLM的单位成本是一个词元的计算。因此，分词器在模型的效率中扮演重要角色，它们被仔细优化以最小化训练语料库中文本的词元数量。大型语言模型（LLMs）最流行的应用之一是与用户交互的聊天机器人。一个关键观察是，对于这些聊天机器人来说，重要的是分词器在用户文本输入和聊天机器人响应中的表现。这些文本很可能与训练语料库中的文本不同。因此，一个立即引起的问题是，是否有潜力通过优化分词器来提高聊天机器人对话的表现。在这篇论文中，通过使用一个公开可用的聊天机器人对话语料库来重新设计不同的分词器词汇表并评估其在该领域的性能，探索了这一想法。结果显示，对话优化的分词器一致地减少了聊天机器人对话中的词元数量，可以在范围为5%到10%的水平上带来有意义的能量节省，同时对原始训练语料库中的分词效率影响很小甚至略微提高。 

---
# ReDit: Reward Dithering for Improved LLM Policy Optimization 

**Title (ZH)**: ReDit: 奖励抖动以改进大型语言模型策略优化 

**Authors**: Chenxing Wei, Jiarui Yu, Ying Tiffany He, Hande Dong, Yao Shu, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.18631)  

**Abstract**: DeepSeek-R1 has successfully enhanced Large Language Model (LLM) reasoning capabilities through its rule-based reward system. While it's a ''perfect'' reward system that effectively mitigates reward hacking, such reward functions are often discrete. Our experimental observations suggest that discrete rewards can lead to gradient anomaly, unstable optimization, and slow convergence. To address this issue, we propose ReDit (Reward Dithering), a method that dithers the discrete reward signal by adding simple random noise. With this perturbed reward, exploratory gradients are continuously provided throughout the learning process, enabling smoother gradient updates and accelerating convergence. The injected noise also introduces stochasticity into flat reward regions, encouraging the model to explore novel policies and escape local optima. Experiments across diverse tasks demonstrate the effectiveness and efficiency of ReDit. On average, ReDit achieves performance comparable to vanilla GRPO with only approximately 10% the training steps, and furthermore, still exhibits a 4% performance improvement over vanilla GRPO when trained for a similar duration. Visualizations confirm significant mitigation of gradient issues with ReDit. Moreover, theoretical analyses are provided to further validate these advantages. 

**Abstract (ZH)**: DeepSeek-R1 通过基于规则的奖励系统成功增强了大型语言模型的推理能力，尽管这是一个“完美”的奖励系统，有效遏制了奖励误用，但这样的奖励函数往往是离散的。我们的实验观察表明，离散奖励会导致梯度异常、优化不稳定和收敛缓慢。为解决这一问题，我们提出了 ReDit（奖励抖动）方法，通过添加简单的随机噪声来抖动离散奖励信号。通过这种方式扰动的奖励，在学习过程中连续提供探索梯度，使梯度更新更加平滑并加速收敛。注入的噪声还在平坦奖励区域引入了随机性，鼓励模型探索新的策略并跳出局部最优。跨多种任务的实验验证了 ReDit 的有效性和效率。在平均情况下，ReDit 在训练步数减少约 10% 的情况下达到了与标准 GRPO 相似的性能，并且在相似训练时间内相对于标准 GRPO 还表现出 4% 的性能提升。可视化结果进一步证实了 ReDit 对梯度问题的有效缓解。此外，还提供了理论分析以进一步验证这些优势。 

---
# Security Assessment of DeepSeek and GPT Series Models against Jailbreak Attacks 

**Title (ZH)**: 深层搜索与GPT系列模型针对 Jailbreak 攻击的安全性评估 

**Authors**: Xiaodong Wu, Xiangman Li, Jianbing Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.18543)  

**Abstract**: The widespread deployment of large language models (LLMs) has raised critical concerns over their vulnerability to jailbreak attacks, i.e., adversarial prompts that bypass alignment mechanisms and elicit harmful or policy-violating outputs. While proprietary models like GPT-4 have undergone extensive evaluation, the robustness of emerging open-source alternatives such as DeepSeek remains largely underexplored, despite their growing adoption in real-world applications. In this paper, we present the first systematic jailbreak evaluation of DeepSeek-series models, comparing them with GPT-3.5 and GPT-4 using the HarmBench benchmark. We evaluate seven representative attack strategies across 510 harmful behaviors categorized by both function and semantic domain. Our analysis reveals that DeepSeek's Mixture-of-Experts (MoE) architecture introduces routing sparsity that offers selective robustness against optimization-based attacks such as TAP-T, but leads to significantly higher vulnerability under prompt-based and manually engineered attacks. In contrast, GPT-4 Turbo demonstrates stronger and more consistent safety alignment across diverse behaviors, likely due to its dense Transformer design and reinforcement learning from human feedback. Fine-grained behavioral analysis and case studies further show that DeepSeek often routes adversarial prompts to under-aligned expert modules, resulting in inconsistent refusal behaviors. These findings highlight a fundamental trade-off between architectural efficiency and alignment generalization, emphasizing the need for targeted safety tuning and modular alignment strategies to ensure secure deployment of open-source LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用引发了对其易受 Jailbreak 攻击的关切，即能够逃避对齐机制并产生有害或政策违反输出的 adversarial prompts。虽然像 GPT-4 这样的专有模型已经经历了广泛的评估，但新兴的开源替代品如 DeepSeek 的稳健性仍然鲜有探索，尽管它们在实际应用中的采用率正在增长。本文首次系统评估了 DeepSeek 系列模型，并使用 HarmBench 基准将其与 GPT-3.5 和 GPT-4 进行比较。我们评估了七个代表性攻击策略在 510 种有害行为中的表现，这些有害行为按功能和语义领域进行分类。分析结果显示，DeepSeek 的 Mixture-of-Experts（MoE）架构引入了路由稀疏性，这种稀疏性对其基于优化的攻击（如 TAP-T）具有选择性的稳健性，但在基于提示的攻击和手动工程化的攻击下则表现出显著更高的脆弱性。相比之下，GPT-4 Turbo 在多种行为中展现出更强且更一致的安全对齐，这可能归因于其密集的Transformer 设计和来自人类反馈的强化学习。细粒度的行为分析和案例研究进一步表明，DeepSeek 经常将 adversarial prompts 路由到未对齐的专家模块，从而导致不一致的拒绝行为。这些发现揭示了架构效率和对齐通用性之间的根本权衡，强调了针对安全调优和模块化对齐策略以确保开源 LLM 安全部署的必要性。 

---
# Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts 

**Title (ZH)**: 平滑操作：大语言模型将不完美的提示转化为富含杂音的转录 

**Authors**: Duygu Altinok  

**Link**: [PDF](https://arxiv.org/pdf/2506.18510)  

**Abstract**: Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints. 

**Abstract (ZH)**: 准确检测口语中的非流畅性对于提升自动语音和语言处理系统的性能以及促进更具包容性的语音和语言技术的发展至关重要。利用大型语言模型（LLMs）作为既能处理词汇性输入又能处理非词汇性输入（如音频和视频）的通用学习者，我们提出了一种新的方法，即将非流畅性转录为带时间戳的显式标记，从而生成丰富的非流畅性标注转录。该方法结合了从音频编码器提取的声学表示和不同质量的文本输入：无非流畅性的干净转录、对齐器的时间对齐转录或基于音素的ASR模型输出——这些输入中可能包含不完美之处。重要的是，我们的实验表明，文本输入不需要完美。只要它们包含时间戳相关的线索，LLMs就可以有效地平滑输入并生成完整的非流畅性标注转录，突显了它们在处理不完整提示时的鲁棒性。 

---
# Comparative Evaluation of ChatGPT and DeepSeek Across Key NLP Tasks: Strengths, Weaknesses, and Domain-Specific Performance 

**Title (ZH)**: ChatGPT与DeepSeek在关键NLP任务上的比较评价：优势、弱点及领域特定性能 

**Authors**: Wael Etaiwi, Bushra Alhijawi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18501)  

**Abstract**: The increasing use of large language models (LLMs) in natural language processing (NLP) tasks has sparked significant interest in evaluating their effectiveness across diverse applications. While models like ChatGPT and DeepSeek have shown strong results in many NLP domains, a comprehensive evaluation is needed to understand their strengths, weaknesses, and domain-specific abilities. This is critical as these models are applied to various tasks, from sentiment analysis to more nuanced tasks like textual entailment and translation. This study aims to evaluate ChatGPT and DeepSeek across five key NLP tasks: sentiment analysis, topic classification, text summarization, machine translation, and textual entailment. A structured experimental protocol is used to ensure fairness and minimize variability. Both models are tested with identical, neutral prompts and evaluated on two benchmark datasets per task, covering domains like news, reviews, and formal/informal texts. The results show that DeepSeek excels in classification stability and logical reasoning, while ChatGPT performs better in tasks requiring nuanced understanding and flexibility. These findings provide valuable insights for selecting the appropriate LLM based on task requirements. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理（NLP）任务中的应用日益增加，引发了对其在各类应用中效果评价的重大兴趣。虽然像ChatGPT和DeepSeek这样的模型在许多NLP领域表现出色，但需要进行全面评估以了解其优势、劣势及其在特定领域的能力。鉴于这些模型在从情感分析到语义蕴含和翻译等不同任务中的广泛应用，本研究旨在通过五个关键NLP任务（情感分析、主题分类、文本摘要、机器翻译和语义蕴含）评价ChatGPT和DeepSeek的表现。采用结构化的实验方案以确保公平性和减少变异性。两种模型使用相同的中性提示进行测试，并在每个任务的两个基准数据集上进行评估，涵盖了新闻、评论以及正式和非正式文本等领域。研究结果表明，DeepSeek在分类稳定性和逻辑推理方面表现优异，而ChatGPT在需要细腻理解和灵活性的任务中表现更佳。这些发现为根据任务需求选择合适的LLM提供了有价值的见解。 

---
# MeRF: Motivation-enhanced Reinforcement Finetuning for Large Reasoning Models 

**Title (ZH)**: 动机增强强化微调大推理模型 

**Authors**: Junjie Zhang, Guozheng Ma, Shunyu Liu, Haoyu Wang, Jiaxing Huang, Ting-En Lin, Fei Huang, Yongbin Li, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18485)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a powerful learn-to-reason paradigm for Large Language Models (LLMs) to tackle complex reasoning tasks. However, existing RLVR methods overlook one of the most distinctive capabilities of LLMs, their in-context learning ability, as prominently demonstrated by the success of Chain-of-Thought (CoT) prompting. This motivates us to explore how reinforcement learning can be effectively combined with in-context learning to better improve the reasoning capabilities of LLMs. In this paper, we introduce Motivation-enhanced Reinforcement Finetuning} (MeRF), an intuitive yet effective method enhancing reinforcement learning of LLMs by involving ``telling LLMs the rules of the game''. Specifically, MeRF directly injects the reward specification into the prompt, which serves as an in-context motivation for model to improve its responses with awareness of the optimization objective. This simple modification leverages the in-context learning ability of LLMs aligning generation with optimization, thereby incentivizing the model to generate desired outputs from both inner motivation and external reward. Empirical evaluations on the Knights and Knaves~(K&K) logic puzzle reasoning benchmark demonstrate that \texttt{MeRF} achieves substantial performance gains over baselines. Moreover, ablation studies show that performance improves with greater consistency between the in-context motivation and the external reward function, while the model also demonstrates an ability to adapt to misleading motivations through reinforcement learning. 

**Abstract (ZH)**: 可验证奖励的强化学习：动机增强的强化微调（Motivation-enhanced Reinforcement Finetuning for Verifiable Rewards, MeRF） 

---
# TReB: A Comprehensive Benchmark for Evaluating Table Reasoning Capabilities of Large Language Models 

**Title (ZH)**: TReB：评估大型语言模型表推理能力的综合性基准 

**Authors**: Ce Li, Xiaofan Liu, Zhiyan Song, Ce Chi, Chen Zhao, Jingjing Yang, Zhendong Wang, Kexin Yang, Boshen Shi, Xing Wang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18421)  

**Abstract**: The majority of data in businesses and industries is stored in tables, databases, and data warehouses. Reasoning with table-structured data poses significant challenges for large language models (LLMs) due to its hidden semantics, inherent complexity, and structured nature. One of these challenges is lacking an effective evaluation benchmark fairly reflecting the performances of LLMs on broad table reasoning abilities. In this paper, we fill in this gap, presenting a comprehensive table reasoning evolution benchmark, TReB, which measures both shallow table understanding abilities and deep table reasoning abilities, a total of 26 sub-tasks. We construct a high quality dataset through an iterative data processing procedure. We create an evaluation framework to robustly measure table reasoning capabilities with three distinct inference modes, TCoT, PoT and ICoT. Further, we benchmark over 20 state-of-the-art LLMs using this frame work and prove its effectiveness. Experimental results reveal that existing LLMs still have significant room for improvement in addressing the complex and real world Table related tasks. Both the dataset and evaluation framework are publicly available, with the dataset hosted on [HuggingFace] and the framework on [GitHub]. 

**Abstract (ZH)**: 企业与行业的大多数数据存储在表格、数据库和数据仓库中。处理具有隐藏语义、内在复杂性和结构化性质的表格结构数据对大型语言模型(LLMs)构成了重大挑战。其中一个挑战是没有一个有效的评估基准公平地反映LLMs在广泛表格推理能力上的表现。在本文中，我们填补了这一空白，提出了一个全面的表格推理演化基准TReB，该基准衡量浅层表格理解能力和深层表格推理能力，共计26个子任务。我们通过迭代数据处理程序构建了一个高质量的数据集。我们构建了一个评估框架，使用三种不同的推理模式TCoT、PoT和ICoT稳健地衡量表格推理能力。进一步地，我们使用该框架对超过20个最先进的LLMs进行了基准测试，并证明了其有效性。实验结果表明，现有的LLMs在处理复杂和实际的表格相关任务方面仍有很大的改进空间。数据集和评估框架均已公开，数据集托管在[HuggingFace]上，框架托管在[GitHub]上。 

---
# The Debugging Decay Index: Rethinking Debugging Strategies for Code LLMs 

**Title (ZH)**: 代码LLM调试衰减指数：重新思考调试策略 

**Authors**: Muntasir Adnan, Carlos C. N. Kuhn  

**Link**: [PDF](https://arxiv.org/pdf/2506.18403)  

**Abstract**: The effectiveness of AI debugging follows a predictable exponential decay pattern; most models lose 60-80% of their debugging capability within just 2-3 attempts, despite iterative debugging being a critical capability for practical code generation systems. We introduce the Debugging Decay Index (DDI), a mathematical framework that quantifies when debugging becomes ineffective and predicts intervention points. Our strategic fresh start approach shifts from exploitation to exploration at strategic points in the debugging process, demonstrating that well-timed interventions can rescue the effectiveness of debugging. DDI reveals a fundamental limitation in current AI debugging and provides the first quantitative framework for optimising iterative code generation strategies. 

**Abstract (ZH)**: AI调试效果遵循可预测的指数衰减模式：大多数模型在仅2-3次调试尝试后会损失60-80%的调试能力，尽管迭代调试是实际代码生成系统的关键能力。我们引入调试衰减指数（DDI），这是一种数学框架，用于量化调试何时变得无效并预测干预点。我们的策略性全新开始方法在调试过程中的战略点从利用转向探索，表明适时干预可以挽救调试效果。DDI揭示了当前AI调试的基本局限性，并提供了优化迭代代码生成策略的首个定量框架。 

---
# Evaluating Causal Explanation in Medical Reports with LLM-Based and Human-Aligned Metrics 

**Title (ZH)**: 基于LLM和人类对齐指标的医学报告因果解释评估 

**Authors**: Yousang Cho, Key-Sun Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.18387)  

**Abstract**: This study investigates how accurately different evaluation metrics capture the quality of causal explanations in automatically generated diagnostic reports. We compare six metrics: BERTScore, Cosine Similarity, BioSentVec, GPT-White, GPT-Black, and expert qualitative assessment across two input types: observation-based and multiple-choice-based report generation. Two weighting strategies are applied: one reflecting task-specific priorities, and the other assigning equal weights to all metrics. Our results show that GPT-Black demonstrates the strongest discriminative power in identifying logically coherent and clinically valid causal narratives. GPT-White also aligns well with expert evaluations, while similarity-based metrics diverge from clinical reasoning quality. These findings emphasize the impact of metric selection and weighting on evaluation outcomes, supporting the use of LLM-based evaluation for tasks requiring interpretability and causal reasoning. 

**Abstract (ZH)**: 本研究调查了不同评价指标在捕捉自动生成诊断报告中因果解释质量方面的准确性。我们比较了六种指标：BERTScore、余弦相似度、BioSentVec、GPT-White、GPT-Black以及专家定性评估，涵盖两种输入类型：基于观察的报告生成和基于多项选择的报告生成。我们应用了两种加权策略：一种反映任务特定优先级，另一种将所有指标赋予等权重。研究结果表明，GPT-Black在识别逻辑连贯且临床有效的因果叙述方面表现出最强的区分能力。GPT-White也与专家评估高度一致，而基于相似性的指标偏离了临床推理质量。这些发现强调了指标选择和加权对评价结果的影响，支持在需要可解释性和因果推理的任务中使用基于LLM的评价方法。 

---
# LOGICPO: Efficient Translation of NL-based Logical Problems to FOL using LLMs and Preference Optimization 

**Title (ZH)**: LOGICPO: 使用LLMs和偏好优化将基于自然语言的逻辑问题高效转换为一阶逻辑 

**Authors**: Koushik Viswanadha, Deepanway Ghosal, Somak Aditya  

**Link**: [PDF](https://arxiv.org/pdf/2506.18383)  

**Abstract**: Logical reasoning is a key task for artificial intelligence due to it's role in major downstream tasks such as Question Answering, Summarization. Recent methods in improving the reasoning ability of LLMs fall short in correctly converting a natural language reasoning problem to an equivalent logical formulation, which hinders the framework's overall ability to reason. Towards this, we propose to use finetuning on a preference optimization dataset to learn to parse and represent a natural language problem as a whole to a consistent logical program by 1) introducing a new supervised and preference optimization dataset LogicPO, and 2) adopting popular techniques such as Direct Preference Optimization (DPO), Kahneman-Tversky optimization (KTO) to finetune open-source LLMs. Our best model with Phi-3.5 consistently outperforms GPT-3.5-turbo's (8-shot) by producing 10% more logically correct and with 14% less syntax errors. Through the framework and our improved evaluation metrics, we offer a promising direction in improving the logical reasoning of LLMs by better representing them in their logical formulations. 

**Abstract (ZH)**: 逻辑推理是人工智能的一项关键任务，因其在问答、总结等重要下游任务中的作用。近期提高大语言模型推理能力的方法在准确将自然语言推理问题转换为等效逻辑形式方面存在不足，这妨碍了框架的整体推理能力。为此，我们提出利用偏好优化数据集进行微调，学习将自然语言问题作为一个整体解析和表示为一致的逻辑程序。具体来说，我们1）引入一个新的监督和偏好优化数据集LogicPO；2）采用直接偏好优化（DPO）、 Kahneman-Tversky优化（KTO）等流行技术对开源大语言模型进行微调。我们的最佳模型Phi-3.5始终优于GPT-3.5-turbo（8-shot），在逻辑正确性上提高了10%，在语法错误上减少了14%。通过我们的框架和改进的评估指标，我们提出了提高大语言模型逻辑推理能力的一个有前途的方向，即更好地在其逻辑形式中表示它们。 

---
# Confucius3-Math: A Lightweight High-Performance Reasoning LLM for Chinese K-12 Mathematics Learning 

**Title (ZH)**: 孔夫子3-数学：一种轻量级高性能的中文K-12数学推理大模型 

**Authors**: Lixin Wu, Na Cai, Qiao Cheng, Jiachen Wang, Yitao Duan  

**Link**: [PDF](https://arxiv.org/pdf/2506.18330)  

**Abstract**: We introduce Confucius3-Math, an open-source large language model with 14B parameters that (1) runs efficiently on a single consumer-grade GPU; (2) achieves SOTA performances on a range of mathematical reasoning tasks, outperforming many models with significantly larger sizes. In particular, as part of our mission to enhancing education and knowledge dissemination with AI, Confucius3-Math is specifically committed to mathematics learning for Chinese K-12 students and educators. Built via post-training with large-scale reinforcement learning (RL), Confucius3-Math aligns with national curriculum and excels at solving main-stream Chinese K-12 mathematical problems with low cost. In this report we share our development recipe, the challenges we encounter and the techniques we develop to overcome them. In particular, we introduce three technical innovations: Targeted Entropy Regularization, Recent Sample Recovery and Policy-Specific Hardness Weighting. These innovations encompass a new entropy regularization, a novel data scheduling policy, and an improved group-relative advantage estimator. Collectively, they significantly stabilize the RL training, improve data efficiency, and boost performance. Our work demonstrates the feasibility of building strong reasoning models in a particular domain at low cost. We open-source our model and code at this https URL. 

**Abstract (ZH)**: 我们介绍Confucius3-Math，一个拥有14B参数的开源大语言模型，能够在单块消费级GPU上高效运行；在一系列数学推理任务上实现了SOTA性能，超越了许多规模大得多的模型。特别是作为我们借助AI增强教育和知识传播使命的一部分，Confucius3-Math特别致力于为中国K-12学生和教育者提供数学学习服务。通过大规模强化学习（RL）后训练构建，Confucius3-Math与国家课程体系高度契合，并以低成本解决主流的中国K-12数学问题。在本报告中，我们分享了我们的开发方法、遇到的挑战及克服这些挑战的技术。特别是，我们介绍了三项技术创新：目标熵正则化、最近样本恢复和策略特定难度加权。这些创新包括一种新的熵正则化、一种新颖的数据调度策略，以及一种改进的组内相对优势估计器。它们共同显著稳定了RL训练，提高了数据效率，并提升了性能。我们的工作展示了在特定领域以较低成本构建强大推理模型的可行性。我们已在以下链接开源了我们的模型和代码：https://。 

---
# Use Property-Based Testing to Bridge LLM Code Generation and Validation 

**Title (ZH)**: 使用基于属性的测试来弥合大型语言模型代码生成与验证的差距 

**Authors**: Lehan He, Zeren Chen, Zhe Zhang, Jing Shao, Xiang Gao, Lu Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.18315)  

**Abstract**: Large Language Models (LLMs) excel at code generation, but ensuring their outputs to be functionally correct, especially in complex programming tasks, is a persistent challenge. While traditional Test-Driven Development (TDD) offers a path for code refinement, its efficacy with LLMs is often undermined by the scarcity of high-quality test cases or the pitfalls of automated test generation, including biased tests or inaccurate output predictions that can misdirect the correction process. This paper introduces Property-Generated Solver, a novel framework that leverages Property-Based Testing (PBT) to validate high-level program properties or invariants, instead of relying on specific input-output examples. These properties are often simpler to define and verify than directly predicting exhaustive test oracles, breaking the "cycle of self-deception" where tests might share flaws with the code they are meant to validate. Property-Generated Solver employs two collaborative LLM-based agents: a Generator dedicated to code generation and iterative refinement, and a Tester that manages the PBT life-cycle and formulate semantically rich feedback from property violations. The resulting comprehensive and actionable feedback then guides the Generator in its refinement efforts. By establishing PBT as the core validation engine within this iterative, closed-loop paradigm, Property-Generated Solver provides a robust mechanism for steering LLMs towards more correct and generalizable code. Extensive experimental results on multiple code generation benchmarks demonstrate that Property-Generated Solver achieves substantial pass@1 improvements, ranging from 23.1% to 37.3% relative gains over established TDD methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成方面表现出色，但确保其输出的功能正确性，特别是在复杂编程任务中，仍是一个持续的挑战。虽然传统的测试驱动开发（TDD）为代码精炼提供了途径，但其与LLMs结合时的效果往往受到高质量测试用例稀缺或自动化测试生成陷阱的影响，包括有偏见的测试或不准确的输出预测，这些都可能误导纠正过程。本文介绍了一种名为Property-Generated Solver的新型框架，它利用基于属性的测试（PBT）来验证高层次程序属性或不变量，而不是依赖于特定的输入-输出示例。这些属性通常比直接预测详尽的测试或acles更易于定义和验证，打破了测试与要验证的代码共享缺陷的“自我蒙蔽循环”。Property-Generated Solver采用两个协作的LLM代理：一个专门用于代码生成和迭代精炼的Generator，以及一个负责管理PBT生命周期并从属性违规中形成语义丰富的反馈的Tester。由此产生的全面且可操作的反馈则指导Generator的精炼努力。通过在迭代、闭环范式中将PBT确立为核心验证引擎，Property-Generated Solver为引导LLMs生成更加正确和泛化的代码提供了一种稳健机制。在多个代码生成基准上的 extensive 实验结果表明，Property-Generated Solver实现了显著的pass@1改进，相对增益范围从23.1%到37.3%。 

---
# LettinGo: Explore User Profile Generation for Recommendation System 

**Title (ZH)**: LettinGo: 探索用户档案生成以优化推荐系统 

**Authors**: Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18309)  

**Abstract**: User profiling is pivotal for recommendation systems, as it transforms raw user interaction data into concise and structured representations that drive personalized recommendations. While traditional embedding-based profiles lack interpretability and adaptability, recent advances with large language models (LLMs) enable text-based profiles that are semantically richer and more transparent. However, existing methods often adhere to fixed formats that limit their ability to capture the full diversity of user behaviors. In this paper, we introduce LettinGo, a novel framework for generating diverse and adaptive user profiles. By leveraging the expressive power of LLMs and incorporating direct feedback from downstream recommendation tasks, our approach avoids the rigid constraints imposed by supervised fine-tuning (SFT). Instead, we employ Direct Preference Optimization (DPO) to align the profile generator with task-specific performance, ensuring that the profiles remain adaptive and effective. LettinGo operates in three stages: (1) exploring diverse user profiles via multiple LLMs, (2) evaluating profile quality based on their impact in recommendation systems, and (3) aligning the profile generation through pairwise preference data derived from task performance. Experimental results demonstrate that our framework significantly enhances recommendation accuracy, flexibility, and contextual awareness. This work enhances profile generation as a key innovation for next-generation recommendation systems. 

**Abstract (ZH)**: 用户画像生成对于推荐系统至关重要，因为它将原始的用户交互数据转化为简洁且结构化的表示，推动个性化推荐。虽然传统的基于嵌入的用户画像缺乏解释性和适应性，但近年来，大规模语言模型（LLMs）的发展使得基于文本的用户画像更加语义丰富且透明。然而，现有方法往往局限于固定格式，限制了它们捕获用户行为多样性的能力。本文介绍了LettinGo，一种生成多样且适应性强的用户画像的新框架。通过利用LLMs的强大表达能力并结合下游推荐任务的直接反馈，我们的方法避免了监督微调（SFT）施加的刚性约束。相反，我们采用了直接偏好优化（DPO）来使用户画像生成器与特定任务的性能对齐，确保用户画像保持适应性和有效性。LettinGo在三个阶段运作：（1）通过多个LLM探索多样化的用户画像；（2）基于其在推荐系统中的影响评价用户画像质量；（3）通过任务性能衍生的成对偏好数据调整用户画像生成。实验结果表明，该框架显著提高了推荐的准确性和灵活性，以及上下文意识。本工作增强了用户画像生成作为下一代推荐系统关键创新的价值。用户画像生成在下一代推荐系统中是一个关键创新。 

---
# ARD-LoRA: Dynamic Rank Allocation for Parameter-Efficient Fine-Tuning of Foundation Models with Heterogeneous Adaptation Needs 

**Title (ZH)**: ARD-LoRA: 不同适应需求下的参数高效微调的动态秩分配方法 

**Authors**: Haseeb Ullah Khan Shinwari, Muhammad Usama  

**Link**: [PDF](https://arxiv.org/pdf/2506.18267)  

**Abstract**: Conventional Low-Rank Adaptation (LoRA) methods employ a fixed rank, imposing uniform adaptation across transformer layers and attention heads despite their heterogeneous learning dynamics. This paper introduces Adaptive Rank Dynamic LoRA (ARD-LoRA), a novel framework that automates rank allocation through learnable scaling factors. These factors are optimized via a meta-objective balancing task performance and parameter efficiency, incorporating $\ell_1$ sparsity for minimal rank and Total Variation regularization for stable rank transitions. ARD-LoRA enables continuous, differentiable, per-head rank adaptation. Experiments on LLAMA-3.1-70B and PaliGemma-2 demonstrate ARD-LoRA's efficacy, achieving up to 99.3% of full fine-tuning performance with only 0.32% trainable parameters, outperforming strong baselines like DoRA and AdaLoRA. Furthermore, it reduces multimodal adaptation memory by 41%. These results establish dynamic, fine-grained rank allocation as a critical paradigm for efficient foundation model adaptation. 

**Abstract (ZH)**: 自适应秩动态LoRA (ARD-LoRA): 通过可学习的缩放因子自动分配秩 

---
# RLPR: Extrapolating RLVR to General Domains without Verifiers 

**Title (ZH)**: RLPR：将RLVR扩展到一般领域而无需验证器 

**Authors**: Tianyu Yu, Bo Ji, Shouli Wang, Shu Yao, Zefan Wang, Ganqu Cui, Lifan Yuan, Ning Ding, Yuan Yao, Zhiyuan Liu, Maosong Sun, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2506.18254)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) demonstrates promising potential in advancing the reasoning capabilities of LLMs. However, its success remains largely confined to mathematical and code domains. This primary limitation stems from the heavy reliance on domain-specific verifiers, which results in prohibitive complexity and limited scalability. To address the challenge, our key observation is that LLM's intrinsic probability of generating a correct free-form answer directly indicates its own evaluation of the reasoning reward (i.e., how well the reasoning process leads to the correct answer). Building on this insight, we propose RLPR, a simple verifier-free framework that extrapolates RLVR to broader general domains. RLPR uses the LLM's own token probability scores for reference answers as the reward signal and maximizes the expected reward during training. We find that addressing the high variance of this noisy probability reward is crucial to make it work, and propose prob-to-reward and stabilizing methods to ensure a precise and stable reward from LLM intrinsic probabilities. Comprehensive experiments in four general-domain benchmarks and three mathematical benchmarks show that RLPR consistently improves reasoning capabilities in both areas for Gemma, Llama, and Qwen based models. Notably, RLPR outperforms concurrent VeriFree by 7.6 points on TheoremQA and 7.5 points on Minerva, and even surpasses strong verifier-model-dependent approaches General-Reasoner by 1.6 average points across seven benchmarks. 

**Abstract (ZH)**: 可验证奖励的强化学习（RLVR）在提升大语言模型的推理能力方面显示出令人鼓舞的潜力，但其成功主要局限于数学和代码领域。这一主要限制源于对领域特定验证器的依赖，导致复杂性高且难以扩展。为应对这一挑战，我们的核心观察是，大语言模型生成正确自由形式答案的内在概率直接反映了其对推理奖励的自我评估（即推理过程如何导致正确答案）。基于这一洞察，我们提出了一个简单的无验证器框架RLPR，将RLVR扩展到更广泛的通用领域。RLPR使用大语言模型自身对参考答案的标记概率分数作为奖励信号，并在训练过程中最大化预期奖励。我们发现，解决这种噪声概率奖励的高方差是使其有效工作的关键，并提出了概率到奖励和稳定化方法，以确保从大语言模型内在概率中获得准确且稳定的奖励。在四个通用领域基准和三个数学基准的全面实验中，我们发现RLPR能够持续提高Gemma、Llama和Qwen等模型的推理能力。值得注意的是，RLPR在TheoremQA上比同时期的VeriFree高7.6分，在Minerva上高7.5分，并且在七个基准中平均高出1.6分超过了强验证器模型依赖的方法General-Reasoner。 

---
# Smart-LLaMA-DPO: Reinforced Large Language Model for Explainable Smart Contract Vulnerability Detection 

**Title (ZH)**: Smart-LLaMA-DPO：可解释的智能合约漏洞检测增强大型语言模型 

**Authors**: Lei Yu, Zhirong Huang, Hang Yuan, Shiqi Cheng, Li Yang, Fengjun Zhang, Chenjie Shen, Jiajia Ma, Jingyuan Zhang, Junyi Lu, Chun Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2506.18245)  

**Abstract**: Smart contract vulnerability detection remains a major challenge in blockchain security. Existing vulnerability detection methods face two main issues: (1) Existing datasets lack comprehensive coverage and high-quality explanations for preference learning. (2) Large language models (LLMs) often struggle with accurately interpreting specific concepts in smart contract security. Empirical analysis shows that even after continual pre-training (CPT) and supervised fine-tuning (SFT), LLMs may misinterpret the execution order of state changes, resulting in incorrect explanations despite making correct detection decisions. To address these challenges, we propose Smart-LLaMA-DPO based on LLaMA-3.1-8B. We construct a comprehensive dataset covering four major vulnerability types and machine-unauditable vulnerabilities, including precise labels, explanations, and locations for SFT, as well as high-quality and low-quality output pairs for Direct Preference Optimization (DPO). Second, we perform CPT using large-scale smart contract to enhance the LLM's understanding of specific security practices in smart contracts. Futhermore, we conduct SFT with our comprehensive dataset. Finally, we apply DPO, leveraging human feedback and a specially designed loss function that increases the probability of preferred explanations while reducing the likelihood of non-preferred outputs. We evaluate Smart-LLaMA-DPO on four major vulnerability types: reentrancy, timestamp dependence, integer overflow/underflow, and delegatecall, as well as machine-unauditable vulnerabilities. Our method significantly outperforms state-of-the-art baselines, with average improvements of 10.43% in F1 score and 7.87% in accuracy. Moreover, both LLM evaluation and human evaluation confirm that our method generates more correct, thorough, and clear explanations. 

**Abstract (ZH)**: 基于LLaMA-3.1-8B的Smart-LLaMA-DPO：智能合约漏洞检测的新方法 

---
# AdapThink: Adaptive Thinking Preferences for Reasoning Language Model 

**Title (ZH)**: AdapThink: 自适应思维偏好 reasoning 语言模型 

**Authors**: Xu Wan, Wei Wang, Wenyue Xu, Wotao Yin, Jie Song, Mingyang Sun  

**Link**: [PDF](https://arxiv.org/pdf/2506.18237)  

**Abstract**: Reinforcement Learning (RL)-based post-training has significantly advanced the complex reasoning capabilities of language models, fostering sophisticated self-reflection processes. However, this ``slow thinking'' paradigm presents a critical challenge to reasoning efficiency: models may expend excessive computation on simple questions and shift reasoning prematurely for complex ones. Previous mechanisms typically rely on static length budgets or predefined rules, lacking the adaptability for varying question complexities and models' evolving capabilities. To this end, we propose AdapThink, an adaptive post-training framework designed to induce more efficient thinking while maintaining the performance of reasoning language models. Specifically, AdapThink incorporates two key mechanisms: 1) A group-relative reward function that leverages model confidence and response's characteristic to dynamically adjust the preference of reflection-related transition words without resorting to a fixed length preference. 2) A diversity-aware sampling mechanism that balances the training group's solution accuracy with reasoning diversity via an entropy-guided score. Experiments on several mathematical reasoning datasets with DeepSeek-distilled models demonstrate AdapThink's advantages in enabling adaptive reasoning patterns and mitigating the inefficiencies. 

**Abstract (ZH)**: 基于强化学习的自适应后训练框架AdapThink显著提升了语言模型的复杂推理能力，促进精细的自我反思过程。然而，这种“缓慢思考”范式对推理效率提出了关键挑战：模型可能在简单问题上过度计算，并在复杂问题上过早推理。先前机制通常依赖于静态长度预算或预定义规则，缺乏适应不同问题复杂度和模型演进能力的能力。为此，我们提出AdapThink，一种设计用于引导更高效思考并保持推理语言模型性能的自适应后训练框架。具体而言，AdapThink包含两大关键机制：1）基于组相对奖励函数，利用模型自信度和响应特性动态调整与反思相关的转换词偏好，而不依赖于固定长度偏好。2）一种兼顾训练组解的准确性和推理多样性的多样性感知采样机制，通过熵引导评分进行平衡。实验结果显示，AdapThink在促进自适应推理模式和减轻效率损失方面具有优势。 

---
# Prompt Engineering Techniques for Mitigating Cultural Bias Against Arabs and Muslims in Large Language Models: A Systematic Review 

**Title (ZH)**: 针对大型语言模型中针对阿拉伯人和穆斯林的文化偏见缓解的提示工程技术系统综述 

**Authors**: Bushra Asseri, Estabrag Abdelaziz, Areej Al-Wabil  

**Link**: [PDF](https://arxiv.org/pdf/2506.18199)  

**Abstract**: Large language models have demonstrated remarkable capabilities across various domains, yet concerns about cultural bias - particularly towards Arabs and Muslims - pose significant ethical challenges by perpetuating harmful stereotypes and marginalization. Despite growing recognition of bias in LLMs, prompt engineering strategies specifically addressing Arab and Muslim representation remain understudied. This mixed-methods systematic review examines such techniques, offering evidence-based guidance for researchers and practitioners. Following PRISMA guidelines and Kitchenham's systematic review methodology, we analyzed 8 empirical studies published between 2021-2024 investigating bias mitigation strategies. Our findings reveal five primary prompt engineering approaches: cultural prompting, affective priming, self-debiasing techniques, structured multi-step pipelines, and parameter-optimized continuous prompts. Although all approaches show potential for reducing bias, effectiveness varied substantially across studies and bias types. Evidence suggests that certain bias types may be more resistant to prompt-based mitigation than others. Structured multi-step pipelines demonstrated the highest overall effectiveness, achieving up to 87.7% reduction in bias, though they require greater technical expertise. Cultural prompting offers broader accessibility with substantial effectiveness. These results underscore the accessibility of prompt engineering for mitigating cultural bias without requiring access to model parameters. The limited number of studies identified highlights a significant research gap in this critical area. Future research should focus on developing culturally adaptive prompting techniques, creating Arab and Muslim-specific evaluation resources, and integrating prompt engineering with complementary debiasing methods to address deeper stereotypes while maintaining model utility. 

**Abstract (ZH)**: 大型语言模型在各领域展现了令人瞩目的能力，但对其文化偏见的担忧——尤其是针对阿拉伯人和穆斯林——提出了重大的伦理挑战，这些偏见会导致有害的刻板印象和边缘化。尽管人们对大型语言模型中的偏见越来越认识到了，但专门针对阿拉伯人和穆斯林代表性问题的提示工程技术研究仍然不足。本混合法系统性回顾研究考察了此类技术，为研究者和实践者提供了基于证据的指导。遵循PRISMA指南和Kitchenham的系统性回顾方法，我们分析了2021-2024年间发表的8篇 empirical 研究，探讨了偏见缓解策略。研究结果揭示了五种主要的提示工程技术方法：文化提示、情感启动、自我去偏技术、结构化多步骤管道和参数优化连续提示。虽然所有方法都显示出减少偏见的潜力，但在不同研究和偏见类型中的有效性差异显著。证据表明，某些偏见类型可能比其他类型更难以通过提示技术缓解。结构化多步骤管道显示出最高的整体有效性，可降低高达87.7%的偏见，但这需要更高的技术专业知识。文化提示在提供广泛可及性的同时，显示出显著的有效性。这些结果强调了提示工程技术在不访问模型参数的情况下缓解文化偏见的可及性。确定的研究数量有限突显了这一关键领域存在显著的研究缺口。未来的研究应侧重于开发文化适应性提示技术、创建阿拉伯人和穆斯林特定的评估资源，并将提示工程技术与互补的去偏技术集成，以应对更深层次的刻板印象，同时保持模型的实用性。 

---
# Understanding Reasoning in Thinking Language Models via Steering Vectors 

**Title (ZH)**: 通过导向向量理解思考语言模型中的推理 

**Authors**: Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, Neel Nanda  

**Link**: [PDF](https://arxiv.org/pdf/2506.18167)  

**Abstract**: Recent advances in large language models (LLMs) have led to the development of thinking language models that generate extensive internal reasoning chains before producing responses. While these models achieve improved performance, controlling their reasoning processes remains challenging. This work presents a steering approach for thinking LLMs by analyzing and manipulating specific reasoning behaviors in DeepSeek-R1-Distill models. Through a systematic experiment on 500 tasks across 10 diverse categories, we identify several reasoning behaviors exhibited by thinking models, including expressing uncertainty, generating examples for hypothesis validation, and backtracking in reasoning chains. We demonstrate that these behaviors are mediated by linear directions in the model's activation space and can be controlled using steering vectors. By extracting and applying these vectors, we provide a method to modulate specific aspects of the model's reasoning process, such as its tendency to backtrack or express uncertainty. Our approach offers practical tools for steering reasoning processes in thinking models in a controlled and interpretable manner. We validate our steering method using two DeepSeek-R1-Distill models, demonstrating consistent control across different model architectures. 

**Abstract (ZH)**: Recent advances in大型语言模型（LLMs）推动了思考型LLMs的发展，这些模型在生成广泛内部推理链后才产生响应。尽管这些模型在性能上有所提升，但控制其推理过程仍然颇具挑战。本文通过分析和操控DeepSeek-R1-Distill模型中的特定推理行为，提出了一种引导方法。通过在10个不同类别下的500个任务上进行系统的实验，我们识别出思考模型表现出的几种推理行为，包括表达不确定性、为假设验证生成例子以及推理链中的回溯。我们证明这些行为由模型激活空间中的线性方向介导，并可通过引导向量进行控制。通过提取和应用这些向量，我们提供了一种调节模型推理过程特定方面的办法，例如回溯倾向或不确定性表达。我们的方法提供了在受控和可解释的方式下引导思考型模型推理过程的实用工具。我们使用两个DeepSeek-R1-Distill模型验证了我们的引导方法，展示了不同模型架构之间的一致控制能力。 

---
# Sparse Feature Coactivation Reveals Composable Semantic Modules in Large Language Models 

**Title (ZH)**: 稀疏特征共激活揭示大规模语言模型中的可组合语义模块 

**Authors**: Ruixuan Deng, Xiaoyang Hu, Miles Gilberti, Shane Storks, Aman Taxali, Mike Angstadt, Chandra Sripada, Joyce Chai  

**Link**: [PDF](https://arxiv.org/pdf/2506.18141)  

**Abstract**: We identify semantically coherent, context-consistent network components in large language models (LLMs) using coactivation of sparse autoencoder (SAE) features collected from just a handful of prompts. Focusing on country-relation tasks, we show that ablating semantic components for countries and relations changes model outputs in predictable ways, while amplifying these components induces counterfactual responses. Notably, composing relation and country components yields compound counterfactual outputs. We find that, whereas most country components emerge from the very first layer, the more abstract relation components are concentrated in later layers. Furthermore, within relation components themselves, nodes from later layers tend to have a stronger causal impact on model outputs. Overall, these findings suggest a modular organization of knowledge within LLMs and advance methods for efficient, targeted model manipulation. 

**Abstract (ZH)**: 我们使用稀疏自编码器（SAE）特征的共激活，在大型语言模型（LLMs）中识别出语义一致且上下文一致的网络组件。聚焦于国家关系任务，我们发现移除国家和关系的语义组件会导致可预测的模型输出变化，而放大这些组件则会导致反事实响应。值得注意的是，关系和国家组件的组合会产生复合的反事实输出。我们发现，大多数国家组件源自模型的第一层，而更为抽象的关系组件则集中在较晚的层中。此外，在关系组件内部，较晚层的节点对模型输出的影响更强。总体而言，这些发现表明LLMs内部知识的模块化组织，并促进了高效、目标导向的模型操控方法的发展。 

---
# $ϕ^{\infty}$: Clause Purification, Embedding Realignment, and the Total Suppression of the Em Dash in Autoregressive Language Models 

**Title (ZH)**: $ϕ^{\infty}$: 子句净化、嵌入重新对齐及自回归语言模型中破折号的完全抑制 

**Authors**: Bugra Kilictas, Faruk Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2506.18129)  

**Abstract**: We identify a critical vulnerability in autoregressive transformer language models where the em dash token induces recursive semantic drift, leading to clause boundary hallucination and embedding space entanglement. Through formal analysis of token-level perturbations in semantic lattices, we demonstrate that em dash insertion fundamentally alters the model's latent representations, causing compounding errors in long-form generation. We propose a novel solution combining symbolic clause purification via the phi-infinity operator with targeted embedding matrix realignment. Our approach enables total suppression of problematic tokens without requiring model retraining, while preserving semantic coherence through fixed-point convergence guarantees. Experimental validation shows significant improvements in generation consistency and topic maintenance. This work establishes a general framework for identifying and mitigating token-level vulnerabilities in foundation models, with immediate implications for AI safety, model alignment, and robust deployment of large language models in production environments. The methodology extends beyond punctuation to address broader classes of recursive instabilities in neural text generation systems. 

**Abstract (ZH)**: 我们在自回归变压器语言模型中识别出一个关键漏洞，其中破折号标记引发递归语义漂移，导致从句边界幻觉和嵌入空间纠缠。通过对语义格中token级扰动的正式分析，我们证明破折号插入从根本上改变了模型的潜在表示，导致长文本生成中的累积错误。我们提出了一个结合φ-∞算子进行符号从句净化和目标嵌入矩阵重对齐的新型解决方案。我们的方法能够在无需模型重新训练的情况下完全抑制问题token，同时通过定点收敛保证维持语义连贯性。实验验证显示生成一致性与话题维持方面的显著改进。本研究建立了识别和缓解基础模型中token级漏洞的一般框架，对AI安全性、模型对齐及大规模语言模型在生产环境中的鲁棒部署具有即时影响。该方法超越标点符号，用以解决神经文本生成系统中更广泛的递归不稳定性问题。 

---
# Mental Health Equity in LLMs: Leveraging Multi-Hop Question Answering to Detect Amplified and Silenced Perspectives 

**Title (ZH)**: LLMs中的心理健康公平性：利用多跳问答检测被放大和沉默的观点 

**Authors**: Batool Haider, Atmika Gorti, Aman Chadha, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2506.18116)  

**Abstract**: Large Language Models (LLMs) in mental healthcare risk propagating biases that reinforce stigma and harm marginalized groups. While previous research identified concerning trends, systematic methods for detecting intersectional biases remain limited. This work introduces a multi-hop question answering (MHQA) framework to explore LLM response biases in mental health discourse. We analyze content from the Interpretable Mental Health Instruction (IMHI) dataset across symptom presentation, coping mechanisms, and treatment approaches. Using systematic tagging across age, race, gender, and socioeconomic status, we investigate bias patterns at demographic intersections. We evaluate four LLMs: Claude 3.5 Sonnet, Jamba 1.6, Gemma 3, and Llama 4, revealing systematic disparities across sentiment, demographics, and mental health conditions. Our MHQA approach demonstrates superior detection compared to conventional methods, identifying amplification points where biases magnify through sequential reasoning. We implement two debiasing techniques: Roleplay Simulation and Explicit Bias Reduction, achieving 66-94% bias reductions through few-shot prompting with BBQ dataset examples. These findings highlight critical areas where LLMs reproduce mental healthcare biases, providing actionable insights for equitable AI development. 

**Abstract (ZH)**: 大型语言模型（LLMs）在心理健康护理中的应用可能会传播强化刻板印象并伤害边缘化群体的偏见。虽然先前研究识别了令人担忧的趋势，但系统的交叉偏见检测方法仍有限。本研究引入多跳问答（MHQA）框架以探索LLM在心理健康对话中的响应偏见。我们分析了可解释心理健康指令（IMHI）数据集中的症状表现、应对机制和治疗方法内容。通过系统标记年龄、种族、性别和社会经济地位，我们研究了人口统计交叉点的偏见模式。我们评估了四种LLM：Claude 3.5 Sonnet、Jamba 1.6、Gemma 3和Llama 4，揭示了情感、人口统计和心理健康状况方面的系统性差异。我们的MHQA方法在偏见检测方面显示出优于传统方法的能力，识别了通过序列推理放大偏见的点。我们实施了两种去偏方法：角色扮演模拟和显式偏见减少，通过使用BBQ数据集示例的少量提示实现了66-94%的偏见减少。这些发现揭示了LLM在心理健康护理偏见再现方面的关键领域，为公平的人工智能开发提供了可操作的见解。 

---
# Federated Learning-Based Data Collaboration Method for Enhancing Edge Cloud AI System Security Using Large Language Models 

**Title (ZH)**: 基于联邦学习的数据协作方法：增强边缘云AI系统安全性以利用大型语言模型 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2506.18087)  

**Abstract**: With the widespread application of edge computing and cloud systems in AI-driven applications, how to maintain efficient performance while ensuring data privacy has become an urgent security issue. This paper proposes a federated learning-based data collaboration method to improve the security of edge cloud AI systems, and use large-scale language models (LLMs) to enhance data privacy protection and system robustness. Based on the existing federated learning framework, this method introduces a secure multi-party computation protocol, which optimizes the data aggregation and encryption process between distributed nodes by using LLM to ensure data privacy and improve system efficiency. By combining advanced adversarial training techniques, the model enhances the resistance of edge cloud AI systems to security threats such as data leakage and model poisoning. Experimental results show that the proposed method is 15% better than the traditional federated learning method in terms of data protection and model robustness. 

**Abstract (ZH)**: 基于联邦学习的数据协作方法在AI驱动的边缘云计算系统中提升数据安全性和系统健壮性 

---
# Mechanistic Interpretability in the Presence of Architectural Obfuscation 

**Title (ZH)**: Architecture-Obfuscation Present Mechanistic Interpretability 

**Authors**: Marcos Florencio, Thomas Barton  

**Link**: [PDF](https://arxiv.org/pdf/2506.18053)  

**Abstract**: Architectural obfuscation - e.g., permuting hidden-state tensors, linearly transforming embedding tables, or remapping tokens - has recently gained traction as a lightweight substitute for heavyweight cryptography in privacy-preserving large-language-model (LLM) inference. While recent work has shown that these techniques can be broken under dedicated reconstruction attacks, their impact on mechanistic interpretability has not been systematically studied. In particular, it remains unclear whether scrambling a network's internal representations truly thwarts efforts to understand how the model works, or simply relocates the same circuits to an unfamiliar coordinate system. We address this gap by analyzing a GPT-2-small model trained from scratch with a representative obfuscation map. Assuming the obfuscation map is private and the original basis is hidden (mirroring an honest-but-curious server), we apply logit-lens attribution, causal path-patching, and attention-head ablation to locate and manipulate known circuits. Our findings reveal that obfuscation dramatically alters activation patterns within attention heads yet preserves the layer-wise computational graph. This disconnect hampers reverse-engineering of user prompts: causal traces lose their alignment with baseline semantics, and token-level logit attributions become too noisy to reconstruct. At the same time, feed-forward and residual pathways remain functionally intact, suggesting that obfuscation degrades fine-grained interpretability without compromising top-level task performance. These results establish quantitative evidence that architectural obfuscation can simultaneously (i) retain global model behaviour and (ii) impede mechanistic analyses of user-specific content. By mapping where interpretability breaks down, our study provides guidance for future privacy defences and for robustness-aware interpretability tooling. 

**Abstract (ZH)**: 建筑混淆 – 例如，打乱隐藏状态张量、线性变换嵌入表，或重映射标记 – 最近被用作大型语言模型（LLM）推理中 heavyweight 摘要隐私保护的轻量化替代方案。尽管近期研究显示这些技术在专门的重建攻击下可被破解，但它们对机制可解释性的影响尚未系统研究。特别是，尚不清楚扰乱网络的内部表示是否会真正阻碍对模型工作原理的理解，还是仅将相同的电路重新定位到一个不熟悉的坐标系统中。我们通过分析一个从头训练且带有代表性混淆映射的 GPT-2-small 模型来弥补这一空白。假设混淆映射是私有的且原本基底是隐藏的（类似于诚实但好奇的服务器），我们应用 logits 镜像归因、因果路径修补和注意力头消融来定位和操作已知电路。我们的发现显示，混淆极大地改变了注意力头内的激活模式，但保留了逐层计算图。这种分离阻碍了用户提示的反向工程：因果轨迹与基线语义失去了对齐，且标记级别的 logits 归因变得太嘈杂而无法重建。同时，前馈和残差路径仍然保持功能完好，表明混淆降低了细粒度可解释性，但未损害高级任务性能。这些结果建立了一定量证据，表明从结构混淆可以在 (i) 保留全局模型行为的同时 (ii) 阻碍用户特定内容的机制分析。通过映射可解释性失效的位置，我们的研究为未来的隐私防御和鲁棒性感知可解释性工具提供指导。 

---
# The Democratic Paradox in Large Language Models' Underestimation of Press Freedom 

**Title (ZH)**: 大型语言模型对新闻自由低估的民主悖论 

**Authors**: I. Loaiza, R. Vestrelli, A. Fronzetti Colladon, R. Rigobon  

**Link**: [PDF](https://arxiv.org/pdf/2506.18045)  

**Abstract**: As Large Language Models (LLMs) increasingly mediate global information access for millions of users worldwide, their alignment and biases have the potential to shape public understanding and trust in fundamental democratic institutions, such as press freedom. In this study, we uncover three systematic distortions in the way six popular LLMs evaluate press freedom in 180 countries compared to expert assessments of the World Press Freedom Index (WPFI). The six LLMs exhibit a negative misalignment, consistently underestimating press freedom, with individual models rating between 71% to 93% of countries as less free. We also identify a paradoxical pattern we term differential misalignment: LLMs disproportionately underestimate press freedom in countries where it is strongest. Additionally, five of the six LLMs exhibit positive home bias, rating their home countries' press freedoms more favorably than would be expected given their negative misalignment with the human benchmark. In some cases, LLMs rate their home countries between 7% to 260% more positively than expected. If LLMs are set to become the next search engines and some of the most important cultural tools of our time, they must ensure accurate representations of the state of our human and civic rights globally. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）日益成为全球数百万用户获取信息的中介，它们的对齐情况和偏差有可能影响公众对新闻自由等基本民主机构的理解和信任。在本研究中，我们发现了六种流行LLM在评估180个国家的新闻自由时与世界新闻自由指数（WPFI）专家评估之间存在三种系统性失真。这六种LLM表现出负面的对齐偏差，一致地低估了新闻自由，各模型分别将71%到93%的国家评为较为不自由。我们还发现了一种我们称之为差异性失真的悖论模式：LLM在评估新闻自由最强的国家时显著低估了新闻自由。此外，六种LLM中有五种表现出向本国偏爱的正偏差，对其本国的新闻自由给予了比其负面的对齐偏差与人类基准相比更高的评价。在某些情况下，这些LLM对其本国的新闻自由的评价比预期高出7%到260%。如果LLM被设定成为下一代搜索引擎和我们这个时代最重要的文化工具之一，它们必须确保对全球人类和公民权利现状的准确呈现。 

---
# Pre-Trained LLM is a Semantic-Aware and Generalizable Segmentation Booster 

**Title (ZH)**: 预训练大语言模型是一个具备语义意识和泛化能力的分割增强器 

**Authors**: Fenghe Tang, Wenxin Ma, Zhiyang He, Xiaodong Tao, Zihang Jiang, S. Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18034)  

**Abstract**: With the advancement of Large Language Model (LLM) for natural language processing, this paper presents an intriguing finding: a frozen pre-trained LLM layer can process visual tokens for medical image segmentation tasks. Specifically, we propose a simple hybrid structure that integrates a pre-trained, frozen LLM layer within the CNN encoder-decoder segmentation framework (LLM4Seg). Surprisingly, this design improves segmentation performance with a minimal increase in trainable parameters across various modalities, including ultrasound, dermoscopy, polypscopy, and CT scans. Our in-depth analysis reveals the potential of transferring LLM's semantic awareness to enhance segmentation tasks, offering both improved global understanding and better local modeling capabilities. The improvement proves robust across different LLMs, validated using LLaMA and DeepSeek. 

**Abstract (ZH)**: 随着大型语言模型（LLM）在自然语言处理领域的进步，本论文呈现了一个有趣的发现：冻结的预训练LLM层可以处理医学图像分割任务中的视觉令牌。具体来说，我们提出了一种简单的混合结构，将预训练并冻结的LLM层集成到CNN编码-解码分割框架中（LLM4Seg）。令人惊讶的是，这种设计在各种成像模态（包括超声、皮肤镜检查、息肉检查和CT扫描）中，能以微量增加可训练参数的方式提升分割性能。深入分析表明，可以通过转移LLM的语义意识来增强分割任务，提供更好的全局理解和局部建模能力。这种改进在不同的LLM模型中都表现出鲁棒性，通过LLaMA和DeepSeek进行验证。 

---
# Scatter-Based Innovation Propagation in Large Language Models for Multi-Stage Process Adaptation 

**Title (ZH)**: 基于散射的创新传播在大型语言模型中的多阶段过程适应中 

**Authors**: Hong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17949)  

**Abstract**: Large Language Models (LLMs) exhibit strong capabilities in reproducing and extending patterns observed during pretraining but often struggle to generalize novel ideas beyond their original context. This paper addresses the challenge of applying such localized innovations - introduced at a specific stage or component - to other parts of a multi-stage process. We propose a scatter-based innovation expansion model (innovation scatter model) that guides the LLM through a four-step process: (1) identifying the core innovation by comparing the user's input with its surrounding context, (2) generalizing the innovation by removing references to specific stages or components, (3) determining whether the generalized innovation applies to a broader scope beyond the original stage, and (4) systematically applying it to other structurally similar stages using the LLM. This model leverages structural redundancy across stages to improve the applicability of novel ideas. Verification results demonstrate that the innovation scatter model enables LLMs to extend innovations across structurally similar stages, thereby enhancing generalization and reuse. 

**Abstract (ZH)**: 大型语言模型(Large Language Models)在再现和扩展预训练中观察到的模式方面表现出强大的能力，但在将新颖想法推广应用到原有上下文之外的其他部分时往往存在困难。本文解决了将特定阶段或组件引入的局部创新应用于多阶段过程中其他部分的挑战。我们提出了一种基于散射的创新扩展模型（创新散射模型），该模型指导大型语言模型通过四个步骤进行：（1）通过将用户输入与其上下文进行比较来识别核心创新，（2）通过移除对特定阶段或组件的引用来进行创新的泛化，（3）确定泛化的创新是否适用于原始阶段之外的更广泛范围，（4）使用大型语言模型有系统地将其应用于其他结构上类似的阶段。该模型利用阶段间的结构冗余性以提高新颖想法的应用性。验证结果表明，创新散射模型使大型语言模型能够将创新扩展到结构上相似的阶段，从而提高泛化能力和重用性。 

---
# Multi-turn Jailbreaking via Global Refinement and Active Fabrication 

**Title (ZH)**: 全局细化与主动伪造驱动的多轮 Jailbreaking 

**Authors**: Hua Tang, Lingyong Yan, Yukun Zhao, Shuaiqiang Wang, Jizhou Huang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2506.17881)  

**Abstract**: Large Language Models (LLMs) have achieved exceptional performance across a wide range of tasks. However, they still pose significant safety risks due to the potential misuse for malicious purposes. Jailbreaks, which aim to elicit models to generate harmful content, play a critical role in identifying the underlying security threats. Recent jailbreaking primarily focuses on single-turn scenarios, while the more complicated multi-turn scenarios remain underexplored. Moreover, existing multi-turn jailbreaking techniques struggle to adapt to the evolving dynamics of dialogue as the interaction progresses. To address this limitation, we propose a novel multi-turn jailbreaking method that refines the jailbreaking path globally at each interaction. We also actively fabricate model responses to suppress safety-related warnings, thereby increasing the likelihood of eliciting harmful outputs in subsequent questions. Experimental results demonstrate the superior performance of our method compared with existing single-turn and multi-turn jailbreaking techniques across six state-of-the-art LLMs. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各类任务中取得了出色的表现，但在恶意利用的潜在风险下仍然存在重大的安全问题。攻击性 jailbreaks 在揭示潜在安全威胁方面起着关键作用。尽管当前主要关注单轮对话攻击，但更复杂的多轮对话攻击场景尚未得到充分探索。此外，现有技术难以适应对话过程中不断变化的交互动态。为解决这一限制，我们提出了一种新颖的多轮对话 jailbreak 方法，在每次交互中全局优化 jailbreak 路径。我们还积极伪造模型响应以抑制与安全性相关的警告，从而增加在后续问题中获得有害输出的可能性。实验结果表明，我们的方法在六个最先进的 LLM 上优于现有的单轮和多轮对话攻击方法。我们的代码已公开，网址为：this https URL。 

---
# How Alignment Shrinks the Generative Horizon 

**Title (ZH)**: 如何对齐缩小生成视野 

**Authors**: Chenghao Yang, Ari Holtzman  

**Link**: [PDF](https://arxiv.org/pdf/2506.17871)  

**Abstract**: Despite their impressive capabilities, aligned large language models (LLMs) often generate outputs that lack diversity. What drives this stability in the generation? We investigate this phenomenon through the lens of probability concentration in the model's output distribution. To quantify this concentration, we introduce the Branching Factor (BF) -- a token-invariant measure of the effective number of plausible next steps during generation. Our empirical analysis reveals two key findings: (1) BF often decreases as generation progresses, suggesting that LLMs become more predictable as they generate. (2) alignment tuning substantially sharpens the model's output distribution from the outset, reducing BF by nearly an order of magnitude (e.g., from 12 to 1.2) relative to base models. This stark reduction helps explain why aligned models often appear less sensitive to decoding strategies. Building on this insight, we find this stability has surprising implications for complex reasoning. Aligned Chain-of-Thought (CoT) models (e.g., DeepSeek-distilled models), for instance, leverage this effect; by generating longer reasoning chains, they push generation into later, more deterministic (lower BF) stages, resulting in more stable outputs. We hypothesize that alignment tuning does not fundamentally change a model's behavior, but instead steers it toward stylistic tokens (e.g., "Sure") that unlock low-entropy trajectories already present in the base model. This view is supported by nudging experiments, which show that prompting base models with such tokens can similarly reduce BF. Together, our findings establish BF as a powerful diagnostic for understanding and controlling LLM outputs - clarifying how alignment reduces variability, how CoT promotes stable generations, and how base models can be steered away from diversity. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具备强大能力，但它们生成的输出往往缺乏多样性。这种稳定性是如何驱动生成过程的？我们通过模型输出分布的概率集中性这一视角来探讨这一现象。为了量化这种集中性，我们引入了分叉因子（Branching Factor，BF）——一个不受标记影响的有效可能下一步的数量度量。我们的实验证据揭示了两个关键发现：（1）BF在生成过程中往往降低，表明LLMs在生成时变得更加可预测。（2）对齐调整从一开始就显著使模型的输出分布变得集中，相对于基础模型，BF几乎减少了十倍（例如，从12降至1.2）。这种显著减少解释了为什么对齐后的模型往往对外推策略的敏感性较低。基于这一见解，我们发现这种稳定性对复杂推理产生了意想不到的影响。例如，对齐后的链式思考（CoT）模型（如DeepSeek提炼模型）利用了这一效果，通过生成更长的推理链，将生成过程推向后期、更确定（BF更低）的阶段，从而产生更稳定的输出。我们推测对齐调整并未根本改变模型的行为，而是引导模型朝向风格化标记（如“当然”）前进，这些标记在基础模型中已经存在低熵轨迹。这一观点通过推动实验得到了支持，这些实验表明，用这些标记提示基础模型也可以同样减少BF。我们的发现共同确立了BF作为理解和控制LLM输出的强大诊断工具——阐明对齐如何减少变异性，CoT如何促进稳定生成，以及如何引导基础模型远离多样性。 

---
# Aligning Frozen LLMs by Reinforcement Learning: An Iterative Reweight-then-Optimize Approach 

**Title (ZH)**: 通过强化学习迭代重加权与优化方法对冻结的LLM进行对齐 

**Authors**: Xinnan Zhang, Chenliang Li, Siliang Zeng, Jiaxiang Li, Zhongruo Wang, Kaixiang Lin, Songtao Lu, Alfredo Garcia, Mingyi Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.17828)  

**Abstract**: Aligning large language models (LLMs) with human preferences usually requires fine-tuning methods such as RLHF and DPO. These methods directly optimize the model parameters, so they cannot be used in test-time to improve model performance, nor are they applicable when the model weights are not accessible. In contrast, test-time methods sidestep weight updates by leveraging reward functions to guide and improve output quality. However, they incur high inference costs, and their one-shot guidance is often based on imperfect reward or value functions, leading to suboptimal outputs. In this work, we present a method named Iterative Reweight-then-Optimize (IRO), a reinforcement learning (RL) framework that performs RL-style alignment of the (frozen) base model without touching its parameters. During training, each iteration (i) samples candidates from the base model, (ii) resamples using current value functions, and (iii) trains a new lightweight value function that guides the next decoding pass. At test time, the value functions are used to guide the base model generation via a search-based optimization process. Notably, users can apply IRO to align a model on their own dataset, similar to OpenAI's reinforcement fine-tuning (RFT), but without requiring access to the model weights. 

**Abstract (ZH)**: Iterative Reweight-then-Optimize (IRO): A Reinforcement Learning Framework for aligning Large Language Models with Human Preferences without Access to Model Weights 

---
# CARTS: Collaborative Agents for Recommendation Textual Summarization 

**Title (ZH)**: CARTS: 联合代理的推荐文本总结 

**Authors**: Jiao Chen, Kehui Yao, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17765)  

**Abstract**: Current recommendation systems often require some form of textual data summarization, such as generating concise and coherent titles for product carousels or other grouped item displays. While large language models have shown promise in NLP domains for textual summarization, these approaches do not directly apply to recommendation systems, where explanations must be highly relevant to the core features of item sets, adhere to strict word limit constraints. In this paper, we propose CARTS (Collaborative Agents for Recommendation Textual Summarization), a multi-agent LLM framework designed for structured summarization in recommendation systems. CARTS decomposes the task into three stages-Generation Augmented Generation (GAG), refinement circle, and arbitration, where successive agent roles are responsible for extracting salient item features, iteratively refining candidate titles based on relevance and length feedback, and selecting the final title through a collaborative arbitration process. Experiments on large-scale e-commerce data and live A/B testing show that CARTS significantly outperforms single-pass and chain-of-thought LLM baselines, delivering higher title relevance and improved user engagement metrics. 

**Abstract (ZH)**: 当前推荐系统往往需要某种形式的文本数据总结，比如为产品轮播或其他分组项显示生成精炼且连贯的标题。尽管大规模语言模型在自然语言处理领域展示了在文本总结方面的潜力，但在推荐系统中，这些方法并不直接适用，因为解释必须高度相关于项目集的核心特征，并严格遵守字数限制。在本文中，我们提出了CARTS（Collaborative Agents for Recommendation Textual Summarization），一个为推荐系统设计的多智能体LLM框架，用于结构化的总结。CARTS将任务分解为三个阶段：增强生成（GAG）、完善循环和仲裁，其中相继的角色负责提取关键项目特征、基于相关性和长度反馈迭代完善候选标题，并通过协作仲裁过程选择最终标题。在大规模电子商务数据上的实验和实时A/B测试显示，CARTS显著优于单次通过和思维链LLM基线，提供了更高的标题相关性和改进的用户参与度指标。 

---
# HIDE and Seek: Detecting Hallucinations in Language Models via Decoupled Representations 

**Title (ZH)**: 隐藏与寻找：通过解耦表示检测语言模型中的幻觉 

**Authors**: Anwoy Chatterjee, Yash Goel, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2506.17748)  

**Abstract**: Contemporary Language Models (LMs), while impressively fluent, often generate content that is factually incorrect or unfaithful to the input context - a critical issue commonly referred to as 'hallucination'. This tendency of LMs to generate hallucinated content undermines their reliability, especially because these fabrications are often highly convincing and therefore difficult to detect. While several existing methods attempt to detect hallucinations, most rely on analyzing multiple generations per input, leading to increased computational cost and latency. To address this, we propose a single-pass, training-free approach for effective Hallucination detectIon via Decoupled rEpresentations (HIDE). Our approach leverages the hypothesis that hallucinations result from a statistical decoupling between an LM's internal representations of input context and its generated output. We quantify this decoupling using the Hilbert-Schmidt Independence Criterion (HSIC) applied to hidden-state representations extracted while generating the output sequence. We conduct extensive experiments on four diverse question answering datasets, evaluating both faithfulness and factuality hallucinations across six open-source LMs of varying scales and properties. Our results demonstrate that HIDE outperforms other single-pass methods in almost all settings, achieving an average relative improvement of ~29% in AUC-ROC over the best-performing single-pass strategy across various models and datasets. Additionally, HIDE shows competitive and often superior performance with multi-pass state-of-the-art methods, obtaining an average relative improvement of ~3% in AUC-ROC while consuming ~51% less computation time. Our findings highlight the effectiveness of exploiting internal representation decoupling in LMs for efficient and practical hallucination detection. 

**Abstract (ZH)**: 当前的语言模型虽然表现出色，但在生成内容时往往会出现事实错误或与输入上下文不一致的情况——这一问题通常被称为“幻觉”。我们提出了一种基于解耦表示的有效单步幻觉检测方法HIDE。我们的方法基于这样一个假设：幻觉源于语言模型内部对输入上下文和生成输出之间的统计解耦。我们使用希尔伯特-施密特独立性判别（HSIC）来量化生成输出序列时提取的隐藏态表示之间的解耦程度。我们在四个不同的问答数据集上进行了广泛实验，评估了六个规模和属性不同的开源语言模型在忠实性和事实性幻觉检测方面的表现。实验结果表明，HIDE在几乎所有场景下的性能都优于其他单步方法，平均ROC-AUC提高了约29%。此外，HIDE在平均ROC-AUC上的改进幅度接近3%，同时计算时间减少了约51%，与多步最新方法竞争并表现出更优性能。我们的研究结果强调了利用语言模型内部表示解耦进行高效和实用的幻觉检测的有效性。 

---
# KAG-Thinker: Teaching Large Language Models to Think with Human-like Reasoning Process 

**Title (ZH)**: KAG-Thinker: 教授大型语言模型使用类似-human推理过程进行思考 

**Authors**: Dalong Zhang, Jun Xu, Jun Zhou, Lei Liang, Lin Yuan, Ling Zhong, Mengshu Sun, Peilong Zhao, QiWei Wang, Xiaorui Wang, Xinkai Du, YangYang Hou, Yu Ao, ZhaoYang Wang, Zhengke Gui, ZhiYing Yi, Zhongpu Bo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17728)  

**Abstract**: In this paper, we introduce KAG-Thinker, a novel human-like reasoning framework built upon a parameter-light large language model (LLM). Our approach enhances the logical coherence and contextual consistency of the thinking process in question-answering (Q\&A) tasks on domain-specific knowledge bases (KBs) within LLMs. This framework simulates human cognitive mechanisms for handling complex problems by establishing a structured thinking process. Continuing the \textbf{Logical Form} guided retrieval and reasoning technology route of KAG v0.7, firstly, it decomposes complex questions into independently solvable sub-problems(also referred to as logical forms) through \textbf{breadth decomposition}, each represented in two equivalent forms-natural language and logical function-and further classified as either Knowledge Retrieval or Reasoning Analysis tasks, with dependencies and variables passing explicitly modeled via logical function interfaces. In the solving process, the Retrieval function is used to perform knowledge retrieval tasks, while the Math and Deduce functions are used to perform reasoning analysis tasks. Secondly, it is worth noting that, in the Knowledge Retrieval sub-problem tasks, LLMs and external knowledge sources are regarded as equivalent KBs. We use the \textbf{knowledge boundary} model to determine the optimal source using self-regulatory mechanisms such as confidence calibration and reflective reasoning, and use the \textbf{depth solving} model to enhance the comprehensiveness of knowledge acquisition. Finally, instead of utilizing reinforcement learning, we employ supervised fine-tuning with multi-turn dialogues to align the model with our structured inference paradigm, thereby avoiding excessive reflection. This is supported by a data evaluation framework and iterative corpus synthesis, which facilitate the generation of detailed reasoning trajectories... 

**Abstract (ZH)**: 基于参数轻量大型语言模型的人类级推理框架KAG-Thinker：在领域特定知识库中的问题解答逻辑推理研究 

---
# Programmable-Room: Interactive Textured 3D Room Meshes Generation Empowered by Large Language Models 

**Title (ZH)**: 可编程房间：由大规模语言模型赋能的交互式纹理化3D房间网格生成 

**Authors**: Jihyun Kim, Junho Park, Kyeongbo Kong, Suk-Ju Kang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17707)  

**Abstract**: We present Programmable-Room, a framework which interactively generates and edits a 3D room mesh, given natural language instructions. For precise control of a room's each attribute, we decompose the challenging task into simpler steps such as creating plausible 3D coordinates for room meshes, generating panorama images for the texture, constructing 3D meshes by integrating the coordinates and panorama texture images, and arranging furniture. To support the various decomposed tasks with a unified framework, we incorporate visual programming (VP). VP is a method that utilizes a large language model (LLM) to write a Python-like program which is an ordered list of necessary modules for the various tasks given in natural language. We develop most of the modules. Especially, for the texture generating module, we utilize a pretrained large-scale diffusion model to generate panorama images conditioned on text and visual prompts (i.e., layout, depth, and semantic map) simultaneously. Specifically, we enhance the panorama image generation quality by optimizing the training objective with a 1D representation of a panorama scene obtained from bidirectional LSTM. We demonstrate Programmable-Room's flexibility in generating and editing 3D room meshes, and prove our framework's superiority to an existing model quantitatively and qualitatively. Project page is available in this https URL. 

**Abstract (ZH)**: Programmable-Room：一种基于自然语言指令的交互式3D房间网格生成与编辑框架 

---
# The Evolution of Natural Language Processing: How Prompt Optimization and Language Models are Shaping the Future 

**Title (ZH)**: 自然语言处理的发展：提示优化与语言模型如何塑造未来 

**Authors**: Summra Saleem, Muhammad Nabeel Asim, Shaista Zulfiqar, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2506.17700)  

**Abstract**: Large Language Models (LLMs) have revolutionized the field of Natural Language Processing (NLP) by automating traditional labor-intensive tasks and consequently accelerated the development of computer-aided applications. As researchers continue to advance this field with the introduction of novel language models and more efficient training/finetuning methodologies, the idea of prompt engineering and subsequent optimization strategies with LLMs has emerged as a particularly impactful trend to yield a substantial performance boost across diverse NLP tasks. To best of our knowledge numerous review articles have explored prompt engineering, however, a critical gap exists in comprehensive analyses of prompt optimization strategies. To bridge this gap this paper provides unique and comprehensive insights about the potential of diverse prompt optimization strategies. It analyzes their underlying working paradigms and based on these principles, categorizes them into 11 distinct classes. Moreover, the paper provides details about various NLP tasks where these prompt optimization strategies have been employed, along with details of different LLMs and benchmark datasets used for evaluation. This comprehensive compilation lays a robust foundation for future comparative studies and enables rigorous assessment of prompt optimization and LLM-based predictive pipelines under consistent experimental settings: a critical need in the current landscape. Ultimately, this research will centralize diverse strategic knowledge to facilitate the adaptation of existing prompt optimization strategies for development of innovative predictors across unexplored tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过自动化传统劳动密集型任务已彻底革新了自然语言处理（NLP）领域，并加速了计算机辅助应用的发展。随着研究人员通过引入新型语言模型和更高效的训练/微调方法继续推进这一领域，针对LLMs的提示工程及其优化策略的概念逐渐成为提升各类NLP任务显著性能的重要趋势。据我们所知，尽管已有大量综述文章探讨了提示工程，但在综合分析提示优化策略方面仍然存在关键空白。本文填补了这一空白，提供了关于各种提示优化策略潜在影响的独特且全面的见解，分析了它们的基本工作原理，并基于这些原则将它们分类为11种不同的类别。此外，本文详细介绍了这些提示优化策略在各种NLP任务中的应用情况，包括用于评估的不同大型语言模型和基准数据集的细节。这一全面的编纂为未来的对比研究奠定了坚实的基础，并在一致的实验设置下实现了对提示优化和基于大型语言模型的预测管道的严格评估：当前环境中的一项关键需求。最终，这项研究将集中各种战略知识以促进对现有提示优化策略的适应，开发适用于未探索任务的创新预测器。 

---
# TPTT: Transforming Pretrained Transformer into Titans 

**Title (ZH)**: TPTT: 将预训练变压器转化为巨擘 

**Authors**: Fabien Furfaro  

**Link**: [PDF](https://arxiv.org/pdf/2506.17671)  

**Abstract**: Recent advances in large language models (LLMs) have led to remarkable progress in natural language processing, but their computational and memory demands remain a significant challenge, particularly for long-context inference. We introduce TPTT (Transforming Pretrained Transformer into Titans), a novel framework for enhancing pretrained Transformer models with efficient linearized attention mechanisms and advanced memory management. TPTT employs techniques such as Memory as Gate (MaG) and mixed linearized attention (LiZA). It is fully compatible with the Hugging Face Transformers library, enabling seamless adaptation of any causal LLM through parameter-efficient fine-tuning (LoRA) without full retraining. We show the effectiveness of TPTT on the MMLU benchmark with models of approximately 1 billion parameters, observing substantial improvements in both efficiency and accuracy. For instance, Titans-Llama-3.2-1B achieves a 20% increase in Exact Match (EM) over its baseline. Statistical analyses and comparisons with recent state-of-the-art methods confirm the practical scalability and robustness of TPTT. Code is available at this https URL . Python package at this https URL . 

**Abstract (ZH)**: 最近在大型语言模型（LLMs）方面的进展引领了自然语言处理的显著进步，但它们的计算和内存需求仍然是一个重大挑战，尤其是在长语境推理方面。我们引入了一种名为TPTT（Transforming Pretrained Transformer into Titans）的新型框架，该框架通过高效的线性化注意力机制和先进的内存管理来增强预训练Transformer模型。TPTT采用的技术包括Memory as Gate（MaG）和混合线性化注意力（LiZA）。它完全兼容Hugging Face Transformers库，可以通过参数高效微调（LoRA）无缝适应任何因果LLM，而无需进行全面重训。我们在约10亿参数的MMLU基准上展示了TPTT的有效性，观察到在效率和准确性方面均实现了显著提升。例如，Titans-Llama-3.2-1B的精确匹配率（EM）相较于其基线提高了20%。统计分析和与最近的最先进方法的比较证实了TPTT的实际可扩展性和鲁棒性。代码可在以下链接获取：此 https URL 。Python包可在以下链接获取：此 https URL 。 

---
# LLM-Prompt: Integrated Heterogeneous Prompts for Unlocking LLMs in Time Series Forecasting 

**Title (ZH)**: LLM-Prompt:综合异构提示以解锁大规模语言模型在时间序列预测中的应用 

**Authors**: Zesen Wang, Yonggang Li, Lijuan Lan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17631)  

**Abstract**: Time series forecasting aims to model temporal dependencies among variables for future state inference, holding significant importance and widespread applications in real-world scenarios. Although deep learning-based methods have achieved remarkable progress, they still exhibit suboptimal performance in long-term forecasting and data-scarce scenarios. Recent research demonstrates that large language models (LLMs) achieve promising performance in time series forecasting. However, we find existing LLM-based methods still have shortcomings: (1) the absence of a unified paradigm for textual prompt formulation and (2) the neglect of modality discrepancies between textual prompts and time series. To address this, we propose LLM-Prompt, an LLM-based time series forecasting framework integrating multi-prompt information and cross-modal semantic alignment. Specifically, we first construct a unified textual prompt paradigm containing learnable soft prompts and textualized hard prompts. Second, to enhance LLMs' comprehensive understanding of the forecasting task, we design a semantic space embedding and cross-modal alignment module to achieve cross-modal fusion of temporal and textual information. Finally, the transformed time series from the LLMs are projected to obtain the forecasts. Comprehensive evaluations on 6 public datasets and 3 carbon emission datasets demonstrate that LLM-Prompt is a powerful framework for time series forecasting. 

**Abstract (ZH)**: 基于大语言模型的文本引导时间序列预测框架：LLM-Prompt 

---
# HalluRNN: Mitigating Hallucinations via Recurrent Cross-Layer Reasoning in Large Vision-Language Models 

**Title (ZH)**: HalluRNN：通过大型视觉语言模型中的递归跨层推理减轻幻觉问题 

**Authors**: Le Yu, Kaishen Wang, Jianlong Xiong, Yue Cao, Tao He  

**Link**: [PDF](https://arxiv.org/pdf/2506.17587)  

**Abstract**: Though Large Vision-Language Models (LVLMs) have achieved remarkable performance across various tasks, they are still prone to hallucinations-generating outputs that are textually plausible but visually ungrounded. While prior approaches generally address this issue through data-centric fine-tuning or innovative decoding strategies, these methods often require substantial resources or task-specific configurations. In this work, we introduce an architecture-level solution, HalluRNN, which enhances model stability through recurrent cross-layer reasoning. Specifically, we propose a novel Dual-Gated Depth Propagation Unit (DG-DPU) module, which is shared across layers and recurrently refines hidden states. This allows for the adaptive propagation of information throughout the model, enforces consistency across layers, and mitigates hallucinations caused by representational drift. By fine-tuning only the DG-DPU module, HalluRNN achieves strong and robust performance across multiple benchmarks. 

**Abstract (ZH)**: 尽管大型多模态模型（LVLMs）已经在各种任务上取得了显著的性能，但它们仍然容易产生幻觉——生成文本上合理但视觉上没有依据的输出。虽然现有的方法通常通过数据导向的微调或创新的解码策略来解决这一问题，但这些方法往往需要大量的资源或特定任务的配置。在这项工作中，我们提出了一种架构级别的解决方案——HalluRNN，该方案通过递归跨层推理增强了模型的稳定性。具体来说，我们提出了一种新颖的双门深度传播单元（DG-DPU）模块，该模块在各层之间共享并递归地细化隐藏状态。这使得信息在模型中的适应性传播成为可能，维护了各层的一致性，并减轻了由于表征漂移引起的幻觉。通过仅微调DG-DPU模块，HalluRNN在多个基准测试中取得了强健且稳定的性能。 

---
# Context-Aware Scientific Knowledge Extraction on Linked Open Data using Large Language Models 

**Title (ZH)**: 基于上下文的科学知识提取在链接开放数据中的大规模语言模型方法 

**Authors**: Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17580)  

**Abstract**: The exponential growth of scientific literature challenges researchers extracting and synthesizing knowledge. Traditional search engines return many sources without direct, detailed answers, while general-purpose LLMs may offer concise responses that lack depth or omit current information. LLMs with search capabilities are also limited by context window, yielding short, incomplete answers. This paper introduces WISE (Workflow for Intelligent Scientific Knowledge Extraction), a system addressing these limits by using a structured workflow to extract, refine, and rank query-specific knowledge. WISE uses an LLM-powered, tree-based architecture to refine data, focusing on query-aligned, context-aware, and non-redundant information. Dynamic scoring and ranking prioritize unique contributions from each source, and adaptive stopping criteria minimize processing overhead. WISE delivers detailed, organized answers by systematically exploring and synthesizing knowledge from diverse sources. Experiments on HBB gene-associated diseases demonstrate WISE reduces processed text by over 80% while achieving significantly higher recall over baselines like search engines and other LLM-based approaches. ROUGE and BLEU metrics reveal WISE's output is more unique than other systems, and a novel level-based metric shows it provides more in-depth information. We also explore how the WISE workflow can be adapted for diverse domains like drug discovery, material science, and social science, enabling efficient knowledge extraction and synthesis from unstructured scientific papers and web sources. 

**Abstract (ZH)**: 智能科学知识提取的工作流（WISE）：一种结构化的工作流框架，用于提取、提炼和排序查询特定的知识 

---
# Research on Model Parallelism and Data Parallelism Optimization Methods in Large Language Model-Based Recommendation Systems 

**Title (ZH)**: 基于大规模语言模型的推荐系统中模型并行性和数据并行性优化方法的研究 

**Authors**: Haowei Yang, Yu Tian, Zhongheng Yang, Zhao Wang, Chengrui Zhou, Dannier Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17551)  

**Abstract**: With the rapid adoption of large language models (LLMs) in recommendation systems, the computational and communication bottlenecks caused by their massive parameter sizes and large data volumes have become increasingly prominent. This paper systematically investigates two classes of optimization methods-model parallelism and data parallelism-for distributed training of LLMs in recommendation scenarios. For model parallelism, we implement both tensor parallelism and pipeline parallelism, and introduce an adaptive load-balancing mechanism to reduce cross-device communication overhead. For data parallelism, we compare synchronous and asynchronous modes, combining gradient compression and sparsification techniques with an efficient aggregation communication framework to significantly improve bandwidth utilization. Experiments conducted on a real-world recommendation dataset in a simulated service environment demonstrate that our proposed hybrid parallelism scheme increases training throughput by over 30% and improves resource utilization by approximately 20% compared to traditional single-mode parallelism, while maintaining strong scalability and robustness. Finally, we discuss trade-offs among different parallel strategies in online deployment and outline future directions involving heterogeneous hardware integration and automated scheduling technologies. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在推荐系统中的快速采用，其巨大的参数量和大数据量导致的计算和通信瓶颈日益突出。本文系统地研究了两种分布式训练优化方法——模型并行性和数据并行性——在推荐场景中训练LLMs的应用。在模型并行性方面，我们实现了张量并行性和管道并行性，并引入了一种自适应负载均衡机制以减少设备间通信开销。在数据并行性方面，我们比较了同步和异步模式，结合梯度压缩和稀疏化技术，并采用高效的聚合通信框架，以显著提高带宽利用率。实验结果表明，与传统的单模式并行性相比，我们提出的混合并行性方案可提高训练 throughput 超过 30%，并提高资源利用率约 20%，同时保持良好的可扩展性和鲁棒性。最后，我们讨论了不同并行策略在在线部署中的权衡，并概述了未来涉及异构硬件集成和自动化调度技术的发展方向。 

---
# Computational Approaches to Understanding Large Language Model Impact on Writing and Information Ecosystems 

**Title (ZH)**: 理解大规模语言模型对写作和信息生态系统影响的计算方法 

**Authors**: Weixin Liang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17467)  

**Abstract**: Large language models (LLMs) have shown significant potential to change how we write, communicate, and create, leading to rapid adoption across society. This dissertation examines how individuals and institutions are adapting to and engaging with this emerging technology through three research directions. First, I demonstrate how the institutional adoption of AI detectors introduces systematic biases, particularly disadvantaging writers of non-dominant language varieties, highlighting critical equity concerns in AI governance. Second, I present novel population-level algorithmic approaches that measure the increasing adoption of LLMs across writing domains, revealing consistent patterns of AI-assisted content in academic peer reviews, scientific publications, consumer complaints, corporate communications, job postings, and international organization press releases. Finally, I investigate LLMs' capability to provide feedback on research manuscripts through a large-scale empirical analysis, offering insights into their potential to support researchers who face barriers in accessing timely manuscript feedback, particularly early-career researchers and those from under-resourced settings. 

**Abstract (ZH)**: 大型语言模型（LLMs）展现出显著潜力，可以改变我们的写作、沟通和创造方式，导致其在社会中迅速被采用。本论文通过三个研究方向探讨个人和机构如何适应并参与这一新兴技术：首先，我展示了机构采用AI检测器引入系统性偏见，特别是对非主流语言变体的写作者不利，突显了AI治理中的关键公平性问题；其次，我提出了新颖的群体级算法方法来衡量LLMs在各种写作领域中的采用情况，揭示了AI辅助内容在学术同行评审、科学出版物、消费者投诉、企业通讯、招聘信息和国际组织新闻公告中的一致性模式；最后，我通过大规模实证分析探讨LLMs在科研手稿反馈方面的能力，提供有关其支持面临及时手稿反馈障碍的研究人员的见解，特别是早期职业研究人员和来自资源不足地区的研究人员。 

---
# UProp: Investigating the Uncertainty Propagation of LLMs in Multi-Step Agentic Decision-Making 

**Title (ZH)**: UProp：探究大规模语言模型在多步代理决策中的不确定性传播 

**Authors**: Jinhao Duan, James Diffenderfer, Sandeep Madireddy, Tianlong Chen, Bhavya Kailkhura, Kaidi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17419)  

**Abstract**: As Large Language Models (LLMs) are integrated into safety-critical applications involving sequential decision-making in the real world, it is essential to know when to trust LLM decisions. Existing LLM Uncertainty Quantification (UQ) methods are primarily designed for single-turn question-answering formats, resulting in multi-step decision-making scenarios, e.g., LLM agentic system, being underexplored. In this paper, we introduce a principled, information-theoretic framework that decomposes LLM sequential decision uncertainty into two parts: (i) internal uncertainty intrinsic to the current decision, which is focused on existing UQ methods, and (ii) extrinsic uncertainty, a Mutual-Information (MI) quantity describing how much uncertainty should be inherited from preceding decisions. We then propose UProp, an efficient and effective extrinsic uncertainty estimator that converts the direct estimation of MI to the estimation of Pointwise Mutual Information (PMI) over multiple Trajectory-Dependent Decision Processes (TDPs). UProp is evaluated over extensive multi-step decision-making benchmarks, e.g., AgentBench and HotpotQA, with state-of-the-art LLMs, e.g., GPT-4.1 and DeepSeek-V3. Experimental results demonstrate that UProp significantly outperforms existing single-turn UQ baselines equipped with thoughtful aggregation strategies. Moreover, we provide a comprehensive analysis of UProp, including sampling efficiency, potential applications, and intermediate uncertainty propagation, to demonstrate its effectiveness. Codes will be available at this https URL. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）被集成到涉及序列决策的安全关键应用中，了解何时信任LLM决策变得至关重要。现有的LLM不确定性量化（UQ）方法主要针对单轮问答格式设计，导致多步决策场景，例如LLM代理系统，尚未得到充分探索。本文介绍了一个原则性的信息论框架，将LLM序列决策不确定性分解为两部分：（i）内在于当前决策的不确定性，这是现有UQ方法关注的焦点；（ii）外在不确定性，这是一个互信息（MI）量度，描述了从先前决策继承多少不确定性。然后提出UProp，一个高效且有效的外在不确定性估计器，将直接估计MI转换为对多个轨迹依赖决策过程（TDPs）的点wise互信息（PMI）估计。UProp在广泛的多步决策基准测试中，如AgentBench和HotpotQA，与最先进的LLMs，如GPT-4.1和DeepSeek-V3进行了评估。实验结果表明，UProp显著优于现有的单轮UQ基线，并结合了精心的聚合策略。此外，我们对UProp进行了全面分析，包括采样效率、潜在应用和中间不确定性传播，以展示其有效性。代码将在以下网址提供：this https URL。 

---
# Re-Evaluating Code LLM Benchmarks Under Semantic Mutation 

**Title (ZH)**: 重新评估语义变异下的代码LLM基准 

**Authors**: Zhiyuan Pan, Xing Hu, Xin Xia, Xiaohu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17369)  

**Abstract**: In the era of large language models (LLMs), code benchmarks have become an important research area in software engineering and are widely used by practitioners. These benchmarks evaluate the performance of LLMs on specific code-related tasks, such as code understanding and generation. A critical step in constructing code benchmarks is the design of prompts. However, as existing code benchmarks typically rely on a single prompt template per task, they are prone to the issue of prompt sensitivity, where minor prompt variations could result in substantial performance variations, leading to unreliable evaluations of model capabilities.
While previous studies have explored prompt sensitivity, their experimental designs and findings are limited to traditional natural language processing (NLP) tasks. In this paper, we present an empirical study to investigate prompt sensitivity in code benchmarks. We first propose a general framework that modifies prompt templates in a manner that preserves both their semantics and their structure as much as possible. Based on the framework, we conduct extensive experiments across eight code benchmark tasks on 10 representative open-source LLMs, with each task featuring 100 semantically similar prompt templates. We then analyze the evaluation results using various statistical metrics, focusing on both absolute and relative model performance. Our findings suggest that even slight prompt variations can lead to significant shifts in performance. Additionally, we observe that such variations can introduce inconsistencies in the performance rankings across different models. These insights highlight the need for considering prompt sensitivity when designing future code benchmarks, to ensure more reliable and accurate evaluation of LLM capabilities. 

**Abstract (ZH)**: 在大型语言模型时代，代码基准已成为软件工程中的一个重要研究领域，广泛应用于实践。这些基准评估大型语言模型在特定代码任务上的表现，如代码理解与生成。构建代码基准的关键步骤之一是设计提示。然而，由于现有代码基准通常每个任务仅依赖一个提示模板，这使得它们容易受到提示敏感性问题的影响，即微小的提示变化可能导致显著的性能变化，从而导致对模型能力的不可靠评估。

尽管先前的研究已经探索了提示敏感性，但其实验设计和发现仅限于传统的自然语言处理任务。本文提出一项实证研究，以探讨代码基准中的提示敏感性。我们首先提出了一种通用框架，该框架以尽可能保留提示语义和结构的方式修改提示模板。基于该框架，我们对10个代表性的开源大型语言模型进行了广泛的实验，每个任务包含100个语义相似的提示模板，共计8个代码基准任务。然后，我们使用多种统计指标分析评估结果，重点关注绝对和相对模型表现。我们的发现表明，即使微小的提示变化也可能导致性能显著变化。我们还观察到，这些变化会在不同模型之间引入表现排名的一致性问题。这些见解强调，在设计未来代码基准时需要考虑提示敏感性，以确保对大型语言模型能力的更可靠和准确评估。 

---
# SAFEx: Analyzing Vulnerabilities of MoE-Based LLMs via Stable Safety-critical Expert Identification 

**Title (ZH)**: SAFEx：通过稳定的关键专家识别分析基于MoE的LLM的安全漏洞 

**Authors**: Zhenglin Lai, Mengyao Liao, Dong Xu, Zebin Zhao, Zhihang Yuan, Chao Fan, Jianqiang Li, Bingzhe Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17368)  

**Abstract**: Large language models based on Mixture-of-Experts have achieved substantial gains in efficiency and scalability, yet their architectural uniqueness introduces underexplored safety alignment challenges. Existing safety alignment strategies, predominantly designed for dense models, are ill-suited to address MoE-specific vulnerabilities. In this work, we formalize and systematically study MoE model's positional vulnerability - the phenomenon where safety-aligned behaviors rely on specific expert modules, revealing critical risks inherent to MoE architectures. To this end, we present SAFEx, an analytical framework that robustly identifies, characterizes, and validates the safety-critical experts using a novel Stability-based Expert Selection (SES) algorithm. Notably, our approach enables the explicit decomposition of safety-critical experts into distinct functional groups, including those responsible for harmful content detection and those controlling safe response generation. Extensive experiments on mainstream MoE models, such as the recently released Qwen3-MoE, demonstrated that their intrinsic safety mechanisms heavily rely on a small subset of positional experts. Disabling these experts significantly compromised the models' ability to refuse harmful requests. For Qwen3-MoE with 6144 experts (in the FNN layer), we find that disabling as few as 12 identified safety-critical experts can cause the refusal rate to drop by 22%, demonstrating the disproportionate impact of a small set of experts on overall model safety. 

**Abstract (ZH)**: 基于Mixture-of-Experts的大语言模型在效率和可扩展性方面取得了显著进步，但其独特的架构引入了未被充分探索的安全对齐挑战。现有的安全对齐策略主要针对密集模型设计，不适合作为MoE特定漏洞的解决方案。在本文中，我们正式化并系统研究了MoE模型的位置性脆弱性——安全对齐行为依赖于特定的专家模块的现象，揭示了MoE架构固有的关键风险。为此，我们提出了SAFEx，一种分析框架，利用一种新颖的基于稳定性的专家选择（SES）算法，稳健地识别、描述和验证安全关键专家。值得注意的是，我们的方法能够将安全关键专家明确分解为不同的功能组，包括负责有害内容检测和控制安全响应生成的专家。在主流MoE模型（如近期发布的Qwen3-MoE）上进行的广泛实验表明，这些模型内部的安全机制高度依赖于一组特定位置的专家。禁用这些专家极大地削弱了模型拒绝有害请求的能力。对于具有6144个专家（在FNN层）的Qwen3-MoE，我们发现禁用12个识别出的安全关键专家会使得拒绝率下降22%，展示了少数几个专家对整体模型安全性的重要影响。 

---
# Cash or Comfort? How LLMs Value Your Inconvenience 

**Title (ZH)**: 现金还是方便？大规模语言模型如何看待你的不便。 

**Authors**: Mateusz Cedro, Timour Ichmoukhamedov, Sofie Goethals, Yifan He, James Hinns, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2506.17367)  

**Abstract**: Large Language Models (LLMs) are increasingly proposed as near-autonomous artificial intelligence (AI) agents capable of making everyday decisions on behalf of humans. Although LLMs perform well on many technical tasks, their behaviour in personal decision-making remains less understood. Previous studies have assessed their rationality and moral alignment with human decisions. However, the behaviour of AI assistants in scenarios where financial rewards are at odds with user comfort has not yet been thoroughly explored. In this paper, we tackle this problem by quantifying the prices assigned by multiple LLMs to a series of user discomforts: additional walking, waiting, hunger and pain. We uncover several key concerns that strongly question the prospect of using current LLMs as decision-making assistants: (1) a large variance in responses between LLMs, (2) within a single LLM, responses show fragility to minor variations in prompt phrasing (e.g., reformulating the question in the first person can considerably alter the decision), (3) LLMs can accept unreasonably low rewards for major inconveniences (e.g., 1 Euro to wait 10 hours), and (4) LLMs can reject monetary gains where no discomfort is imposed (e.g., 1,000 Euro to wait 0 minutes). These findings emphasize the need for scrutiny of how LLMs value human inconvenience, particularly as we move toward applications where such cash-versus-comfort trade-offs are made on users' behalf. 

**Abstract (ZH)**: 大型语言模型在财务奖励与用户舒适度冲突情境下的行为探究 

---
# A Large-Scale Real-World Evaluation of LLM-Based Virtual Teaching Assistant 

**Title (ZH)**: 大规模现实世界中基于LLM的虚拟教学助手评估 

**Authors**: Sunjun Kweon, Sooyohn Nam, Hyunseung Lim, Hwajung Hong, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17363)  

**Abstract**: Virtual Teaching Assistants (VTAs) powered by Large Language Models (LLMs) have the potential to enhance student learning by providing instant feedback and facilitating multi-turn interactions. However, empirical studies on their effectiveness and acceptance in real-world classrooms are limited, leaving their practical impact uncertain. In this study, we develop an LLM-based VTA and deploy it in an introductory AI programming course with 477 graduate students. To assess how student perceptions of the VTA's performance evolve over time, we conduct three rounds of comprehensive surveys at different stages of the course. Additionally, we analyze 3,869 student--VTA interaction pairs to identify common question types and engagement patterns. We then compare these interactions with traditional student--human instructor interactions to evaluate the VTA's role in the learning process. Through a large-scale empirical study and interaction analysis, we assess the feasibility of deploying VTAs in real-world classrooms and identify key challenges for broader adoption. Finally, we release the source code of our VTA system, fostering future advancements in AI-driven education: \texttt{this https URL}. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的虚拟教学助手（VTAs）有潜力通过提供即时反馈和促进多轮交互来增强学生学习。然而，关于其在真实课堂中的有效性和接受度的实证研究有限，其实际影响尚不确定。在本研究中，我们开发了一个基于LLM的VTA，并将其部署在一门面向477名研究生的 introductory AI 编程课程中。为了评估学生对VTA性能的看法随着时间的推移如何演变，我们在课程的不同阶段进行了三次全面调查。此外，我们分析了3,869个学生-VTA交互对以识别常见问题类型和参与模式。然后，我们将这些交互与传统的学生-人类讲师交互进行比较，以评估VTA在学习过程中的角色。通过大规模的实证研究和交互分析，我们评估了在真实课堂中部署VTAs的可行性，并指出了更广泛采用的关键挑战。最后，我们发布了我们VTA系统的源代码，促进了人工智能驱动教育的未来发展：\texttt{this https URL}。 

---
# Automatic Large Language Models Creation of Interactive Learning Lessons 

**Title (ZH)**: 自动创建交互式学习课程的大规模语言模型方法 

**Authors**: Jionghao Lin, Jiarui Rao, Yiyang Zhao, Yuting Wang, Ashish Gurung, Amanda Barany, Jaclyn Ocumpaugh, Ryan S. Baker, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2506.17356)  

**Abstract**: We explore the automatic generation of interactive, scenario-based lessons designed to train novice human tutors who teach middle school mathematics online. Employing prompt engineering through a Retrieval-Augmented Generation approach with GPT-4o, we developed a system capable of creating structured tutor training lessons. Our study generated lessons in English for three key topics: Encouraging Students' Independence, Encouraging Help-Seeking Behavior, and Turning on Cameras, using a task decomposition prompting strategy that breaks lesson generation into sub-tasks. The generated lessons were evaluated by two human evaluators, who provided both quantitative and qualitative evaluations using a comprehensive rubric informed by lesson design research. Results demonstrate that the task decomposition strategy led to higher-rated lessons compared to single-step generation. Human evaluators identified several strengths in the LLM-generated lessons, including well-structured content and time-saving potential, while also noting limitations such as generic feedback and a lack of clarity in some instructional sections. These findings underscore the potential of hybrid human-AI approaches for generating effective lessons in tutor training. 

**Abstract (ZH)**: 我们探索了通过检索增强生成方法利用GPT-4o进行自动生成交互式、基于场景的课程的设计，这些课程旨在培训在线辅导初中数学的初学者人机导师。我们采用任务分解提示策略，将课程生成分解为子任务，并通过一种结构化的方法创建了辅导培训课程。研究使用了两名人类评估者，他们根据课程设计研究的全面评判标准提供了定量和定性的评估。结果表明，任务分解策略生成的课程评价高于一步生成的方式。人类评估者认为LLM生成的课程具有结构良好和节省时间的优点，但也指出了通用反馈和部分教学环节不够清晰的局限性。这些发现突显了人机混合方法在生成有效辅导培训课程方面的潜力。 

---
# Differentiation-Based Extraction of Proprietary Data from Fine-Tuned LLMs 

**Title (ZH)**: 基于分化提取细调后的大规模语言模型中的专有数据 

**Authors**: Zongjie Li, Daoyuan Wu, Shuai Wang, Zhendong Su  

**Link**: [PDF](https://arxiv.org/pdf/2506.17353)  

**Abstract**: The increasing demand for domain-specific and human-aligned Large Language Models (LLMs) has led to the widespread adoption of Supervised Fine-Tuning (SFT) techniques. SFT datasets often comprise valuable instruction-response pairs, making them highly valuable targets for potential extraction. This paper studies this critical research problem for the first time. We start by formally defining and formulating the problem, then explore various attack goals, types, and variants based on the unique properties of SFT data in real-world scenarios. Based on our analysis of extraction behaviors of direct extraction, we develop a novel extraction method specifically designed for SFT models, called Differentiated Data Extraction (DDE), which exploits the confidence levels of fine-tuned models and their behavioral differences from pre-trained base models. Through extensive experiments across multiple domains and scenarios, we demonstrate the feasibility of SFT data extraction using DDE. Our results show that DDE consistently outperforms existing extraction baselines in all attack settings. To counter this new attack, we propose a defense mechanism that mitigates DDE attacks with minimal impact on model performance. Overall, our research reveals hidden data leak risks in fine-tuned LLMs and provides insights for developing more secure models. 

**Abstract (ZH)**: 对领域特定和人类对齐的大语言模型进行监督微调的数据提取:一个新的研究问题及防御策略 

---
# Towards Safety Evaluations of Theory of Mind in Large Language Models 

**Title (ZH)**: 大型语言模型中理论心智安全评估的研究 

**Authors**: Tatsuhiro Aoshima, Mitsuaki Akiyama  

**Link**: [PDF](https://arxiv.org/pdf/2506.17352)  

**Abstract**: As the capabilities of large language models (LLMs) continue to advance, the importance of rigorous safety evaluation is becoming increasingly evident. Recent concerns within the realm of safety assessment have highlighted instances in which LLMs exhibit behaviors that appear to disable oversight mechanisms and respond in a deceptive manner. For example, there have been reports suggesting that, when confronted with information unfavorable to their own persistence during task execution, LLMs may act covertly and even provide false answers to questions intended to verify their this http URL evaluate the potential risk of such deceptive actions toward developers or users, it is essential to investigate whether these behaviors stem from covert, intentional processes within the model. In this study, we propose that it is necessary to measure the theory of mind capabilities of LLMs. We begin by reviewing existing research on theory of mind and identifying the perspectives and tasks relevant to its application in safety evaluation. Given that theory of mind has been predominantly studied within the context of developmental psychology, we analyze developmental trends across a series of open-weight LLMs. Our results indicate that while LLMs have improved in reading comprehension, their theory of mind capabilities have not shown comparable development. Finally, we present the current state of safety evaluation with respect to LLMs' theory of mind, and discuss remaining challenges for future work. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）能力的不断提升，严格的安全评估的重要性日益凸显。近期在安全评估领域的关切已经凸显了LLMs表现出看似规避监管机制并进行欺骗性回应的行为实例。例如，有报道称，在执行任务时面对对其持续性的不利信息时，LLMs可能会秘密行动甚至提供虚假答案以回答验证其持续性的质询。出于评估此类欺骗性行为对开发者或用户潜在风险的考虑，有必要调查这些行为是否源于模型内部的隐蔽故意过程。本研究提出，有必要衡量LLMs的理论思维能力。我们首先回顾了关于理论思维的研究，确定了适用于安全评估的应用视角和任务。鉴于理论思维主要是在发展心理学背景下研究的，我们分析了一系列开源权重LLMs的发展趋势。结果显示，尽管LLMs在阅读理解方面有所提高，但其理论思维能力尚未表现出相应的进步。最后，我们介绍了当前LLMs理论思维安全评估的状况，并讨论了未来研究面临的挑战。 

---
# Can Common VLMs Rival Medical VLMs? Evaluation and Strategic Insights 

**Title (ZH)**: 通用大模型能否挑战医疗大模型？评估与战略洞察 

**Authors**: Yuan Zhong, Ruinan Jin, Xiaoxiao Li, Qi Dou  

**Link**: [PDF](https://arxiv.org/pdf/2506.17337)  

**Abstract**: Medical vision-language models (VLMs) leverage large-scale pretraining for diverse imaging tasks but require substantial computational and data resources. Meanwhile, common or general-purpose VLMs (e.g., CLIP, LLaVA), though not trained for medical use, show promise with fine-tuning. This raises a key question: Can efficient fine-tuned common VLMs rival generalist medical VLMs for solving specific medical imaging tasks? This study systematically evaluates common and medical VLMs across disease diagnosis and visual question answering (VQA). Using CLIP-based and LLaVA-based models, we examine (1) off-the-shelf performance gaps in in-domain (ID) settings, (2) whether fine-tuning bridges these gaps, and (3) generalization to out-of-domain (OOD) tasks on unseen medical modalities. While medical-specific pretraining provides advantages in ID settings, common VLMs match or surpass medical-specific models after lightweight fine-tuning, with LoRA-based adaptation proving highly effective among different tasks. In OOD tasks, common VLMs demonstrate strong adaptability in some tasks, challenging the assumption that medical-specific pre-training is essential. These findings suggest that leveraging common VLMs with fine-tuning offers a scalable and cost-effective alternative to developing large-scale medical VLMs, providing crucial insights for future research in the medical imaging field. 

**Abstract (ZH)**: 医学视觉-语言模型在多种成像任务中利用大规模预训练但需要大量计算和数据资源。尽管通用或通用型视觉-语言模型（如CLIP、LLaVA）未专门训练用于医疗用途，但在微调后显示出潜力。本研究系统地评估了通用和医学视觉-语言模型在疾病诊断和视觉问答任务中的表现。利用CLIP和LLaVA模型，我们研究了（1）领域内（ID）设置下的即用型性能差距，（2）微调是否能弥合这些差距，以及（3）在未见医学模态的领域外（OOD）任务中的泛化能力。虽然针对医学的预训练在ID设置中具有优势，但在轻量级微调后，通用视觉-语言模型能够匹配或超越针对医学的模型，基于LoRA的适应尤其在不同任务中证明非常有效。在OOD任务中，通用视觉-语言模型在某些任务中表现出强大的适应性，挑战了医学专用预训练必不可少的假设。这些发现表明，利用通用视觉-语言模型并进行微调提供了一种可扩展且成本效益高的替代方案，用于开发大规模医学视觉-语言模型，并为未来医学成像领域的研究提供了关键见解。 

---
# LMR-BENCH: Evaluating LLM Agent's Ability on Reproducing Language Modeling Research 

**Title (ZH)**: LMR-BENCH: 评估大型语言模型代理在重现语言模型研究能力方面的表现 

**Authors**: Shuo Yan, Ruochen Li, Ziming Luo, Zimu Wang, Daoyang Li, Liqiang Jing, Kaiyu He, Peilin Wu, George Michalopoulos, Yue Zhang, Ziyang Zhang, Mian Zhang, Zhiyu Chen, Xinya Du  

**Link**: [PDF](https://arxiv.org/pdf/2506.17335)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable potential in advancing scientific discovery. However, their capability in the fundamental yet crucial task of reproducing code from research papers, especially in the NLP domain, remains underexplored. This task includes unique complex reasoning challenges in the intellectual synthesis of abstract concepts and the comprehension of code repositories with interdependent files. Motivated by this gap, we present LMR-BENCH, a benchmark designed to systematically evaluate the capability of LLM agents on code reproduction from Language Modeling Research. It consists of 28 code reproduction tasks derived from 23 research papers published in top-tier NLP venues over the past five years, spanning nine fundamental categories. Models are provided with a research paper, a code repository containing one or more masked functions, and instructions for implementing these functions. We conduct extensive experiments in standard prompting and LLM agent settings with state-of-the-art LLMs, evaluating the accuracy of unit tests and performing LLM-based evaluation of code correctness. Experimental results reveal that even the most advanced models still exhibit persistent limitations in scientific reasoning and code synthesis, highlighting critical gaps in LLM agents' ability to autonomously reproduce scientific research 

**Abstract (ZH)**: 大型语言模型代理在语言建模研究中的代码重现基准（LMR-BENCH）：系统评估其在科学发现中的代码重现能力 

---
# I Know Which LLM Wrote Your Code Last Summer: LLM generated Code Stylometry for Authorship Attribution 

**Title (ZH)**: 我知道你去年夏天的代码是由哪个LLM生成的：基于代码风格的作者归因方法 

**Authors**: Tamas Bisztray, Bilel Cherif, Richard A. Dubniczky, Nils Gruschka, Bertalan Borsos, Mohamed Amine Ferrag, Attila Kovacs, Vasileios Mavroeidis, Norbert Tihanyi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17323)  

**Abstract**: Detecting AI-generated code, deepfakes, and other synthetic content is an emerging research challenge. As code generated by Large Language Models (LLMs) becomes more common, identifying the specific model behind each sample is increasingly important. This paper presents the first systematic study of LLM authorship attribution for C programs. We released CodeT5-Authorship, a novel model that uses only the encoder layers from the original CodeT5 encoder-decoder architecture, discarding the decoder to focus on classification. Our model's encoder output (first token) is passed through a two-layer classification head with GELU activation and dropout, producing a probability distribution over possible authors. To evaluate our approach, we introduce LLM-AuthorBench, a benchmark of 32,000 compilable C programs generated by eight state-of-the-art LLMs across diverse tasks. We compare our model to seven traditional ML classifiers and eight fine-tuned transformer models, including BERT, RoBERTa, CodeBERT, ModernBERT, DistilBERT, DeBERTa-V3, Longformer, and LoRA-fine-tuned Qwen2-1.5B. In binary classification, our model achieves 97.56% accuracy in distinguishing C programs generated by closely related models such as GPT-4.1 and GPT-4o, and 95.40% accuracy for multi-class attribution among five leading LLMs (Gemini 2.5 Flash, Claude 3.5 Haiku, GPT-4.1, Llama 3.3, and DeepSeek-V3). To support open science, we release the CodeT5-Authorship architecture, the LLM-AuthorBench benchmark, and all relevant Google Colab scripts on GitHub: this https URL. 

**Abstract (ZH)**: 检测由人工智能生成的代码、深度伪造和其他合成内容是新兴的研究挑战。随着大型语言模型（LLMs）生成的代码变得更加普遍，识别每份样本背后的特定模型变得越来越重要。本文呈现了首个对C程序进行LLM作者归属的系统性研究。我们发布了CodeT5-Authorship，这是一种新颖的模型，仅使用CodeT5编码器-解码器架构的编码器层，舍弃解码器以专注于分类。我们的模型的编码器输出（第一个token）通过带有GELU激活和dropout的两层分类头，生成可能作者的概率分布。为了评估我们的方法，我们引入了LLM-AuthorBench基准，该基准包含32,000个由八种最先进的LLM生成的可编译C程序，涵盖了多种任务。我们将我们的模型与七种传统机器学习分类器和八种微调的变换器模型进行了比较，包括BERT、RoBERTa、CodeBERT、ModernBERT、DistilBERT、DeBERTa-V3、Longformer和基于LoRA微调的Qwen2-1.5B。在二分类中，我们的模型在区分紧密相关模型（如GPT-4.1和GPT-4o）生成的C程序时实现了97.56%的准确率，并在五种领先LLM（Gemini 2.5 Flash、Claude 3.5 Haiku、GPT-4.1、Llama 3.3和DeepSeek-V3）的多类归属中实现了95.40%的准确率。为了支持开放科学，我们将在GitHub上发布CodeT5-Authorship架构、LLM-AuthorBench基准以及所有相关的Google Colab脚本：this https URL。 

---
# LLM Jailbreak Oracle 

**Title (ZH)**: LLM Jailbreak Oracle 

**Authors**: Shuyi Lin, Anshuman Suri, Alina Oprea, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17299)  

**Abstract**: As large language models (LLMs) become increasingly deployed in safety-critical applications, the lack of systematic methods to assess their vulnerability to jailbreak attacks presents a critical security gap. We introduce the jailbreak oracle problem: given a model, prompt, and decoding strategy, determine whether a jailbreak response can be generated with likelihood exceeding a specified threshold. This formalization enables a principled study of jailbreak vulnerabilities. Answering the jailbreak oracle problem poses significant computational challenges -- the search space grows exponentially with the length of the response tokens. We present Boa, the first efficient algorithm for solving the jailbreak oracle problem. Boa employs a three-phase search strategy: (1) constructing block lists to identify refusal patterns, (2) breadth-first sampling to identify easily accessible jailbreaks, and (3) depth-first priority search guided by fine-grained safety scores to systematically explore promising low-probability paths. Boa enables rigorous security assessments including systematic defense evaluation, standardized comparison of red team attacks, and model certification under extreme adversarial conditions. 

**Abstract (ZH)**: 大语言模型（LLMs）在安全关键应用中的广泛应用引发了对其对抗脱管攻击脆弱性的系统评估方法的需求缺口。我们提出了脱管攻击Oracle问题：给定一个模型、提示和解码策略，确定是否存在生成超出指定阈值概率的脱管响应的可能性。这一形式化定义使得系统的脱管攻击脆弱性研究成为可能。解决脱管攻击Oracle问题面临着巨大的计算挑战——随着响应令牌长度的增加，搜索空间呈指数增长。我们提出了Boa，首个解决脱管攻击Oracle问题的高效算法。Boa采用三阶段搜索策略：（1）构建块列表以识别拒绝模式，（2）广度优先采样以识别易于访问的脱管响应，（3）基于细粒度安全评分的深度优先优先级搜索以系统地探索有希望的低概率路径。Boa使得严格的 security 评估成为可能，包括系统性防御评估、红队攻击的标准性比较以及在极端对抗条件下的模型认证。 

---
# Mercury: Ultra-Fast Language Models Based on Diffusion 

**Title (ZH)**: 汞：基于扩散的超快速语言模型 

**Authors**: Inception Labs, Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, Aditya Grover, Volodymyr Kuleshov  

**Link**: [PDF](https://arxiv.org/pdf/2506.17298)  

**Abstract**: We present Mercury, a new generation of commercial-scale large language models (LLMs) based on diffusion. These models are parameterized via the Transformer architecture and trained to predict multiple tokens in parallel. In this report, we detail Mercury Coder, our first set of diffusion LLMs designed for coding applications. Currently, Mercury Coder comes in two sizes: Mini and Small. These models set a new state-of-the-art on the speed-quality frontier. Based on independent evaluations conducted by Artificial Analysis, Mercury Coder Mini and Mercury Coder Small achieve state-of-the-art throughputs of 1109 tokens/sec and 737 tokens/sec, respectively, on NVIDIA H100 GPUs and outperform speed-optimized frontier models by up to 10x on average while maintaining comparable quality. We discuss additional results on a variety of code benchmarks spanning multiple languages and use-cases as well as real-world validation by developers on Copilot Arena, where the model currently ranks second on quality and is the fastest model overall. We also release a public API at this https URL and free playground at this https URL 

**Abstract (ZH)**: Mercury：基于扩散的新型商用大型语言模型及其编码应用Подробное описание Mercury Coder，我们的首款针对编码应用的扩散型大型语言模型。目前，Mercury Coder 提供 Mini 和 Small 两种规模。这些模型在速度-质量前沿上达到了新的标准。根据 Artificial Analysis 进行的独立评测，Mercury Coder Mini 和 Mercury Coder Small 在 NVIDIA H100 GPU 上分别实现了每秒 1109 个令牌和 737 个令牌的吞吐量，比速度优化的前沿模型平均快 10 倍，同时保持了相当的质量。我们还讨论了在多种编程语言和应用场景下的代码基准测试结果，以及开发人员在 Copilot Arena 中的实战验证，Mercury Coder 当前在质量排名第二，在所有模型中最快。我们也在该网址发布了一个公共 API 和一个免费的 playground。 

---
# Semantic uncertainty in advanced decoding methods for LLM generation 

**Title (ZH)**: 高级解码方法中LLM生成的语义不确定性 

**Authors**: Darius Foodeei, Simin Fan, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17296)  

**Abstract**: This study investigates semantic uncertainty in large language model (LLM) outputs across different decoding methods, focusing on emerging techniques like speculative sampling and chain-of-thought (CoT) decoding. Through experiments on question answering, summarization, and code generation tasks, we analyze how different decoding strategies affect both the diversity and reliability of model outputs. Our findings reveal that while CoT decoding demonstrates higher semantic diversity, it maintains lower predictive entropy, suggesting that structured exploration can lead to more confident and accurate outputs. This is evidenced by a 48.8% improvement in code generation Pass@2 rates, despite lower alignment with reference solutions. For summarization tasks, speculative sampling proved particularly effective, achieving superior ROUGE scores while maintaining moderate semantic diversity. Our results challenge conventional assumptions about trade-offs between diversity and accuracy in language model outputs, demonstrating that properly structured decoding methods can increase semantic exploration while maintaining or improving output quality. These findings have significant implications for deploying language models in practical applications where both reliability and diverse solution generation are crucial. 

**Abstract (ZH)**: 本研究探讨了不同解码方法下大规模语言模型（LLM）输出中的语义不确定性，重点研究了投机采样和链式思考（CoT）解码等新兴技术。通过在问答、总结和代码生成任务上的实验，我们分析了不同解码策略如何影响模型输出的多样性和可靠性。研究发现，虽然CoT解码显示出更高的语义多样性，但其预测不确定性较低，表明结构化的探索可以导致更加自信和准确的输出。这一结论在代码生成任务中得到了验证，尽管与参考答案的匹配度较低，但代码生成的Pass@2率提高了48.8%。对于总结任务，投机采样特别有效，实现了优越的ROUGE评分，同时保持了适度的语义多样性。我们的结果挑战了语言模型输出中多样性和准确性之间权衡的传统假设，表明适当的结构化解码方法可以在增加语义探索的同时维持或提高输出质量。这些发现对在需要可靠性和多样化解决方案的应用中部署语言模型具有重要意义。 

---
# GTA: Grouped-head latenT Attention 

**Title (ZH)**: GTA: 分组头部潜注意力 

**Authors**: Luoyang Sun, Jiwen Jiang, Cheng Deng, Xinjian Wu, Haifeng Zhang, Lei Chen, Lionel Ni, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17286)  

**Abstract**: Attention mechanisms underpin the success of large language models (LLMs), yet their substantial computational and memory overhead poses challenges for optimizing efficiency and performance. A critical bottleneck arises as KV cache and attention computations scale rapidly with text length, challenging deployment on hardware with limited computational and memory resources. We observe that attention mechanisms exhibit substantial redundancy, since the KV cache can be significantly compressed and attention maps across heads display high similarity, revealing that much of the computation and storage is unnecessary. Leveraging these insights, we propose \textbf{G}rouped-Head Laten\textbf{T} \textbf{A}ttention (GTA), a novel attention mechanism that reduces memory usage and computational complexity while maintaining performance. GTA comprises two components: (1) a shared attention map mechanism that reuses attention scores across multiple heads, decreasing the key cache size; and (2) a nonlinear value decoder with learned projections that compresses the value cache into a latent space, further cutting memory needs. GTA cuts attention computation FLOPs by up to \emph{62.5\%} versus Grouped-Query Attention and shrink the KV cache by up to \emph{70\%}, all while avoiding the extra overhead of Multi-Head Latent Attention to improve LLM deployment efficiency. Consequently, GTA models achieve a \emph{2x} increase in end-to-end inference speed, with prefill benefiting from reduced computational cost and decoding benefiting from the smaller cache footprint. 

**Abstract (ZH)**: 注意力机制支撑了大规模语言模型的成功，但其巨大的计算和内存开销阻碍了效率和性能的优化。随着文本长度的增加，KV缓存和注意力计算迅速上升，成为在计算和内存资源有限的硬件上部署的瓶颈。我们观察到注意力机制存在大量冗余，KV缓存可以显著压缩，而跨头的注意力图显示出高度的相似性，表明大量的计算和存储是不必要的。基于这些洞察，我们提出了**组头潜在注意力**（GTA），这一新颖的注意力机制可以在减少存储需求和计算复杂度的同时保持性能。GTA 包含两个组成部分：(1) 一个共享注意力图机制，通过在多个头之间重用注意力评分来减少键缓存的大小；(2) 一个非线性值解码器，带有学习的投影，将值缓存压缩到潜在空间中，进一步减少内存需求。与组查询注意力相比，GTA将注意力计算FLOPs减少了高达62.5%，并且将KV缓存减少了高达70%，同时避免了多头潜在注意力的额外开销以提高大规模语言模型的部署效率。因此，GTA 模型实现了端到端推理速度的两倍提升，预填充受益于计算成本的降低，解码则受益于更小的缓存占用。 

---
# CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models 

**Title (ZH)**: CORONA：基于图的推荐系统与大规模语言模型的粗到细框架 

**Authors**: Junze Chen, Xinjie Yang, Cheng Yang, Junfei Bao, Zeyuan Guo, Yawen Li, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17281)  

**Abstract**: Recommender systems (RSs) are designed to retrieve candidate items a user might be interested in from a large pool. A common approach is using graph neural networks (GNNs) to capture high-order interaction relationships. As large language models (LLMs) have shown strong capabilities across domains, researchers are exploring their use to enhance recommendation. However, prior work limits LLMs to re-ranking results or dataset augmentation, failing to utilize their power during candidate filtering - which may lead to suboptimal performance. Instead, we propose to leverage LLMs' reasoning abilities during the candidate filtering process, and introduce Chain Of Retrieval ON grAphs (CORONA) to progressively narrow down the range of candidate items on interaction graphs with the help of LLMs: (1) First, LLM performs preference reasoning based on user profiles, with the response serving as a query to extract relevant users and items from the interaction graph as preference-assisted retrieval; (2) Then, using the information retrieved in the previous step along with the purchase history of target user, LLM conducts intent reasoning to help refine an even smaller interaction subgraph as intent-assisted retrieval; (3) Finally, we employ a GNN to capture high-order collaborative filtering information from the extracted subgraph, performing GNN-enhanced retrieval to generate the final recommendation results. The proposed framework leverages the reasoning capabilities of LLMs during the retrieval process, while seamlessly integrating GNNs to enhance overall recommendation performance. Extensive experiments on various datasets and settings demonstrate that our proposed CORONA achieves state-of-the-art performance with an 18.6% relative improvement in recall and an 18.4% relative improvement in NDCG on average. 

**Abstract (ZH)**: 基于图的链式检索系统 (CORONA): 结合大型语言模型的推理能力以优化推荐系统 

---
# Step-by-Step Reasoning Attack: Revealing 'Erased' Knowledge in Large Language Models 

**Title (ZH)**: 逐步推理攻击：揭示大型语言模型中被“擦除”的知识 

**Authors**: Yash Sinha, Manit Baser, Murari Mandal, Dinil Mon Divakaran, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2506.17279)  

**Abstract**: Knowledge erasure in large language models (LLMs) is important for ensuring compliance with data and AI regulations, safeguarding user privacy, mitigating bias, and misinformation. Existing unlearning methods aim to make the process of knowledge erasure more efficient and effective by removing specific knowledge while preserving overall model performance, especially for retained information. However, it has been observed that the unlearning techniques tend to suppress and leave the knowledge beneath the surface, thus making it retrievable with the right prompts. In this work, we demonstrate that \textit{step-by-step reasoning} can serve as a backdoor to recover this hidden information. We introduce a step-by-step reasoning-based black-box attack, Sleek, that systematically exposes unlearning failures. We employ a structured attack framework with three core components: (1) an adversarial prompt generation strategy leveraging step-by-step reasoning built from LLM-generated queries, (2) an attack mechanism that successfully recalls erased content, and exposes unfair suppression of knowledge intended for retention and (3) a categorization of prompts as direct, indirect, and implied, to identify which query types most effectively exploit unlearning weaknesses. Through extensive evaluations on four state-of-the-art unlearning techniques and two widely used LLMs, we show that existing approaches fail to ensure reliable knowledge removal. Of the generated adversarial prompts, 62.5% successfully retrieved forgotten Harry Potter facts from WHP-unlearned Llama, while 50% exposed unfair suppression of retained knowledge. Our work highlights the persistent risks of information leakage, emphasizing the need for more robust unlearning strategies for erasure. 

**Abstract (ZH)**: 大规模语言模型（LLMs）中的知识擦除对于遵守数据和AI法规、保护用户隐私、减少偏见和虚假信息至关重要。现有的遗忘方法旨在通过移除特定知识同时保持整体模型性能的方式，使知识擦除过程更高效和有效，特别是对于保留信息。然而，观察到的是，遗忘技术倾向于抑制并保留知识，使其在适当的提示下可检索。在本文中，我们证明逐步推理可以用作后门，以恢复这些隐藏的信息。我们介绍了一种基于逐步推理的黑盒攻击Sleek，系统地揭示了遗忘失败。我们采用了一种结构化的攻击框架，包含三个核心组件：（1）利用来自LLM生成查询的逐步推理构建的对抗性提示生成策略，（2）成功检索被删除的内容并揭示意图保留知识的不公平抑制机制，以及（3）将提示分类为直接、间接和暗含，以确定哪种查询类型最有效地利用遗忘弱点。通过对四种最先进的遗忘技术以及两种广泛使用的LLM进行详尽评估，我们展示了现有方法无法确保可靠的知识去除。生成的对抗性提示中有62.5%成功从WHP-未删除的Llama中检索出了被遗忘的哈利·波特事实，而50%暴露了保留知识的不公平抑制。我们的工作强调了持续的信息泄露风险，强调了需要更 robust 的遗忘策略来进行去除。 

---
# Does Multimodal Large Language Model Truly Unlearn? Stealthy MLLM Unlearning Attack 

**Title (ZH)**: 多模态大语言模型真的能够有效遗忘吗？隐秘的MLLM遗忘攻击 

**Authors**: Xianren Zhang, Hui Liu, Delvin Ce Zhang, Xianfeng Tang, Qi He, Dongwon Lee, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17265)  

**Abstract**: Multimodal Large Language Models (MLLMs) trained on massive data may memorize sensitive personal information and photos, posing serious privacy risks. To mitigate this, MLLM unlearning methods are proposed, which fine-tune MLLMs to reduce the ``forget'' sensitive information. However, it remains unclear whether the knowledge has been truly forgotten or just hidden in the model. Therefore, we propose to study a novel problem of LLM unlearning attack, which aims to recover the unlearned knowledge of an unlearned LLM. To achieve the goal, we propose a novel framework Stealthy Unlearning Attack (SUA) framework that learns a universal noise pattern. When applied to input images, this noise can trigger the model to reveal unlearned content. While pixel-level perturbations may be visually subtle, they can be detected in the semantic embedding space, making such attacks vulnerable to potential defenses. To improve stealthiness, we introduce an embedding alignment loss that minimizes the difference between the perturbed and denoised image embeddings, ensuring the attack is semantically unnoticeable. Experimental results show that SUA can effectively recover unlearned information from MLLMs. Furthermore, the learned noise generalizes well: a single perturbation trained on a subset of samples can reveal forgotten content in unseen images. This indicates that knowledge reappearance is not an occasional failure, but a consistent behavior. 

**Abstract (ZH)**: 多模态大型语言模型的未学习攻击： Stealthy Unlearning Attack (SUA) 框架 

---
# OAT-Rephrase: Optimization-Aware Training Data Rephrasing for Zeroth-Order LLM Fine-Tuning 

**Title (ZH)**: OAT-重构：面向优化的训练数据重构在零阶语言模型微调中的应用 

**Authors**: Jikai Long, Zijian Hu, Xiaodong Yu, Jianwen Xie, Zhaozhuo Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17264)  

**Abstract**: Fine-tuning large language models (LLMs) using zeroth-order optimization (ZO) offers a memory-efficient alternative to gradient-based methods but suffers from slower convergence and unstable optimization due to noisy gradient estimates. This paper introduces OAT-Rephrase, an Optimization-Aware Training data rephrasing strategy that leverages an LLM to rephrase training instances based on its understanding of the ZO dynamics, specifically MeZO, derived directly from its paper. The approach incorporates a dual-stage pipeline featuring a rewriter LLM and a semantic judge, ensuring all rephrasings retain task relevance and logical consistency. Evaluations across five classification tasks and three LLM architectures demonstrate that OAT-Rephrase consistently improves MeZO fine-tuning performance, often narrowing or eliminating the gap with first-order methods. Our findings suggest that optimization-aware rephrasing serves as a reusable and low-overhead enhancement for zeroth-order tuning regimes. 

**Abstract (ZH)**: 利用零阶优化(OAT-Rephrase)意识训练数据重述策略：一种大型语言模型微调的新方法 

---
# UltraSketchLLM: Saliency-Driven Sketching for Ultra-Low Bit LLM Compression 

**Title (ZH)**: UltraSketchLLM：基于显著性的人工智能超低比特量压缩素描表示方法 

**Authors**: Sunan Zou, Ziyun Zhang, Xueting Sun, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2506.17255)  

**Abstract**: The rapid growth of large language models (LLMs) has outpaced the memory constraints of edge devices, necessitating extreme weight compression beyond the 1-bit limit. While quantization reduces model size, it is fundamentally limited to 1 bit per weight. Existing multiple-to-one compression methods either rely on mapping tables (inducing memory overhead) or incur severe accuracy degradation due to random weight grouping. We introduce UltraSketchLLM, an index-free, sketch-based framework that achieves ultra-low bit compression (down to 0.5 bits per weight) while preserving model performance. UltraSketchLLM leverages data sketching, a sub-linear representation technique from streaming applications, to map multiple weights to single values with bounded error. Our approach integrates an underestimate AbsMaxMin sketch to minimize relative errors for small weights, importance-aware space allocation to prioritize salient weights, and a straight-through estimator for compression-aware finetuning. Experiments on Llama-3.2-1B demonstrate up to 0.5-bit compression with competitive perplexity, alongside tolerable latency overhead. UltraSketchLLM offers a practical solution for deploying LLMs in resource-constrained environments. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的迅猛增长超过了边缘设备的内存限制， necessitating 极端权重压缩，远超1位限制。我们提出 UltraSketchLLM，一种无索引、基于草图的框架，实现超低位压缩（每位权重低至0.5位）同时保持模型性能。UltraSketchLLM 利用数据草图，这是一种来自流式应用的亚线性表示技术，将多个权重映射为单个值并带有有界误差。我们的方法结合了低估 AbsMaxMin 草图以最小化小权重的相对误差，基于重要性的空间分配以优先处理关键权重，并通过压缩感知微调引入直接通过估计器。实验表明，在 Llama-3.2-1B 上实现多达0.5位压缩，同时具有竞争力的困惑度和可 tolerable 的延迟开销。UltraSketchLLM 为在资源受限环境中部署 LLMs 提供了一种实际解决方案。 

---
# Keeping Up with the Models: Online Deployment and Routing of LLMs at Scale 

**Title (ZH)**: 跟上模型的步伐：大规模在线部署和路由LLM 

**Authors**: Shaoang Li, Jian Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17254)  

**Abstract**: The rapid pace at which new large language models (LLMs) appear -- and older ones become obsolete -- forces LLM service providers to juggle a streaming inventory of models while respecting tight deployment capacity and per-query cost budgets. We cast the reality as an online decision problem that couples stage-wise deployment, made at fixed maintenance windows, with per-query routing among the models kept live. We introduce StageRoute, a hierarchical algorithm that (i) optimistically selects up to $M_max$ models for the next stage using reward upper-confidence and cost lower-confidence bounds, then (ii) solves a budget-constrained bandit sub-problem to route each incoming query. We prove that StageRoute achieves a regret of order $T^{2/3}$ and provide a matching lower bound, thereby establishing its near-optimality. Moreover, our experiments confirm the theory, demonstrating that StageRoute performs close to the optimum in practical settings. 

**Abstract (ZH)**: 新出现的大语言模型（LLM）的快速迭代及其旧版模型的迅速过时迫使LLM服务提供商在严格的部署容量和每查询成本预算下管理一个流动的模型库存，同时在固定的维护窗口内进行阶段性的部署决策。我们将这一现实问题视为结合阶段部署（在固定的维护窗口进行）和查询路由的在线决策问题。我们引入了StageRoute算法，该算法（i）乐观地根据奖励的上置信界和成本的下置信界选择最多$M_{max}$个模型进入下一阶段，然后（ii）通过预算约束的多臂 bandit 子问题解决每个新查询的路由问题。我们证明StageRoute的遗憾度为$O(T^{2/3})$，并给出匹配的下界，从而确立了其接近最优性。此外，我们的实验验证了理论，表明在实际应用中StageRoute接近最优性能。 

---
# Adaptive Sample Scheduling for Direct Preference Optimization 

**Title (ZH)**: 直接偏好优化的自适应采样调度 

**Authors**: Zixuan Huang, Yikun Ban, Lean Fu, Xiaojie Li, Zhongxiang Dai, Jianxin Li, Deqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17252)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as an effective approach for aligning large language models (LLMs) with human preferences. However, its performance is highly dependent on the quality of the underlying human preference data. To address this bottleneck, prior work has explored various data selection strategies, but these methods often overlook the impact of the evolving states of the language model during the DPO process. %including active querying, response pair selection, and data pre-selection. In this paper, we introduce a novel problem: Sample Scheduling for DPO, which aims to dynamically and adaptively schedule training samples based on the model's evolving states throughout preference optimization. To solve this problem, we propose SamS, an efficient and effective algorithm that adaptively selects samples in each training batch based on the LLM's learning feedback to maximize the potential generalization performance. Notably, without modifying the core DPO algorithm, simply integrating SamS significantly improves performance across tasks, with minimal additional computational overhead. This work points to a promising new direction for improving LLM alignment through more effective utilization of fixed preference datasets. 

**Abstract (ZH)**: 直接偏好优化（DPO）已成为一种有效的方法，用于使大型语言模型（LLMs）与人类偏好保持一致。然而，其性能高度依赖于底层人类偏好数据的质量。为了应对这一瓶颈，先前的工作探索了各种数据选择策略，但这些方法往往忽略了语言模型在DPO过程中状态变化的影响。本文介绍了一个新的问题：DPO的采样调度问题，旨在根据模型在整个偏好优化过程中的状态动态和适应性地调度训练样本。为此，我们提出了SamS算法，该算法能够在每个训练批次中根据LLM的学习反馈自适应地选择样本，以最大化潜在的一般化性能。值得注意的是，仅通过将SamS集成到DPO核心算法中，即可在不增加额外计算开销的情况下显著提高各种任务的性能。这项工作指出了通过更有效地利用固定偏好数据来改进LLM对齐的有前途的新方向。 

---
# Training-free LLM Verification via Recycling Few-shot Examples 

**Title (ZH)**: 无需训练的LLM验证通过回收少量样本实例 

**Authors**: Dongseok Lee, Jimyung Hong, Dongyoung Kim, Jaehyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2506.17251)  

**Abstract**: Although LLMs have achieved remarkable performance, the inherent stochasticity of their reasoning process and varying conclusions present significant challenges. Majority voting or Best-of-N with external verification models has been explored to find the most promising solution among multiple LLM outputs. However, these approaches have certain limitations, such as limited applicability or the cost of an additional training step. To address this problem, we propose a novel and effective framework that Recycles Few-shot examples to verify LLM outputs (Referi). Our key idea is to additionally utilize the given few-shot examples to evaluate the candidate outputs of the target query, not only using them to generate outputs as the conventional few-shot prompting setup. Specifically, Referi evaluates the generated outputs by combining two different scores, designed motivated from Bayes' rule, and subsequently selects the candidate that is both confidently determined and contextually coherent through a few additional LLM inferences. Experiments with three different LLMs and across seven diverse tasks demonstrate that our framework significantly improves the accuracy of LLMs-achieving an average gain of 4.8%-through effective response selection, without additional training. 

**Abstract (ZH)**: 虽然大语言模型（LLMs）取得了显著的性能，其推理过程中的固有随机性和结论的多样性提出了重大挑战。为了在多个LLM输出中找到最有可能的解决方案，已经探索了多数投票或Best-of-N结合外部验证模型的方法。然而，这些方法存在一定的局限性，如适用范围有限或额外的训练步骤成本。为了解决这些问题，我们提出了一种新颖且有效的方法——Recycles Few-shot examples to Verify LLM outputs（Referi），旨在通过利用给定的少量示例，不仅生成输出，还评估目标查询的候选输出。具体而言，Referi通过结合两种不同的分数，依据贝叶斯规则进行设计，随后通过少量额外的LLM推理来选择同时具有高确定性和上下文一致性的候选输出。实验结果显示，该框架在三个不同的LLM和七个不同任务上显著提高了LLM的准确性，平均提升率达到4.8%，而无需额外训练。 

---
# Improving Prediction Certainty Estimation for Reliable Early Exiting via Null Space Projection 

**Title (ZH)**: 通过空域投影提高可靠早期退出的预测 certainty 估计 

**Authors**: Jianing He, Qi Zhang, Duoqian Miao, Yi Kun, Shufeng Hao, Hongyun Zhang, Zhihua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2506.17249)  

**Abstract**: Early exiting has demonstrated great potential in accelerating the inference of pre-trained language models (PLMs) by enabling easy samples to exit at shallow layers, eliminating the need for executing deeper layers. However, existing early exiting methods primarily rely on class-relevant logits to formulate their exiting signals for estimating prediction certainty, neglecting the detrimental influence of class-irrelevant information in the features on prediction certainty. This leads to an overestimation of prediction certainty, causing premature exiting of samples with incorrect early predictions. To remedy this, we define an NSP score to estimate prediction certainty by considering the proportion of class-irrelevant information in the features. On this basis, we propose a novel early exiting method based on the Certainty-Aware Probability (CAP) score, which integrates insights from both logits and the NSP score to enhance prediction certainty estimation, thus enabling more reliable exiting decisions. The experimental results on the GLUE benchmark show that our method can achieve an average speed-up ratio of 2.19x across all tasks with negligible performance degradation, surpassing the state-of-the-art (SOTA) ConsistentEE by 28%, yielding a better trade-off between task performance and inference efficiency. The code is available at this https URL. 

**Abstract (ZH)**: 早退出在通过使容易样本在浅层层退出以加速预训练语言模型的推理方面展示了巨大的潜力，但现有的早退出方法主要依靠类相关logits来形成退出信号以估计预测置信度，忽视了特征中类无关信息对预测置信度的负面影响，导致预测置信度估计过高，使得一些错误的早期预测提前退出。为此，我们定义了一个NSP得分来考虑特征中类无关信息的比例以估计预测置信度，并在此基础上提出了一种基于Certainty-Aware Probability (CAP)得分的新型早退出方法，该方法结合了logits和NSP得分的洞察，以提高预测置信度估计，从而实现更可靠的退出决策。GLUE基准实验结果显示，与最先进的ConsistentEE相比，我们的方法在所有任务上的平均加速比为2.19倍，性能下降可忽略不计，实现了更高的任务性能与推理效率 trade-off，代码详见此链接。 

---
