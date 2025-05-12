# Neuro-Symbolic Concepts 

**Title (ZH)**: 神经符号概念 

**Authors**: Jiayuan Mao, Joshua B. Tenenbaum, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06191)  

**Abstract**: This article presents a concept-centric paradigm for building agents that can learn continually and reason flexibly. The concept-centric agent utilizes a vocabulary of neuro-symbolic concepts. These concepts, such as object, relation, and action concepts, are grounded on sensory inputs and actuation outputs. They are also compositional, allowing for the creation of novel concepts through their structural combination. To facilitate learning and reasoning, the concepts are typed and represented using a combination of symbolic programs and neural network representations. Leveraging such neuro-symbolic concepts, the agent can efficiently learn and recombine them to solve various tasks across different domains, ranging from 2D images, videos, 3D scenes, and robotic manipulation tasks. This concept-centric framework offers several advantages, including data efficiency, compositional generalization, continual learning, and zero-shot transfer. 

**Abstract (ZH)**: 基于概念的代理构建范式：实现持续学习与灵活推理 

---
# Free and Fair Hardware: A Pathway to Copyright Infringement-Free Verilog Generation using LLMs 

**Title (ZH)**: 免费且公正的硬件：利用大语言模型实现版权侵权-free 的 Verilog 生成途径 

**Authors**: Sam Bush, Matthew DeLorenzo, Phat Tieu, Jeyavijayan Rajendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.06096)  

**Abstract**: Limitations in Large Language Model (LLM) capabilities for hardware design tasks, such as generating functional Verilog codes, have motivated various fine-tuning optimizations utilizing curated hardware datasets from open-source repositories. However, these datasets remain limited in size and contain minimal checks on licensing for reuse, resulting in potential copyright violations by fine-tuned LLMs. Therefore, we propose an evaluation benchmark to estimate the risk of Verilog-trained LLMs to generate copyright-protected codes. To minimize this risk, we present an open-source Verilog dataset, FreeSet, containing over 220k files, along with the automated dataset curation framework utilized to provide additional guarantees of fair-use Verilog data. We then execute an LLM fine-tuning framework consisting of continual pre-training, resulting in a fine-tuned Llama model for Verilog, FreeV. Our results indicate that FreeV demonstrates the smallest risk of copyright-infringement among prior works, with only a 3% violation rate. Furthermore, experimental results demonstrate improvements in Verilog generation functionality over its baseline model, improving VerilogEval pass@10 rates by over 10%. 

**Abstract (ZH)**: 大型语言模型（LLM）在硬件设计任务中的能力限制，如生成功能性的Verilog代码，推动了利用开源仓库中的精心策划的硬件数据集的各种精细调整优化方法。然而，这些数据集仍然规模有限，并且在再利用时的许可证检查较少，导致潜在的版权侵权风险。因此，我们提出了一种评估基准来估计Verilog训练的LLM生成受版权保护代码的风险。为了最小化这种风险，我们提供了一个包含超过220k个文件的开源Verilog数据集FreeSet，以及用于提供有关公平使用Verilog数据的额外保障的自动数据策展框架。然后，我们执行了一个持续预训练的LLM精细调整框架，从而得到一个用于Verilog的精细调整的Llama模型FreeV。我们的结果表明，FreeV在先前工作中展示了最小的版权侵权风险，其中只有3%的违规率。此外，实验结果还显示，FreeV在Verilog生成功能方面优于其基线模型，VerilogEval pass@10得分提高了超过10%。 

---
# Seqret: Mining Rule Sets from Event Sequences 

**Title (ZH)**: Seqret：从事件序列中挖掘规则集 

**Authors**: Aleena Siji, Joscha Cüppers, Osman Ali Mian, Jilles Vreeken  

**Link**: [PDF](https://arxiv.org/pdf/2505.06049)  

**Abstract**: Summarizing event sequences is a key aspect of data mining. Most existing methods neglect conditional dependencies and focus on discovering sequential patterns only. In this paper, we study the problem of discovering both conditional and unconditional dependencies from event sequence data. We do so by discovering rules of the form $X \rightarrow Y$ where $X$ and $Y$ are sequential patterns. Rules like these are simple to understand and provide a clear description of the relation between the antecedent and the consequent. To discover succinct and non-redundant sets of rules we formalize the problem in terms of the Minimum Description Length principle. As the search space is enormous and does not exhibit helpful structure, we propose the Seqret method to discover high-quality rule sets in practice. Through extensive empirical evaluation we show that unlike the state of the art, Seqret ably recovers the ground truth on synthetic datasets and finds useful rules from real datasets. 

**Abstract (ZH)**: 总结事件序列是数据挖掘中的关键方面。大多数现有方法忽视了条件依赖关系，仅专注于发现序列模式。本文研究从事件序列数据中发现条件和无条件依赖关系的问题。通过发现形式为 $X \rightarrow Y$ 的规则，其中 $X$ 和 $Y$ 是序列模式，来解决这一问题。这类规则易于理解，并且清晰地描述了前件和后件之间的关系。为了发现简洁且不冗余的规则集，本文从最小描述长度原理出发形式化该问题。由于搜索空间巨大且缺乏帮助性的结构，本文提出了Seqret方法以在实践中发现高质量的规则集。通过广泛的实证评估，本文展示了Seqret能够在合成数据集上恢复真实情况，并从真实数据集中发现有用规则，优于现有方法。 

---
# Why Are You Wrong? Counterfactual Explanations for Language Grounding with 3D Objects 

**Title (ZH)**: 你为什么错误？基于3D物体的语言定位反事实解释 

**Authors**: Tobias Preintner, Weixuan Yuan, Qi Huang, Adrian König, Thomas Bäck, Elena Raponi, Niki van Stein  

**Link**: [PDF](https://arxiv.org/pdf/2505.06030)  

**Abstract**: Combining natural language and geometric shapes is an emerging research area with multiple applications in robotics and language-assisted design. A crucial task in this domain is object referent identification, which involves selecting a 3D object given a textual description of the target. Variability in language descriptions and spatial relationships of 3D objects makes this a complex task, increasing the need to better understand the behavior of neural network models in this domain. However, limited research has been conducted in this area. Specifically, when a model makes an incorrect prediction despite being provided with a seemingly correct object description, practitioners are left wondering: "Why is the model wrong?". In this work, we present a method answering this question by generating counterfactual examples. Our method takes a misclassified sample, which includes two objects and a text description, and generates an alternative yet similar formulation that would have resulted in a correct prediction by the model. We have evaluated our approach with data from the ShapeTalk dataset along with three distinct models. Our counterfactual examples maintain the structure of the original description, are semantically similar and meaningful. They reveal weaknesses in the description, model bias and enhance the understanding of the models behavior. Theses insights help practitioners to better interact with systems as well as engineers to improve models. 

**Abstract (ZH)**: 结合自然语言和几何形状在机器人技术和语言辅助设计中的应用是一项新兴的研究领域。该领域的一个关键任务是对象referent识别，即根据目标的文本描述选择一个3D对象。由于语言描述和3D对象的空间关系的不确定性，这一任务变得十分复杂，增加了对该领域神经网络模型行为更好地理解的需求。然而，在这方面的研究尚有限。特别是，当模型在提供看似正确对象描述的情况下做出错误预测时，实践者会疑惑：“模型为什么错了？”本文提出了一种方法来回答这个问题，通过生成对抗性示例。该方法接受一个分类错误的样本，包含两个对象和一个文本描述，并生成一个替代但相似的表述，该表述本会使模型做出正确的预测。我们使用ShapeTalk数据集及三种不同的模型评估了该方法。我们的对抗性示例保留了原始描述的结构，具有语义上的相似性和意义性。它们揭示了描述中的弱点、模型偏见，并增强了对模型行为的理解。这些见解有助于实践者更好地与系统互动，以及工程师改进模型。 

---
# ArtRAG: Retrieval-Augmented Generation with Structured Context for Visual Art Understanding 

**Title (ZH)**: ArtRAG：带有结构化上下文的检索增强生成，用于视觉艺术理解 

**Authors**: Shuai Wang, Ivona Najdenkoska, Hongyi Zhu, Stevan Rudinac, Monika Kackovic, Nachoem Wijnberg, Marcel Worring  

**Link**: [PDF](https://arxiv.org/pdf/2505.06020)  

**Abstract**: Understanding visual art requires reasoning across multiple perspectives -- cultural, historical, and stylistic -- beyond mere object recognition. While recent multimodal large language models (MLLMs) perform well on general image captioning, they often fail to capture the nuanced interpretations that fine art demands. We propose ArtRAG, a novel, training-free framework that combines structured knowledge with retrieval-augmented generation (RAG) for multi-perspective artwork explanation. ArtRAG automatically constructs an Art Context Knowledge Graph (ACKG) from domain-specific textual sources, organizing entities such as artists, movements, themes, and historical events into a rich, interpretable graph. At inference time, a multi-granular structured retriever selects semantically and topologically relevant subgraphs to guide generation. This enables MLLMs to produce contextually grounded, culturally informed art descriptions. Experiments on the SemArt and Artpedia datasets show that ArtRAG outperforms several heavily trained baselines. Human evaluations further confirm that ArtRAG generates coherent, insightful, and culturally enriched interpretations. 

**Abstract (ZH)**: 理解视觉艺术需要从文化、历史和风格等多视角进行推理，而不仅仅是对象识别。尽管近期的多模态大型语言模型（MLLMs）在通用图像描述任务上表现良好，但往往无法捕捉到艺术品所需的微妙解读。我们提出ArtRAG，这是一种新颖的无需训练框架，结合结构化知识与检索增强生成（RAG）进行多视角艺术作品解释。ArtRAG自动从领域特定的文本来源构建艺术上下文知识图谱（ACKG），将艺术家、流派、主题和历史事件组织成一个丰富且可解释的图。在推理阶段，多粒度结构化检索器选择语义和拓扑相关的子图以指导生成，从而使MLLMs产生基于上下文、文化背景的艺术描述。在SemArt和Artpedia数据集上的实验表明，ArtRAG优于多个经过大量训练的基本模型。人类评估进一步证实，ArtRAG生成了连贯、见解深刻且文化丰富的解释。 

---
# Pseudo-Boolean d-DNNF Compilation for Expressive Feature Modeling Constructs 

**Title (ZH)**: 伪布尔d-DNNF编译在表达性特征建模构造中的应用 

**Authors**: Chico Sundermann, Stefan Vill, Elias Kuiter, Sebastian Krieter, Thomas Thüm, Matthias Tichy  

**Link**: [PDF](https://arxiv.org/pdf/2505.05976)  

**Abstract**: Configurable systems typically consist of reusable assets that have dependencies between each other. To specify such dependencies, feature models are commonly used. As feature models in practice are often complex, automated reasoning is typically employed to analyze the dependencies. Here, the de facto standard is translating the feature model to conjunctive normal form (CNF) to enable employing off-the-shelf tools, such as SAT or #SAT solvers. However, modern feature-modeling dialects often contain constructs, such as cardinality constraints, that are ill-suited for conversion to CNF. This mismatch between the input of reasoning engines and the available feature-modeling dialects limits the applicability of the more expressive constructs. In this work, we shorten this gap between expressive constructs and scalable automated reasoning. Our contribution is twofold: First, we provide a pseudo-Boolean encoding for feature models, which facilitates smaller representations of commonly employed constructs compared to Boolean encoding. Second, we propose a novel method to compile pseudo-Boolean formulas to Boolean d-DNNF. With the compiled d-DNNFs, we can resort to a plethora of efficient analyses already used in feature modeling. Our empirical evaluation shows that our proposal substantially outperforms the state-of-the-art based on CNF inputs for expressive constructs. For every considered dataset representing different feature models and feature-modeling constructs, the feature models can be significantly faster translated to pseudo-Boolean than to CNF. Overall, deriving d-DNNFs from a feature model with the targeted expressive constraints can be substantially accelerated using our pseudo-Boolean approach. Furthermore, our approach is competitive on feature models with only basic constructs. 

**Abstract (ZH)**: 可配置系统通常由具有彼此依赖关系的可重用资产组成。为了指定这些依赖关系，通常使用特征模型。由于特征模型在实践中往往非常复杂，因此通常会使用自动推理来分析这些依赖关系。目前的标准做法是将特征模型转换为合取范式（CNF），以便使用现成的工具，如SAT或#SAT求解器。然而，现代特征建模方言中包含的一些构造，如基数约束，不适合转换为CNF。这种推理引擎输入与可用的特征建模方言之间的不匹配限制了更富有表现力的构造的应用。本文在富有表现力的构造与可扩展自动推理之间缩短了这一差距。我们的贡献有两个方面：首先，我们提供了一种伪布尔编码方法，相对于布尔编码，这种方法对于常用构造提供了更小的表示形式。其次，我们提出了一种将伪布尔公式编译为布尔d-DNNF的新方法。使用编译后的d-DNNF，我们可以利用已经在特征建模中广泛使用的各种高效分析方法。我们的实证评估表明，与基于CNF输入的现有方法相比，我们的提议在处理富有表现力的构造时显著表现出色。对于每一种代表不同特征模型和特征建模构造的数据集，特征模型可以显著更快地转化为伪布尔表达式，而非CNF。总体而言，使用我们的伪布尔方法从特征模型导出带目标富有表现力约束的d-DNNF可以显著加速计算。此外，对于仅包含基本构造的特征模型，我们的方法具有竞争力。 

---
# Combining Abstract Argumentation and Machine Learning for Efficiently Analyzing Low-Level Process Event Streams 

**Title (ZH)**: 结合抽象论辩与机器学习高效分析低层级过程事件流 

**Authors**: Bettina Fazzinga, Sergio Flesca, Filippo Furfaro, Luigi Pontieri, Francesco Scala  

**Link**: [PDF](https://arxiv.org/pdf/2505.05880)  

**Abstract**: Monitoring and analyzing process traces is a critical task for modern companies and organizations. In scenarios where there is a gap between trace events and reference business activities, this entails an interpretation problem, amounting to translating each event of any ongoing trace into the corresponding step of the activity instance. Building on a recent approach that frames the interpretation problem as an acceptance problem within an Abstract Argumentation Framework (AAF), one can elegantly analyze plausible event interpretations (possibly in an aggregated form), as well as offer explanations for those that conflict with prior process knowledge. Since, in settings where event-to-activity mapping is highly uncertain (or simply under-specified) this reasoning-based approach may yield lowly-informative results and heavy computation, one can think of discovering a sequencetagging model, trained to suggest highly-probable candidate event interpretations in a context-aware way. However, training such a model optimally may require using a large amount of manually-annotated example traces. Considering the urgent need of developing Green AI solutions enabling environmental and societal sustainability (with reduced labor/computational costs and carbon footprint), we propose a data/computation-efficient neuro-symbolic approach to the problem, where the candidate interpretations returned by the example-driven sequence tagger is refined by the AAF-based reasoner. This allows us to also leverage prior knowledge to compensate for the scarcity of example data, as confirmed by experimental results; clearly, this property is particularly useful in settings where data annotation and model optimization costs are subject to stringent constraints. 

**Abstract (ZH)**: 基于抽象论证框架的进程踪迹监控与分析：一种数据和计算高效的方法 

---
# APOLLO: Automated LLM and Lean Collaboration for Advanced Formal Reasoning 

**Title (ZH)**: APOLLO: 自动化大语言模型和精简协作以实现高级形式化推理 

**Authors**: Azim Ospanov, Roozbeh Yousefzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.05758)  

**Abstract**: Formal reasoning and automated theorem proving constitute a challenging subfield of machine learning, in which machines are tasked with proving mathematical theorems using formal languages like Lean. A formal verification system can check whether a formal proof is correct or not almost instantaneously, but generating a completely correct formal proof with large language models (LLMs) remains a formidable task. The usual approach in the literature is to prompt the LLM many times (up to several thousands) until one of the generated proofs passes the verification system. In this work, we present APOLLO (Automated PrOof repair via LLM and Lean cOllaboration), a modular, model-agnostic pipeline that combines the strengths of the Lean compiler with an LLM's reasoning abilities to achieve better proof-generation results at a low sampling budget. Apollo directs a fully automated process in which the LLM generates proofs for theorems, a set of agents analyze the proofs, fix the syntax errors, identify the mistakes in the proofs using Lean, isolate failing sub-lemmas, utilize automated solvers, and invoke an LLM on each remaining goal with a low top-K budget. The repaired sub-proofs are recombined and reverified, iterating up to a user-controlled maximum number of attempts. On the miniF2F benchmark, we establish a new state-of-the-art accuracy of 75.0% among 7B-parameter models while keeping the sampling budget below one thousand. Moreover, Apollo raises the state-of-the-art accuracy for Goedel-Prover-SFT to 65.6% while cutting sample complexity from 25,600 to a few hundred. General-purpose models (o3-mini, o4-mini) jump from 3-7% to over 40% accuracy. Our results demonstrate that targeted, compiler-guided repair of LLM outputs yields dramatic gains in both efficiency and correctness, suggesting a general paradigm for scalable automated theorem proving. 

**Abstract (ZH)**: 自动推理与形式化定理证明是机器学习的一个具有挑战性的子领域，其中机器使用如Lean的形式语言来证明数学定理。形式验证系统几乎可以瞬时检查形式证明的正确性，但使用大语言模型（LLMs）生成完全正确的形式证明仍是一个艰巨的任务。文献中通常的方法是在验证系统中不断提示LLM（多达数千次），直到生成的证明之一通过验证系统。在本工作中，我们提出了APOLLO（自动推理验证通过LLM和Lean合作），一个模块化、模型无关的管道，结合了Lean编译器的力量和LLM的推理能力，在较低的采样预算下实现更好的证明生成结果。Apollo指导一个全自动过程，在该过程中，LLM生成定理的证明，一组代理分析证明、修正语法错误、使用Lean识别证明中的错误、隔离失败的子引理、利用自动求解器，并对剩余目标调用每个LLM，使用较低的Top-K预算。修复的子证明被重新组合和重新验证，迭代到用户控制的最大尝试次数。在miniF2F基准测试中，我们在7B参数模型中建立了一个新的最佳准确率75.0%，同时保持采样预算低于一千。此外，Apollo将Goedel-Prover-SFT的最佳准确率提高到65.6%，并将样本复杂度从25,600降低到几百。通用模型（o3-mini, o4-mini）的准确率从3-7%提高到超过40%。我们的结果表明，目标导向、编译器引导的LLM输出修复在效率和正确性方面产生了巨大的提升，这表明了一种可扩展自动定理证明的一般范式。 

---
# Pretraining a Shared Q-Network for Data-Efficient Offline Reinforcement Learning 

**Title (ZH)**: 基于数据高效离线强化学习的共享Q网络预训练 

**Authors**: Jongchan Park, Mingyu Park, Donghwan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.05701)  

**Abstract**: Offline reinforcement learning (RL) aims to learn a policy from a static dataset without further interactions with the environment. Collecting sufficiently large datasets for offline RL is exhausting since this data collection requires colossus interactions with environments and becomes tricky when the interaction with the environment is restricted. Hence, how an agent learns the best policy with a minimal static dataset is a crucial issue in offline RL, similar to the sample efficiency problem in online RL. In this paper, we propose a simple yet effective plug-and-play pretraining method to initialize a feature of a $Q$-network to enhance data efficiency in offline RL. Specifically, we introduce a shared $Q$-network structure that outputs predictions of the next state and $Q$-value. We pretrain the shared $Q$-network through a supervised regression task that predicts a next state and trains the shared $Q$-network using diverse offline RL methods. Through extensive experiments, we empirically demonstrate that our method enhances the performance of existing popular offline RL methods on the D4RL, Robomimic and V-D4RL benchmarks. Furthermore, we show that our method significantly boosts data-efficient offline RL across various data qualities and data distributions trough D4RL and ExoRL benchmarks. Notably, our method adapted with only 10% of the dataset outperforms standard algorithms even with full datasets. 

**Abstract (ZH)**: 离线强化学习（Offline RL）旨在从静态数据集中学习策略，而不与环境进一步交互。由于数据收集需要与环境进行大量交互，而在环境交互受限的情况下变得更加困难，如何使用最小的静态数据集让代理学习到最佳策略成为离线RL中一个关键问题，类似于在线RL中的样本效率问题。在本文中，我们提出了一种简单而有效的插即用预训练方法，用于初始化Q网络的特征以增强离线RL的数据效率。具体地，我们引入了一种共享的Q网络结构，该结构输出下一个状态和Q值的预测。我们通过监督回归任务对共享的Q网络进行预训练，该任务预测下一个状态，并使用多种离线RL方法训练共享的Q网络。通过广泛的实验，我们实证证明，我们的方法在D4RL、Robomimic和V-D4RL基准上增强了现有流行离线RL方法的性能。此外，我们展示了我们的方法在D4RL和ExoRL基准上显著增强了不同数据质量和分布下离线RL的数据效率。值得注意的是，即使只用10%的数据集，我们的方法也能在数据集完整的情况下超越标准算法。 

---
# Prompted Meta-Learning for Few-shot Knowledge Graph Completion 

**Title (ZH)**: Prompted元学习在少样本知识图谱补全中的应用 

**Authors**: Han Wu, Jie Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05684)  

**Abstract**: Few-shot knowledge graph completion (KGC) has obtained significant attention due to its practical applications in real-world scenarios, where new knowledge often emerges with limited available data. While most existing methods for few-shot KGC have predominantly focused on leveraging relational information, rich semantics inherent in KGs have been largely overlooked. To address this gap, we propose a novel prompted meta-learning (PromptMeta) framework that seamlessly integrates meta-semantics with relational information for few-shot KGC. PrompMeta has two key innovations: (1) a meta-semantic prompt pool that captures and consolidates high-level meta-semantics, enabling effective knowledge transfer and adaptation to rare and newly emerging relations. (2) a learnable fusion prompt that dynamically combines meta-semantic information with task-specific relational information tailored to different few-shot tasks. Both components are optimized together with model parameters within a meta-learning framework. Extensive experiments on two benchmark datasets demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: Few-shot知识图谱补全（KGC）因其在现实场景中的实际应用而获得了显著关注，尤其是在有限可用数据的情况下新知识不断涌现。尽管大多数现有的few-shot KGC方法主要侧重于利用关系信息，但KG中固有的丰富语义尚未得到充分关注。为解决这一问题，我们提出了一种新的提示元学习（PromptMeta）框架，该框架无缝地将元语义与关系信息结合用于few-shot KGC。PromptMeta 的两个关键创新包括：（1）一个元语义提示池，用于捕获和整合高级别元语义，从而实现有效的知识迁移和对稀有及新出现关系的适应；（2）一个可学习融合提示，能够动态地将特定任务的关系信息与元语义信息结合。这两个组件在元学习框架中与模型参数一起优化。在两个基准数据集上的广泛实验显示了我们方法的有效性。 

---
# Leveraging Large Language Models for enzymatic reaction prediction and characterization 

**Title (ZH)**: 利用大规模语言模型进行酶促反应预测与表征 

**Authors**: Lorenzo Di Fruscia, Jana Marie Weber  

**Link**: [PDF](https://arxiv.org/pdf/2505.05616)  

**Abstract**: Predicting enzymatic reactions is crucial for applications in biocatalysis, metabolic engineering, and drug discovery, yet it remains a complex and resource-intensive task. Large Language Models (LLMs) have recently demonstrated remarkable success in various scientific domains, e.g., through their ability to generalize knowledge, reason over complex structures, and leverage in-context learning strategies. In this study, we systematically evaluate the capability of LLMs, particularly the Llama-3.1 family (8B and 70B), across three core biochemical tasks: Enzyme Commission number prediction, forward synthesis, and retrosynthesis. We compare single-task and multitask learning strategies, employing parameter-efficient fine-tuning via LoRA adapters. Additionally, we assess performance across different data regimes to explore their adaptability in low-data settings. Our results demonstrate that fine-tuned LLMs capture biochemical knowledge, with multitask learning enhancing forward- and retrosynthesis predictions by leveraging shared enzymatic information. We also identify key limitations, for example challenges in hierarchical EC classification schemes, highlighting areas for further improvement in LLM-driven biochemical modeling. 

**Abstract (ZH)**: 预测酶促反应对于生物催化、代谢工程和药物发现的应用至关重要，但这项任务依然复杂且资源密集。大规模语言模型（LLMs）最近在各个科学领域展现了显著的成功，例如通过其泛化知识、处理复杂结构和利用上下文学习策略的能力。在这项研究中，我们系统地评估了LLMs，特别是Llama-3.1家族（8B和70B参数版本）在三大核心生化任务中的能力：酶委分类号预测、正向合成和逆向合成。我们比较了单任务和多任务学习策略，并通过LoRA适配器进行参数高效微调。此外，我们还评估了不同数据集对模型性能的影响，以探索其在数据稀缺环境中的适应性。研究结果表明，微调后的LLMs能够捕捉生化知识，多任务学习通过利用共享的酶促信息提高了正向合成和逆向合成的预测能力。我们还指出了关键的局限性，如在分层酶分类方案中的挑战，这突显了LLM驱动的生化建模进一步改进的必要性。 

---
# scDrugMap: Benchmarking Large Foundation Models for Drug Response Prediction 

**Title (ZH)**: scDrugMap: 评估大型基础模型在药物响应预测中的性能 

**Authors**: Qing Wang, Yining Pan, Minghao Zhou, Zijia Tang, Yanfei Wang, Guangyu Wang, Qianqian Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.05612)  

**Abstract**: Drug resistance presents a major challenge in cancer therapy. Single cell profiling offers insights into cellular heterogeneity, yet the application of large-scale foundation models for predicting drug response in single cell data remains underexplored. To address this, we developed scDrugMap, an integrated framework featuring both a Python command-line interface and a web server for drug response prediction. scDrugMap evaluates a wide range of foundation models, including eight single-cell models and two large language models, using a curated dataset of over 326,000 cells in the primary collection and 18,800 cells in the validation set, spanning 36 datasets and diverse tissue and cancer types. We benchmarked model performance under pooled-data and cross-data evaluation settings, employing both layer freezing and Low-Rank Adaptation (LoRA) fine-tuning strategies. In the pooled-data scenario, scFoundation achieved the best performance, with mean F1 scores of 0.971 (layer freezing) and 0.947 (fine-tuning), outperforming the lowest-performing model by over 50%. In the cross-data setting, UCE excelled post fine-tuning (mean F1: 0.774), while scGPT led in zero-shot learning (mean F1: 0.858). Overall, scDrugMap provides the first large-scale benchmark of foundation models for drug response prediction in single-cell data and serves as a user-friendly, flexible platform for advancing drug discovery and translational research. 

**Abstract (ZH)**: 药物抗性是癌症治疗中的一个重要挑战。单细胞测序提供了细胞异质性的见解，但大规模基础模型在单细胞数据中预测药物反应的应用尚未得到充分探索。为了解决这一问题，我们开发了scDrugMap，这是一种集成了Python命令行界面和网络服务器的综合框架，用于药物反应预测。scDrugMap评估了包括八种单细胞模型和两种大型语言模型在内的多种基础模型，使用了一个包含326,000个细胞的初级数据集和18,800个细胞的验证集，覆盖了36个数据集和多种组织及癌症类型。我们在混合数据和跨数据评估场景下对模型性能进行了基准测试，使用了层冻结和低秩适应（LoRA）微调策略。在混合数据场景下，scFoundation表现最佳，平均F1得分为0.971（层冻结）和0.947（微调），比最低性能模型高出50%以上。在跨数据场景下，UCE在微调后表现最佳（平均F1得分：0.774），而scGPT在零样本学习中表现出色（平均F1得分：0.858）。总体而言，scDrugMap为单细胞数据中的药物反应预测提供了第一个大规模基础模型基准，并提供了一个用户友好且灵活的平台，用于推动药物发现和转化研究。 

---
# HiBayES: A Hierarchical Bayesian Modeling Framework for AI Evaluation Statistics 

**Title (ZH)**: HiBayES：一种用于AI评估统计的分层贝叶斯建模框架 

**Authors**: Lennart Luettgau, Harry Coppock, Magda Dubois, Christopher Summerfield, Cozmin Ududec  

**Link**: [PDF](https://arxiv.org/pdf/2505.05602)  

**Abstract**: As Large Language Models (LLMs) and other AI systems evolve, robustly estimating their capabilities from inherently stochastic outputs while systematically quantifying uncertainty in these estimates becomes increasingly important. Further, advanced AI evaluations often have a nested hierarchical structure, exhibit high levels of complexity, and come with high costs in testing the most advanced AI systems. To address these challenges, we introduce HiBayES, a generalizable Hierarchical Bayesian modeling framework for AI Evaluation Statistics. HiBayES supports robust inferences in classical question-answer benchmarks and advanced agentic evaluations, particularly in low-data scenarios (e.g., < 20 data points per evaluation). Built on Generalized Linear Models (GLMs), Bayesian data analysis, and formal model comparison, HiBayES provides principled uncertainty quantification and robust parameter estimation. This paper offers a comprehensive introduction to HiBayES, including illustrative examples, comparisons to conventional statistical methods, and practical guidance for implementing multilevel Bayesian GLMs. Additionally, we provide a HiBayES software package [4] (Beta version) for out-of-the-box implementation. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）和其他AI系统的发展，从本质上具有随机性的输出中稳健估计其能力和系统地量化这些估计的不确定性变得越来越重要。此外，先进的AI评估往往具有嵌套的层级结构，表现出极高的复杂性，并且测试最先进AI系统需要高成本。为了解决这些问题，我们介绍了HiBayES，一种适用于AI评估统计的泛化层级贝叶斯建模框架。HiBayES 支持在经典问答基准测试和高级代理评估中进行稳健的推断，尤其是在低数据场景（例如，每个评估少于20个数据点）中。基于广义线性模型（GLMs）、贝叶斯数据分析和正式模型比较，HiBayES 提供了原则性的不确定性量化和稳健的参数估计。本文全面介绍了HiBayES，包括示例说明、与传统统计方法的比较以及实施多层次贝叶斯GLMs 的实用指导，并提供了HiBayES 软件包（Beta版本）以供直接使用。 

---
# Safety by Measurement: A Systematic Literature Review of AI Safety Evaluation Methods 

**Title (ZH)**: 基于度量的安全性保障：人工智能安全性评估方法系统文献综述 

**Authors**: Markov Grey, Charbel-Raphaël Segerie  

**Link**: [PDF](https://arxiv.org/pdf/2505.05541)  

**Abstract**: As frontier AI systems advance toward transformative capabilities, we need a parallel transformation in how we measure and evaluate these systems to ensure safety and inform governance. While benchmarks have been the primary method for estimating model capabilities, they often fail to establish true upper bounds or predict deployment behavior. This literature review consolidates the rapidly evolving field of AI safety evaluations, proposing a systematic taxonomy around three dimensions: what properties we measure, how we measure them, and how these measurements integrate into frameworks. We show how evaluations go beyond benchmarks by measuring what models can do when pushed to the limit (capabilities), the behavioral tendencies exhibited by default (propensities), and whether our safety measures remain effective even when faced with subversive adversarial AI (control). These properties are measured through behavioral techniques like scaffolding, red teaming and supervised fine-tuning, alongside internal techniques such as representation analysis and mechanistic interpretability. We provide deeper explanations of some safety-critical capabilities like cybersecurity exploitation, deception, autonomous replication, and situational awareness, alongside concerning propensities like power-seeking and scheming. The review explores how these evaluation methods integrate into governance frameworks to translate results into concrete development decisions. We also highlight challenges to safety evaluations - proving absence of capabilities, potential model sandbagging, and incentives for "safetywashing" - while identifying promising research directions. By synthesizing scattered resources, this literature review aims to provide a central reference point for understanding AI safety evaluations. 

**Abstract (ZH)**: 随着前沿人工智能系统向变革能力迈进，我们需要在衡量和评估这些系统方面进行相应的变革，以确保安全并指导治理。本文综述了快速发展的AI安全评估领域，提出了围绕三个维度的系统分类：我们测量的属性、如何测量以及这些测量如何整合到框架中。我们展示了评估方法如何超越基准，通过测量模型在极限状态下能做什么（能力）、默认情况下表现出的行为倾向（倾向性），以及即使面对颠覆性 adversarial AI 时，我们的安全措施是否依然有效（控制）。这些属性通过行为技术（如支架测试、红队测试和监督微调）以及内部技术（如表示分析和机制可解释性）进行衡量。我们对一些关键的安全能力（如网络空间利用、欺骗、自主复制和态势感知）进行了深入解释，并对令人担忧的倾向性（如追求权力和密谋）进行了讨论。本文综述了这些评估方法如何整合进治理框架，将结果转化为具体的开发决策。同时，我们指出了安全评估面临的挑战——证明不存在某项能力、潜在的模型水货以及追求“安全漂白”的激励机制——并指出了有希望的研究方向。通过整合零散的资源，本文综述旨在提供一个关于AI安全评估的理解中心参考点。 

---
# Let Humanoids Hike! Integrative Skill Development on Complex Trails 

**Title (ZH)**: 让类人形机器人远足！在复杂山路上的综合技能开发 

**Authors**: Kwan-Yee Lin, Stella X.Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06218)  

**Abstract**: Hiking on complex trails demands balance, agility, and adaptive decision-making over unpredictable terrain. Current humanoid research remains fragmented and inadequate for hiking: locomotion focuses on motor skills without long-term goals or situational awareness, while semantic navigation overlooks real-world embodiment and local terrain variability. We propose training humanoids to hike on complex trails, driving integrative skill development across visual perception, decision making, and motor execution. We develop a learning framework, LEGO-H, that enables a vision-equipped humanoid robot to hike complex trails autonomously. We introduce two technical innovations: 1) A temporal vision transformer variant - tailored into Hierarchical Reinforcement Learning framework - anticipates future local goals to guide movement, seamlessly integrating locomotion with goal-directed navigation. 2) Latent representations of joint movement patterns, combined with hierarchical metric learning - enhance Privileged Learning scheme - enable smooth policy transfer from privileged training to onboard execution. These components allow LEGO-H to handle diverse physical and environmental challenges without relying on predefined motion patterns. Experiments across varied simulated trails and robot morphologies highlight LEGO-H's versatility and robustness, positioning hiking as a compelling testbed for embodied autonomy and LEGO-H as a baseline for future humanoid development. 

**Abstract (ZH)**: 复杂地形徒步 demands 平衡、灵活性和适应性决策：当前的人形机器人研究仍然碎片化且不足以应对徒步：运动功能侧重于动作技能而缺乏长期目标或情境意识，而语义导航则忽视了现实世界的实体化和局部地形的变异性。我们提出训练人形机器人在复杂地形上徒步，以促进跨视觉感知、决策制定和运动执行的综合技能发展。我们开发了一种学习框架LEGO-H，使装备视觉的人形机器人能够自主徒步复杂地形。我们引入了两项技术创新：1) 一个时间视觉变换器变体，定制整合到层次强化学习框架中，预测未来局部目标以指导运动，无缝地将运动与目标导向导航相结合。2) 关节运动模式的潜在表示与层次度量学习结合，增强特权学习方案，使策略从特权训练平滑转移到船上执行。这些组件使LEGO-H能够处理多样的物理和环境挑战，而不依赖于预定义的运动模式。跨越各种模拟地形和机器人形态的实验突显了LEGO-H的多样性和鲁棒性，将徒步作为 embodied autonomy 的有吸引力测试平台，并将LEGO-H作为未来人形机器人发展的baseline。 

---
# Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies 

**Title (ZH)**: 基于查询驱动的生物医学研究文档级科学证据提取 

**Authors**: Massimiliano Pronesti, Joao Bettencourt-Silva, Paul Flanagan, Alessandra Pascale, Oisin Redmond, Anya Belz, Yufang Hou  

**Link**: [PDF](https://arxiv.org/pdf/2505.06186)  

**Abstract**: Extracting scientific evidence from biomedical studies for clinical research questions (e.g., Does stem cell transplantation improve quality of life in patients with medically refractory Crohn's disease compared to placebo?) is a crucial step in synthesising biomedical evidence. In this paper, we focus on the task of document-level scientific evidence extraction for clinical questions with conflicting evidence. To support this task, we create a dataset called CochraneForest, leveraging forest plots from Cochrane systematic reviews. It comprises 202 annotated forest plots, associated clinical research questions, full texts of studies, and study-specific conclusions. Building on CochraneForest, we propose URCA (Uniform Retrieval Clustered Augmentation), a retrieval-augmented generation framework designed to tackle the unique challenges of evidence extraction. Our experiments show that URCA outperforms the best existing methods by up to 10.3% in F1 score on this task. However, the results also underscore the complexity of CochraneForest, establishing it as a challenging testbed for advancing automated evidence synthesis systems. 

**Abstract (ZH)**: 从生物医学研究中提取科学证据以回答临床研究问题（例如，干细胞移植是否能改善医源性难治性克罗恩病患者的生活质量，与安慰剂相比？）是合成生物医学证据的关键步骤。本文聚焦于具有矛盾证据的临床问题的文档级别科学证据提取任务。为了支持这一任务，我们利用Cochrane系统评价中的森林图创建了CochraneForest数据集，该数据集包含202个标注的森林图、相关的临床研究问题、研究的全文及其特定结论。基于CochraneForest，我们提出了URCA（Uniform Retrieval Clustered Augmentation）检索增强生成框架，旨在应对证据提取的独特挑战。实验结果表明，URCA在F1分数上比现有最佳方法高10.3%，但结果也突显了CochraneForest的复杂性，将其确立为推动自动化证据合成系统的挑战性测试床。 

---
# Turbo-ICL: In-Context Learning-Based Turbo Equalization 

**Title (ZH)**: Turbo-ICL：基于上下文学习的 Turbo 等化 

**Authors**: Zihang Song, Matteo Zecchin, Bipin Rajendran, Osvaldo Simeone  

**Link**: [PDF](https://arxiv.org/pdf/2505.06175)  

**Abstract**: This paper introduces a novel in-context learning (ICL) framework, inspired by large language models (LLMs), for soft-input soft-output channel equalization in coded multiple-input multiple-output (MIMO) systems. The proposed approach learns to infer posterior symbol distributions directly from a prompt of pilot signals and decoder feedback. A key innovation is the use of prompt augmentation to incorporate extrinsic information from the decoder output as additional context, enabling the ICL model to refine its symbol estimates iteratively across turbo decoding iterations. Two model variants, based on Transformer and state-space architectures, are developed and evaluated. Extensive simulations demonstrate that, when traditional linear assumptions break down, e.g., in the presence of low-resolution quantization, ICL equalizers consistently outperform conventional model-based baselines, even when the latter are provided with perfect channel state information. Results also highlight the advantage of Transformer-based models under limited training diversity, as well as the efficiency of state-space models in resource-constrained scenarios. 

**Abstract (ZH)**: 本文提出了一种受大规模语言模型启发的新型上下文学习（ICL）框架，用于编码的多输入多输出（MIMO）系统中的软输入-软输出信道均衡。所提出的方法直接从探针信号和解码器反馈中学习后验符号分布。一个关键创新是通过探针增强引入解码器输出的外部信息作为额外上下文，使ICL模型能够在Turbo解码迭代过程中迭代 refinement 其符号估计。基于Transformer和状态空间架构发展了两种模型变体，并进行了评估。广泛的仿真实验表明，当传统的线性假设不成立，例如在低分辨率量化存在的情况下，ICL均衡器始终优于传统的基于模型的基础方案，即使后者拥有完美的信道状态信息。结果还强调了在有限的训练多样性下基于Transformer模型的优势，以及基于状态空间模型在资源受限场景下的效率。 

---
# MM-Skin: Enhancing Dermatology Vision-Language Model with an Image-Text Dataset Derived from Textbooks 

**Title (ZH)**: MM-Skin: 通过源自教科书的图像-文本数据集增强皮肤病视力语言模型 

**Authors**: Wenqi Zeng, Yuqi Sun, Chenxi Ma, Weimin Tan, Bo Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.06152)  

**Abstract**: Medical vision-language models (VLMs) have shown promise as clinical assistants across various medical fields. However, specialized dermatology VLM capable of delivering professional and detailed diagnostic analysis remains underdeveloped, primarily due to less specialized text descriptions in current dermatology multimodal datasets. To address this issue, we propose MM-Skin, the first large-scale multimodal dermatology dataset that encompasses 3 imaging modalities, including clinical, dermoscopic, and pathological and nearly 10k high-quality image-text pairs collected from professional textbooks. In addition, we generate over 27k diverse, instruction-following vision question answering (VQA) samples (9 times the size of current largest dermatology VQA dataset). Leveraging public datasets and MM-Skin, we developed SkinVL, a dermatology-specific VLM designed for precise and nuanced skin disease interpretation. Comprehensive benchmark evaluations of SkinVL on VQA, supervised fine-tuning (SFT) and zero-shot classification tasks across 8 datasets, reveal its exceptional performance for skin diseases in comparison to both general and medical VLM models. The introduction of MM-Skin and SkinVL offers a meaningful contribution to advancing the development of clinical dermatology VLM assistants. MM-Skin is available at this https URL 

**Abstract (ZH)**: 医学视觉语言模型（VLMs）在 various 医疗领域展现出作为临床助手的前景。然而，专门用于皮肤科且能够提供专业详细诊断分析的视觉语言模型仍处于初步发展阶段，主要原因是当前皮肤科多模态数据集中文本描述不够专门化。为了解决这一问题，我们提出了 MM-Skin，这是首个包含 3 种成像模态（临床、皮肤镜和病理）的大规模多模态皮肤科数据集，积累了近 10,000 个高质量图像-文本对，来自专业教科书。此外，我们还生成了超过 27,000 个多样且遵循指令的视觉问答（VQA）样本（比当前最大的皮肤科 VQA 数据集大 9 倍）。利用公开数据集和 MM-Skin，我们开发了 SkinVL，这是一种针对皮肤疾病的专属性视觉语言模型，旨在进行精确和细腻的皮肤疾病解释。在 8 个数据集上的视觉问答、监督微调和零样本分类基准评估中，SkinVL 在皮肤疾病任务上的表现明显优于通用和医学视觉语言模型。MM-Skin 的引入和 SkinVL 的开发对推动临床皮肤科视觉语言模型助手的发展具有重要意义。MM-Skin 可在以下链接获取：this https URL。 

---
# A Scaling Law for Token Efficiency in LLM Fine-Tuning Under Fixed Compute Budgets 

**Title (ZH)**: 固定计算预算下LLM细调中令牌效率的标度律 

**Authors**: Ryan Lagasse, Aidan Kiernans, Avijit Ghosh, Shiri Dori-Hacohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06150)  

**Abstract**: We introduce a scaling law for fine-tuning large language models (LLMs) under fixed compute budgets that explicitly accounts for data composition. Conventional approaches measure training data solely by total tokens, yet the number of examples and their average token length -- what we term \emph{dataset volume} -- play a decisive role in model performance. Our formulation is tuned following established procedures. Experiments on the BRICC dataset \cite{salavati2024reducing} and subsets of the MMLU dataset \cite{hendrycks2021measuringmassivemultitasklanguage}, evaluated under multiple subsampling strategies, reveal that data composition significantly affects token efficiency. These results motivate refined scaling laws for practical LLM fine-tuning in resource-constrained settings. 

**Abstract (ZH)**: 我们介绍了一种在固定计算预算下调整大型语言模型（LLMs）的缩放定律，该定律明确考虑了数据组成。 

---
# Efficient Sensorimotor Learning for Open-world Robot Manipulation 

**Title (ZH)**: 开放世界机器人操作的高效感觉运动学习 

**Authors**: Yifeng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.06136)  

**Abstract**: This dissertation considers Open-world Robot Manipulation, a manipulation problem where a robot must generalize or quickly adapt to new objects, scenes, or tasks for which it has not been pre-programmed or pre-trained. This dissertation tackles the problem using a methodology of efficient sensorimotor learning. The key to enabling efficient sensorimotor learning lies in leveraging regular patterns that exist in limited amounts of demonstration data. These patterns, referred to as ``regularity,'' enable the data-efficient learning of generalizable manipulation skills. This dissertation offers a new perspective on formulating manipulation problems through the lens of regularity. Building upon this notion, we introduce three major contributions. First, we introduce methods that endow robots with object-centric priors, allowing them to learn generalizable, closed-loop sensorimotor policies from a small number of teleoperation demonstrations. Second, we introduce methods that constitute robots' spatial understanding, unlocking their ability to imitate manipulation skills from in-the-wild video observations. Last but not least, we introduce methods that enable robots to identify reusable skills from their past experiences, resulting in systems that can continually imitate multiple tasks in a sequential manner. Altogether, the contributions of this dissertation help lay the groundwork for building general-purpose personal robots that can quickly adapt to new situations or tasks with low-cost data collection and interact easily with humans. By enabling robots to learn and generalize from limited data, this dissertation takes a step toward realizing the vision of intelligent robotic assistants that can be seamlessly integrated into everyday scenarios. 

**Abstract (ZH)**: 本论文探讨开放世界机器人操作问题，即机器人必须在没有预先编程或训练的情况下，泛化或快速适应新对象、场景或任务的操作问题。本论文通过高效传感器运动学习的方法来应对这一问题。使高效传感器运动学习成为可能的关键在于利用有限演示数据中存在的规律模式。这些模式被称为“规律性”，能够使机器人高效学习可泛化的操作技能。本论文从规律性的视角出发，提出了操作问题的一种新视角，并提出了三项主要贡献。首先，我们介绍了使机器人具备以对象为中心的先验的方法，从而使其能够从少量的遥控演示中学习通用的闭环传感器运动策略。其次，我们介绍了使机器人具备空间理解能力的方法，从而解锁它们从野生视频观察中模仿操作技能的能力。最后，我们介绍了使机器人能够识别从过往经验中可重用的技能的方法，从而构建出能够以序列方式持续模仿多种任务的系统。综上所述，本论文的贡献为构建能够在低成本数据收集下快速适应新情况或任务的通用个人机器人奠定了基础，并使机器人能够容易地与人类交互。通过使机器人能够从有限的数据中学习和泛化，本论文朝着实现无缝集成到日常生活场景中的智能机器人助手愿景迈出了一步。 

---
# Wasserstein Distances Made Explainable: Insights into Dataset Shifts and Transport Phenomena 

**Title (ZH)**: Wasserstein 距离的可解释性：数据集偏移和传输现象的见解 

**Authors**: Philip Naumann, Jacob Kauffmann, Grégoire Montavon  

**Link**: [PDF](https://arxiv.org/pdf/2505.06123)  

**Abstract**: Wasserstein distances provide a powerful framework for comparing data distributions. They can be used to analyze processes over time or to detect inhomogeneities within data. However, simply calculating the Wasserstein distance or analyzing the corresponding transport map (or coupling) may not be sufficient for understanding what factors contribute to a high or low Wasserstein distance. In this work, we propose a novel solution based on Explainable AI that allows us to efficiently and accurately attribute Wasserstein distances to various data components, including data subgroups, input features, or interpretable subspaces. Our method achieves high accuracy across diverse datasets and Wasserstein distance specifications, and its practical utility is demonstrated in two use cases. 

**Abstract (ZH)**: Wasserstein 距离提供了一个强大的框架用于比较数据分布，可以用于分析数据随时间的变化或检测数据中的不均匀性。然而，仅仅计算 Wasserstein 距离或分析相应的运输映射（或耦合）可能不足以理解导致高或低 Wasserstein 距离的因素。在本文中，我们提出了一种基于可解释人工智能的新解决方案，该解决方案能够高效且准确地将 Wasserstein 距离归因于各种数据组件，包括数据子组、输入特征或可解释子空间。我们的方法在多种数据集和 Wasserstein 距离规格下实现了高准确性，并在两个应用场景中证明了其实用价值。 

---
# The Application of Deep Learning for Lymph Node Segmentation: A Systematic Review 

**Title (ZH)**: 深度学习在淋巴结分割中的应用：一项系统性回顾 

**Authors**: Jingguo Qu, Xinyang Han, Man-Lik Chui, Yao Pu, Simon Takadiyi Gunda, Ziman Chen, Jing Qin, Ann Dorothy King, Winnie Chiu-Wing Chu, Jing Cai, Michael Tin-Cheung Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.06118)  

**Abstract**: Automatic lymph node segmentation is the cornerstone for advances in computer vision tasks for early detection and staging of cancer. Traditional segmentation methods are constrained by manual delineation and variability in operator proficiency, limiting their ability to achieve high accuracy. The introduction of deep learning technologies offers new possibilities for improving the accuracy of lymph node image analysis. This study evaluates the application of deep learning in lymph node segmentation and discusses the methodologies of various deep learning architectures such as convolutional neural networks, encoder-decoder networks, and transformers in analyzing medical imaging data across different modalities. Despite the advancements, it still confronts challenges like the shape diversity of lymph nodes, the scarcity of accurately labeled datasets, and the inadequate development of methods that are robust and generalizable across different imaging modalities. To the best of our knowledge, this is the first study that provides a comprehensive overview of the application of deep learning techniques in lymph node segmentation task. Furthermore, this study also explores potential future research directions, including multimodal fusion techniques, transfer learning, and the use of large-scale pre-trained models to overcome current limitations while enhancing cancer diagnosis and treatment planning strategies. 

**Abstract (ZH)**: 自动淋巴结分割是推进早期癌症检测和分期的计算机视觉任务中的基石。传统的分割方法受限于手动勾画和操作者熟练程度的差异，限制了其达到高准确性的能力。深度学习技术的引入为提高淋巴结图像分析的准确性提供了新的可能性。本研究评估了深度学习在淋巴结分割中的应用，并讨论了各种深度学习架构（如卷积神经网络、编码器-解码器网络和变压器）在不同医学成像模态数据分析中的方法学。尽管取得了进展，但仍然面临淋巴结形状多样性、准确标注数据集稀缺以及方法难以在不同成像模态之间稳健且普适的问题。据我们所知，这是第一篇全面概述深度学习技术在淋巴结分割任务中应用的研究。此外，本研究还探讨了潜在的未来研究方向，包括多模态融合技术、迁移学习以及使用大规模预训练模型来克服当前局限性，从而增强癌症诊断和治疗规划策略。 

---
# UniVLA: Learning to Act Anywhere with Task-centric Latent Actions 

**Title (ZH)**: UniVLA: 学习在任何地方执行任务导向的潜在动作 

**Authors**: Qingwen Bu, Yanting Yang, Jisong Cai, Shenyuan Gao, Guanghui Ren, Maoqing Yao, Ping Luo, Hongyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.06111)  

**Abstract**: A generalist robot should perform effectively across various environments. However, most existing approaches heavily rely on scaling action-annotated data to enhance their capabilities. Consequently, they are often limited to single physical specification and struggle to learn transferable knowledge across different embodiments and environments. To confront these limitations, we propose UniVLA, a new framework for learning cross-embodiment vision-language-action (VLA) policies. Our key innovation is to derive task-centric action representations from videos with a latent action model. This enables us to exploit extensive data across a wide spectrum of embodiments and perspectives. To mitigate the effect of task-irrelevant dynamics, we incorporate language instructions and establish a latent action model within the DINO feature space. Learned from internet-scale videos, the generalist policy can be deployed to various robots through efficient latent action decoding. We obtain state-of-the-art results across multiple manipulation and navigation benchmarks, as well as real-robot deployments. UniVLA achieves superior performance over OpenVLA with less than 1/20 of pretraining compute and 1/10 of downstream data. Continuous performance improvements are observed as heterogeneous data, even including human videos, are incorporated into the training pipeline. The results underscore UniVLA's potential to facilitate scalable and efficient robot policy learning. 

**Abstract (ZH)**: 一种通用机器人应该在各种环境中有效执行。然而，大多数现有方法严重依赖于扩展动作标注数据以提升其能力，从而导致它们往往受限于单一的物理规格，并且难以在不同的体态和环境中学习可迁移的知识。为应对这些限制，我们提出了一种新的框架UniVLA，用于学习跨体态的视觉-语言-动作（VLA）策略。我们的关键创新是从视频中通过潜在动作模型提取以任务为中心的动作表示，这使得可以从广泛的体态和视角中获取大量数据。为减轻与任务无关的动力学影响，我们引入了语言指令并在DINO特征空间中建立潜在动作模型。通过从互联网规模的视频中学习到的通用策略，可以通过高效的潜在动作解码应用于各种机器人。我们在多个操作和导航基准测试以及实际机器人部署中取得了最先进的结果。与OpenVLA相比，UniVLA在预训练计算量不到其1/20、下游数据量不到其1/10的情况下实现了更优性能。随着异构数据，甚至包括人类视频的数据被纳入训练管道，持续的性能提升被观察到。结果强调了UniVLA在促进可扩展和高效的机器人策略学习方面具有巨大潜力。 

---
# Multimodal Sentiment Analysis on CMU-MOSEI Dataset using Transformer-based Models 

**Title (ZH)**: 基于Transformer模型的CMU-MOSEI多模态情感分析 

**Authors**: Jugal Gajjar, Kaustik Ranaware  

**Link**: [PDF](https://arxiv.org/pdf/2505.06110)  

**Abstract**: This project performs multimodal sentiment analysis using the CMU-MOSEI dataset, using transformer-based models with early fusion to integrate text, audio, and visual modalities. We employ BERT-based encoders for each modality, extracting embeddings that are concatenated before classification. The model achieves strong performance, with 97.87\% 7-class accuracy and a 0.9682 F1-score on the test set, demonstrating the effectiveness of early fusion in capturing cross-modal interactions. The training utilized Adam optimization (lr=1e-4), dropout (0.3), and early stopping to ensure generalization and robustness. Results highlight the superiority of transformer architectures in modeling multimodal sentiment, with a low MAE (0.1060) indicating precise sentiment intensity prediction. Future work may compare fusion strategies or enhance interpretability. This approach utilizes multimodal learning by effectively combining linguistic, acoustic, and visual cues for sentiment analysis. 

**Abstract (ZH)**: 本项目使用CMU-MOSEI数据集进行多模态情感分析，采用基于Transformer的模型并采用早期融合策略整合文本、音频和视觉模态。我们为每个模态使用BERT编码器提取嵌入，然后在分类前进行拼接。模型在测试集上达到97.87%的7类准确率和0.9682的F1分数，证明了早期融合在捕捉跨模态交互方面的有效性。训练过程中采用Adam优化（学习率1e-4）、dropout（0.3）和早停策略以确保泛化能力和稳健性。结果表明，Transformer架构在建模多模态情感方面具有优势，较低的MAE（0.1060）表明情感强度预测的精确性。未来工作可能会比较不同的融合策略或增强可解释性。本方法通过有效结合语言、声学和视觉线索来进行情感分析。 

---
# LLMs Outperform Experts on Challenging Biology Benchmarks 

**Title (ZH)**: LLMs在具有挑战性的生物学基准测试中 surpass 专家 

**Authors**: Lennart Justen  

**Link**: [PDF](https://arxiv.org/pdf/2505.06108)  

**Abstract**: This study systematically evaluates 27 frontier Large Language Models on eight diverse biology benchmarks spanning molecular biology, genetics, cloning, virology, and biosecurity. Models from major AI developers released between November 2022 and April 2025 were assessed through ten independent runs per benchmark. The findings reveal dramatic improvements in biological capabilities. Top model performance increased more than 4-fold on the challenging text-only subset of the Virology Capabilities Test over the study period, with the top model now performing twice as well as expert virologists. Several models now match or exceed expert-level performance on other challenging benchmarks, including LAB-Bench CloningScenarios and the biology subsets of GPQA and WMDP. Contrary to expectations, chain-of-thought did not substantially improve performance over zero-shot evaluation, while extended reasoning features in o3-mini and Claude 3.7 Sonnet typically improved performance as predicted by inference scaling. Benchmarks such as PubMedQA and the MMLU and WMDP biology subsets exhibited performance plateaus well below 100%, suggesting benchmark saturation and errors in the underlying benchmark data. The analysis highlights the need for more sophisticated evaluation methodologies as AI systems continue to advance. 

**Abstract (ZH)**: 本研究系统性评估了27个前沿大型语言模型在涵盖分子生物学、遗传学、克隆、病毒学和生物安全等八项多元生物学基准测试中的表现。从2022年11月到2025年4月期间发布的主流AI开发者的模型，每项基准测试进行了十次独立运行评估。研究结果揭示了生物能力方面的显著进步。顶级模型在病毒学能力测试的挑战性纯文本子集中的表现，在研究期间提升了4倍以上，顶级模型现在比专家病毒学家的绩效高出一倍。多项模型在其他挑战性基准测试中达到了或超过了专家级绩效，包括LAB-Bench克隆场景和GPQA及WMDP的生物学子集。与预期相反，推理链条并未在零样本评估中显著提升性能，而o3-mini和Claude 3.7 Sonnet中的扩展推理特征通常如预期的推断缩放所示，提升了性能。诸如PubMedQA和MMLU及WMDP的生物学子集等基准测试的表现 plateau 处于低于100%的水平，这表明基准测试饱和和基础数据中的错误。分析强调，随着AI系统继续进步，需要更加复杂的评估方法。 

---
# UniSymNet: A Unified Symbolic Network Guided by Transformer 

**Title (ZH)**: UniSymNet：由Transformer引导的统一符号网络 

**Authors**: Xinxin Li, Juan Zhang, Da Li, Xingyu Liu, Jin Xu, Junping Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06091)  

**Abstract**: Symbolic Regression (SR) is a powerful technique for automatically discovering mathematical expressions from input data. Mainstream SR algorithms search for the optimal symbolic tree in a vast function space, but the increasing complexity of the tree structure limits their performance. Inspired by neural networks, symbolic networks have emerged as a promising new paradigm. However, most existing symbolic networks still face certain challenges: binary nonlinear operators $\{\times, ÷\}$ cannot be naturally extended to multivariate operators, and training with fixed architecture often leads to higher complexity and overfitting. In this work, we propose a Unified Symbolic Network that unifies nonlinear binary operators into nested unary operators and define the conditions under which UniSymNet can reduce complexity. Moreover, we pre-train a Transformer model with a novel label encoding method to guide structural selection, and adopt objective-specific optimization strategies to learn the parameters of the symbolic network. UniSymNet shows high fitting accuracy, excellent symbolic solution rate, and relatively low expression complexity, achieving competitive performance on low-dimensional Standard Benchmarks and high-dimensional SRBench. 

**Abstract (ZH)**: 统一符号网络（UniSymNet）：统一非线性二元运算并减小表达式复杂度 

---
# Assessing Tenstorrent's RISC-V MatMul Acceleration Capabilities 

**Title (ZH)**: 评估Tenstorrent的RISC-V矩阵乘法加速能力 

**Authors**: Hiari Pizzini Cavagna, Daniele Cesarini, Andrea Bartolini  

**Link**: [PDF](https://arxiv.org/pdf/2505.06085)  

**Abstract**: The increasing demand for generative AI as Large Language Models (LLMs) services has driven the need for specialized hardware architectures that optimize computational efficiency and energy consumption. This paper evaluates the performance of the Tenstorrent Grayskull e75 RISC-V accelerator for basic linear algebra kernels at reduced numerical precision, a fundamental operation in LLM computations. We present a detailed characterization of Grayskull's execution model, gridsize, matrix dimensions, data formats, and numerical precision impact computational efficiency. Furthermore, we compare Grayskull's performance against state-of-the-art architectures with tensor acceleration, including Intel Sapphire Rapids processors and two NVIDIA GPUs (V100 and A100). Whilst NVIDIA GPUs dominate raw performance, Grayskull demonstrates a competitive trade-off between power consumption and computational throughput, reaching a peak of 1.55 TFLOPs/Watt with BF16. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）服务对生成人工智能日益增长的需求，已经推动了优化计算效率和能耗的专用硬件架构的发展。本文评估了Tenstorrent Grayskull e75 RISC-V加速器在降低数值精度下基本线性代数内核的性能，这是LLM计算中的基本操作。本文详细分析了Grayskull的执行模型、网格尺寸、矩阵维度、数据格式及其对计算效率的影响。此外，我们将Grayskull的性能与具有张量加速的最新架构进行了比较，包括Intel Sapphire Rapids处理器和两台NVIDIA GPU（V100和A100）。虽然NVIDIA GPU在原始性能上占据优势，但Grayskull在能耗和计算吞吐量之间的权衡表现出竞争力，BF16精度下的峰值性能达到1.55 TFLOPs/W。 

---
# PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks 

**Title (ZH)**: PYRREGULAR：统一的时间序列框架，包含不规则时间序列分类基准 

**Authors**: Francesco Spinnato, Cristiano Landi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06047)  

**Abstract**: Irregular temporal data, characterized by varying recording frequencies, differing observation durations, and missing values, presents significant challenges across fields like mobility, healthcare, and environmental science. Existing research communities often overlook or address these challenges in isolation, leading to fragmented tools and methods. To bridge this gap, we introduce a unified framework, and the first standardized dataset repository for irregular time series classification, built on a common array format to enhance interoperability. This repository comprises 34 datasets on which we benchmark 12 classifier models from diverse domains and communities. This work aims to centralize research efforts and enable a more robust evaluation of irregular temporal data analysis methods. 

**Abstract (ZH)**: 不规则时间序列数据，特征表现为记录频率各异、观测持续时间不同以及缺失值，给移动性、医疗保健和环境科学等领域带来了重大挑战。现有研究社区往往忽视或孤立地解决这些挑战，导致工具和方法碎片化。为弥补这一差距，我们提出了一种统一框架，并构建了首个标准化的时间序列分类数据集仓库，基于统一数组格式以增强互操作性。该仓库包含34个数据集，在这些数据集上我们对12种来自不同领域和社区的分类器模型进行了基准测试。这项工作旨在集中研究力量，并促进对不规则时间序列数据分析方法的更稳健评估。 

---
# Universal Approximation Theorem for Deep Q-Learning via FBSDE System 

**Title (ZH)**: 深度Q学习通过FBSDE系统的一种普遍逼近定理 

**Authors**: Qian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06023)  

**Abstract**: The approximation capabilities of Deep Q-Networks (DQNs) are commonly justified by general Universal Approximation Theorems (UATs) that do not leverage the intrinsic structural properties of the optimal Q-function, the solution to a Bellman equation. This paper establishes a UAT for a class of DQNs whose architecture is designed to emulate the iterative refinement process inherent in Bellman updates. A central element of our analysis is the propagation of regularity: while the transformation induced by a single Bellman operator application exhibits regularity, for which Backward Stochastic Differential Equations (BSDEs) theory provides analytical tools, the uniform regularity of the entire sequence of value iteration iterates--specifically, their uniform Lipschitz continuity on compact domains under standard Lipschitz assumptions on the problem data--is derived from finite-horizon dynamic programming principles. We demonstrate that layers of a deep residual network, conceived as neural operators acting on function spaces, can approximate the action of the Bellman operator. The resulting approximation theorem is thus intrinsically linked to the control problem's structure, offering a proof technique wherein network depth directly corresponds to iterations of value function refinement, accompanied by controlled error propagation. This perspective reveals a dynamic systems view of the network's operation on a space of value functions. 

**Abstract (ZH)**: Deep Q-Networks的逼近能力：基于贝尔曼更新内在结构的统一逼近定理 

---
# Minimal Sequent Calculus for Teaching First-Order Logic: Lessons Learned 

**Title (ZH)**: 面向教学的命题逻辑最小 sequent 算法：经验教训 

**Authors**: Jørgen Villadsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.05988)  

**Abstract**: MiniCalc is a web app for teaching first-order logic based on a minimal sequent calculus. As an option the proofs can be verified in the Isabelle proof assistant. We present the lessons learned using the tool in recent years at our university. 

**Abstract (ZH)**: MiniCalc 是一个基于极小化 sequent  calculus 教授一阶逻辑的网络应用。作为一种选择，证明可以在 Isabelle 证明辅助系统中验证。我们介绍了近年来在我们大学使用该工具所获得的经验。 

---
# A Noise-Resilient Semi-Supervised Graph Autoencoder for Overlapping Semantic Community Detection 

**Title (ZH)**: 一种抗噪声的半监督图自编码器及其在重叠语义社区检测中的应用 

**Authors**: Abdelfateh Bekkair, Slimane Bellaouar, Slimane Oulad-Naoui  

**Link**: [PDF](https://arxiv.org/pdf/2505.05965)  

**Abstract**: Community detection in networks with overlapping structures remains a significant challenge, particularly in noisy real-world environments where integrating topology, node attributes, and prior information is critical. To address this, we propose a semi-supervised graph autoencoder that combines graph multi-head attention and modularity maximization to robustly detect overlapping communities. The model learns semantic representations by fusing structural, attribute, and prior knowledge while explicitly addressing noise in node features. Key innovations include a noise-resistant architecture and a semantic semi-supervised design optimized for community quality through modularity constraints. Experiments demonstrate superior performance the model outperforms state-of-the-art methods in overlapping community detection (improvements in NMI and F1-score) and exhibits exceptional robustness to attribute noise, maintaining stable performance under 60\% feature corruption. These results highlight the importance of integrating attribute semantics and structural patterns for accurate community discovery in complex networks. 

**Abstract (ZH)**: 具有重叠结构的网络中的社区检测仍然是一个显著的挑战，特别是在嘈杂的现实环境中，拓扑结构、节点属性和先验信息的集成尤为关键。为了解决这一问题，我们提出了一种半监督图自动编码器，该模型结合了图多头注意力和模块化最大化方法，以稳健地检测重叠社区。该模型通过融合结构、属性和先验知识来学习语义表示，并明确地解决了节点特征中的噪声问题。关键创新包括抗噪架构和通过模块化约束优化的语义半监督设计，旨在提高社区质量。实验结果表明，该模型在重叠社区检测（NMI和F1分值提升）中性能 superior，并且对属性噪声表现出色的鲁棒性，在节点特征 corruption 达到 60% 的情况下仍能保持稳定的性能。这些结果突显了在复杂网络中准确发现社区的重要性，需要整合属性语义和结构模式。 

---
# Elastic Weight Consolidation for Full-Parameter Continual Pre-Training of Gemma2 

**Title (ZH)**: Gemma2全参数连续预训练的弹性权重聚合 

**Authors**: Vytenis Šliogeris, Povilas Daniušis, Artūras Nakvosas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05946)  

**Abstract**: This technical report describes an experiment on autoregressive pre-training of Gemma2 2 billion parameter large language model (LLM) with 10\% on the Lithuanian language component of CulturaX from the point of view of continual learning. We apply elastic weight consolidation (EWC) to the full set of the model's parameters and investigate language understanding benchmarks, consisting of Arc, Belebele, Gsm8K, Hellaswag, MMLU, TruthfulQA, and Winogrande sets (both in English and Lithuanian versions), and perplexity benchmarks. We empirically demonstrate that EWC regularisation allows us not only to mitigate catastrophic forgetting effects but also that it is potentially beneficial for learning of the new task with LLMs. 

**Abstract (ZH)**: 本技术报告从持续学习的角度描述了使用CulturaX中的10%立陶宛语言组件对Gemma2 2亿参数大型语言模型（LLM）进行自回归预训练的实验。我们对模型的所有参数应用弹性权重聚合（EWC），并调查了由Arc、Belebele、Gsm8K、Hellaswag、MMLU、TruthfulQA和Winogrande数据集（包括英语和立陶宛语版本）组成的语言理解基准和困惑度基准。我们实验证明，EWC正则化不仅可以减轻灾难性遗忘效应，还可能对使用LLM学习新任务有益。 

---
# Achieving 3D Attention via Triplet Squeeze and Excitation Block 

**Title (ZH)**: 通过三重挤压与激励模块实现3D注意力 

**Authors**: Maan Alhazmi, Abdulrahman Altahhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.05943)  

**Abstract**: The emergence of ConvNeXt and its variants has reaffirmed the conceptual and structural suitability of CNN-based models for vision tasks, re-establishing them as key players in image classification in general, and in facial expression recognition (FER) in particular. In this paper, we propose a new set of models that build on these advancements by incorporating a new set of attention mechanisms that combines Triplet attention with Squeeze-and-Excitation (TripSE) in four different variants. We demonstrate the effectiveness of these variants by applying them to the ResNet18, DenseNet and ConvNext architectures to validate their versatility and impact. Our study shows that incorporating a TripSE block in these CNN models boosts their performances, particularly for the ConvNeXt architecture, indicating its utility. We evaluate the proposed mechanisms and associated models across four datasets, namely CIFAR100, ImageNet, FER2013 and AffectNet datasets, where ConvNext with TripSE achieves state-of-the-art results with an accuracy of \textbf{78.27\%} on the popular FER2013 dataset, a new feat for this dataset. 

**Abstract (ZH)**: ConvNeXt及其变种的出现再次证实了基于CNN的模型在视觉任务中的概念和结构适宜性，重新确立了它们在图像分类以及面部表情识别（FER）中的关键地位。本文提出了一种新的模型集合，通过将Triple Attention与Squeeze-and-Excitation (TripSE)结合，在四种不同的变体中引入新的注意力机制。我们通过将这些变体应用于ResNet18、DenseNet和ConvNeXt架构，验证了它们的通用性和影响。研究表明，在这些CNN模型中加入TripSE模块可以提升其性能，特别是在ConvNeXt架构中效果显著，显示了其实用性。我们在CIFAR100、ImageNet、FER2013和AffectNet四个数据集中评估了所提出机制及其相关模型，其中使用ConvNeXt和TripSE的模型在流行的数据集FER2013上取得了78.27%的准确率，是该数据集的一个新成就。 

---
# IRNN: Innovation-driven Recurrent Neural Network for Time-Series Data Modeling and Prediction 

**Title (ZH)**: IRNN：创新驱动的循环神经网络及其在时间序列数据建模与预测中的应用 

**Authors**: Yifan Zhou, Yibo Wang, Chao Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05916)  

**Abstract**: Many real-world datasets are time series that are sequentially collected and contain rich temporal information. Thus, a common interest in practice is to capture dynamics of time series and predict their future evolutions. To this end, the recurrent neural network (RNN) has been a prevalent and effective machine learning option, which admits a nonlinear state-space model representation. Motivated by the resemblance between RNN and Kalman filter (KF) for linear state-space models, we propose in this paper Innovation-driven RNN (IRNN), a novel RNN architecture tailored to time-series data modeling and prediction tasks. By adapting the concept of "innovation" from KF to RNN, past prediction errors are adopted as additional input signals to update hidden states of RNN and boost prediction performance. Since innovation data depend on network parameters, existing training algorithms for RNN do not apply to IRNN straightforwardly. Thus, a tailored training algorithm dubbed input updating-based back-propagation through time (IU-BPTT) is further proposed, which alternates between updating innovations and optimizing network parameters via gradient descent. Experiments on real-world benchmark datasets show that the integration of innovations into various forms of RNN leads to remarkably improved prediction accuracy of IRNN without increasing the training cost substantially. 

**Abstract (ZH)**: 创新驱动递归神经网络：一种适用于时间序列数据建模和预测的新架构 

---
# Examining the Source of Defects from a Mechanical Perspective for 3D Anomaly Detection 

**Title (ZH)**: 从机械角度探究3D异常检测中缺陷的来源 

**Authors**: Hanzhe Liang, Aoran Wang, Jie Zhou, Xin Jin, Can Gao, Jinbao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05901)  

**Abstract**: In this paper, we go beyond identifying anomalies only in structural terms and think about better anomaly detection motivated by anomaly causes. Most anomalies are regarded as the result of unpredictable defective forces from internal and external sources, and their opposite forces are sought to correct the anomalies. We introduced a Mechanics Complementary framework for 3D anomaly detection (MC4AD) to generate internal and external Corrective forces for each point. A Diverse Anomaly-Generation (DA-Gen) module is first proposed to simulate various anomalies. Then, we present a Corrective Force Prediction Network (CFP-Net) with complementary representations for point-level representation to simulate the different contributions of internal and external corrective forces. A combined loss was proposed, including a new symmetric loss and an overall loss, to constrain the corrective forces properly. As a highlight, we consider 3D anomaly detection in industry more comprehensively, creating a hierarchical quality control strategy based on a three-way decision and contributing a dataset named Anomaly-IntraVariance with intraclass variance to evaluate the model. On the proposed and existing five datasets, we obtained nine state-of-the-art performers with the minimum parameters and the fastest inference speed. The source is available at this https URL 

**Abstract (ZH)**: 本文超越了仅从结构角度识别异常的做法，从异常成因出发，提出更好的异常检测方法。大多数异常被视为来自内部和外部的不可预测缺陷力量的结果，我们引入了一个机械互补框架（MC4AD）用于3D异常检测，以生成每个点的内部和外部矫正力。首先提出了一个多样性异常生成（DA-Gen）模块来模拟各种异常。然后，我们提出了一种点级表示的互补表示矫正力预测网络（CFP-Net），以模拟内部和外部矫正力的不同贡献。提出了一种结合损失函数，包括一种新的对称损失和总体损失，以适当地约束矫正力。作为亮点，本文更全面地考虑了工业中的3D异常检测，基于三元决策创建了分级质量控制策略，并贡献了一个名为Anomaly-IntraVariance的数据集，用于评估模型。在所提出的和现有的五个数据集上，我们获得了最少参数和最快推理速度的九个当前最佳表现者。源代码可在以下链接获取。 

---
# Leveraging Vision-Language Models for Visual Grounding and Analysis of Automotive UI 

**Title (ZH)**: 利用视觉语言模型进行汽车UI的视觉定位与分析 

**Authors**: Benjamin Raphael Ernhofer, Daniil Prokhorov, Jannica Langner, Dominik Bollmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.05895)  

**Abstract**: Modern automotive infotainment systems require intelligent and adaptive solutions to handle frequent User Interface (UI) updates and diverse design variations. We introduce a vision-language framework for understanding and interacting with automotive infotainment systems, enabling seamless adaptation across different UI designs. To further support research in this field, we release AutomotiveUI-Bench-4K, an open-source dataset of 998 images with 4,208 annotations. Additionally, we present a synthetic data pipeline to generate training data. We fine-tune a Molmo-7B-based model using Low-Rank Adaptation (LoRa) and incorporating reasoning generated by our pipeline, along with visual grounding and evaluation capabilities. The fine-tuned Evaluative Large Action Model (ELAM) achieves strong performance on AutomotiveUI-Bench-4K (model and dataset are available on Hugging Face) and demonstrating strong cross-domain generalization, including a +5.2% improvement on ScreenSpot over the baseline model. Notably, our approach achieves 80.4% average accuracy on ScreenSpot, closely matching or even surpassing specialized models for desktop, mobile, and web, such as ShowUI, despite being trained for the infotainment domain. This research investigates how data collection and subsequent fine-tuning can lead to AI-driven progress within automotive UI understanding and interaction. The applied method is cost-efficient and fine-tuned models can be deployed on consumer-grade GPUs. 

**Abstract (ZH)**: 现代汽车娱乐系统需要智能且适应性强的解决方案来处理频繁的用户界面更新和多样化的设计变化。我们介绍了一种vision-language框架，用于理解和与汽车娱乐系统交互， enabling无缝跨不同UI设计的适应性。为了进一步支持该领域的研究，我们发布了AutomotiveUI-Bench-4K，这是一个包含998张图像和4,208个注释的开源数据集。此外，我们呈现了一种合成数据管道来生成训练数据。我们使用低秩适应（LoRa）并结合由我们的管道生成的推理、视觉定位和评估能力，对基于Molmo-7B的模型进行了微调。微调后的Evaluative Large Action Model (ELAM)在AutomotiveUI-Bench-4K上表现出较强的效果（模型和数据集可在Hugging Face上获得），并在跨域泛化方面表现出色，包括在ScreenSpot上比基线模型提高了5.2%。值得注意的是，我们的方法在ScreenSpot上的平均准确率为80.4%，接近或甚至超过了专门为桌面、移动和网络设计的ShowUI等专用模型，尽管它是为娱乐系统领域训练的。本研究探讨了数据收集和后续微调如何推动汽车UI理解与交互的AI驱动进步。应用的方法经济高效，微调后的模型可以部署在消费级GPU上。 

---
# LightNobel: Improving Sequence Length Limitation in Protein Structure Prediction Model via Adaptive Activation Quantization 

**Title (ZH)**: LightNobel：通过自适应激活量化的蛋白质结构预测模型中序列长度限制的改进 

**Authors**: Seunghee Han, Soongyu Choi, Joo-Young Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.05893)  

**Abstract**: Recent advances in Protein Structure Prediction Models (PPMs), such as AlphaFold2 and ESMFold, have revolutionized computational biology by achieving unprecedented accuracy in predicting three-dimensional protein folding structures. However, these models face significant scalability challenges, particularly when processing proteins with long amino acid sequences (e.g., sequence length > 1,000). The primary bottleneck that arises from the exponential growth in activation sizes is driven by the unique data structure in PPM, which introduces an additional dimension that leads to substantial memory and computational demands. These limitations have hindered the effective scaling of PPM for real-world applications, such as analyzing large proteins or complex multimers with critical biological and pharmaceutical relevance.
In this paper, we present LightNobel, the first hardware-software co-designed accelerator developed to overcome scalability limitations on the sequence length in PPM. At the software level, we propose Token-wise Adaptive Activation Quantization (AAQ), which leverages unique token-wise characteristics, such as distogram patterns in PPM activations, to enable fine-grained quantization techniques without compromising accuracy. At the hardware level, LightNobel integrates the multi-precision reconfigurable matrix processing unit (RMPU) and versatile vector processing unit (VVPU) to enable the efficient execution of AAQ. Through these innovations, LightNobel achieves up to 8.44x, 8.41x speedup and 37.29x, 43.35x higher power efficiency over the latest NVIDIA A100 and H100 GPUs, respectively, while maintaining negligible accuracy loss. It also reduces the peak memory requirement up to 120.05x in PPM, enabling scalable processing for proteins with long sequences. 

**Abstract (ZH)**: Recent Advances in Protein Structure Prediction Models (PPMs) and Their Scalability Challenges: Introducing LightNobel, a Hardware-software Co-designed Accelerator 

---
# Multi-Modal Molecular Representation Learning via Structure Awareness 

**Title (ZH)**: 基于结构意识的多模态分子表示学习 

**Authors**: Rong Yin, Ruyue Liu, Xiaoshuai Hao, Xingrui Zhou, Yong Liu, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05877)  

**Abstract**: Accurate extraction of molecular representations is a critical step in the drug discovery process. In recent years, significant progress has been made in molecular representation learning methods, among which multi-modal molecular representation methods based on images, and 2D/3D topologies have become increasingly mainstream. However, existing these multi-modal approaches often directly fuse information from different modalities, overlooking the potential of intermodal interactions and failing to adequately capture the complex higher-order relationships and invariant features between molecules. To overcome these challenges, we propose a structure-awareness-based multi-modal self-supervised molecular representation pre-training framework (MMSA) designed to enhance molecular graph representations by leveraging invariant knowledge between molecules. The framework consists of two main modules: the multi-modal molecular representation learning module and the structure-awareness module. The multi-modal molecular representation learning module collaboratively processes information from different modalities of the same molecule to overcome intermodal differences and generate a unified molecular embedding. Subsequently, the structure-awareness module enhances the molecular representation by constructing a hypergraph structure to model higher-order correlations between molecules. This module also introduces a memory mechanism for storing typical molecular representations, aligning them with memory anchors in the memory bank to integrate invariant knowledge, thereby improving the model generalization ability. Extensive experiments have demonstrated the effectiveness of MMSA, which achieves state-of-the-art performance on the MoleculeNet benchmark, with average ROC-AUC improvements ranging from 1.8% to 9.6% over baseline methods. 

**Abstract (ZH)**: 准确提取分子表示是药物发现过程中一个关键步骤。近年来，在分子表示学习方法方面取得了显著进展，其中基于图像和2D/3D拓扑的多模态分子表示方法变得日益主流。然而，现有的多模态方法通常直接融合不同模态的信息，忽视了模态间相互作用的潜力，并未能充分捕捉分子之间的复杂高阶关系和不变特征。为克服这些挑战，我们提出了一种基于结构意识的多模态自监督分子表示预训练框架（MMSA），旨在通过利用分子间的不变知识来增强分子图表示。该框架由两个主要模块组成：多模态分子表示学习模块和结构意识模块。多模态分子表示学习模块协作处理同一分子不同模态的信息，以克服模态间差异并生成统一的分子嵌入。随后，结构意识模块通过构建超图结构来建模分子间的高阶相关性，从而增强分子表示。该模块还引入了记忆机制，用于存储典型分子表示，并将它们与记忆库中的记忆锚进行对齐，以整合不变知识，从而提高模型的泛化能力。广泛实验表明，MMSA 的有效性，在MoleculeNet基准测试上的性能显著优于基线方法，AUC改善范围从1.8%到9.6%。 

---
# Towards Facial Image Compression with Consistency Preserving Diffusion Prior 

**Title (ZH)**: 面向一致性保留扩散先验的面部图像压缩 

**Authors**: Yimin Zhou, Yichong Xia, Bin Chen, Baoyi An, Haoqian Wang, Zhi Wang, Yaowei Wang, Zikun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.05870)  

**Abstract**: With the widespread application of facial image data across various domains, the efficient storage and transmission of facial images has garnered significant attention. However, the existing learned face image compression methods often produce unsatisfactory reconstructed image quality at low bit rates. Simply adapting diffusion-based compression methods to facial compression tasks results in reconstructed images that perform poorly in downstream applications due to insufficient preservation of high-frequency information. To further explore the diffusion prior in facial image compression, we propose Facial Image Compression with a Stable Diffusion Prior (FaSDiff), a method that preserves consistency through frequency enhancement. FaSDiff employs a high-frequency-sensitive compressor in an end-to-end framework to capture fine image details and produce robust visual prompts. Additionally, we introduce a hybrid low-frequency enhancement module that disentangles low-frequency facial semantics and stably modulates the diffusion prior alongside visual prompts. The proposed modules allow FaSDiff to leverage diffusion priors for superior human visual perception while minimizing performance loss in machine vision due to semantic inconsistency. Extensive experiments show that FaSDiff outperforms state-of-the-art methods in balancing human visual quality and machine vision accuracy. The code will be released after the paper is accepted. 

**Abstract (ZH)**: 面部图像压缩中稳定扩散先验的方法（FaSDiff） 

---
# Generative Discovery of Partial Differential Equations by Learning from Math Handbooks 

**Title (ZH)**: 由从数学手册学习生成偏微分方程的发现方法 

**Authors**: Hao Xu, Yuntian Chen, Rui Cao, Tianning Tang, Mengge Du, Jian Li, Adrian H. Callaghan, Dongxiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05869)  

**Abstract**: Data driven discovery of partial differential equations (PDEs) is a promising approach for uncovering the underlying laws governing complex systems. However, purely data driven techniques face the dilemma of balancing search space with optimization efficiency. This study introduces a knowledge guided approach that incorporates existing PDEs documented in a mathematical handbook to facilitate the discovery process. These PDEs are encoded as sentence like structures composed of operators and basic terms, and used to train a generative model, called EqGPT, which enables the generation of free form PDEs. A loop of generation evaluation optimization is constructed to autonomously identify the most suitable PDE. Experimental results demonstrate that this framework can recover a variety of PDE forms with high accuracy and computational efficiency, particularly in cases involving complex temporal derivatives or intricate spatial terms, which are often beyond the reach of conventional methods. The approach also exhibits generalizability to irregular spatial domains and higher dimensional settings. Notably, it succeeds in discovering a previously unreported PDE governing strongly nonlinear surface gravity waves propagating toward breaking, based on real world experimental data, highlighting its applicability to practical scenarios and its potential to support scientific discovery. 

**Abstract (ZH)**: 基于知识指导的数据驱动偏微分方程发现方法 

---
# Evolutionary ecology of words 

**Title (ZH)**: 词的进化生态学 

**Authors**: Reiji Suzuki, Takaya Arita  

**Link**: [PDF](https://arxiv.org/pdf/2505.05863)  

**Abstract**: We propose a model for the evolutionary ecology of words as one attempt to extend evolutionary game theory and agent-based models by utilizing the rich linguistic expressions of Large Language Models (LLMs). Our model enables the emergence and evolution of diverse and infinite options for interactions among agents. Within the population, each agent possesses a short word (or phrase) generated by an LLM and moves within a spatial environment. When agents become adjacent, the outcome of their interaction is determined by the LLM based on the relationship between their words, with the loser's word being replaced by the winner's. Word mutations, also based on LLM outputs, may occur. We conducted preliminary experiments assuming that ``strong animal species" would survive. The results showed that from an initial population consisting of well-known species, many species emerged both gradually and in a punctuated equilibrium manner. Each trial demonstrated the unique evolution of diverse populations, with one type of large species becoming dominant, such as terrestrial animals, marine life, or extinct species, which were ecologically specialized and adapted ones across diverse extreme habitats. We also conducted a long-term experiment with a large population, demonstrating the emergence and coexistence of diverse species. 

**Abstract (ZH)**: 我们提出了一种词汇演化生态学模型，旨在通过利用大型语言模型（LLMs）丰富的语言表达，扩展进化游戏理论和基于代理的模型。该模型能够促进代理之间多样且无限的互动方式的产生和演变。在该群体中，每个代理拥有由LLM生成的短词（或短语），并在空间环境中移动。当代理相邻时，其互动的结果由LLM根据它们之间词汇的关系来决定，输者词汇被胜者词汇取代。词汇突变也基于LLM的输出。我们在假设“强势动物物种”会生存的前提下进行了初步实验。结果显示，从最初由知名物种组成的人口开始，许多物种以渐进和间断平衡的方式逐渐出现。每次试验都展示了不同群体的独特演化，其中一类大型物种成为主导，如陆生动物、海洋生物或适应不同极端环境的生态特化和适应物种。我们还进行了长期大规模实验，展示了多种物种的出现与共存。 

---
# AgentXploit: End-to-End Redteaming of Black-Box AI Agents 

**Title (ZH)**: AgentXploit: 黑箱AI代理的整体红队评估 

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.05849)  

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites. 

**Abstract (ZH)**: 大型语言模型（LLMs）强大的规划和推理能力促进了能够利用外部工具并与日益复杂的环境交互的基于代理的系统的发展。然而，这些强大的功能也引入了一个关键的安全风险：间接提示注入，这是一种高级攻击向量，通过操纵上下文信息而非直接用户提示来 compromize 这些代理的核心——LLM。本文提出了一种通用的灰盒模糊测试框架AgentXploit，旨在自动发现并利用多样化的LLM代理中的间接提示注入漏洞。我们的方法首先构建了一个高质量的初始种子库，然后使用基于蒙特卡洛树搜索（MCTS）的种子选择算法迭代优化输入，从而最大限度地提高发现代理弱点的可能性。我们在两个公开基准AgentDojo和VWA-adv上评估了AgentXploit，分别针对基于o3-mini和GPT-4o的代理，成功率分别为71%和70%，比基线攻击性能几乎翻了一番。此外，AgentXploit在未见过的任务和内部LLM上表现出强大的迁移性，并在针对防御措施方面取得了有希望的结果。除了基准测试评估，我们还在真实的环境中应用了我们的攻击，成功误导代理导航到任意URL，包括恶意网站。 

---
# MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design 

**Title (ZH)**: MxMoE: 混合精度量化在MoE中的准确性和性能协同设计 

**Authors**: Haojie Duanmu, Xiuhong Li, Zhihang Yuan, Size Zheng, Jiangfei Duan, Xingcheng Zhang, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.05799)  

**Abstract**: Mixture-of-Experts (MoE) models face deployment challenges due to their large parameter counts and computational demands. We explore quantization for MoE models and highlight two key insights: 1) linear blocks exhibit varying quantization sensitivity, and 2) divergent expert activation frequencies create heterogeneous computational characteristics. Based on these observations, we introduce MxMoE, a mixed-precision optimization framework for MoE models that considers both algorithmic and system perspectives. MxMoE navigates the design space defined by parameter sensitivity, expert activation dynamics, and hardware resources to derive efficient mixed-precision configurations. Additionally, MxMoE automatically generates optimized mixed-precision GroupGEMM kernels, enabling parallel execution of GEMMs with different precisions. Evaluations show that MxMoE outperforms existing methods, achieving 2.4 lower Wikitext-2 perplexity than GPTQ at 2.25-bit and delivering up to 3.4x speedup over full precision, as well as up to 29.4% speedup over uniform quantization at equivalent accuracy with 5-bit weight-activation quantization. Our code is available at this https URL. 

**Abstract (ZH)**: MoE模型因参数量大和计算需求高而面临部署挑战。我们探索了MoE模型的量化方法，并强调了两个关键见解：1) 线性层的量化敏感性各异，2) 专家激活频率的差异造成了计算特性的异质性。基于这些观察，我们引入了MxMoE，这是一个考虑算法和系统视角的混合精度优化框架，旨在为MoE模型设计高效的混合精度配置。此外，MxMoE还自动生成了优化的混合精度GroupGEMM内核，能够并行执行不同精度的GEMM操作。评估结果显示，MxMoE优于现有方法，在2.25位的量化下Wikitext-2困惑度降低了2.4倍，并且与全精度方法相比可以提速3.4倍，同时在5位权重-激活量化下达到相当的准确度时，比均匀量化提速29.4%。我们的代码可在以下链接获取：this https URL。 

---
# Human-in-the-Loop AI for HVAC Management Enhancing Comfort and Energy Efficiency 

**Title (ZH)**: 带有人类在环的AI在HVAC管理中的应用：提高舒适度和能源效率 

**Authors**: Xinyu Liang, Frits de Nijs, Buser Say, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05796)  

**Abstract**: Heating, Ventilation, and Air Conditioning (HVAC) systems account for approximately 38% of building energy consumption globally, making them one of the most energy-intensive services. The increasing emphasis on energy efficiency and sustainability, combined with the need for enhanced occupant comfort, presents a significant challenge for traditional HVAC systems. These systems often fail to dynamically adjust to real-time changes in electricity market rates or individual comfort preferences, leading to increased energy costs and reduced comfort. In response, we propose a Human-in-the-Loop (HITL) Artificial Intelligence framework that optimizes HVAC performance by incorporating real-time user feedback and responding to fluctuating electricity prices. Unlike conventional systems that require predefined information about occupancy or comfort levels, our approach learns and adapts based on ongoing user input. By integrating the occupancy prediction model with reinforcement learning, the system improves operational efficiency and reduces energy costs in line with electricity market dynamics, thereby contributing to demand response initiatives. Through simulations, we demonstrate that our method achieves significant cost reductions compared to baseline approaches while maintaining or enhancing occupant comfort. This feedback-driven approach ensures personalized comfort control without the need for predefined settings, offering a scalable solution that balances individual preferences with economic and environmental goals. 

**Abstract (ZH)**: 基于 humans-in-the-loop 人工 intelligence 的 HVAC 系统优化：实现实时用户反馈与波动电价下的舒适与节能 

---
# What Is Next for LLMs? Next-Generation AI Computing Hardware Using Photonic Chips 

**Title (ZH)**: LLMs的下一步将是新一代基于光子芯片的AI计算硬件。 

**Authors**: Renjie Li, Wenjie Wei, Qi Xin, Xiaoli Liu, Sixuan Mao, Erik Ma, Zijian Chen, Malu Zhang, Haizhou Li, Zhaoyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05794)  

**Abstract**: Large language models (LLMs) are rapidly pushing the limits of contemporary computing hardware. For example, training GPT-3 has been estimated to consume around 1300 MWh of electricity, and projections suggest future models may require city-scale (gigawatt) power budgets. These demands motivate exploration of computing paradigms beyond conventional von Neumann architectures. This review surveys emerging photonic hardware optimized for next-generation generative AI computing. We discuss integrated photonic neural network architectures (e.g., Mach-Zehnder interferometer meshes, lasers, wavelength-multiplexed microring resonators) that perform ultrafast matrix operations. We also examine promising alternative neuromorphic devices, including spiking neural network circuits and hybrid spintronic-photonic synapses, which combine memory and processing. The integration of two-dimensional materials (graphene, TMDCs) into silicon photonic platforms is reviewed for tunable modulators and on-chip synaptic elements. Transformer-based LLM architectures (self-attention and feed-forward layers) are analyzed in this context, identifying strategies and challenges for mapping dynamic matrix multiplications onto these novel hardware substrates. We then dissect the mechanisms of mainstream LLMs, such as ChatGPT, DeepSeek, and LLaMA, highlighting their architectural similarities and differences. We synthesize state-of-the-art components, algorithms, and integration methods, highlighting key advances and open issues in scaling such systems to mega-sized LLM models. We find that photonic computing systems could potentially surpass electronic processors by orders of magnitude in throughput and energy efficiency, but require breakthroughs in memory, especially for long-context windows and long token sequences, and in storage of ultra-large datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）正快速推动当代计算硬件的极限。例如，训练GPT-3耗电约1300 MWh，预计未来模型可能需要城市规模（吉瓦）级别的电力预算。这些需求促使我们探索超越传统冯·诺依曼架构的计算范式。本文综述了适用于下一代生成性AI计算的新兴光电硬件。我们讨论了集成的光电神经网络架构（如Mach-Zehnder干涉仪网格、激光器、波长复用微环谐振器），它们执行超高速矩阵操作。我们还考察了有前途的类脑替代器件，包括脉冲神经网络电路和结合了存储和处理功能的自旋电子-光电突触。我们审核了将二维材料（石墨烯、过渡金属二硫属化合物）集成到硅光电平台中的可调调制器和片上突触元件。在这一背景下分析了基于变换器的LLM架构（自我注意力层和前向传播层），确定了将动态矩阵乘法映射到这些新型硬件基板上的策略和挑战。然后剖析了主流LLM的机制，如ChatGPT、DeepSeek和LLaMA，突显了它们的架构异同。我们综合了最先进的组件、算法和集成方法，突显了扩展此类系统以支持巨型LLM模型的关键进展和开放问题。我们发现，光电计算系统在吞吐量和能效方面可能比电子处理器高出几个数量级，但需要在内存方面取得突破，特别是对于长上下文窗口和长令牌序列，并需要存储超大数据集。 

---
# FlowHFT: Flow Policy Induced Optimal High-Frequency Trading under Diverse Market Conditions 

**Title (ZH)**: FlowHFT: 流动性政策诱导下的最优高频交易策略在多变市场环境中的应用 

**Authors**: Yang Li, Zhi Chen, Steve Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05784)  

**Abstract**: High-frequency trading (HFT) is an investing strategy that continuously monitors market states and places bid and ask orders at millisecond speeds. Traditional HFT approaches fit models with historical data and assume that future market states follow similar patterns. This limits the effectiveness of any single model to the specific conditions it was trained for. Additionally, these models achieve optimal solutions only under specific market conditions, such as assumptions about stock price's stochastic process, stable order flow, and the absence of sudden volatility. Real-world markets, however, are dynamic, diverse, and frequently volatile. To address these challenges, we propose the FlowHFT, a novel imitation learning framework based on flow matching policy. FlowHFT simultaneously learns strategies from numerous expert models, each proficient in particular market scenarios. As a result, our framework can adaptively adjust investment decisions according to the prevailing market state. Furthermore, FlowHFT incorporates a grid-search fine-tuning mechanism. This allows it to refine strategies and achieve superior performance even in complex or extreme market scenarios where expert strategies may be suboptimal. We test FlowHFT in multiple market environments. We first show that flow matching policy is applicable in stochastic market environments, thus enabling FlowHFT to learn trading strategies under different market conditions. Notably, our single framework consistently achieves performance superior to the best expert for each market condition. 

**Abstract (ZH)**: 基于流匹配策略的仿돼学习交易系统（FlowHFT）：适应动态市场的多样策略学习框架 

---
# PyResBugs: A Dataset of Residual Python Bugs for Natural Language-Driven Fault Injection 

**Title (ZH)**: PyResBugs: 一种用于自然语言驱动故障注入的 Python 残差bug数据集 

**Authors**: Domenico Cotroneo, Giuseppe De Rosa, Pietro Liguori  

**Link**: [PDF](https://arxiv.org/pdf/2505.05777)  

**Abstract**: This paper presents PyResBugs, a curated dataset of residual bugs, i.e., defects that persist undetected during traditional testing but later surface in production, collected from major Python frameworks. Each bug in the dataset is paired with its corresponding fault-free (fixed) version and annotated with multi-level natural language (NL) descriptions. These NL descriptions enable natural language-driven fault injection, offering a novel approach to simulating real-world faults in software systems. By bridging the gap between software fault injection techniques and real-world representativeness, PyResBugs provides researchers with a high-quality resource for advancing AI-driven automated testing in Python systems. 

**Abstract (ZH)**: PyResBugs：一个精心编纂的持久未被发现缺陷数据集及其在Python框架中的应用 

---
# Predicting Diabetic Macular Edema Treatment Responses Using OCT: Dataset and Methods of APTOS Competition 

**Title (ZH)**: 使用OCT预测糖尿病黄斑水肿治疗反应：APTOS竞赛的数据集和方法 

**Authors**: Weiyi Zhang, Peranut Chotcomwongse, Yinwen Li, Pusheng Xu, Ruijie Yao, Lianhao Zhou, Yuxuan Zhou, Hui Feng, Qiping Zhou, Xinyue Wang, Shoujin Huang, Zihao Jin, Florence H.T. Chung, Shujun Wang, Yalin Zheng, Mingguang He, Danli Shi, Paisan Ruamviboonsuk  

**Link**: [PDF](https://arxiv.org/pdf/2505.05768)  

**Abstract**: Diabetic macular edema (DME) significantly contributes to visual impairment in diabetic patients. Treatment responses to intravitreal therapies vary, highlighting the need for patient stratification to predict therapeutic benefits and enable personalized strategies. To our knowledge, this study is the first to explore pre-treatment stratification for predicting DME treatment responses. To advance this research, we organized the 2nd Asia-Pacific Tele-Ophthalmology Society (APTOS) Big Data Competition in 2021. The competition focused on improving predictive accuracy for anti-VEGF therapy responses using ophthalmic OCT images. We provided a dataset containing tens of thousands of OCT images from 2,000 patients with labels across four sub-tasks. This paper details the competition's structure, dataset, leading methods, and evaluation metrics. The competition attracted strong scientific community participation, with 170 teams initially registering and 41 reaching the final round. The top-performing team achieved an AUC of 80.06%, highlighting the potential of AI in personalized DME treatment and clinical decision-making. 

**Abstract (ZH)**: 糖尿病黄斑水肿（DME）显著导致糖尿病患者的视觉损伤。不同的眼内治疗反应差异显著，强调了根据患者进行分层预测治疗效果和实施个性化策略的必要性。据我们所知，本研究是首次探索预处理分层以预测DME治疗反应的研究。为了推进这项研究，我们于2021年举办了第二届亚太视网膜电信学会（APTOS）大数据竞赛。该竞赛的重点是利用眼底OCT图像提高抗VEGF治疗反应的预测准确性。我们提供了一个包含来自2000名患者、涵盖四个子任务标签的数万张OCT图像的数据集。本文详细介绍了竞赛的结构、数据集、领先方法和评估指标。竞赛吸引了强大的科学界参与，共有170支队伍注册，最终有41支队伍进入决赛。表现最佳的队伍实现了80.06%的AUC，突显了人工智能在个性化DME治疗和临床决策中的潜力。 

---
# Multi-Agent Systems for Robotic Autonomy with LLMs 

**Title (ZH)**: 基于大语言模型的机器人自主性的多Agent系统 

**Authors**: Junhong Chen, Ziqi Yang, Haoyuan G Xu, Dandan Zhang, George Mylonas  

**Link**: [PDF](https://arxiv.org/pdf/2505.05762)  

**Abstract**: Since the advent of Large Language Models (LLMs), various research based on such models have maintained significant academic attention and impact, especially in AI and robotics. In this paper, we propose a multi-agent framework with LLMs to construct an integrated system for robotic task analysis, mechanical design, and path generation. The framework includes three core agents: Task Analyst, Robot Designer, and Reinforcement Learning Designer. Outputs are formatted as multimodal results, such as code files or technical reports, for stronger understandability and usability. To evaluate generalizability comparatively, we conducted experiments with models from both GPT and DeepSeek. Results demonstrate that the proposed system can design feasible robots with control strategies when appropriate task inputs are provided, exhibiting substantial potential for enhancing the efficiency and accessibility of robotic system development in research and industrial applications. 

**Abstract (ZH)**: 自大型语言模型（LLMs）问世以来，基于此类模型的各类研究一直保持了显著的学术关注和影响，特别是在人工智能和机器人领域。本文提出了一种由LLMs支持的多Agent框架，用于构建集成的机器人任务分析、机械设计和路径生成系统。该框架包含三个核心Agent：任务分析师、机器人设计师和强化学习设计师。输出结果采用多模态格式，如代码文件或技术报告，以增强理解和实用性。为比较通用性，我们使用来自GPT和DeepSeek的模型进行了实验。结果表明，在适当的任务输入下，所提系统能够设计出可实现的机器人并制定控制策略，显示出在研究和工业应用中增强机器人系统开发效率和可访问性的巨大潜力。 

---
# Evolutionary thoughts: integration of large language models and evolutionary algorithms 

**Title (ZH)**: 演化思想：大型语言模型与演化算法的集成 

**Authors**: Antonio Jimeno Yepes, Pieter Barnard  

**Link**: [PDF](https://arxiv.org/pdf/2505.05756)  

**Abstract**: Large Language Models (LLMs) have unveiled remarkable capabilities in understanding and generating both natural language and code, but LLM reasoning is prone to hallucination and struggle with complex, novel scenarios, often getting stuck on partial or incorrect solutions. However, the inherent ability of Evolutionary Algorithms (EAs) to explore extensive and complex search spaces makes them particularly effective in scenarios where traditional optimization methodologies may falter. However, EAs explore a vast search space when applied to complex problems.
To address the computational bottleneck of evaluating large populations, particularly crucial for complex evolutionary tasks, we introduce a highly efficient evaluation framework. This implementation maintains compatibility with existing primitive definitions, ensuring the generation of valid individuals.
Using LLMs, we propose an enhanced evolutionary search strategy that enables a more focused exploration of expansive solution spaces. LLMs facilitate the generation of superior candidate solutions, as evidenced by empirical results demonstrating their efficacy in producing improved outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）在理解和生成自然语言和代码方面展现了非凡的能力，但在处理复杂和新颖的情景时，其推理容易产生幻觉并常常陷入部分或不正确的解决方案。然而，进化算法（EAs）固有的能力使其能够在传统优化方法可能失效的情景中特别有效。但是，当应用于复杂问题时，EAs会探索一个广阔的搜索空间。为了应对评估大量个体的计算瓶颈，特别是在复杂的进化任务中尤为重要，我们提出了一种高效的家庭评估框架。该实现保持了与现有基本定义的兼容性，确保生成有效的个体。利用LLMs，我们提出了一种增强的进化搜索策略，能够更集中地探索广泛的解空间。实验结果证明了其生成改进结果的有效性。 

---
# Towards Embodiment Scaling Laws in Robot Locomotion 

**Title (ZH)**: 向机器人运动中的本体规模化律研究 

**Authors**: Bo Ai, Liu Dai, Nico Bohlinger, Dichen Li, Tongzhou Mu, Zhanxin Wu, K. Fay, Henrik I. Christensen, Jan Peters, Hao Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.05753)  

**Abstract**: Developing generalist agents that can operate across diverse tasks, environments, and physical embodiments is a grand challenge in robotics and artificial intelligence. In this work, we focus on the axis of embodiment and investigate embodiment scaling laws$\unicode{x2013}$the hypothesis that increasing the number of training embodiments improves generalization to unseen ones. Using robot locomotion as a test bed, we procedurally generate a dataset of $\sim$1,000 varied embodiments, spanning humanoids, quadrupeds, and hexapods, and train generalist policies capable of handling diverse observation and action spaces on random subsets. We find that increasing the number of training embodiments improves generalization to unseen ones, and scaling embodiments is more effective in enabling embodiment-level generalization than scaling data on small, fixed sets of embodiments. Notably, our best policy, trained on the full dataset, zero-shot transfers to novel embodiments in the real world, such as Unitree Go2 and H1. These results represent a step toward general embodied intelligence, with potential relevance to adaptive control for configurable robots, co-design of morphology and control, and beyond. 

**Abstract (ZH)**: 开发能够在多样任务、环境和物理载体间操作的一般性代理是机器人技术和人工智能领域的重大挑战。本项工作关注载体维度，探讨载体扩展律——增加训练载体的数量能够提高对未见载体的泛化能力。以机器人行动为实验平台，我们程序生成了一个包含约1,000个不同载体的数据集，涵盖了类人型、四足和六足载体，并在随机子集上训练能够处理多样观察和行动空间的一般性策略。研究发现，增加训练载体的数量能够提高对未见载体的泛化能力，相较于在小规模固定载体集上扩展数据，扩展载体更能促进载体级泛化。值得注意的是，我们最好的策略在完整数据集上训练，在现实世界中零样本 transfer 到新的载体，如 Unitree Go2 和 H1。这些结果标志着向通用 embodiable 智能迈出的一步，对于可配置机器人中的自适应控制、形态和控制的协同设计等具有潜在意义。 

---
# Accurate and Efficient Multivariate Time Series Forecasting via Offline Clustering 

**Title (ZH)**: 基于离线聚类的准确高效多变量时间序列预报 

**Authors**: Yiming Niu, Jinliang Deng, Lulu Zhang, Zimu Zhou, Yongxin Tong  

**Link**: [PDF](https://arxiv.org/pdf/2505.05738)  

**Abstract**: Accurate and efficient multivariate time series (MTS) forecasting is essential for applications such as traffic management and weather prediction, which depend on capturing long-range temporal dependencies and interactions between entities. Existing methods, particularly those based on Transformer architectures, compute pairwise dependencies across all time steps, leading to a computational complexity that scales quadratically with the length of the input. To overcome these challenges, we introduce the Forecaster with Offline Clustering Using Segments (FOCUS), a novel approach to MTS forecasting that simplifies long-range dependency modeling through the use of prototypes extracted via offline clustering. These prototypes encapsulate high-level events in the real-world system underlying the data, summarizing the key characteristics of similar time segments. In the online phase, FOCUS dynamically adapts these patterns to the current input and captures dependencies between the input segment and high-level events, enabling both accurate and efficient forecasting. By identifying prototypes during the offline clustering phase, FOCUS reduces the computational complexity of modeling long-range dependencies in the online phase to linear scaling. Extensive experiments across diverse benchmarks demonstrate that FOCUS achieves state-of-the-art accuracy while significantly reducing computational costs. 

**Abstract (ZH)**: 基于离线聚类分段的准确高效多变量时间序列 Forecasting (FOCUS) 

---
# HyperspectralMAE: The Hyperspectral Imagery Classification Model using Fourier-Encoded Dual-Branch Masked Autoencoder 

**Title (ZH)**: 超光谱MAE：基于傅里叶编码双分支掩蔽自编码器的超光谱图像分类模型 

**Authors**: Wooyoung Jeong, Hyun Jae Park, Seonghun Jeong, Jong Wook Jang, Tae Hoon Lim, Dae Seoung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.05710)  

**Abstract**: Hyperspectral imagery provides rich spectral detail but poses unique challenges because of its high dimensionality in both spatial and spectral domains. We propose \textit{HyperspectralMAE}, a Transformer-based foundation model for hyperspectral data that employs a \textit{dual masking} strategy: during pre-training we randomly occlude 50\% of spatial patches and 50\% of spectral bands. This forces the model to learn representations capable of reconstructing missing information across both dimensions. To encode spectral order, we introduce learnable harmonic Fourier positional embeddings based on wavelength. The reconstruction objective combines mean-squared error (MSE) with the spectral angle mapper (SAM) to balance pixel-level accuracy and spectral-shape fidelity.
The resulting model contains about $1.8\times10^{8}$ parameters and produces 768-dimensional embeddings, giving it sufficient capacity for transfer learning. We pre-trained HyperspectralMAE on two large hyperspectral corpora -- NASA EO-1 Hyperion ($\sim$1\,600 scenes, $\sim$$3\times10^{11}$ pixel spectra) and DLR EnMAP Level-0 ($\sim$1\,300 scenes, $\sim$$3\times10^{11}$ pixel spectra) -- and fine-tuned it for land-cover classification on the Indian Pines benchmark. HyperspectralMAE achieves state-of-the-art transfer-learning accuracy on Indian Pines, confirming that masked dual-dimensional pre-training yields robust spectral-spatial representations. These results demonstrate that dual masking and wavelength-aware embeddings advance hyperspectral image reconstruction and downstream analysis. 

**Abstract (ZH)**: 高光谱成像提供了丰富的光谱细节，但由于其在空间和光谱域的高维度性，带来了独特的挑战。我们提出了一种基于Transformer的高光谱数据基础模型HyperspectralMAE，采用了双遮罩策略：在预训练过程中，随机遮罩50%的空间patches和50%的光谱波段。这迫使模型学习能够重建跨两个维度缺失信息的表示。为了编码光谱顺序，我们引入了基于波长的可学习谐波傅里叶位置嵌入。重构目标结合均方误差（MSE）和光谱角映射（SAM），以平衡像素级准确性和光谱形状保真度。

HyperspectralMAE包含约$1.8\times10^{8}$个参数，生成768维的嵌入，给其迁移学习提供了足够的容量。我们使用两个大型高光谱数据集——NASA EO-1 Hyperion（约1600个场景，约$3\times10^{11}$像素光谱）和DLR EnMAP Level-0（约1300个场景，约$3\times10^{11}$像素光谱）对HyperspectralMAE进行了预训练，并针对印度-pinus基准进行了土地覆盖分类微调。HyperspectralMAE在印度-pinus上达到了最先进的迁移学习精度，证实了双遮罩和双维度预训练能够产生健壮的光谱-空间表示。这些结果表明，双遮罩和波长感知嵌入推动了高光谱图像重建和下游分析的进步。 

---
# Assessing Robustness to Spurious Correlations in Post-Training Language Models 

**Title (ZH)**: 评估后训练语言模型对虚假相关性的鲁棒性 

**Authors**: Julia Shuieh, Prasann Singhal, Apaar Shanker, John Heyer, George Pu, Samuel Denton  

**Link**: [PDF](https://arxiv.org/pdf/2505.05704)  

**Abstract**: Supervised and preference-based fine-tuning techniques have become popular for aligning large language models (LLMs) with user intent and correctness criteria. However, real-world training data often exhibits spurious correlations -- arising from biases, dataset artifacts, or other "shortcut" features -- that can compromise a model's performance or generalization. In this paper, we systematically evaluate three post-training algorithms -- Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and KTO (Kahneman-Tversky Optimization) -- across a diverse set of synthetic tasks and spuriousness conditions. Our tasks span mathematical reasoning, constrained instruction-following, and document-grounded question answering. We vary the degree of spurious correlation (10% vs. 90%) and investigate two forms of artifacts: "Feature Ambiguity" and "Distributional Narrowness." Our results show that the models often but not always degrade under higher spuriousness. The preference-based methods (DPO/KTO) can demonstrate relative robustness in mathematical reasoning tasks. By contrast, SFT maintains stronger performance in complex, context-intensive tasks. These findings highlight that no single post-training strategy universally outperforms in all scenarios; the best choice depends on the type of target task and the nature of spurious correlations. 

**Abstract (ZH)**: 监督和偏好导向的微调技术已成为将大规模语言模型与用户意图和正确性标准对齐的流行方法。然而，实际的训练数据中往往存在虚假相关性——这些虚假相关性来源于偏差、数据集特征或其他“捷径”特征——这可能损害模型的性能或泛化能力。在本文中，我们系统性地评估了三种后训练算法——监督微调（SFT）、直接偏好优化（DPO）和KTO（凯汉曼-特维斯基优化）——在多种合成任务和虚假相关性条件下的表现。我们的任务涵盖了数学推理、受约束的指令遵循以及文档导向的问题回答。我们改变了虚假相关性的程度（10% vs. 90%）并探讨了两种形式的数据集特征：特征歧义性和分布狭窄性。结果显示，模型在较高的虚假相关性下表现通常但不总是较差。偏好导向的方法（DPO/KTO）在数学推理任务中展现出相对较高的鲁棒性。相比之下，监督微调（SFT）在复杂、上下文密集型任务中保持更强的表现。这些发现表明，并没有一种后训练策略在所有场景中都能普遍表现更优；最佳选择取决于目标任务的类型和虚假相关性的性质。 

---
# Interactive Diabetes Risk Prediction Using Explainable Machine Learning: A Dash-Based Approach with SHAP, LIME, and Comorbidity Insights 

**Title (ZH)**: 基于Dash的可解释机器学习的交互式糖尿病风险预测：SHAP、LIME和共病洞察方法 

**Authors**: Udaya Allani  

**Link**: [PDF](https://arxiv.org/pdf/2505.05683)  

**Abstract**: This study presents a web-based interactive health risk prediction tool designed to assess diabetes risk using machine learning models. Built on the 2015 CDC BRFSS dataset, the study evaluates models including Logistic Regression, Random Forest, XGBoost, LightGBM, KNN, and Neural Networks under original, SMOTE, and undersampling strategies. LightGBM with undersampling achieved the best recall, making it ideal for risk detection. The tool integrates SHAP and LIME to explain predictions and highlights comorbidity correlations using Pearson analysis. A Dash-based UI enables user-friendly interaction with model predictions, personalized suggestions, and feature insights, supporting data-driven health awareness. 

**Abstract (ZH)**: 基于Web的交互式糖尿病风险预测工具：机器学习模型评估及解释 

---
# Lost in OCR Translation? Vision-Based Approaches to Robust Document Retrieval 

**Title (ZH)**: 迷失在OCR翻译中？基于视觉的稳健文档检索方法 

**Authors**: Alexander Most, Joseph Winjum, Ayan Biswas, Shawn Jones, Nishath Rajiv Ranasinghe, Dan O'Malley, Manish Bhattarai  

**Link**: [PDF](https://arxiv.org/pdf/2505.05666)  

**Abstract**: Retrieval-Augmented Generation (RAG) has become a popular technique for enhancing the reliability and utility of Large Language Models (LLMs) by grounding responses in external documents. Traditional RAG systems rely on Optical Character Recognition (OCR) to first process scanned documents into text. However, even state-of-the-art OCRs can introduce errors, especially in degraded or complex documents. Recent vision-language approaches, such as ColPali, propose direct visual embedding of documents, eliminating the need for OCR. This study presents a systematic comparison between a vision-based RAG system (ColPali) and more traditional OCR-based pipelines utilizing Llama 3.2 (90B) and Nougat OCR across varying document qualities. Beyond conventional retrieval accuracy metrics, we introduce a semantic answer evaluation benchmark to assess end-to-end question-answering performance. Our findings indicate that while vision-based RAG performs well on documents it has been fine-tuned on, OCR-based RAG is better able to generalize to unseen documents of varying quality. We highlight the key trade-offs between computational efficiency and semantic accuracy, offering practical guidance for RAG practitioners in selecting between OCR-dependent and vision-based document retrieval systems in production environments. 

**Abstract (ZH)**: 基于视觉的检索增强生成(RAG)与基于OCR的RAG系统在大型语言模型中的系统性比较 

---
# Adaptive Stress Testing Black-Box LLM Planners 

**Title (ZH)**: 自适应压力测试黑盒LLM规划器 

**Authors**: Neeloy Chakraborty, John Pohovey, Melkior Ornik, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2505.05665)  

**Abstract**: Large language models (LLMs) have recently demonstrated success in generalizing across decision-making tasks including planning, control and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. We argue that detecting such failures is necessary, especially in safety-critical scenarios. Existing black-box methods often detect hallucinations by identifying inconsistencies across multiple samples. Many of these approaches typically introduce prompt perturbations like randomizing detail order or generating adversarial inputs, with the intuition that a confident model should produce stable outputs. We first perform a manual case study showing that other forms of perturbations (e.g., adding noise, removing sensor details) cause LLMs to hallucinate in a driving environment. We then propose a novel method for efficiently searching the space of prompt perturbations using Adaptive Stress Testing (AST) with Monte-Carlo Tree Search (MCTS). Our AST formulation enables discovery of scenarios and prompts that cause language models to act with high uncertainty. By generating MCTS prompt perturbation trees across diverse scenarios, we show that offline analyses can be used at runtime to automatically generate prompts that influence model uncertainty, and to inform real-time trust assessments of an LLM. 

**Abstract (ZH)**: 大型语言模型（LLMs）在规划、控制和预测等决策任务上展现出泛化的成功，但它们产生不安全和不 desired 输出的趋势带来了风险。我们认为，在安全关键场景中检测此类失败是必要的。现有黑盒方法通常通过识别多份样本之间的不一致性来检测幻觉。许多这些方法通常引入提示扰动，例如随机化细节顺序或生成对抗性输入，其直觉是自信的模型应产生稳定输出。我们首先进行了一项手动案例研究，表明其他形式的扰动（例如，添加噪声、移除传感器细节）会使LLMs在驾驶环境中产生幻觉。然后，我们提出了一种使用自适应压力测试（AST）和蒙特卡洛树搜索（MCTS）高效搜索提示扰动空间的新方法。我们的AST公式化使我们能够发现导致语言模型高不确定性行为的场景和提示。通过跨多种场景生成MCTS提示扰动树，我们展示了脱机分析可以在运行时自动生成影响模型不确定性的提示，并为LLM提供实时信任评估信息。 

---
# Closing the Loop: Motion Prediction Models beyond Open-Loop Benchmarks 

**Title (ZH)**: 闭环连接：超越开环基准的运动预测模型 

**Authors**: Mohamed-Khalil Bouzidi, Christian Schlauch, Nicole Scheuerer, Yue Yao, Nadja Klein, Daniel Göhring, Jörg Reichardt  

**Link**: [PDF](https://arxiv.org/pdf/2505.05638)  

**Abstract**: Fueled by motion prediction competitions and benchmarks, recent years have seen the emergence of increasingly large learning based prediction models, many with millions of parameters, focused on improving open-loop prediction accuracy by mere centimeters. However, these benchmarks fail to assess whether such improvements translate to better performance when integrated into an autonomous driving stack. In this work, we systematically evaluate the interplay between state-of-the-art motion predictors and motion planners. Our results show that higher open-loop accuracy does not always correlate with better closed-loop driving behavior and that other factors, such as temporal consistency of predictions and planner compatibility, also play a critical role. Furthermore, we investigate downsized variants of these models, and, surprisingly, find that in some cases models with up to 86% fewer parameters yield comparable or even superior closed-loop driving performance. Our code is available at this https URL. 

**Abstract (ZH)**: 受运动预测竞赛和基准的推动，近年来出现了越来越多基于学习的大规模预测模型，许多模型包含数以百万计的参数，旨在通过几厘米改进开环预测精度。然而，这些基准未能评估这种改进在集成到自动驾驶堆栈中时是否会转化为更好的表现。在本文中，我们系统地评估了最先进的运动预测器与运动规划器之间的相互作用。我们的结果表明，更高的开环精度并不总是与更好的闭环驾驶行为相关，并且其他因素，如预测的时间一致性以及规划器的兼容性，也起着关键作用。此外，我们研究了这些模型的缩小版本，令人惊讶地发现，在某些情况下，参数少至86%的模型在闭环驾驶性能方面可与甚至优于更大的模型。我们的代码可在以下链接获取。 

---
# Looking Beyond Language Priors: Enhancing Visual Comprehension and Attention in Multimodal Models 

**Title (ZH)**: 超越语言先验：增强多模态模型中的视觉理解与注意力 

**Authors**: Aarti Ghatkesar, Uddeshya Upadhyay, Ganesh Venkatesh  

**Link**: [PDF](https://arxiv.org/pdf/2505.05626)  

**Abstract**: Achieving deep alignment between vision and language remains a central challenge for Multimodal Large Language Models (MLLMs). These models often fail to fully leverage visual input, defaulting to strong language priors. Our approach first provides insights into how MLLMs internally build visual understanding of image regions and then introduces techniques to amplify this capability. Specifically, we explore techniques designed both to deepen the model's understanding of visual content and to ensure that these visual insights actively guide language generation. We demonstrate the superior multimodal understanding of our resultant model through a detailed upstream analysis quantifying its ability to predict visually-dependent tokens as well as 10 pt boost on visually challenging tasks. 

**Abstract (ZH)**: 实现视觉与语言的深层对齐仍然是多模态大规模语言模型（MLLMs）面临的核心挑战。我们的方法首先深入探讨MLLMs如何内部构建对图像区域的视觉理解，然后引入技术以增强这一能力。具体来说，我们探索了既加深模型对视觉内容理解又能确保这些视觉洞察积极引导语言生成的技术。我们通过详细的上游分析展示了所得模型的优越多模态理解能力，该分析定量评估了模型预测视觉依赖性标记的能力，并在视觉挑战性任务上获得了10个百分点的提升。 

---
# SPIN-ODE: Stiff Physics-Informed Neural ODE for Chemical Reaction Rate Estimation 

**Title (ZH)**: SPIN-ODE：刚性物理学约束神经ODE在化学反应速率估计中的应用 

**Authors**: Wenqing Peng, Zhi-Song Liu, Michael Boy  

**Link**: [PDF](https://arxiv.org/pdf/2505.05625)  

**Abstract**: Estimating rate constants from complex chemical reactions is essential for advancing detailed chemistry. However, the stiffness inherent in real-world atmospheric chemistry systems poses severe challenges, leading to training instability and poor convergence that hinder effective rate constant estimation using learning-based approaches. To address this, we propose a Stiff Physics-Informed Neural ODE framework (SPIN-ODE) for chemical reaction modelling. Our method introduces a three-stage optimisation process: first, a latent neural ODE learns the continuous and differentiable trajectory between chemical concentrations and their time derivatives; second, an explicit Chemical Reaction Neural Network (CRNN) extracts the underlying rate coefficients based on the learned dynamics; and third, fine-tune CRNN using a neural ODE solver to further improve rate coefficient estimation. Extensive experiments on both synthetic and newly proposed real-world datasets validate the effectiveness and robustness of our approach. As the first work on stiff Neural ODEs for chemical rate coefficient discovery, our study opens promising directions for integrating neural networks with detailed chemistry. 

**Abstract (ZH)**: 基于Stiff物理信息神经ODE的化学反应建模方法 

---
# CityNavAgent: Aerial Vision-and-Language Navigation with Hierarchical Semantic Planning and Global Memory 

**Title (ZH)**: 城市导航代理：具有层次语义规划和全局记忆的航空气象与语言导航 

**Authors**: Weichen Zhang, Chen Gao, Shiquan Yu, Ruiying Peng, Baining Zhao, Qian Zhang, Jinqiang Cui, Xinlei Chen, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.05622)  

**Abstract**: Aerial vision-and-language navigation (VLN), requiring drones to interpret natural language instructions and navigate complex urban environments, emerges as a critical embodied AI challenge that bridges human-robot interaction, 3D spatial reasoning, and real-world deployment. Although existing ground VLN agents achieved notable results in indoor and outdoor settings, they struggle in aerial VLN due to the absence of predefined navigation graphs and the exponentially expanding action space in long-horizon exploration. In this work, we propose \textbf{CityNavAgent}, a large language model (LLM)-empowered agent that significantly reduces the navigation complexity for urban aerial VLN. Specifically, we design a hierarchical semantic planning module (HSPM) that decomposes the long-horizon task into sub-goals with different semantic levels. The agent reaches the target progressively by achieving sub-goals with different capacities of the LLM. Additionally, a global memory module storing historical trajectories into a topological graph is developed to simplify navigation for visited targets. Extensive benchmark experiments show that our method achieves state-of-the-art performance with significant improvement. Further experiments demonstrate the effectiveness of different modules of CityNavAgent for aerial VLN in continuous city environments. The code is available at \href{this https URL}{link}. 

**Abstract (ZH)**: 基于无人机的视听说导（VLN）：一种大型语言模型赋能的城市空中VLN代理 

---
# Enhancing Satellite Object Localization with Dilated Convolutions and Attention-aided Spatial Pooling 

**Title (ZH)**: 使用膨胀卷积和注意力辅助空间池化增强卫星目标定位 

**Authors**: Seraj Al Mahmud Mostafa, Chenxi Wang, Jia Yue, Yuta Hozumi, Jianwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05599)  

**Abstract**: Object localization in satellite imagery is particularly challenging due to the high variability of objects, low spatial resolution, and interference from noise and dominant features such as clouds and city lights. In this research, we focus on three satellite datasets: upper atmospheric Gravity Waves (GW), mesospheric Bores (Bore), and Ocean Eddies (OE), each presenting its own unique challenges. These challenges include the variability in the scale and appearance of the main object patterns, where the size, shape, and feature extent of objects of interest can differ significantly. To address these challenges, we introduce YOLO-DCAP, a novel enhanced version of YOLOv5 designed to improve object localization in these complex scenarios. YOLO-DCAP incorporates a Multi-scale Dilated Residual Convolution (MDRC) block to capture multi-scale features at scale with varying dilation rates, and an Attention-aided Spatial Pooling (AaSP) module to focus on the global relevant spatial regions, enhancing feature selection. These structural improvements help to better localize objects in satellite imagery. Experimental results demonstrate that YOLO-DCAP significantly outperforms both the YOLO base model and state-of-the-art approaches, achieving an average improvement of 20.95% in mAP50 and 32.23% in IoU over the base model, and 7.35% and 9.84% respectively over state-of-the-art alternatives, consistently across all three satellite datasets. These consistent gains across all three satellite datasets highlight the robustness and generalizability of the proposed approach. Our code is open sourced at this https URL. 

**Abstract (ZH)**: 卫星影像中的目标定位特别具有挑战性，由于对象的高度变异性、低空间分辨率，以及来自噪声、云彩和城市灯光等主要特征的干扰。本研究聚焦于三种卫星数据集：高层大气重力波（GW）、中层大气波（Bore）和海洋涡旋（OE），每种数据集都具有其独特的挑战。这些挑战包括主要对象模式在尺度和外观上的变异性，导致感兴趣对象的大小、形状和特征扩展范围存在显著差异。为应对这些挑战，我们介绍了一种名为YOLO-DCAP的新型增强版YOLOv5，旨在改善这些复杂场景中的目标定位能力。YOLO-DCAP结合了多尺度空洞残差卷积（MDRC）模块来捕捉不同膨胀率下的多尺度特征，并结合了注意力辅助空间聚类（AaSP）模块以聚焦于全局相关空间区域，从而增强特征选择。结构上的改进有助于在卫星影像中更好地定位物体。实验结果显示，YOLO-DCAP在mAP50和IoU方面显著优于YOLO基模型和最先进的方法，分别提高了20.95%和32.23%，相对于最先进的替代方法，分别提高了7.35%和9.84%，在所有三个卫星数据集中表现出一致的改善。这些一致的收益突显了所提出方法的稳健性和泛化能力。我们的代码已开源，可通过此链接获取：https://github.com/Alibaba-Qwen/YOLO-DCAP 

---
# Trading Under Uncertainty: A Distribution-Based Strategy for Futures Markets Using FutureQuant Transformer 

**Title (ZH)**: 在不确定性条件下的交易：一种基于分布的FutureQuant变换器策略用于期货市场 

**Authors**: Wenhao Guo, Yuda Wang, Zeqiao Huang, Changjiang Zhang, Shumin ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.05595)  

**Abstract**: In the complex landscape of traditional futures trading, where vast data and variables like real-time Limit Order Books (LOB) complicate price predictions, we introduce the FutureQuant Transformer model, leveraging attention mechanisms to navigate these challenges. Unlike conventional models focused on point predictions, the FutureQuant model excels in forecasting the range and volatility of future prices, thus offering richer insights for trading strategies. Its ability to parse and learn from intricate market patterns allows for enhanced decision-making, significantly improving risk management and achieving a notable average gain of 0.1193% per 30-minute trade over state-of-the-art models with a simple algorithm using factors such as RSI, ATR, and Bollinger Bands. This innovation marks a substantial leap forward in predictive analytics within the volatile domain of futures trading. 

**Abstract (ZH)**: 在传统期货交易的复杂景观中，通过利用注意机制处理实时限价订单簿等大量数据和变量以预测价格，我们引入了FutureQuant变换器模型。该模型在预测未来价格范围和波动性方面优于传统的仅关注点预测的模型，为交易策略提供了更丰富的洞察。其解析和学习复杂市场模式的能力增强了决策制定，显著改善了风险管理，并在30分钟的交易中实现了相对于最先进的模型和简单算法（如RSI、ATR和布林带）的平均收益提升0.1193%。这一创新标志着在期货交易这个波动性领域中预测分析的一大进步。 

---
# ReactDance: Progressive-Granular Representation for Long-Term Coherent Reactive Dance Generation 

**Title (ZH)**: ReactDance：渐进细粒度表示生成长期连贯反应舞蹈 

**Authors**: Jingzhong Lin, Yuanyuan Qi, Xinru Li, Wenxuan Huang, Xiangfeng Xu, Bangyan Li, Xuejiao Wang, Gaoqi He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05589)  

**Abstract**: Reactive dance generation (RDG) produces follower movements conditioned on guiding dancer and music while ensuring spatial coordination and temporal coherence. However, existing methods overemphasize global constraints and optimization, overlooking local information, such as fine-grained spatial interactions and localized temporal context. Therefore, we present ReactDance, a novel diffusion-based framework for high-fidelity RDG with long-term coherence and multi-scale controllability. Unlike existing methods that struggle with interaction fidelity, synchronization, and temporal consistency in duet synthesis, our approach introduces two key innovations: 1)Group Residual Finite Scalar Quantization (GRFSQ), a multi-scale disentangled motion representation that captures interaction semantics from coarse body rhythms to fine-grained joint dynamics, and 2)Blockwise Local Context (BLC), a sampling strategy eliminating error accumulation in long sequence generation via local block causal masking and periodic positional encoding. Built on the decoupled multi-scale GRFSQ representation, we implement a diffusion model withLayer-Decoupled Classifier-free Guidance (LDCFG), allowing granular control over motion semantics across scales. Extensive experiments on standard benchmarks demonstrate that ReactDance surpasses existing methods, achieving state-of-the-art performance. 

**Abstract (ZH)**: 基于扩散的高保真反应舞动生成框架ReactDance 

---
# Flight Validation of Learning-Based Trajectory Optimization for the Astrobee Free-Flyer 

**Title (ZH)**: 基于学习的轨迹优化在Astrobee自由飞行器上的飞行验证 

**Authors**: Somrita Banerjee, Abhishek Cauligi, Marco Pavone  

**Link**: [PDF](https://arxiv.org/pdf/2505.05588)  

**Abstract**: Although widely used in commercial and industrial robotics, trajectory optimization has seen limited use in space applications due to its high computational demands. In this work, we present flight results from experiments with the Astrobee free-flying robot on board the International Space Station (ISS), that demonstrate how machine learning can accelerate on-board trajectory optimization while preserving theoretical solver guarantees. To the best of the authors' knowledge, this is the first-ever demonstration of learning-based control on the ISS. Our approach leverages the GuSTO sequential convex programming framework and uses a neural network, trained offline, to map problem parameters to effective initial ``warm-start'' trajectories, paving the way for faster real-time optimization on resource-constrained space platforms. 

**Abstract (ZH)**: 尽管轨迹优化在商业和工业机器人中得到广泛应用，但由于其高度的计算需求，在太空应用中的使用受到限制。在本文中，我们展示了国际空间站（ISS）上自由飞行的Astrobee机器人实验的飞行结果，证明了机器学习可以加速在轨轨迹优化，同时保持理论求解器的保证。据作者所知，这是首次在ISS上演示基于学习的控制。我们的方法利用了GuSTO序列凸规划框架，并使用一个离线训练的神经网络将问题参数映射到有效的初始“预热”轨迹，为在资源受限的太空平台上实现更快的实时优化铺平了道路。 

---
# PyTDC: A multimodal machine learning training, evaluation, and inference platform for biomedical foundation models 

**Title (ZH)**: PyTDC：面向生物医学基础模型的多模态机器学习训练、评估和推理平台 

**Authors**: Alejandro Velez-Arce, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2505.05577)  

**Abstract**: Existing biomedical benchmarks do not provide end-to-end infrastructure for training, evaluation, and inference of models that integrate multimodal biological data and a broad range of machine learning tasks in therapeutics. We present PyTDC, an open-source machine-learning platform providing streamlined training, evaluation, and inference software for multimodal biological AI models. PyTDC unifies distributed, heterogeneous, continuously updated data sources and model weights and standardizes benchmarking and inference endpoints. This paper discusses the components of PyTDC's architecture and, to our knowledge, the first-of-its-kind case study on the introduced single-cell drug-target nomination ML task. We find state-of-the-art methods in graph representation learning and domain-specific methods from graph theory perform poorly on this task. Though we find a context-aware geometric deep learning method that outperforms the evaluated SoTA and domain-specific baseline methods, the model is unable to generalize to unseen cell types or incorporate additional modalities, highlighting PyTDC's capacity to facilitate an exciting avenue of research developing multimodal, context-aware, foundation models for open problems in biomedical AI. 

**Abstract (ZH)**: 现有的生物医药基准尚未提供从训练、评估到推理的端到端基础设施，用于整合多模态生物数据和广泛药物治疗机器学习任务的模型。我们提出PyTDC，一个开源机器学习平台，提供多模态生物AI模型的简化训练、评估和推理软件。PyTDC 统一了分布式、异构的并持续更新的数据源和模型权重，并标准化了基准测试和推理端点。本文讨论了PyTDC 架构的组件，并提供了这项工作中介绍的第一个案例研究，即单细胞药物靶点提名的机器学习任务。我们发现，图表示学习的最新方法和特定领域的图理论方法在这项任务上表现不佳。尽管我们发现一种基于语境的几何深度学习方法在评估的最新方法和领域特定基线方法中表现出色，但该模型无法泛化到未见的细胞类型或整合额外的模态，突显了PyTDC 在促进为生物医药AI 开放问题开发多模态、基于语境的基石模型的研究方面的能力。 

---
# Prompt to Polyp: Clinically-Aware Medical Image Synthesis with Diffusion Models 

**Title (ZH)**: 从息肉到息肉：基于临床意识的医疗图像合成 

**Authors**: Mikhail Chaichuk, Sushant Gautam, Steven Hicks, Elena Tutubalina  

**Link**: [PDF](https://arxiv.org/pdf/2505.05573)  

**Abstract**: The generation of realistic medical images from text descriptions has significant potential to address data scarcity challenges in healthcare AI while preserving patient privacy. This paper presents a comprehensive study of text-to-image synthesis in the medical domain, comparing two distinct approaches: (1) fine-tuning large pre-trained latent diffusion models and (2) training small, domain-specific models. We introduce a novel model named MSDM, an optimized architecture based on Stable Diffusion that integrates a clinical text encoder, variational autoencoder, and cross-attention mechanisms to better align medical text prompts with generated images. Our study compares two approaches: fine-tuning large pre-trained models (FLUX, Kandinsky) versus training compact domain-specific models (MSDM). Evaluation across colonoscopy (MedVQA-GI) and radiology (ROCOv2) datasets reveals that while large models achieve higher fidelity, our optimized MSDM delivers comparable quality with lower computational costs. Quantitative metrics and qualitative evaluations by medical experts reveal strengths and limitations of each approach. 

**Abstract (ZH)**: 从文本描述生成真实医疗图像在医疗保健AI中具有解决数据稀缺挑战的潜在价值，同时保护患者隐私。本文对医疗领域的文本到图像合成进行了全面研究，比较了两种不同的方法：（1）微调大型预训练潜扩散模型和（2）训练小型领域特定模型。我们介绍了一种名为MSDM的新模型，这是一种基于Stable Diffusion的优化架构，集成了临床文本编码器、变分自编码器和交叉注意力机制，以更好地使医学文本提示与生成的图像对齐。我们的研究比较了两种方法：微调大型预训练模型（FLUX，Kandinsky）与训练紧凑的领域特定模型（MSDM）。在结肠镜检查（MedVQA-GI）和放射学（ROCOv2）数据集上的评估表明，虽然大型模型具有更高的保真度，但我们的优化MSDM模型以较低的计算成本提供了可比较的质量。定量指标和医学专家的定性评估揭示了每种方法的优缺点。 

---
# Griffin: Towards a Graph-Centric Relational Database Foundation Model 

**Title (ZH)**: Griffin: 朝向基于图的关系数据库基础模型 

**Authors**: Yanbo Wang, Xiyuan Wang, Quan Gan, Minjie Wang, Qibin Yang, David Wipf, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05568)  

**Abstract**: We introduce Griffin, the first foundation model attemptation designed specifically for Relational Databases (RDBs). Unlike previous smaller models focused on single RDB tasks, Griffin unifies the data encoder and task decoder to handle diverse tasks. Additionally, we enhance the architecture by incorporating a cross-attention module and a novel aggregator. Griffin utilizes pretraining on both single-table and RDB datasets, employing advanced encoders for categorical, numerical, and metadata features, along with innovative components such as cross-attention modules and enhanced message-passing neural networks (MPNNs) to capture the complexities of relational data. Evaluated on large-scale, heterogeneous, and temporal graphs extracted from RDBs across various domains (spanning over 150 million nodes), Griffin demonstrates superior or comparable performance to individually trained models, excels in low-data scenarios, and shows strong transferability with similarity and diversity in pretraining across new datasets and tasks, highlighting its potential as a universally applicable foundation model for RDBs. Code available at this https URL. 

**Abstract (ZH)**: 我们引入了Griffin，这是首个专门设计用于关系数据库（RDBs）的基础模型尝试。与之前专注于单一RDB任务的小型模型不同，Griffin将数据编码器和任务解码器统一起来，以处理各种任务。此外，我们通过引入跨注意力模块和新型聚合器增强了模型架构。Griffin在单表和RDB数据集上进行预训练，采用先进的编码器来处理类别型、数值型和元数据特征，并结合了跨注意力模块和增强的消息传递神经网络（MPNN）等创新组件，以捕捉关系数据的复杂性。在跨越多个领域（包含超过1.5亿个节点）的大规模、异构和时序图上进行评估，Griffin显示出优于或可与独立训练模型相媲美的性能，在少量数据场景下表现出色，并在新数据集和任务的预训练相似性和多样性方面表现出强大的迁移学习能力，突显了其作为RDBs通用基础模型的潜力。代码可在以下链接获取：this https URL。 

---
# Would You Rely on an Eerie Agent? A Systematic Review of the Impact of the Uncanny Valley Effect on Trust in Human-Agent Interaction 

**Title (ZH)**: 你会依赖一位毛骨悚然的代理吗？关于Uncanny Valley效应对人类-代理互动中信任影响的系统性 review 

**Authors**: Ahdiyeh Alipour, Tilo Hartmann, Maryam Alimardani  

**Link**: [PDF](https://arxiv.org/pdf/2505.05543)  

**Abstract**: Trust is a fundamental component of human-agent interaction. With the increasing presence of artificial agents in daily life, it is essential to understand how people perceive and trust these agents. One of the key challenges affecting this perception is the Uncanny Valley Effect (UVE), where increasingly human-like artificial beings can be perceived as eerie or repelling. Despite growing interest in trust and the UVE, existing research varies widely in terms of how these concepts are defined and operationalized. This inconsistency raises important questions about how and under what conditions the UVE influences trust in agents. A systematic understanding of their relationship is currently lacking. This review aims to examine the impact of the UVE on human trust in agents and to identify methodological patterns, limitations, and gaps in the existing empirical literature. Following PRISMA guidelines, a systematic search identified 53 empirical studies that investigated both UVE-related constructs and trust or trust-related outcomes. Studies were analyzed based on a structured set of categories, including types of agents and interactions, methodological and measurement approaches, and key findings. The results of our systematic review reveal that most studies rely on static images or hypothetical scenarios with limited real-time interaction, and the majority use subjective trust measures. This review offers a novel framework for classifying trust measurement approaches with regard to the best-practice criteria for empirically investigating the UVE. As the first systematic attempt to map the intersection of UVE and trust, this review contributes to a deeper understanding of their interplay and offers a foundation for future research. Keywords: the uncanny valley effect, trust, human-likeness, affinity response, human-agent interaction 

**Abstract (ZH)**: 信任是人机交互的基本组成部分。随着日常生活中的人工智能代理越来越多，理解人们如何感知和信任这些代理变得至关重要。影响这种感知的一个关键挑战是超乎寻常谷效应（UVE），即越来越像人类的人工生物可能会被感知为恐怖或令人反感。尽管在信任和UVE方面已有越来越多的兴趣，但现有研究在这些概念的定义和操作化方面存在很大差异。这种不一致性提出了一个重要问题：UVE如何以及在什么条件下影响代理的信任。目前对它们之间关系的理解还不够系统。此次综述旨在考察UVE对人类对代理的信任的影响，并识别现有实证文献中的方法论模式、限制和空白。根据PRISMA指南，系统搜索确定了53项研究，这些研究探讨了与UVE相关的构念以及信任或信任相关的结果。研究是基于一个结构化的分类体系进行分析，包括代理和互动的类型、方法论和测量方法，以及关键发现。系统综述的结果表明，大多数研究依赖于静态图像或假设场景，且缺乏即时互动，大多数使用主观信任测量指标。此次综述提供了一个新颖的框架来分类信任测量方法，并符合实证研究UVE的最佳实践标准。作为第一个系统尝试映射UVE和信任交叉的研究，此次综述加深了对它们相互作用的理解，并为未来的研究奠定了基础。关键词：超乎寻常谷效应，信任，拟人类化，亲和反应，人机交互。 

---
# Cardioformer: Advancing AI in ECG Analysis with Multi-Granularity Patching and ResNet 

**Title (ZH)**: Cardioformer: 采用多粒度patches和ResNet推动ECG分析的AI技术 

**Authors**: Md Kamrujjaman Mobin, Md Saiful Islam, Sadik Al Barid, Md Masum  

**Link**: [PDF](https://arxiv.org/pdf/2505.05538)  

**Abstract**: Electrocardiogram (ECG) classification is crucial for automated cardiac disease diagnosis, yet existing methods often struggle to capture local morphological details and long-range temporal dependencies simultaneously. To address these challenges, we propose Cardioformer, a novel multi-granularity hybrid model that integrates cross-channel patching, hierarchical residual learning, and a two-stage self-attention mechanism. Cardioformer first encodes multi-scale token embeddings to capture fine-grained local features and global contextual information and then selectively fuses these representations through intra- and inter-granularity self-attention. Extensive evaluations on three benchmark ECG datasets under subject-independent settings demonstrate that model consistently outperforms four state-of-the-art baselines. Our Cardioformer model achieves the AUROC of 96.34$\pm$0.11, 89.99$\pm$0.12, and 95.59$\pm$1.66 in MIMIC-IV, PTB-XL and PTB dataset respectively outperforming PatchTST, Reformer, Transformer, and Medformer models. It also demonstrates strong cross-dataset generalization, achieving 49.18% AUROC on PTB and 68.41% on PTB-XL when trained on MIMIC-IV. These findings underscore the potential of Cardioformer to advance automated ECG analysis, paving the way for more accurate and robust cardiovascular disease diagnosis. We release the source code at this https URL. 

**Abstract (ZH)**: 心电图（ECG）分类对于自动心脏疾病诊断至关重要，但现有方法往往难以同时捕捉局部形态细节和长时间序列依赖性。为了解决这些挑战，我们提出了一种名为Cardioformer的新型多粒度混合模型，该模型结合了跨通道补丁技术、分层残差学习以及两阶段自注意力机制。Cardioformer首先通过多尺度标记嵌入来捕捉细粒度的局部特征和全局上下文信息，然后通过跨粒度自注意力机制选择性地融合这些表示。在独立受试者设置下对三个基准ECG数据集进行了广泛的评估，结果表明该模型始终优于四个最先进的基线模型。我们的Cardioformer模型在MIMIC-IV、PTB-XL和PTB数据集上的AUROC分别为96.34±0.11、89.99±0.12和95.59±1.66，分别优于PatchTST、Reformer、Transformer和Medformer模型。此外，Cardioformer还展示了强大的跨数据集泛化能力，在使用MIMIC-IV训练时，PTB数据集和PTB-XL数据集的AUROC分别为49.18%和68.41%。这些发现突显了Cardioformer在推进自动化ECG分析方面的潜力，为其在更准确和稳健的心血管疾病诊断中的应用铺平了道路。我们将在以下链接发布源代码：这个 https URL。 

---
# Rethinking Graph Contrastive Learning through Relative Similarity Preservation 

**Title (ZH)**: 重新思考基于相对相似性的图对比学习 

**Authors**: Zhiyuan Ning, Pengfei Wang, Ziyue Qiao, Pengyang Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.05533)  

**Abstract**: Graph contrastive learning (GCL) has achieved remarkable success by following the computer vision paradigm of preserving absolute similarity between augmented views. However, this approach faces fundamental challenges in graphs due to their discrete, non-Euclidean nature -- view generation often breaks semantic validity and similarity verification becomes unreliable. Through analyzing 11 real-world graphs, we discover a universal pattern transcending the homophily-heterophily dichotomy: label consistency systematically diminishes as structural distance increases, manifesting as smooth decay in homophily graphs and oscillatory decay in heterophily graphs. We establish theoretical guarantees for this pattern through random walk theory, proving label distribution convergence and characterizing the mechanisms behind different decay behaviors. This discovery reveals that graphs naturally encode relative similarity patterns, where structurally closer nodes exhibit collectively stronger semantic relationships. Leveraging this insight, we propose RELGCL, a novel GCL framework with complementary pairwise and listwise implementations that preserve these inherent patterns through collective similarity objectives. Extensive experiments demonstrate that our method consistently outperforms 20 existing approaches across both homophily and heterophily graphs, validating the effectiveness of leveraging natural relative similarity over artificial absolute similarity. 

**Abstract (ZH)**: 图对比学习（GCL）通过保留增强视图之间的绝对相似性，在遵循计算机视觉范式方面取得了显著成功。然而，这一方法在图中面临着根本性的挑战，因为图具有离散的、非欧几里得的本质——视图生成往往破坏了语义的有效性和相似性验证变得不可靠。通过对11个实际图的分析，我们发现超越同构-异构二分法的普遍模式：标签一致性系统地随着结构距离增加而减弱，表现为同构图中的平滑衰减和异构图中的振荡衰减。通过随机游走理论，我们为这一模式提供了理论保证，证明了标签分布的收敛性，并表征了不同衰减行为背后的机制。这一发现揭示了图自然地编码相对相似性模式，其中结构更接近的节点展示了更强的语义关系。借助这一见解，我们提出了RELGCL，一个具有互补的成对和列表实现的新型GCL框架，通过集体相似性目标保留这些内在模式。广泛的实验表明，我们的方法在同构图和异构图中都优于20种现有方法，验证了利用自然的相对相似性而非人工的绝对相似性的有效性。 

---
# Low-bit Model Quantization for Deep Neural Networks: A Survey 

**Title (ZH)**: 低比特模型量化的深度神经网络综述 

**Authors**: Kai Liu, Qian Zheng, Kaiwen Tao, Zhiteng Li, Haotong Qin, Wenbo Li, Yong Guo, Xianglong Liu, Linghe Kong, Guihai Chen, Yulun Zhang, Xiaokang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05530)  

**Abstract**: With unprecedented rapid development, deep neural networks (DNNs) have deeply influenced almost all fields. However, their heavy computation costs and model sizes are usually unacceptable in real-world deployment. Model quantization, an effective weight-lighting technique, has become an indispensable procedure in the whole deployment pipeline. The essence of quantization acceleration is the conversion from continuous floating-point numbers to discrete integer ones, which significantly speeds up the memory I/O and calculation, i.e., addition and multiplication. However, performance degradation also comes with the conversion because of the loss of precision. Therefore, it has become increasingly popular and critical to investigate how to perform the conversion and how to compensate for the information loss. This article surveys the recent five-year progress towards low-bit quantization on DNNs. We discuss and compare the state-of-the-art quantization methods and classify them into 8 main categories and 24 sub-categories according to their core techniques. Furthermore, we shed light on the potential research opportunities in the field of model quantization. A curated list of model quantization is provided at this https URL. 

**Abstract (ZH)**: 在前所未有的迅猛发展下，深度神经网络（DNNs）几乎影响了所有领域。然而，它们沉重的计算成本和模型规模通常在现实部署中不可接受。模型量化，一种有效的权重轻量化技术，已成为整个部署管道中不可或缺的步骤。量化加速的本质是将连续的浮点数转换为离散的整数，这显著加快了内存I/O和计算，即加法和乘法。然而，这种转换也会伴随着精度的损失，从而导致性能下降。因此，研究如何进行这种转换以及如何补偿信息损失变得越来越流行和关键。本文回顾了过去五年DNN低比特量化的发展。我们讨论并比较了最先进的量化方法，并根据其核心技术将其分类为8个主要类别和24个子类别。此外，我们还指出了模型量化领域潜在的研究机会。提供了经过筛选的模型量化列表，详见：this https URL。 

---
# X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP 

**Title (ZH)**: X-_transfer 攻击：面向 CLIP 的超转移对抗攻击 

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey  

**Link**: [PDF](https://arxiv.org/pdf/2505.05528)  

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{this https URL}{GitHub repository}. 

**Abstract (ZH)**: X-Transfer: 一种揭示CLIP模型普遍对抗脆弱性的新型攻击方法 

---
# ADMM-Based Training for Spiking Neural Networks 

**Title (ZH)**: 基于ADMM的脉冲神经网络训练方法 

**Authors**: Giovanni Perin, Cesare Bidini, Riccardo Mazzieri, Michele Rossi  

**Link**: [PDF](https://arxiv.org/pdf/2505.05527)  

**Abstract**: In recent years, spiking neural networks (SNNs) have gained momentum due to their high potential in time-series processing combined with minimal energy consumption. However, they still lack a dedicated and efficient training algorithm. The popular backpropagation with surrogate gradients, adapted from stochastic gradient descent (SGD)-derived algorithms, has several drawbacks when used as an optimizer for SNNs. Specifically, it suffers from low scalability and numerical imprecision. In this paper, we propose a novel SNN training method based on the alternating direction method of multipliers (ADMM). Our ADMM-based training aims to solve the problem of the SNN step function's non-differentiability. We formulate the problem, derive closed-form updates, and empirically show the optimizer's convergence properties, great potential, and possible new research directions to improve the method in a simulated proof-of-concept. 

**Abstract (ZH)**: 近年来，由于其在时间序列处理中高い潜能以及极低的能量消耗，脉冲神经网络（SNNs）获得了广泛关注。然而，它们仍然缺乏专门且高效的训练算法。流行的基于替代梯度的反向传播算法尽管是从随机梯度下降（SGD）衍生而来，但作为SNN的优化器时存在诸多不足，特别是可扩展性和数值精确性较差的问题。本文提出了一种基于交替方向乘子方法（ADMM）的新型SNN训练方法，旨在解决SNN阶跃函数非可微的问题。我们对该问题进行了形式化描述，推导出闭合形式的更新规则，并通过仿真实验验证了优化器的收敛性、巨大潜力及其改进的新研究方向。 

---
# GenAI in Entrepreneurship: a systematic review of generative artificial intelligence in entrepreneurship research: current issues and future directions 

**Title (ZH)**: GenAI在创业中的应用：生成式人工智能在创业研究中的系统回顾：现有问题与未来方向 

**Authors**: Anna Kusetogullari, Huseyin Kusetogullari, Martin Andersson, Tony Gorschek  

**Link**: [PDF](https://arxiv.org/pdf/2505.05523)  

**Abstract**: Generative Artificial Intelligence (GenAI) and Large Language Models (LLMs) are recognized to have significant effects on industry and business dynamics, not least because of their impact on the preconditions for entrepreneurship. There is still a lack of knowledge of GenAI as a theme in entrepreneurship research. This paper presents a systematic literature review aimed at identifying and analyzing the evolving landscape of research on the effects of GenAI on entrepreneurship. We analyze 83 peer-reviewed articles obtained from leading academic databases: Web of Science and Scopus. Using natural language processing and unsupervised machine learning techniques with TF-IDF vectorization, Principal Component Analysis (PCA), and hierarchical clustering, five major thematic clusters are identified: (1) Digital Transformation and Behavioral Models, (2) GenAI-Enhanced Education and Learning Systems, (3) Sustainable Innovation and Strategic AI Impact, (4) Business Models and Market Trends, and (5) Data-Driven Technological Trends in Entrepreneurship. Based on the review, we discuss future research directions, gaps in the current literature, as well as ethical concerns raised in the literature. We highlight the need for more macro-level research on GenAI and LLMs as external enablers for entrepreneurship and for research on effective regulatory frameworks that facilitate business experimentation, innovation, and further technology development. 

**Abstract (ZH)**: 生成式人工智能（GenAI）和大型语言模型（LLMs）对行业和商业动态产生了显著影响，尤其是在创业的先决条件方面。尽管如此，关于GenAI作为创业研究主题的现有知识仍然不足。本文提出了一项系统性文献综述，旨在识别和分析GenAI对创业影响的研究景观演变。我们分析了来自Web of Science和Scopus等领先学术数据库的83篇同行评审文章。通过使用自然语言处理和无监督机器学习技术，结合TF-IDF向量化、主成分分析（PCA）和层次聚类，我们识别出五大主题集群：（1）数字化转型与行为模型，（2）GenAI增强的教育和学习系统，（3）可持续创新和战略AI影响，（4）商业模式和市场趋势，（5）创业中的数据驱动技术趋势。基于综述，我们讨论了未来研究方向、当前文献中的知识空白以及文献中提出的伦理问题。我们强调了需要更多关于GenAI和LLMs作为外在促进因素的宏观层面研究，以及研究有助于业务试验、创新和技术发展的有效监管框架的必要性。 

---
# Continuous Thought Machines 

**Title (ZH)**: 连续思维机器 

**Authors**: Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, Llion Jones  

**Link**: [PDF](https://arxiv.org/pdf/2505.05522)  

**Abstract**: Biological brains demonstrate complex neural activity, where the timing and interplay between neurons is critical to how brains process information. Most deep learning architectures simplify neural activity by abstracting away temporal dynamics. In this paper we challenge that paradigm. By incorporating neuron-level processing and synchronization, we can effectively reintroduce neural timing as a foundational element. We present the Continuous Thought Machine (CTM), a model designed to leverage neural dynamics as its core representation. The CTM has two core innovations: (1) neuron-level temporal processing, where each neuron uses unique weight parameters to process a history of incoming signals; and (2) neural synchronization employed as a latent representation. The CTM aims to strike a balance between oversimplified neuron abstractions that improve computational efficiency, and biological realism. It operates at a level of abstraction that effectively captures essential temporal dynamics while remaining computationally tractable for deep learning. We demonstrate the CTM's strong performance and versatility across a range of challenging tasks, including ImageNet-1K classification, solving 2D mazes, sorting, parity computation, question-answering, and RL tasks. Beyond displaying rich internal representations and offering a natural avenue for interpretation owing to its internal process, the CTM is able to perform tasks that require complex sequential reasoning. The CTM can also leverage adaptive compute, where it can stop earlier for simpler tasks, or keep computing when faced with more challenging instances. The goal of this work is to share the CTM and its associated innovations, rather than pushing for new state-of-the-art results. To that end, we believe the CTM represents a significant step toward developing more biologically plausible and powerful artificial intelligence systems. 

**Abstract (ZH)**: 生物大脑表现出复杂的神经活动，其中神经元之间的时间关系和相互作用对大脑处理信息至关重要。大多数深度学习架构通过抽象掉时间动态来简化神经活动。本文挑战了这一范式。通过引入神经元级别的时间处理和同步，我们可以有效地重新引入神经元时间作为基本要素。我们提出了连续思维机器（CTM）模型，该模型旨在将其核心表示基于神经动力学。CTM的两大创新分别是：（1）神经元级别的时间处理，其中每个神经元使用独特的权重参数处理输入信号的历史；（2）将神经元同步作为潜在表示的应用。CTM旨在在简化神经元抽象以提高计算效率和生物现实性之间找到平衡。它以一种有效捕捉关键时间动态的抽象级别运行，同时保持深度学习的计算可行性。本文展示了CTM在一系列具有挑战性的任务（包括ImageNet-1K分类、解决2D迷宫、排序、奇偶性计算、问答和强化学习任务）上的强大性能和灵活性。除了展示丰富的内部表示和由于其内部过程自然而提供易于解释的道路外，CTM还能够执行需要复杂序列推理的任务。CTM还可以利用自适应计算，对于简单的任务可以在早期停止计算，而在面对更具有挑战性的实例时继续计算。本文的目标是分享CTM及其相关创新，而非追求新的领先成果。我们认为，CTM代表了朝着开发更加生物可信且强大的人工智能系统迈出的重要一步。 

---
# GaMNet: A Hybrid Network with Gabor Fusion and NMamba for Efficient 3D Glioma Segmentation 

**Title (ZH)**: GaMNet: 结合Gabor融合和NMamba的高效脑胶质瘤三维分割混合网络 

**Authors**: Chengwei Ye, Huanzhen Zhang, Yufei Lin, Kangsheng Wang, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05520)  

**Abstract**: Gliomas are aggressive brain tumors that pose serious health risks. Deep learning aids in lesion segmentation, but CNN and Transformer-based models often lack context modeling or demand heavy computation, limiting real-time use on mobile medical devices. We propose GaMNet, integrating the NMamba module for global modeling and a multi-scale CNN for efficient local feature extraction. To improve interpretability and mimic the human visual system, we apply Gabor filters at multiple scales. Our method achieves high segmentation accuracy with fewer parameters and faster computation. Extensive experiments show GaMNet outperforms existing methods, notably reducing false positives and negatives, which enhances the reliability of clinical diagnosis. 

**Abstract (ZH)**: 基于NMamba模块和多尺度CNN的GaMNet在胶质瘤分割中的应用 

---
# AI-powered virtual eye: perspective, challenges and opportunities 

**Title (ZH)**: AI赋能的虚拟眼睛：视角、挑战与机遇 

**Authors**: Yue Wu, Yibo Guo, Yulong Yan, Jiancheng Yang, Xin Zhou, Ching-Yu Cheng, Danli Shi, Mingguang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.05516)  

**Abstract**: We envision the "virtual eye" as a next-generation, AI-powered platform that uses interconnected foundation models to simulate the eye's intricate structure and biological function across all scales. Advances in AI, imaging, and multiomics provide a fertile ground for constructing a universal, high-fidelity digital replica of the human eye. This perspective traces the evolution from early mechanistic and rule-based models to contemporary AI-driven approaches, integrating in a unified model with multimodal, multiscale, dynamic predictive capabilities and embedded feedback mechanisms. We propose a development roadmap emphasizing the roles of large-scale multimodal datasets, generative AI, foundation models, agent-based architectures, and interactive interfaces. Despite challenges in interpretability, ethics, data processing and evaluation, the virtual eye holds the potential to revolutionize personalized ophthalmic care and accelerate research into ocular health and disease. 

**Abstract (ZH)**: 我们设想“虚拟眼”是一个下一代、基于AI的平台，利用互联互通的基础模型来模拟眼睛复杂的结构和生物学功能，涵盖所有尺度。随着AI、成像技术和多组学的发展，构建一个通用的、高保真的人类眼睛数字复制品有了肥沃的土壤。本文追溯从早期机制性和规则性模型到当前基于AI的方法的发展历程，整合了多模态、多尺度、动态预测能力和嵌入式反馈机制的统一模型。我们提出了一个开发路线图，强调大规模多模态数据集、生成型AI、基础模型、基于代理的架构以及交互式界面的作用。尽管在可解释性、伦理、数据处理和评估方面存在挑战，“虚拟眼”仍有潜力革新个性化眼科护理，并加速眼科健康和疾病研究的步伐。 

---
# Preliminary Explorations with GPT-4o(mni) Native Image Generation 

**Title (ZH)**: 最初探究：GPT-4o(mni)原生图像生成 

**Authors**: Pu Cao, Feng Zhou, Junyi Ji, Qingye Kong, Zhixiang Lv, Mingjian Zhang, Xuekun Zhao, Siqi Wu, Yinghui Lin, Qing Song, Lu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.05501)  

**Abstract**: Recently, the visual generation ability by GPT-4o(mni) has been unlocked by OpenAI. It demonstrates a very remarkable generation capability with excellent multimodal condition understanding and varied task instructions. In this paper, we aim to explore the capabilities of GPT-4o across various tasks. Inspired by previous study, we constructed a task taxonomy along with a carefully curated set of test samples to conduct a comprehensive qualitative test. Benefiting from GPT-4o's powerful multimodal comprehension, its image-generation process demonstrates abilities surpassing those of traditional image-generation tasks. Thus, regarding the dimensions of model capabilities, we evaluate its performance across six task categories: traditional image generation tasks, discriminative tasks, knowledge-based generation, commonsense-based generation, spatially-aware image generation, and temporally-aware image generation. These tasks not only assess the quality and conditional alignment of the model's outputs but also probe deeper into GPT-4o's understanding of real-world concepts. Our results reveal that GPT-4o performs impressively well in general-purpose synthesis tasks, showing strong capabilities in text-to-image generation, visual stylization, and low-level image processing. However, significant limitations remain in its ability to perform precise spatial reasoning, instruction-grounded generation, and consistent temporal prediction. Furthermore, when faced with knowledge-intensive or domain-specific scenarios, such as scientific illustrations or mathematical plots, the model often exhibits hallucinations, factual errors, or structural inconsistencies. These findings suggest that while GPT-4o marks a substantial advancement in unified multimodal generation, there is still a long way to go before it can be reliably applied to professional or safety-critical domains. 

**Abstract (ZH)**: GPT-4o在各种任务中的生成能力探索：从传统图像生成到时空感知生成 

---
# An Overview of the Prospects and Challenges of Using Artificial Intelligence for Energy Management Systems in Microgrids 

**Title (ZH)**: 微网中使用人工智能进行能源管理系统前景与挑战综述 

**Authors**: Noor ul Misbah Khanum, Hayssam Dahrouj, Ramesh C. Bansal, Hissam Mouayad Tawfik  

**Link**: [PDF](https://arxiv.org/pdf/2505.05498)  

**Abstract**: Microgrids have emerged as a pivotal solution in the quest for a sustainable and energy-efficient future. While microgrids offer numerous advantages, they are also prone to issues related to reliably forecasting renewable energy demand and production, protecting against cyberattacks, controlling operational costs, optimizing power flow, and regulating the performance of energy management systems (EMS). Tackling these energy management challenges is essential to facilitate microgrid applications and seamlessly incorporate renewable energy resources. Artificial intelligence (AI) has recently demonstrated immense potential for optimizing energy management in microgrids, providing efficient and reliable solutions. This paper highlights the combined benefits of enabling AI-based methodologies in the energy management systems of microgrids by examining the applicability and efficiency of AI-based EMS in achieving specific technical and economic objectives. The paper also points out several future research directions that promise to spearhead AI-driven EMS, namely the development of self-healing microgrids, integration with blockchain technology, use of Internet of things (IoT), and addressing interpretability, data privacy, scalability, and the prospects to generative AI in the context of future AI-based EMS. 

**Abstract (ZH)**: 微电网已经 emerged as a pivotal solution in the quest for a sustainable and energy-efficient future. While microgrids offer numerous advantages, they are also prone to issues related to reliably forecasting renewable energy demand and production, protecting against cyberattacks, controlling operational costs, optimizing power flow, and regulating the performance of energy management systems (EMS). Tackling these energy management challenges is essential to facilitate microgrid applications and seamlessly incorporate renewable energy resources. Artificial intelligence (AI) has recently demonstrated immense potential for optimizing energy management in microgrids, providing efficient and reliable solutions. This paper highlights the combined benefits of enabling AI-based methodologies in the energy management systems of microgrids by examining the applicability and efficiency of AI-based EMS in achieving specific technical and economic objectives. The paper also points out several future research directions that promise to spearhead AI-driven EMS, namely the development of self-healing microgrids, integration with blockchain technology, use of Internet of things (IoT), and addressing interpretability, data privacy, scalability, and the prospects to generative AI in the context of future AI-based EMS。

微电网已成为实现可持续和能源高效未来的关键解决方案。尽管微电网具有诸多优势，但它们也容易受到可靠预测可再生能...

Artificial Intelligence-Based Energy Management Systems in Microgrids: Challenges, Benefits, and Future Research Directions 

---
# An Automated LLM-based Pipeline for Asset-Level Database Creation to Assess Deforestation Impact 

**Title (ZH)**: 基于LLM的自动资产级数据库创建流水线以评估森林砍伐影响 

**Authors**: Avanija Menon, Ovidiu Serban  

**Link**: [PDF](https://arxiv.org/pdf/2505.05494)  

**Abstract**: The European Union Deforestation Regulation (EUDR) requires companies to prove their products do not contribute to deforestation, creating a critical demand for precise, asset-level environmental impact data. Current databases lack the necessary detail, relying heavily on broad financial metrics and manual data collection, which limits regulatory compliance and accurate environmental modeling. This study presents an automated, end-to-end data extraction pipeline that uses LLMs to create, clean, and validate structured databases, specifically targeting sectors with a high risk of deforestation. The pipeline introduces Instructional, Role-Based, Zero-Shot Chain-of-Thought (IRZ-CoT) prompting to enhance data extraction accuracy and a Retrieval-Augmented Validation (RAV) process that integrates real-time web searches for improved data reliability. Applied to SEC EDGAR filings in the Mining, Oil & Gas, and Utilities sectors, the pipeline demonstrates significant improvements over traditional zero-shot prompting approaches, particularly in extraction accuracy and validation coverage. This work advances NLP-driven automation for regulatory compliance, CSR (Corporate Social Responsibility), and ESG, with broad sectoral applicability. 

**Abstract (ZH)**: 欧洲联盟毁林法规（EUDR）要求企业证明其产品不会导致毁林，从而对精确的资产级别环境影响数据产生了关键性需求。当前数据库缺乏必要的细节，严重依赖于宽泛的财务指标和手动数据收集，这限制了监管合规性和准确的环境模型构建。本研究提出了一种自动化、端到端的数据提取管道，利用大语言模型（LLMs）创建、清洁和验证结构化数据库，特别针对毁林风险高的行业。该管道引入了指令、角色基于、零样本推理链（IRZ-CoT）提示以提高数据提取准确性，并结合了检索增强验证（RAV）流程，通过实时网络搜索提高数据可靠性。在矿业、石油与天然气及公用事业行业的SEC EDGAR申报文件中应用该管道，表现出显著优于传统零样本提示方法的改进，尤其是在提取准确性和验证覆盖面方面。这项工作推进了基于NLP的自动化在监管合规、企业社会责任（CSR）和ESG领域的应用，具有广泛的行业适用性。 

---
# DetoxAI: a Python Toolkit for Debiasing Deep Learning Models in Computer Vision 

**Title (ZH)**: DetoxAI：计算机视觉中深度学习模型去偏见的Python工具包 

**Authors**: Ignacy Stępka, Lukasz Sztukiewicz, Michał Wiliński, Jerzy Stefanowski  

**Link**: [PDF](https://arxiv.org/pdf/2505.05492)  

**Abstract**: While machine learning fairness has made significant progress in recent years, most existing solutions focus on tabular data and are poorly suited for vision-based classification tasks, which rely heavily on deep learning. To bridge this gap, we introduce DetoxAI, an open-source Python library for improving fairness in deep learning vision classifiers through post-hoc debiasing. DetoxAI implements state-of-the-art debiasing algorithms, fairness metrics, and visualization tools. It supports debiasing via interventions in internal representations and includes attribution-based visualization tools and quantitative algorithmic fairness metrics to show how bias is mitigated. This paper presents the motivation, design, and use cases of DetoxAI, demonstrating its tangible value to engineers and researchers. 

**Abstract (ZH)**: 尽管机器学习公平性在近年来取得了显著进展，但现有的大多数解决方案主要针对表格式数据，而不适合依赖深度学习的视觉分类任务。为解决这一问题，我们引入了DetoxAI，这是一个用于通过后处理去偏见来提高深度学习视觉分类器公平性的开源Python库。DetoxAI实现了最先进的去偏见算法、公平性指标和可视化工具。它支持通过内部表示的干预进行去偏见，并包括基于 Attribution 的可视化工具和定量的算法公平性指标，以展示如何减轻偏见。本文介绍了DetoxAI的动机、设计和应用场景，展示了其对工程师和研究人员的实际价值。 

---
# MDDFNet: Mamba-based Dynamic Dual Fusion Network for Traffic Sign Detection 

**Title (ZH)**: MDDFNet：基于Mamba的动态双模融合网络用于交通标志检测 

**Authors**: TianYi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.05491)  

**Abstract**: The Detection of small objects, especially traffic signs, is a critical sub-task in object detection and autonomous driving. Despite signficant progress in previous research, two main challenges remain. First, the issue of feature extraction being too singular. Second, the detection process struggles to efectively handle objects of varying sizes or scales. These problems are also prevalent in general object detection tasks. To address these challenges, we propose a novel object detection network, Mamba-based Dynamic Dual Fusion Network (MDDFNet), for traffic sign detection. The network integrates a dynamic dual fusion module and a Mamba-based backbone to simultaneously tackle the aforementioned issues. Specifically, the dynamic dual fusion module utilizes multiple branches to consolidate various spatial and semantic information, thus enhancing feature diversity. The Mamba-based backbone leverages global feature fusion and local feature interaction, combining features in an adaptive manner to generate unique classification characteristics. Extensive experiments conducted on the TT100K (Tsinghua-Tencent 100K) datasets demonstrate that MDDFNet outperforms other state-of-the-art detectors, maintaining real-time processing capabilities of single-stage models while achieving superior performance. This confirms the efectiveness of MDDFNet in detecting small traffic signs. 

**Abstract (ZH)**: 基于Mamba的动态双分支融合网络在交通标志检测中的应用 

---
# FedAvgen: Metadata for Model Aggregation In Communication Systems 

**Title (ZH)**: FedAvgen：通信系统中模型聚合的元数据 

**Authors**: Anthony Kiggundu, Dennis Krummacker, Hans D. Schotten  

**Link**: [PDF](https://arxiv.org/pdf/2505.05486)  

**Abstract**: To improve business efficiency and minimize costs, Artificial Intelligence (AI) practitioners have adopted a shift from formulating models from scratch towards sharing pretrained models. The pretrained models are then aggregated into a global model with higher generalization capabilities, which is afterwards distributed to the client devices. This approach is known as federated learning and inherently utilizes different techniques to select the candidate client models averaged to obtain the global model. This approach, in the case of communication systems, faces challenges arising from the existential diversity in device profiles. The multiplicity in profiles motivates our conceptual assessment of a metaheuristic algorithm (FedAvgen), which relates each pretrained model with its weight space as metadata, to a phenotype and genotype, respectively. This parent-child genetic evolution characterizes the global averaging step in federated learning. We then compare the results of our approach to two widely adopted baseline federated learning algorithms like Federated Averaging (FedAvg) and Federated Stochastic Gradient Descent (FedSGD). 

**Abstract (ZH)**: 为了提高商务效率并降低成本，人工智能从业者转向了共享预训练模型的方式，而不是从头构建模型。预训练模型被聚合为具有更高泛化能力的全局模型，随后分发到客户端设备。这种方法称为联邦学习，通过不同的技术选择候选客户端模型的平均值来构建全局模型。在通信系统中，多样的设备配置带来了挑战。这种多样性促使我们对一种元启发式算法（FedAvgen）进行概念性评估，将每个预训练模型与其权重空间分别对应于表型和基因型。父母-子女的遗传进化特性描述了联邦学习中的全局平均步骤。我们随后将这种方法的结果与两种广泛采用的基线联邦学习算法（Federated Averaging (FedAvg) 和 Federated Stochastic Gradient Descent (FedSGD)）进行了比较。 

---
# Structure & Quality: Conceptual and Formal Foundations for the Mind-Body Problem 

**Title (ZH)**: 结构与质量：心灵-身体问题的概念基础与形式基础 

**Authors**: Ryan Williams  

**Link**: [PDF](https://arxiv.org/pdf/2505.05481)  

**Abstract**: This paper explores the hard problem of consciousness from a different perspective. Instead of drawing distinctions between the physical and the mental, an exploration of a more foundational relationship is examined: the relationship between structure and quality.
Information-theoretic measures are developed to quantify the mutual determinability between structure and quality, including a novel Q-S space for analyzing fidelity between the two domains. This novel space naturally points toward a five-fold categorization of possible relationships between structural and qualitative properties, illustrating each through conceptual and formal models.
The ontological implications of each category are examined, shedding light on debates around functionalism, emergentism, idealism, panpsychism, and neutral monism.
This new line of inquiry has established a framework for deriving theoretical constraints on qualitative systems undergoing evolution that is explored in my companion paper, Qualia & Natural Selection. 

**Abstract (ZH)**: 这篇论文从一个新的视角探索意识的难题。而不是区分物理和心理，而是探讨一种更为基础的关系：结构与质量之间的关系。发展了信息论措施来量化结构与质量之间的相互决定性，包括一种新颖的Q-S空间，用于分析两个领域的忠实度。这种新型空间自然地导向对结构与质量属性之间五种可能关系的分类，通过概念和形式模型展示每种分类。探讨了每种分类的本体论含义，阐明了功能主义、涌现主义、唯心主义、泛心论和中立一元论的争论。这一新的研究方向为在进化过程中质性系统的理论约束提供了框架，该框架在本文的姊妹论文《质态与自然选择》中进行了探索。 

---
# CLAM: Continuous Latent Action Models for Robot Learning from Unlabeled Demonstrations 

**Title (ZH)**: CLAM：连续潜在动作模型用于机器人从无标签示范学习 

**Authors**: Anthony Liang, Pavel Czempin, Matthew Hong, Yutai Zhou, Erdem Biyik, Stephen Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04999)  

**Abstract**: Learning robot policies using imitation learning requires collecting large amounts of costly action-labeled expert demonstrations, which fundamentally limits the scale of training data. A promising approach to address this bottleneck is to harness the abundance of unlabeled observations-e.g., from video demonstrations-to learn latent action labels in an unsupervised way. However, we find that existing methods struggle when applied to complex robot tasks requiring fine-grained motions. We design continuous latent action models (CLAM) which incorporate two key ingredients we find necessary for learning to solve complex continuous control tasks from unlabeled observation data: (a) using continuous latent action labels instead of discrete representations, and (b) jointly training an action decoder to ensure that the latent action space can be easily grounded to real actions with relatively few labeled examples. Importantly, the labeled examples can be collected from non-optimal play data, enabling CLAM to learn performant policies without access to any action-labeled expert data. We demonstrate on continuous control benchmarks in DMControl (locomotion) and MetaWorld (manipulation), as well as on a real WidowX robot arm that CLAM significantly outperforms prior state-of-the-art methods, remarkably with a 2-3x improvement in task success rate compared to the best baseline. Videos and code can be found at this http URL. 

**Abstract (ZH)**: 使用模仿学习学习机器人策略需要收集大量昂贵的动作标注专家示例，这从根本上限制了训练数据的规模。通过利用未标注观察数据（例如来自视频示例的观察数据）以无监督方式学习潜在动作标签来解决这一瓶颈是一种有前景的方法。然而，我们发现现有方法在处理需要精细动作的复杂机器人任务时表现不佳。我们设计了连续潜在动作模型（CLAM），该模型包含学习复杂连续控制任务所需的关键成分：（a）使用连续潜在动作标签代替离散表示，以及（b）联合训练一个动作解码器以确保潜在动作空间可以相对较少的标注示例的支持下易于与真实动作对接。重要的是，这些标注示例可以从非最优玩法数据中收集，从而使CLAM能够在不访问任何动作标注专家数据的情况下学习表现良好的策略。我们在DMControl（移动性）和MetaWorld（操作性）的连续控制基准测试中以及在真实WidowX机器人臂上展示了CLAM显著优于先前的最佳方法，与最好的基线相比，任务成功率提高了2-3倍。更多视频和代码请参见此链接。 

---
