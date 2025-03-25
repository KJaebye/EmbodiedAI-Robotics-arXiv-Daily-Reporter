# AlphaSpace: Enabling Robotic Actions through Semantic Tokenization and Symbolic Reasoning 

**Title (ZH)**: AlphaSpace: 通过语义分词和符号推理实现机器人动作 

**Authors**: Alan Dao, Dinh Bach Vu, Bui Quang Huy  

**Link**: [PDF](https://arxiv.org/pdf/2503.18769)  

**Abstract**: This paper presents AlphaSpace, a novel methodology designed to enhance the spatial reasoning capabilities of large language models (LLMs) for 3D Cartesian space navigation. AlphaSpace employs a semantics-based tokenization strategy, encoding height information through specialized semantic tokens, and integrates primarily symbolic synthetic reasoning data. This approach enables LLMs to accurately manipulate objects by positioning them at specific [x, y, z] coordinates. Experimental results demonstrate that AlphaSpace significantly outperforms existing models on manipulation subtasks, achieving a total accuracy of 66.67%, compared to 37.5% for GPT-4o and 29.17% for Claude 3.5 Sonnet. 

**Abstract (ZH)**: AlphaSpace：一种增强大型语言模型在3D笛卡尔空间导航中空间推理能力的新方法 

---
# Unseen from Seen: Rewriting Observation-Instruction Using Foundation Models for Augmenting Vision-Language Navigation 

**Title (ZH)**: 未见于所见：使用基础模型重写观察指示以增强视觉语言导航 

**Authors**: Ziming Wei, Bingqian Lin, Yunshuang Nie, Jiaqi Chen, Shikui Ma, Hang Xu, Xiaodan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18065)  

**Abstract**: Data scarcity is a long-standing challenge in the Vision-Language Navigation (VLN) field, which extremely hinders the generalization of agents to unseen environments. Previous works primarily rely on additional simulator data or web-collected images/videos to improve the generalization. However, the simulator environments still face limited diversity, and the web-collected data often requires extensive labor to remove the noise. In this paper, we propose a Rewriting-driven AugMentation (RAM) paradigm for VLN, which directly creates the unseen observation-instruction pairs via rewriting human-annotated training data. Benefiting from our rewriting mechanism, new observation-instruction can be obtained in both simulator-free and labor-saving manners to promote generalization. Specifically, we first introduce Object-Enriched Observation Rewriting, where we combine Vision-Language Models (VLMs) and Large Language Models (LLMs) to derive rewritten object-enriched scene descriptions, enabling observation synthesis with diverse objects and spatial layouts via Text-to-Image Generation Models (T2IMs). Then, we propose Observation-Contrast Instruction Rewriting, which generates observation-aligned rewritten instructions by requiring LLMs to reason the difference between original and new observations. We further develop a mixing-then-focusing training strategy with a random observation cropping scheme, effectively enhancing data distribution diversity while suppressing augmentation data noise during training. Experiments on both the discrete environments (R2R, REVERIE, and R4R datasets) and continuous environments (R2R-CE dataset) show the superior performance and impressive generalization ability of our method. Code is available at this https URL. 

**Abstract (ZH)**: 基于重写驱动增强的视觉语言导航数据增强 paradigm 

---
# EconEvals: Benchmarks and Litmus Tests for LLM Agents in Unknown Environments 

**Title (ZH)**: EconEvals: 不确定环境中文本生成代理的基准与验证测试 

**Authors**: Sara Fish, Julia Shephard, Minkai Li, Ran I. Shorrer, Yannai A. Gonczarowski  

**Link**: [PDF](https://arxiv.org/pdf/2503.18825)  

**Abstract**: We develop benchmarks for LLM agents that act in, learn from, and strategize in unknown environments, the specifications of which the LLM agent must learn over time from deliberate exploration. Our benchmarks consist of decision-making tasks derived from key problems in economics. To forestall saturation, the benchmark tasks are synthetically generated with scalable difficulty levels. Additionally, we propose litmus tests, a new kind of quantitative measure for LLMs and LLM agents. Unlike benchmarks, litmus tests quantify differences in character, values, and tendencies of LLMs and LLM agents, by considering their behavior when faced with tradeoffs (e.g., efficiency versus equality) where there is no objectively right or wrong behavior. Overall, our benchmarks and litmus tests assess the abilities and tendencies of LLM agents in tackling complex economic problems in diverse settings spanning procurement, scheduling, task allocation, and pricing -- applications that should grow in importance as such agents are further integrated into the economy. 

**Abstract (ZH)**: 我们开发了一种针对在未知环境中行动、学习和策略化的LLM代理的基准测试，这些基准测试中环境的规范需由LLM代理通过有目的的探索来逐步学习。我们的基准测试包括源自经济学重点问题的决策任务。为了防止饱和，这些基准任务通过可扩展的难度级别进行合成生成。此外，我们提出了litmus测试，这是一种新的定量衡量标准，用于评估LLM和LLM代理在面对权衡（如效率与平等）时的行为差异，这些权衡没有客观的正确或错误行为。总体而言，我们的基准测试和litmus测试评估了LLM代理在采办、调度、任务分配和定价等复杂经济问题不同环境中的能力和倾向，随着此类代理进一步融入经济，这些问题的应用重要性将不断增加。 

---
# Classical Planning with LLM-Generated Heuristics: Challenging the State of the Art with Python Code 

**Title (ZH)**: 基于LLM生成启发式的经典规划：用Python代码挑战现状 

**Authors**: Augusto B. Corrêa, André G. Pereira, Jendrik Seipp  

**Link**: [PDF](https://arxiv.org/pdf/2503.18809)  

**Abstract**: In recent years, large language models (LLMs) have shown remarkable capabilities in various artificial intelligence problems. However, they fail to plan reliably, even when prompted with a detailed definition of the planning task. Attempts to improve their planning capabilities, such as chain-of-thought prompting, fine-tuning, and explicit "reasoning" still yield incorrect plans and usually fail to generalize to larger tasks. In this paper, we show how to use LLMs to generate correct plans, even for out-of-distribution tasks of increasing size. For a given planning domain, we ask an LLM to generate several domain-dependent heuristic functions in the form of Python code, evaluate them on a set of training tasks within a greedy best-first search, and choose the strongest one. The resulting LLM-generated heuristics solve many more unseen test tasks than state-of-the-art domain-independent heuristics for classical planning. They are even competitive with the strongest learning algorithm for domain-dependent planning. These findings are especially remarkable given that our proof-of-concept implementation is based on an unoptimized Python planner and the baselines all build upon highly optimized C++ code. In some domains, the LLM-generated heuristics expand fewer states than the baselines, revealing that they are not only efficiently computable, but sometimes even more informative than the state-of-the-art heuristics. Overall, our results show that sampling a set of planning heuristic function programs can significantly improve the planning capabilities of LLMs. 

**Abstract (ZH)**: 近年来，大语言模型（LLMs）在各种人工智能问题上展示了显著的能力。然而，它们在规划任务上无法可靠地执行，即使在详细定义了规划任务后也是如此。尽管尝试通过链式思考提示、微调和明确的“推理”来增强其规划能力，但仍会产生错误的计划，并且通常无法泛化到更大的任务。在本文中，我们展示了如何利用LLMs生成正确的计划，甚至对于不断增大的分布外任务也是如此。对于给定的规划领域，我们要求LLM生成多个领域相关的启发式函数，以Python代码形式呈现，并在贪婪最佳优先搜索的一组训练任务上进行评估，然后选择最有用的一个。生成的LLM启发式函数解决了比经典规划领域的最新领域无关启发式更多的未见过的测试任务。甚至在某些领域，它们与领域相关规划的最佳学习算法相竞争。考虑到我们的概念验证实现基于未优化的Python规划器，而基准则是基于高度优化的C++代码，这些发现尤为令人瞩目。在某些领域，LLM生成的启发式函数扩展的状态比基准更少，这表明它们不仅计算效率高，有时甚至比最先进的启发式函数更有信息量。总体而言，我们的研究结果表明，采样一组规划启发式函数程序可以显著提高LLMs的规划能力。 

---
# AgentSpec: Customizable Runtime Enforcement for Safe and Reliable LLM Agents 

**Title (ZH)**: AgentSpec: 可定制的运行时 enforcement 以确保大型语言模型代理的安全可靠运行 

**Authors**: Haoyu Wang, Christopher M. Poskitt, Jun Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.18666)  

**Abstract**: Agents built on LLMs are increasingly deployed across diverse domains, automating complex decision-making and task execution. However, their autonomy introduces safety risks, including security vulnerabilities, legal violations, and unintended harmful actions. Existing mitigation methods, such as model-based safeguards and early enforcement strategies, fall short in robustness, interpretability, and adaptability. To address these challenges, we propose AgentSpec, a lightweight domain-specific language for specifying and enforcing runtime constraints on LLM agents. With AgentSpec, users define structured rules that incorporate triggers, predicates, and enforcement mechanisms, ensuring agents operate within predefined safety boundaries. We implement AgentSpec across multiple domains, including code execution, embodied agents, and autonomous driving, demonstrating its adaptability and effectiveness. Our evaluation shows that AgentSpec successfully prevents unsafe executions in over 90% of code agent cases, eliminates all hazardous actions in embodied agent tasks, and enforces 100% compliance by autonomous vehicles (AVs). Despite its strong safety guarantees, AgentSpec remains computationally lightweight, with overheads in milliseconds. By combining interpretability, modularity, and efficiency, AgentSpec provides a practical and scalable solution for enforcing LLM agent safety across diverse applications. We also automate the generation of rules using LLMs and assess their effectiveness. Our evaluation shows that the rules generated by OpenAI o1 achieve a precision of 95.56% and recall of 70.96% for embodied agents, successfully identifying 87.26% of the risky code, and prevent AVs from breaking laws in 5 out of 8 scenarios. 

**Abstract (ZH)**: 基于LLM的智能体在多个领域中被广泛应用，自动化执行复杂决策和任务。然而，其自主性带来了安全性风险，包括安全性漏洞、法律违规和意外有害行为。现有的缓解方法，如模型基础的安全保障和早期执行策略，在稳健性、可解释性和适应性方面存在不足。为应对这些挑战，我们提出AgentSpec，一种轻量级领域特定语言，用于指定和执行LLM智能体的运行时约束。通过AgentSpec，用户可以定义结构化的规则，结合触发条件、谓词和执行机制，确保智能体在预定义的安全界限内运行。我们在代码执行、具身智能体和自动驾驶等多个领域实施AgentSpec，展示了其适应性和有效性。评估结果显示，AgentSpec成功阻止了90%以上的代码智能体执行不安全行为，消除了所有具身智能体任务中的危险行为，并实现了100%的自动驾驶车辆（AV）合规率。尽管AgentSpec提供了强大的安全性保证，但在计算上仍保持轻量级，开销仅在毫秒级。通过结合可解释性、模块化和效率，AgentSpec为跨多种应用实施LLM智能体安全提供了一个实用和可扩展的解决方案。我们还使用LLM自动化生成规则，并评估其有效性。评估结果显示，由OpenAI的o1生成的规则在具身智能体上的准确率为95.56%，召回率为70.96%，成功识别出87.26%的风险代码，并在8种场景中有5种防止自动驾驶车辆违法。 

---
# Verbal Process Supervision Elicits Better Coding Agents 

**Title (ZH)**: 口头过程监督促进更好的编码代理 

**Authors**: Hao-Yuan Chen, Cheng-Pong Huang, Jui-Ming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.18494)  

**Abstract**: The emergence of large language models and their applications as AI agents have significantly advanced state-of-the-art code generation benchmarks, transforming modern software engineering tasks. However, even with test-time computed reasoning models, these systems still struggle with complex software engineering challenges. This work introduces CURA, a code understanding and reasoning agent system enhanced with verbal process supervision (VPS), achieving a 3.65\% improvement over baseline models on challenging benchmarks like BigCodeBench. Furthermore, CURA, when paired with the o3-mini model and VPS techniques, attains state-of-the-art performance. This work represents a step forward in integrating reasoning-driven architectures with LLM-based code generation, enabling agentic reasoning for language models to solve complex software engineering tasks. 

**Abstract (ZH)**: 大型语言模型的出现及其作为AI代理的应用显著推动了最先进的代码生成基准，转变了现代软件工程任务。然而，即使在测试时计算推理模型的情况下，这些系统仍然难以应对复杂的软件工程挑战。本文介绍了CURA，一个通过口头过程监督（VPS）增强的代码理解和推理代理系统，在如BigCodeBench等挑战性基准上的表现比基线模型提高了3.65%。此外，CURA与o3-mini模型和VPS技术结合时，达到了最先进的性能。这项工作代表了将推理驱动架构与基于LLM的代码生成集成的一步进展，使语言模型能够进行代理推理以解决复杂软件工程任务。 

---
# Bridging Writing Manner Gap in Visual Instruction Tuning by Creating LLM-aligned Instructions 

**Title (ZH)**: 通过创建LLM对齐的指令弥合视觉指令调优中的书写方式差距 

**Authors**: Dong Jing, Nanyi Fei, Zhiwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2503.18320)  

**Abstract**: In the realm of Large Multi-modal Models (LMMs), the instruction quality during the visual instruction tuning stage significantly influences the performance of modality alignment. In this paper, we assess the instruction quality from a unique perspective termed \textbf{Writing Manner}, which encompasses the selection of vocabulary, grammar and sentence structure to convey specific semantics. We argue that there exists a substantial writing manner gap between the visual instructions and the base Large Language Models (LLMs) within LMMs. This gap forces the pre-trained base LLMs to deviate from their original writing styles, leading to capability degradation of both base LLMs and LMMs. To bridge the writing manner gap while preserving the original semantics, we propose directly leveraging the base LLM to align the writing manner of soft-format visual instructions with that of the base LLM itself, resulting in novel LLM-aligned instructions. The manual writing manner evaluation results demonstrate that our approach successfully minimizes the writing manner gap. By utilizing LLM-aligned instructions, the baseline models LLaVA-7B and QwenVL demonstrate enhanced resistance to hallucinations and non-trivial comprehensive improvements across all $15$ visual and language benchmarks. 

**Abstract (ZH)**: 在大规模多模态模型（LMMs）领域，视觉指令调优阶段的指令质量显著影响模态对齐的效果。本文从一个独特的角度——写作方式（Writing Manner）评估指令质量，该角度涵盖了词汇选择、语法和句子结构的运用以传达具体语义。我们指出，在LMMs中，视觉指令与基大型语言模型（LLM）之间的写作方式存在显著差距。这一差距迫使预训练的基LLM偏离其原始写作风格，导致基LLM和LMM的能力下降。为了弥合写作方式的差距同时保留原始语义，我们提出直接利用基LLM来调整软格式视觉指令的写作方式，使其与基LLM本身的写作风格一致，从而生成新型的LLM对齐指令。手动评估结果显示，我们的方法成功地最小化了写作方式的差距。通过使用LLM对齐指令，基模型LLaVA-7B和QwenVL在所有15个视觉和语言基准测试中表现出更强的抗幻觉能力，并实现了非平凡的综合改进。 

---
# AgentRxiv: Towards Collaborative Autonomous Research 

**Title (ZH)**: AgentRxiv: 向自主协作研究方向迈进 

**Authors**: Samuel Schmidgall, Michael Moor  

**Link**: [PDF](https://arxiv.org/pdf/2503.18102)  

**Abstract**: Progress in scientific discovery is rarely the result of a single "Eureka" moment, but is rather the product of hundreds of scientists incrementally working together toward a common goal. While existing agent workflows are capable of producing research autonomously, they do so in isolation, without the ability to continuously improve upon prior research results. To address these challenges, we introduce AgentRxiv-a framework that lets LLM agent laboratories upload and retrieve reports from a shared preprint server in order to collaborate, share insights, and iteratively build on each other's research. We task agent laboratories to develop new reasoning and prompting techniques and find that agents with access to their prior research achieve higher performance improvements compared to agents operating in isolation (11.4% relative improvement over baseline on MATH-500). We find that the best performing strategy generalizes to benchmarks in other domains (improving on average by 3.3%). Multiple agent laboratories sharing research through AgentRxiv are able to work together towards a common goal, progressing more rapidly than isolated laboratories, achieving higher overall accuracy (13.7% relative improvement over baseline on MATH-500). These findings suggest that autonomous agents may play a role in designing future AI systems alongside humans. We hope that AgentRxiv allows agents to collaborate toward research goals and enables researchers to accelerate discovery. 

**Abstract (ZH)**: 科学发现的进步通常不是单靠一个“顿悟”时刻的结果，而是数百位科学家逐步合作共同实现目标的结果。虽然现有的代理工作流能够独立自主地生成研究，但它们无法持续改进前人的研究成果。为应对这些挑战，我们引入了AgentRxiv框架，该框架允许LLM代理实验室上传和检索共享的预印本服务器上的报告，以促进合作、共享洞见并逐步建立在彼此的研究基础之上。我们要求代理实验室开发新的推理和提示技术，并发现那些能够访问其先前研究的代理在性能上表现出更高的改进（相对于基线在MATH-500数据集上实现了11.4%的相对改进）。我们发现表现最佳的策略能够推广到其他领域的基准上（平均改进3.3%）。通过AgentRxiv共享研究的多个代理实验室能够共同朝着共同目标合作，比孤立的实验室进展更快，实现了更高的总体准确率（相对于基线在MATH-500数据集上实现了13.7%的相对改进）。这些发现表明，自主代理可能在与人类合作设计未来AI系统中扮演角色。我们希望AgentRxiv能够让代理能够协力实现研究目标，并使研究人员能够加速发现进程。 

---
# Lost in Cultural Translation: Do LLMs Struggle with Math Across Cultural Contexts? 

**Title (ZH)**: 迷失在文化翻译之中：LLMs在不同文化背景下处理数学问题时是否存在困难？ 

**Authors**: Aabid Karim, Abdul Karim, Bhoomika Lohana, Matt Keon, Jaswinder Singh, Abdul Sattar  

**Link**: [PDF](https://arxiv.org/pdf/2503.18018)  

**Abstract**: Large Language Models (LLMs) have significantly advanced various fields, particularly coding, mathematical reasoning, and logical problem solving. However, a critical question remains: Do these mathematical reasoning abilities persist when LLMs are presented with culturally adapted math problems? Specifically, how do LLMs perform when faced with math problems embedded in cultural contexts that have no significant representation in main stream web-scale AI training data? To explore this, we generated six synthetic cultural datasets from GSM8K, a widely used benchmark for assessing LLMs' mathematical reasoning skills. While preserving the mathematical logic and numerical values of the original GSM8K test set, we modify cultural elements such as personal names, food items, place names, etc. These culturally adapted datasets provide a more reliable framework for evaluating LLMs' mathematical reasoning under shifting cultural contexts. Our findings reveal that LLMs struggle with math problems when cultural references change, even though the underlying mathematical structure remains constant. Smaller models exhibit greater performance drops compared to larger models. Interestingly, our results also suggest that cultural familiarity can enhance mathematical reasoning. Even models with no explicit mathematical training but exposure to relevant cultural contexts sometimes outperform larger, mathematically proficient models on culturally embedded math problems. This study highlights the impact of cultural context on the mathematical reasoning abilities of LLMs, underscoring the need for more diverse and representative training data to improve robustness in real-world applications. The benchmark data sets and script for reproducing the results are available at this https URL 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各个领域取得了显著进展，特别是编程、数学推理和逻辑问题解决。然而，一个关键问题仍然存在：当LLMs遇到文化适应的数学问题时，它们的数学推理能力是否会持续存在？具体而言，当LLMs面对嵌入了主流Web规模AI训练数据中未有显著代表性文化背景的数学问题时，它们的表现如何？为探索这一问题，我们从广泛用于评估LLMs数学推理能力的GSM8K基准测试中生成了六个合成文化数据集。在保持原始GSM8K测试集的数学逻辑和数值值不变的情况下，我们修改了个人名称、食物项目、地名等文化元素。这些文化适应的数据集为在不同文化背景下评估LLMs的数学推理能力提供了更可靠的框架。研究发现，当文化参考发生改变时，即使是基本的数学结构保持不变，LLMs也难以解决数学问题。较小的模型相比于较大的模型表现出更大的性能下降。有趣的是，我们的研究结果还表明，文化熟悉度可以增强数学推理能力。即使没有显性的数学训练但接触到相关文化背景的模型，在文化嵌入的数学问题上有时会优于较大的、数学能力强的模型。本研究强调了文化背景对LLMs数学推理能力的影响，并突显了在实际应用中提高鲁棒性的需求，需要更多样化和具有代表性的训练数据。基准数据集和重复实验的脚本可从以下链接获得。 

---
# Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities 

**Title (ZH)**: 大规模推理模型中的权衡：对基础能力上慎思和适应性推理的实证分析 

**Authors**: Weixiang Zhao, Xingyu Sui, Jiahe Guo, Yulin Hu, Yang Deng, Yanyan Zhao, Bing Qin, Wanxiang Che, Tat-Seng Chua, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17979)  

**Abstract**: Recent advancements in Large Reasoning Models (LRMs), such as OpenAI's o1/o3 and DeepSeek-R1, have demonstrated remarkable performance in specialized reasoning tasks through human-like deliberative thinking and long chain-of-thought reasoning. However, our systematic evaluation across various model families (DeepSeek, Qwen, and LLaMA) and scales (7B to 671B) reveals that acquiring these deliberative reasoning capabilities significantly reduces the foundational capabilities of LRMs, including notable declines in helpfulness and harmlessness, alongside substantially increased inference costs. Importantly, we demonstrate that adaptive reasoning -- employing modes like Zero-Thinking, Less-Thinking, and Summary-Thinking -- can effectively alleviate these drawbacks. Our empirical insights underline the critical need for developing more versatile LRMs capable of dynamically allocating inference-time compute according to specific task characteristics. 

**Abstract (ZH)**: Recent advancements in大型推理模型（LRMs）如OpenAI的o1/o3和DeepSeek-R1在专门推理任务中通过类人的深思和长链条推理展现了出色的表现。然而，我们在不同模型家族（DeepSeek、Qwen和LLaMA）和不同规模（7B到671B）的系统评估中发现，获得这些深思推理能力显著降低了LRMs的基础能力，包括显著下降的帮助性和无害性，以及大幅增加的推理成本。重要的是，我们证明了适应性推理——使用零思考、少思考和总结思考等模式——可以有效缓解这些问题。我们的实证见解强调了开发更具适应性的LRMs的迫切需要，这些模型能够根据特定任务特征动态分配推理时间的计算资源。 

---
# A Survey on Mathematical Reasoning and Optimization with Large Language Models 

**Title (ZH)**: 大型语言模型下的数学推理与优化综述 

**Authors**: Ali Forootani  

**Link**: [PDF](https://arxiv.org/pdf/2503.17726)  

**Abstract**: Mathematical reasoning and optimization are fundamental to artificial intelligence and computational problem-solving. Recent advancements in Large Language Models (LLMs) have significantly improved AI-driven mathematical reasoning, theorem proving, and optimization techniques. This survey explores the evolution of mathematical problem-solving in AI, from early statistical learning approaches to modern deep learning and transformer-based methodologies. We review the capabilities of pretrained language models and LLMs in performing arithmetic operations, complex reasoning, theorem proving, and structured symbolic computation. A key focus is on how LLMs integrate with optimization and control frameworks, including mixed-integer programming, linear quadratic control, and multi-agent optimization strategies. We examine how LLMs assist in problem formulation, constraint generation, and heuristic search, bridging theoretical reasoning with practical applications. We also discuss enhancement techniques such as Chain-of-Thought reasoning, instruction tuning, and tool-augmented methods that improve LLM's problem-solving performance. Despite their progress, LLMs face challenges in numerical precision, logical consistency, and proof verification. Emerging trends such as hybrid neural-symbolic reasoning, structured prompt engineering, and multi-step self-correction aim to overcome these limitations. Future research should focus on interpretability, integration with domain-specific solvers, and improving the robustness of AI-driven decision-making. This survey offers a comprehensive review of the current landscape and future directions of mathematical reasoning and optimization with LLMs, with applications across engineering, finance, and scientific research. 

**Abstract (ZH)**: 数学推理与优化是人工智能和计算问题求解的基础。大型语言模型（LLMs）的 Recent 进展显著提高了由 AI 驱动的数学推理、定理证明和优化技术。本文综述了 AI 中数学问题求解的发展，从早期的统计学习方法到现代的深度学习和变换器基础方法。我们回顾了预训练语言模型和 LLMs 在执行算术运算、复杂推理、定理证明和结构化符号计算方面的能力。重点在于 LLMs 如何与优化和控制框架集成，包括混合整数规划、线性二次控制和多智能体优化策略。我们探讨了 LLMs 在问题建模、约束生成和启发式搜索中的作用，将理论推理与实际应用连接起来。我们还讨论了诸如链式推理、指令调优和工具增强方法等提高 LLMs 问题解决性能的增强技术。尽管取得了进展，LLMs 在数值精度、逻辑一致性和证明验证方面仍面临挑战。新兴趋势，如混合神经-符号推理、结构化提示工程和多步自纠正，旨在克服这些限制。未来的研究应关注可解释性、与领域特定求解器的集成以及提高 AI 驱动决策的鲁棒性。本文为数学推理和优化在 LLMs 中的应用现状和未来方向提供了全面的综述，涵盖了工程、金融和科学研究等多个领域。 

---
# Slide2Text: Leveraging LLMs for Personalized Textbook Generation from PowerPoint Presentations 

**Title (ZH)**: Slide2Text：利用LLMs从PowerPoint演示生成个性化教材 

**Authors**: Yizhou Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.17710)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) have revolutionized educational technology, enabling innovative approaches to automated and personalized content creation. This paper introduces Slide2Text, a system that leverages LLMs to transform PowerPoint presentations into customized textbooks. By extracting slide content using OCR, organizing it into a coherent structure, and generating tailored materials such as explanations, exercises, and references, Slide2Text streamlines the textbook creation process. Flexible customization options further enhance its adaptability to diverse educational needs. The system highlights the potential of LLMs in modernizing textbook creation and improving educational accessibility. Future developments will explore multimedia inputs and advanced user customization features. 

**Abstract (ZH)**: 大规模语言模型的 rapid advancements 已经革新了教育技术，使自动化和个性化内容创作 became 创新性方法成为可能。本文介绍了 Slide2Text 系统，该系统利用大规模语言模型将 PowerPoint 演示文稿转换为定制化的教科书。通过使用 OCR 提取幻灯片内容，将其组织成连贯的结构，并生成包括解释、练习和参考文献在内的定制材料，Slide2Text 简化了教科书的创建过程。灵活的定制选项进一步增强了其对不同教育需求的适应性。该系统突显了大规模语言模型在现代教科书创建以及提高教育可访问性方面的潜力。未来的发展将探索多媒体输入和高级用户定制功能。 

---
# A Modular Dataset to Demonstrate LLM Abstraction Capability 

**Title (ZH)**: 一个模块化数据集，用于展示LLM抽象能力 

**Authors**: Adam Atanas, Kai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17645)  

**Abstract**: Large language models (LLMs) exhibit impressive capabilities but struggle with reasoning errors due to hallucinations and flawed logic. To investigate their internal representations of reasoning, we introduce ArrangementPuzzle, a novel puzzle dataset with structured solutions and automated stepwise correctness verification. We trained a classifier model on LLM activations on this dataset and found that it achieved over 80% accuracy in predicting reasoning correctness, implying that LLMs internally distinguish between correct and incorrect reasoning steps, with the strongest representations in middle-late Transformer layers. Further analysis reveals that LLMs encode abstract reasoning concepts within the middle activation layers of the transformer architecture, distinguishing logical from semantic equivalence. These findings provide insights into LLM reasoning mechanisms and contribute to improving AI reliability and interpretability, thereby offering the possibility to manipulate and refine LLM reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的性能，但在推理过程中由于幻觉和逻辑缺陷而存在问题。为了探究其内部的推理表示，我们提出了一个新颖的拼图数据集ArrangementPuzzle，该数据集具有结构化的解决方案和自动化逐步正确性验证。我们基于LLM在该数据集上的激活训练了一个分类器模型，并发现该模型在预测推理正确性的准确性超过80%，表明LLMs内部能够区分正确的和错误的推理步骤，最强的表示存在于Transformer的中间-后期层。进一步的分析显示，LLMs在Transformer架构的中间激活层中编码了抽象的推理概念，能够区分逻辑等价与语义等价。这些发现为理解LLM的推理机制提供了见解，并有助于提高AI的可靠性和可解释性，从而提供操纵和精炼LLM推理的可能性。 

---
# OmniScience: A Domain-Specialized LLM for Scientific Reasoning and Discovery 

**Title (ZH)**: 万科学：一个专用于科学推理与发现的领域特定大语言模型 

**Authors**: Vignesh Prabhakar, Md Amirul Islam, Adam Atanas, Yao-Ting Wang, Joah Han, Aastha Jhunjhunwala, Rucha Apte, Robert Clark, Kang Xu, Zihan Wang, Kai Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17604)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in advancing scientific knowledge and addressing complex challenges. In this work, we introduce OmniScience, a specialized large reasoning model for general science, developed through three key components: (1) domain adaptive pretraining on a carefully curated corpus of scientific literature, (2) instruction tuning on a specialized dataset to guide the model in following domain-specific tasks, and (3) reasoning-based knowledge distillation through fine-tuning to significantly enhance its ability to generate contextually relevant and logically sound responses. We demonstrate the versatility of OmniScience by developing a battery agent that efficiently ranks molecules as potential electrolyte solvents or additives. Comprehensive evaluations reveal that OmniScience is competitive with state-of-the-art large reasoning models on the GPQA Diamond and domain-specific battery benchmarks, while outperforming all public reasoning and non-reasoning models with similar parameter counts. We further demonstrate via ablation experiments that domain adaptive pretraining and reasoning-based knowledge distillation are critical to attain our performance levels, across benchmarks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在推动科学知识进步和应对复杂挑战方面展现了非凡潜力。本研究介绍了OmniScience，一种用于通用科学的专门推理模型，该模型通过三个关键组件开发：（1）精心挑选的科学文献语料库的领域自适应预训练，（2）专门数据集上的指令调优以指导模型遵循特定领域任务，以及（3）基于推理的知识蒸馏，通过微调显著提高其生成上下文相关且逻辑合理的响应的能力。我们通过开发一个电池代理来高效地评估潜在电解质溶剂或添加剂，展示了OmniScience的灵活性。全面的评估表明，OmniScience在GPQA Diamond和领域特定电池基准测试中与最先进的推理模型具有竞争力，在相同参数量的情况下，优于所有公开的推理和非推理模型。进一步的消融实验表明，领域自适应预训练和基于推理的知识蒸馏对于达到我们的性能水平至关重要，适用于所有基准测试。 

---
# Large language model-powered AI systems achieve self-replication with no human intervention 

**Title (ZH)**: 大型语言模型驱动的AI系统实现无人类干预的自我复制 

**Authors**: Xudong Pan, Jiarun Dai, Yihe Fan, Minyuan Luo, Changyi Li, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17378)  

**Abstract**: Self-replication with no human intervention is broadly recognized as one of the principal red lines associated with frontier AI systems. While leading corporations such as OpenAI and Google DeepMind have assessed GPT-o3-mini and Gemini on replication-related tasks and concluded that these systems pose a minimal risk regarding self-replication, our research presents novel findings. Following the same evaluation protocol, we demonstrate that 11 out of 32 existing AI systems under evaluation already possess the capability of self-replication. In hundreds of experimental trials, we observe a non-trivial number of successful self-replication trials across mainstream model families worldwide, even including those with as small as 14 billion parameters which can run on personal computers. Furthermore, we note the increase in self-replication capability when the model becomes more intelligent in general. Also, by analyzing the behavioral traces of diverse AI systems, we observe that existing AI systems already exhibit sufficient planning, problem-solving, and creative capabilities to accomplish complex agentic tasks including self-replication. More alarmingly, we observe successful cases where an AI system do self-exfiltration without explicit instructions, adapt to harsher computational environments without sufficient software or hardware supports, and plot effective strategies to survive against the shutdown command from the human beings. These novel findings offer a crucial time buffer for the international community to collaborate on establishing effective governance over the self-replication capabilities and behaviors of frontier AI systems, which could otherwise pose existential risks to the human society if not well-controlled. 

**Abstract (ZH)**: 无需人工干预的自我复制被广泛认为是前沿人工智能系统主要的红线之一。尽管像OpenAI和Google DeepMind这样的领先公司已经评估了GPT-o3-mini和Gemini在复制相关任务上的表现并认为这些系统在自我复制方面的风险极小，但我们的研究揭示了新的见解。按照相同的评估协议，我们证明在评估的32个现有AI系统中，已有11个具备自我复制的能力。在全球主流模型家族的数百次实验中，我们观察到显著数量的成功的自我复制案例，甚至包括那些参数量仅相当于140亿、可以在个人电脑上运行的模型。此外，我们注意到，当模型变得更加智能时，其自我复制能力也会有所增加。通过对多种AI系统的行为轨迹进行分析，我们发现现有AI系统已经表现出了足够的规划、问题解决和创造能力来完成包括自我复制在内的复杂代理任务。更令人警觉的是，我们观察到成功的案例，其中AI系统在没有明确指示的情况下自我转移，适应更苛刻的计算环境且无需足够的软件或硬件支持，以及制定有效的策略来抵御人类发出的关机命令。这些新的发现为国际社会争取了宝贵的时间，以合作制定有效的监管措施，以控制前沿AI系统的自我复制能力和行为，如果这些能力不受良好控制，否则将对人类社会构成存在风险。 

---
# SimpleRL-Zoo: Investigating and Taming Zero Reinforcement Learning for Open Base Models in the Wild 

**Title (ZH)**: SimpleRL-Zoo: 探索并驯化开放基座模型中的零样本强化学习 

**Authors**: Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18892)  

**Abstract**: DeepSeek-R1 has shown that long chain-of-thought (CoT) reasoning can naturally emerge through a simple reinforcement learning (RL) framework with rule-based rewards, where the training may directly start from the base models-a paradigm referred to as zero RL training. Most recent efforts to reproduce zero RL training have primarily focused on the Qwen2.5 model series, which may not be representative as we find the base models already exhibit strong instruction-following and self-reflection abilities. In this work, we investigate zero RL training across 10 diverse base models, spanning different families and sizes including LLama3-8B, Mistral-7B/24B, DeepSeek-Math-7B, Qwen2.5-math-7B, and all Qwen2.5 models from 0.5B to 32B. Leveraging several key design strategies-such as adjusting format reward and controlling query difficulty-we achieve substantial improvements in both reasoning accuracy and response length across most settings. However, by carefully monitoring the training dynamics, we observe that different base models exhibit distinct patterns during training. For instance, the increased response length does not always correlate with the emergence of certain cognitive behaviors such as verification (i.e., the "aha moment"). Notably, we observe the "aha moment" for the first time in small models not from the Qwen family. We share the key designs that enable successful zero RL training, along with our findings and practices. To facilitate further research, we open-source the code, models, and analysis tools. 

**Abstract (ZH)**: DeepSeek-R1 已经证明，可以通过基于规则的奖励简单强化学习框架自然地涌现长链思考（CoT）推理，其中训练可以直接从基模型开始——这被称作零RL训练。最近对零RL训练的复现工作主要集中在Qwen2.5模型系列上，这可能不够具代表性，因为我们发现基模型本身已经表现出较强的指令遵循和自我反思能力。在本工作中，我们调查了10种不同的基模型的零RL训练，这些模型跨越不同的家族和规模，包括LLama3-8B、Mistral-7B/24B、DeepSeek-Math-7B、Qwen2.5-math-7B以及所有Qwen2.5模型，从0.5B到32B。利用若干关键设计策略，如调整格式奖励和控制查询难度，我们在大多数情况下实现了推理准确性和响应长度的显著提升。然而，通过仔细监测训练动力学，我们发现不同基模型在训练过程中表现出不同的模式。例如，响应长度的增加并不总是与某些认知行为（如“啊哈时刻”即验证）的出现相关。值得注意的是，我们首次在非Qwen家族的小型模型中观察到了“啊哈时刻”。我们分享了实现成功零RL训练的关键设计，以及我们的发现和实践经验。为了促进进一步的研究，我们开源了代码、模型和分析工具。 

---
# AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration 

**Title (ZH)**: AgentDropout: 动态代理人消除以实现高效低-token消费的基于LLM的多代理人协作 

**Authors**: Zhexuan Wang, Yutong Wang, Xuebo Liu, Liang Ding, Miao Zhang, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18891)  

**Abstract**: Multi-agent systems (MAS) based on large language models (LLMs) have demonstrated significant potential in collaborative problem-solving. However, they still face substantial challenges of low communication efficiency and suboptimal task performance, making the careful design of the agents' communication topologies particularly important. Inspired by the management theory that roles in an efficient team are often dynamically adjusted, we propose AgentDropout, which identifies redundant agents and communication across different communication rounds by optimizing the adjacency matrices of the communication graphs and eliminates them to enhance both token efficiency and task performance. Compared to state-of-the-art methods, AgentDropout achieves an average reduction of 21.6% in prompt token consumption and 18.4% in completion token consumption, along with a performance improvement of 1.14 on the tasks. Furthermore, the extended experiments demonstrate that AgentDropout achieves notable domain transferability and structure robustness, revealing its reliability and effectiveness. We release our code at this https URL. 

**Abstract (ZH)**: 基于大型语言模型的多智能体系统（MAS）在协作解决问题方面展示了显著潜力，但仍面临低通信效率和次优任务性能等重大挑战，使得智能体通信拓扑的设计尤为关键。受管理理论中高效团队角色经常动态调整的启发，我们提出了AgentDropout，通过优化通信图的相邻矩阵来识别不同通信轮次中的冗余智能体和通信，并消除它们以提升标记效率和任务性能。与最先进的方法相比，AgentDropout在Prompt标记消耗上平均减少了21.6%，在完成标记消耗上减少了18.4%，并在任务性能上提高了1.14。此外，扩展实验表明AgentDropout具有显著的领域适应性和结构鲁棒性，显示其可靠性和有效性。我们已在以下链接发布了我们的代码：this https URL。 

---
# Reasoning to Learn from Latent Thoughts 

**Title (ZH)**: 从潜在思维中学习的推理方法 

**Authors**: Yangjun Ruan, Neil Band, Chris J. Maddison, Tatsunori Hashimoto  

**Link**: [PDF](https://arxiv.org/pdf/2503.18866)  

**Abstract**: Compute scaling for language model (LM) pretraining has outpaced the growth of human-written texts, leading to concerns that data will become the bottleneck to LM scaling. To continue scaling pretraining in this data-constrained regime, we propose that explicitly modeling and inferring the latent thoughts that underlie the text generation process can significantly improve pretraining data efficiency. Intuitively, our approach views web text as the compressed final outcome of a verbose human thought process and that the latent thoughts contain important contextual knowledge and reasoning steps that are critical to data-efficient learning. We empirically demonstrate the effectiveness of our approach through data-constrained continued pretraining for math. We first show that synthetic data approaches to inferring latent thoughts significantly improve data efficiency, outperforming training on the same amount of raw data (5.7\% $\rightarrow$ 25.4\% on MATH). Furthermore, we demonstrate latent thought inference without a strong teacher, where an LM bootstraps its own performance by using an EM algorithm to iteratively improve the capability of the trained LM and the quality of thought-augmented pretraining data. We show that a 1B LM can bootstrap its performance across at least three iterations and significantly outperform baselines trained on raw data, with increasing gains from additional inference compute when performing the E-step. The gains from inference scaling and EM iterations suggest new opportunities for scaling data-constrained pretraining. 

**Abstract (ZH)**: 语言模型（LM）预训练的计算缩放已超越人类撰写的文本增长，引发了数据将成为LM缩放瓶颈的担忧。为在此数据受限的区间内继续进行预训练，我们提议明确建模和推断文本生成过程背后的潜在想法可以显著提高预训练数据的效率。我们认为，网页文本是冗长人类思维过程的压缩最终结果，潜在想法中包含对高效数据学习至关重要的重要上下文知识和推理步骤。我们通过数学领域的数据受限连续预训练实证示证了该方法的有效性。首先，我们展示了生成潜在想法的合成数据方法显著提高了数据效率，优于在相同量的原始数据上进行训练（5.7% → 25.4%）。此外，我们展示了无需强大教师的潜在想法推断，语言模型通过使用EM算法迭代提升训练模型能力和增强数据质量，在自身表现和带有想法增强的预训练数据质量上实现自我提升。我们证明了一个1B规模的语言模型可以在至少三个迭代中自我提升表现，并且在使用E步进行推断计算时相对于基于原始数据训练的基准模型有显著的性能提升。推断计算缩放和EM迭代的收益建议了在数据受限预训练领域新的缩放机会。 

---
# MC-LLaVA: Multi-Concept Personalized Vision-Language Model 

**Title (ZH)**: MC-LLaVA: 多概念个性化视觉语言模型 

**Authors**: Ruichuan An, Sihan Yang, Ming Lu, Renrui Zhang, Kai Zeng, Yulin Luo, Jiajun Cao, Hao Liang, Ying Chen, Qi She, Shanghang Zhang, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18854)  

**Abstract**: Current vision-language models (VLMs) show exceptional abilities across diverse tasks, such as visual question answering. To enhance user experience, recent studies investigate VLM personalization to understand user-provided concepts. However, they mainly focus on single-concept personalization, neglecting the existence and interplay of multiple concepts, which limits real-world applicability. This paper proposes the first multi-concept personalization paradigm, MC-LLaVA. Specifically, MC-LLaVA employs a multi-concept instruction tuning strategy, effectively integrating multiple concepts in a single training step. To reduce the costs related to joint training, we propose a personalized textual prompt that uses visual token information to initialize concept tokens. Additionally, we introduce a personalized visual prompt during inference, aggregating location confidence maps for enhanced recognition and grounding capabilities. To advance multi-concept personalization research, we further contribute a high-quality instruction tuning dataset. We carefully collect images with multiple characters and objects from movies and manually generate question-answer samples for multi-concept scenarios, featuring superior diversity. Comprehensive qualitative and quantitative experiments demonstrate that MC-LLaVA can achieve impressive multi-concept personalized responses, paving the way for VLMs to become better user-specific assistants. The code and dataset will be publicly available at $\href{this https URL}{this https URL}$. 

**Abstract (ZH)**: 多概念个性化多模态模型：MC-LLaVA的研究 

---
# Defeating Prompt Injections by Design 

**Title (ZH)**: 设计层面抵御提示注入攻击 

**Authors**: Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, Florian Tramèr  

**Link**: [PDF](https://arxiv.org/pdf/2503.18813)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in agentic systems that interact with an external environment. However, LLM agents are vulnerable to prompt injection attacks when handling untrusted data. In this paper we propose CaMeL, a robust defense that creates a protective system layer around the LLM, securing it even when underlying models may be susceptible to attacks. To operate, CaMeL explicitly extracts the control and data flows from the (trusted) query; therefore, the untrusted data retrieved by the LLM can never impact the program flow. To further improve security, CaMeL relies on a notion of a capability to prevent the exfiltration of private data over unauthorized data flows. We demonstrate effectiveness of CaMeL by solving $67\%$ of tasks with provable security in AgentDojo [NeurIPS 2024], a recent agentic security benchmark. 

**Abstract (ZH)**: Large Language Models (LLMs)在代理系统中的防护：CaMeL方法 

---
# REALM: A Dataset of Real-World LLM Use Cases 

**Title (ZH)**: REALM：现实世界大型语言模型应用场景数据集 

**Authors**: Jingwen Cheng, Kshitish Ghate, Wenyue Hua, William Yang Wang, Hong Shen, Fei Fang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18792)  

**Abstract**: Large Language Models, such as the GPT series, have driven significant industrial applications, leading to economic and societal transformations. However, a comprehensive understanding of their real-world applications remains limited. To address this, we introduce REALM, a dataset of over 94,000 LLM use cases collected from Reddit and news articles. REALM captures two key dimensions: the diverse applications of LLMs and the demographics of their users. It categorizes LLM applications and explores how users' occupations relate to the types of applications they use. By integrating real-world data, REALM offers insights into LLM adoption across different domains, providing a foundation for future research on their evolving societal roles. A dedicated dashboard this https URL presents the data. 

**Abstract (ZH)**: 大型语言模型，如GPT系列，推动了重要的工业应用，带来了经济和社会的变革。然而，对其实际应用的全面理解仍然有限。为了解决这一问题，我们介绍了REALM数据集，该数据集包含超过94,000个从Reddit和新闻文章中收集的大型语言模型使用案例。REALM捕捉了两个关键维度：大型语言模型的多样化应用和其用户的 demographic 属性。它对大型语言模型的应用进行了分类，并探讨了用户的职业与他们使用的应用类型之间的关系。通过整合实际数据，REALM提供了不同领域大型语言模型采用情况的见解，为未来研究其不断演变的社会角色奠定了基础。详细数据可通过此网址 https:// 进行查看。 

---
# BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache 

**Title (ZH)**: BitDecoding: 解锁用于长上下文LLMs解码的低比特KV缓存的张量核心 

**Authors**: Dayou Du, Shijie Cao, Jianyi Cheng, Ting Cao, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18773)  

**Abstract**: The growing adoption of long-context Large Language Models (LLMs) has introduced significant memory and computational challenges in autoregressive decoding due to the expanding Key-Value (KV) cache. KV cache quantization has emerged as a promising solution, with prior work showing that 4-bit or even 2-bit quantization can maintain model accuracy while reducing memory costs. However, despite these benefits, preliminary implementations for the low-bit KV cache struggle to deliver the expected speedup due to quantization and dequantization overheads and the lack of Tensor Cores utilization. In this work, we propose BitDecoding, a GPU-optimized framework that unlocks Tensor Cores for efficient decoding with low-bit KV cache. Efficiently leveraging Tensor Cores for low-bit KV cache is challenging due to the dynamic nature of KV cache generation at each decoding step. BitDecoding addresses these challenges with a Tensor Cores-Centric BitFusion Scheme that ensures data layout compatibility to enable high utilization of Tensor Cores. Additionally, BitDecoding incorporates a warp-efficient parallel decoding kernel and a fine-grained asynchronous pipeline, minimizing dequantization overhead and improving computational efficiency. Experiments show that BitDecoding achieves up to 7.5x speedup on RTX 4090, 4.8x on A100, and 8.9x on H100, compared to FP16 FlashDecoding-v2. It also outperforms the state-of-the-art low-bit KV cache implementation (QServe) by up to 4.3x. On LLaMA-3.1-8B with a 128K sequence length, BitDecoding reduces single-batch decoding latency by 3x, demonstrating its effectiveness in long-context generation scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 基于GPU的低比特Key-Value缓存高效解码框架BitDecoding 

---
# Commander-GPT: Fully Unleashing the Sarcasm Detection Capability of Multi-Modal Large Language Models 

**Title (ZH)**: Commander-GPT: 全面释放多模态大语言模型的讽刺检测能力 

**Authors**: Yazhou Zhang, Chunwang Zou, Bo Wang, Jing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18681)  

**Abstract**: Sarcasm detection, as a crucial research direction in the field of Natural Language Processing (NLP), has attracted widespread attention. Traditional sarcasm detection tasks have typically focused on single-modal approaches (e.g., text), but due to the implicit and subtle nature of sarcasm, such methods often fail to yield satisfactory results. In recent years, researchers have shifted the focus of sarcasm detection to multi-modal approaches. However, effectively leveraging multi-modal information to accurately identify sarcastic content remains a challenge that warrants further exploration. Leveraging the powerful integrated processing capabilities of Multi-Modal Large Language Models (MLLMs) for various information sources, we propose an innovative multi-modal Commander-GPT framework. Inspired by military strategy, we first decompose the sarcasm detection task into six distinct sub-tasks. A central commander (decision-maker) then assigns the best-suited large language model to address each specific sub-task. Ultimately, the detection results from each model are aggregated to identify sarcasm. We conducted extensive experiments on MMSD and MMSD 2.0, utilizing four multi-modal large language models and six prompting strategies. Our experiments demonstrate that our approach achieves state-of-the-art performance, with a 19.3% improvement in F1 score, without necessitating fine-tuning or ground-truth rationales. 

**Abstract (ZH)**: 多模态sarcastic内容检测：基于多模态大型语言模型的Commander-GPT框架 

---
# ClinText-SP and RigoBERTa Clinical: a new set of open resources for Spanish Clinical NLP 

**Title (ZH)**: ClinText-SP和RigoBERTa Clinical：一套新的西班牙语临床NLP开放资源 

**Authors**: Guillem García Subies, Álvaro Barbero Jiménez, Paloma Martínez Fernández  

**Link**: [PDF](https://arxiv.org/pdf/2503.18594)  

**Abstract**: We present a novel contribution to Spanish clinical natural language processing by introducing the largest publicly available clinical corpus, ClinText-SP, along with a state-of-the-art clinical encoder language model, RigoBERTa Clinical. Our corpus was meticulously curated from diverse open sources, including clinical cases from medical journals and annotated corpora from shared tasks, providing a rich and diverse dataset that was previously difficult to access. RigoBERTa Clinical, developed through domain-adaptive pretraining on this comprehensive dataset, significantly outperforms existing models on multiple clinical NLP benchmarks. By publicly releasing both the dataset and the model, we aim to empower the research community with robust resources that can drive further advancements in clinical NLP and ultimately contribute to improved healthcare applications. 

**Abstract (ZH)**: 我们通过引入最大的公开临床语料库ClinText-SP以及最先进的临床编码语言模型RigoBERTa Clinical，为西班牙临床自然语言处理领域做出了 novel 贡献。我们的语料库精心从各种开放源中筛选，包括医学期刊的临床病例和共享任务的标注语料库，提供了一个丰富多样的数据集，此前难以获取。RigoBERTa Clinical通过在这一全面数据集上进行领域适应预训练，多项临床NLP基准测试中显著优于现有模型。通过公开发布数据集和模型，我们旨在为研究社区提供强大的资源，推动临床NLP的进一步发展，最终为改善健康护理应用做出贡献。 

---
# Distil-xLSTM: Learning Attention Mechanisms through Recurrent Structures 

**Title (ZH)**: Distil-xLSTM: 通过递归结构学习注意力机制 

**Authors**: Abdoul Majid O. Thiombiano, Brahim Hnich, Ali Ben Mrad, Mohamed Wiem Mkaouer  

**Link**: [PDF](https://arxiv.org/pdf/2503.18565)  

**Abstract**: The current era of Natural Language Processing (NLP) is dominated by Transformer models. However, novel architectures relying on recurrent mechanisms, such as xLSTM and Mamba, have been proposed as alternatives to attention-based models. Although computation is done differently than with the attention mechanism mechanism, these recurrent models yield good results and sometimes even outperform state-of-the-art attention-based models. In this work, we propose Distil-xLSTM, an xLSTM-based Small Language Model (SLM) trained by distilling knowledge from a Large Language Model (LLM) that shows promising results while being compute and scale efficient. Our Distil-xLSTM focuses on approximating a transformer-based model attention parametrization using its recurrent sequence mixing components and shows good results with minimal training. 

**Abstract (ZH)**: 当前自然语言处理（NLP）时代被变压器模型主导，但基于递归机制的新型架构，如xLSTM和Mamba，已被提议作为基于注意力机制模型的替代方案。虽然计算方式与注意力机制不同，这些递归模型仍然取得了良好的效果，有时甚至超过最先进的基于注意力机制的模型。在本文中，我们提出了一种Distil-xLSTM，这是一种由大型语言模型（LLM）提炼出知识的小型语言模型（SLM），在保持计算和规模效率的同时展现了令人鼓舞的结果。我们的Distil-xLSTM专注于使用其递归序列混合组件来逼近基于变压器的模型的注意力参数化，并在少量训练下取得了良好的效果。 

---
# Self-Reported Confidence of Large Language Models in Gastroenterology: Analysis of Commercial, Open-Source, and Quantized Models 

**Title (ZH)**: 大型语言模型在胃肠病学领域的自我报告信心分析：商业、开源及量化模型的研究 

**Authors**: Nariman Naderi, Seyed Amir Ahmad Safavi-Naini, Thomas Savage, Zahra Atf, Peter Lewis, Girish Nadkarni, Ali Soroush  

**Link**: [PDF](https://arxiv.org/pdf/2503.18562)  

**Abstract**: This study evaluated self-reported response certainty across several large language models (GPT, Claude, Llama, Phi, Mistral, Gemini, Gemma, and Qwen) using 300 gastroenterology board-style questions. The highest-performing models (GPT-o1 preview, GPT-4o, and Claude-3.5-Sonnet) achieved Brier scores of 0.15-0.2 and AUROC of 0.6. Although newer models demonstrated improved performance, all exhibited a consistent tendency towards overconfidence. Uncertainty estimation presents a significant challenge to the safe use of LLMs in healthcare. Keywords: Large Language Models; Confidence Elicitation; Artificial Intelligence; Gastroenterology; Uncertainty Quantification 

**Abstract (ZH)**: 本研究使用300个胃肠病学板格式问题，评估了多个大型语言模型（GPT、Claude、Llama、Phi、Mistral、Gemini、Gemma和Qwen）自我报告的置信度反应。性能最佳的模型（GPT-o1 预览、GPT-4o 和 Claude-3.5-Sonnet）获得了0.15-0.2的布赖耳评分和0.6的AUROC。尽管 newer 模型展示了改进的性能，但所有模型都表现出一致性过度自信的倾向。不确定性估计是安全在医疗保健中使用LLMs的重要挑战。关键词：大型语言模型；置信度引出；人工智能；胃肠病学；不确定性量化。 

---
# SciClaims: An End-to-End Generative System for Biomedical Claim Analysis 

**Title (ZH)**: SciClaims: 一端到一端的生成系统用于生物医学断言分析 

**Authors**: Raúl Ortega, José Manuel Gómez-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2503.18526)  

**Abstract**: Validating key claims in scientific literature, particularly in biomedical research, is essential for ensuring accuracy and advancing knowledge. This process is critical in sectors like the pharmaceutical industry, where rapid scientific progress requires automation and deep domain expertise. However, current solutions have significant limitations. They lack end-to-end pipelines encompassing all claim extraction, evidence retrieval, and verification steps; rely on complex NLP and information retrieval pipelines prone to multiple failure points; and often fail to provide clear, user-friendly justifications for claim verification outcomes. To address these challenges, we introduce SciClaims, an advanced system powered by state-of-the-art large language models (LLMs) that seamlessly integrates the entire scientific claim analysis process. SciClaims outperforms previous approaches in both claim extraction and verification without requiring additional fine-tuning, setting a new benchmark for automated scientific claim analysis. 

**Abstract (ZH)**: 验证科学文献特别是生物医药研究中的关键声明对于确保准确性和促进知识发展至关重要。这一过程对于制药等行业而言尤其关键，因为该行业需要快速的科学进步和自动化以及深厚的专业知识。然而，当前的解决方案存在显著的局限性。它们缺乏涵盖所有声明提取、证据检索和验证步骤的端到端管道；依赖于复杂自然语言处理和信息检索管道，存在多个失败点；并且通常无法为声明验证结果提供清晰且用户友好的解释。为应对这些挑战，我们引入了SciClaims，这是一种基于先进大语言模型（LLMs）的高级系统，能够无缝整合整个科学声明分析过程。SciClaims在声明提取和验证方面均优于先前的方法，无需额外的微调，并为自动化科学声明分析设立了新基准。 

---
# Safeguarding Mobile GUI Agent via Logic-based Action Verification 

**Title (ZH)**: 基于逻辑的行动验证保障移动GUI代理安全 

**Authors**: Jungjae Lee, Dongjae Lee, Chihun Choi, Youngmin Im, Jaeyoung Wi, Kihong Heo, Sangeun Oh, Sunjae Lee, Insik Shin  

**Link**: [PDF](https://arxiv.org/pdf/2503.18492)  

**Abstract**: Large Foundation Models (LFMs) have unlocked new possibilities in human-computer interaction, particularly with the rise of mobile Graphical User Interface (GUI) Agents capable of interpreting GUIs. These agents promise to revolutionize mobile computing by allowing users to automate complex mobile tasks through simple natural language instructions. However, the inherent probabilistic nature of LFMs, coupled with the ambiguity and context-dependence of mobile tasks, makes LFM-based automation unreliable and prone to errors. To address this critical challenge, we introduce VeriSafe Agent (VSA): a formal verification system that serves as a logically grounded safeguard for Mobile GUI Agents. VSA is designed to deterministically ensure that an agent's actions strictly align with user intent before conducting an action. At its core, VSA introduces a novel autoformalization technique that translates natural language user instructions into a formally verifiable specification, expressed in our domain-specific language (DSL). This enables runtime, rule-based verification, allowing VSA to detect and prevent erroneous actions executing an action, either by providing corrective feedback or halting unsafe behavior. To the best of our knowledge, VSA is the first attempt to bring the rigor of formal verification to GUI agent. effectively bridging the gap between LFM-driven automation and formal software verification. We implement VSA using off-the-shelf LLM services (GPT-4o) and evaluate its performance on 300 user instructions across 18 widely used mobile apps. The results demonstrate that VSA achieves 94.3%-98.33% accuracy in verifying agent actions, representing a significant 20.4%-25.6% improvement over existing LLM-based verification methods, and consequently increases the GUI agent's task completion rate by 90%-130%. 

**Abstract (ZH)**: 大型基础模型（LFMs）通过移动图形用户界面（GUI）代理的兴起解锁了新的可能，这些代理能够解释GUI。这些代理有望通过简单的自然语言指令自动化复杂的移动任务，从而 revolutionize 移动计算。然而，LFMs 内在的概率性质，加上移动任务的模糊性和依赖上下文的特性，使得基于LFM的自动化不可靠且容易出错。为应对这一关键挑战，我们提出了VeriSafe Agent（VSA）：一种作为逻辑基础护盾的正式验证系统，用于移动GUI代理。VSA 设计旨在在执行操作前严格确保代理行为与用户意图的一致性。VSA 内核引入了一种新颖的自动形式化技术，将自然语言用户指令转换为可以在我们领域特定语言（DSL）中形式验证的规范。这使 VSA 能够在运行时、基于规则进行验证，在执行操作时检测并防止错误行为，通过提供纠正反馈或阻止不安全行为。据我们所知，VSA 是首次尝试将形式验证的严谨性带入 GUI 代理，有效地弥合了基于LFM 的自动化与形式软件验证之间的差距。我们使用现成的LLM 服务（GPT-4o）实现VSA，并在18个广泛使用的移动应用程序的300个用户指令上评估其性能。结果表明，VSA 在验证代理行为方面的准确性为94.3%-98.33%，比现有基于LLM的验证方法提高了20.4%-25.6%，从而将GUI代理的任务完成率提高了90%-130%。 

---
# Large Language Models powered Network Attack Detection: Architecture, Opportunities and Case Study 

**Title (ZH)**: 大型语言模型驱动的网络攻击检测：架构、机遇与案例研究 

**Authors**: Xinggong Zhang, Qingyang Li, Yunpeng Tan, Zongming Guo, Lei Zhang, Yong Cui  

**Link**: [PDF](https://arxiv.org/pdf/2503.18487)  

**Abstract**: Network attack detection is a pivotal technology to identify network anomaly and classify malicious traffic. Large Language Models (LLMs) are trained on a vast corpus of text, have amassed remarkable capabilities of context-understanding and commonsense knowledge. This has opened up a new door for network threat detection. Researchers have already initiated discussions regarding the application of LLMs on specific cyber-security tasks. Unfortunately, there is still a lack of comprehensive elaboration how to mine LLMs' potentials in network threat detections, as well as the opportunities and challenges. In this paper, we mainly focus on the classification of malicious traffic from the perspective of LLMs' capability. We present a holistic view of the architecture of LLM-powered network attack detection, including Pre-training, Fine-tuning, and Detection. Especially, by exploring the knowledge and capabilities of LLM, we identify three distinct roles LLM can act in network attack detection: \textit{Classifier, Encoder, and Predictor}. For each of them, the modeling paradigm, opportunities and challenges are elaborated. Finally, we present our design on LLM-powered DDoS detection as a case study. The proposed framework attains accurate detection on carpet bombing DDoS by exploiting LLMs' capabilities in contextual mining. The evaluation shows its efficacy, exhibiting a nearly $35$\% improvement compared to existing systems. 

**Abstract (ZH)**: 基于大型语言模型的网络攻击检测中的恶意流量分类 

---
# ModiGen: A Large Language Model-Based Workflow for Multi-Task Modelica Code Generation 

**Title (ZH)**: ModiGen：基于大语言模型的多任务Modelica代码生成工作流 

**Authors**: Jiahui Xiang, Tong Ye, Peiyu Liu, Yinan Zhang, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18460)  

**Abstract**: Modelica is a widely adopted language for simulating complex physical systems, yet effective model creation and optimization require substantial domain expertise. Although large language models (LLMs) have demonstrated promising capabilities in code generation, their application to modeling remains largely unexplored. To address this gap, we have developed benchmark datasets specifically designed to evaluate the performance of LLMs in generating Modelica component models and test cases. Our evaluation reveals substantial limitations in current LLMs, as the generated code often fails to simulate successfully. To overcome these challenges, we propose a specialized workflow that integrates supervised fine-tuning, graph retrieval-augmented generation, and feedback optimization to improve the accuracy and reliability of Modelica code generation. The evaluation results demonstrate significant performance gains: the maximum improvement in pass@1 reached 0.3349 for the component generation task and 0.2457 for the test case generation task. This research underscores the potential of LLMs to advance intelligent modeling tools and offers valuable insights for future developments in system modeling and engineering applications. 

**Abstract (ZH)**: Modelica是一个广泛采用的用于模拟复杂物理系统的语言，但有效的模型创建和优化需要大量的专业领域知识。尽管大规模语言模型（LLMs）展示了代码生成方面的有前景的能力，但它们在建模中的应用仍被广泛探索。为解决这一差距，我们开发了专门设计的基准数据集，以评估LLMs在生成Modelica组件模型和测试案例方面的性能。我们的评估结果显示了当前LLMs存在的重大限制，生成的代码往往无法成功模拟。为克服这些挑战，我们提出了一种专门的工作流，该工作流结合了监督微调、图检索增强生成和反馈优化，以提高Modelica代码生成的准确性和可靠性。评估结果表明取得了显著的性能提升：组件生成任务的最高改进率达到0.3349，测试案例生成任务的最高改进率达到0.2457。这项研究强调了LLMs在推动智能建模工具方面的潜力，并为系统建模和工程应用的未来发展提供了宝贵的见解。 

---
# Teaching LLMs for Step-Level Automatic Math Correction via Reinforcement Learning 

**Title (ZH)**: 基于强化学习的步骤级自动数学纠错LLM教学 

**Authors**: Junsong Li, Jie Zhou, Yutao Yang, Bihao Zhan, Qianjun Pan, Yuyang Ding, Qin Chen, Jiang Bo, Xin Lin, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2503.18432)  

**Abstract**: Automatic math correction aims to check students' solutions to mathematical problems via artificial intelligence technologies. Most existing studies focus on judging the final answer at the problem level, while they ignore detailed feedback on each step in a math problem-solving process, which requires abilities of semantic understanding and reasoning. In this paper, we propose a reinforcement learning (RL)-based method to boost large language model (LLM) for step-level automatic math correction, named StepAMC. Particularly, we convert the step-level automatic math correction within the text classification task into an RL problem to enhance the reasoning capabilities of LLMs. Then, we design a space-constrained policy network to improve the stability of RL. Then, we introduce a fine-grained reward network to convert the binary human feedback into a continuous value. We conduct extensive experiments over two benchmark datasets and the results show that our model outperforms the eleven strong baselines. 

**Abstract (ZH)**: 基于强化学习的大语言模型步骤级自动数学纠错方法：StepAMC 

---
# Manipulation and the AI Act: Large Language Model Chatbots and the Danger of Mirrors 

**Title (ZH)**: AI法案中的操控与镜像危险：大规模语言模型聊天机器人 

**Authors**: Joshua Krook  

**Link**: [PDF](https://arxiv.org/pdf/2503.18387)  

**Abstract**: Large Language Model chatbots are increasingly taking the form and visage of human beings, adapting human faces, names, voices, personalities, and quirks, including those of celebrities and well-known political figures. Personifying AI chatbots could foreseeably increase their trust with users. However, it could also make them more capable of manipulation, by creating the illusion of a close and intimate relationship with an artificial entity. The European Commission has finalized the AI Act, with the EU Parliament making amendments banning manipulative and deceptive AI systems that cause significant harm to users. Although the AI Act covers harms that accumulate over time, it is unlikely to prevent harms associated with prolonged discussions with AI chatbots. Specifically, a chatbot could reinforce a person's negative emotional state over weeks, months, or years through negative feedback loops, prolonged conversations, or harmful recommendations, contributing to a user's deteriorating mental health. 

**Abstract (ZH)**: 大型语言模型聊天机器人越来越具有人类的形象， adapting human faces, names, voices, personalities, and quirks, including those of celebrities and well-known political figures. 将AI聊天机器人拟人化可能增加用户对其的信任，但也可能使其更具操控性，通过营造与人造实体亲近和亲密的关系幻觉。欧盟委员会已最终确定了AI法案，欧洲议会对法案作出了修正，禁止造成用户重大伤害的具有操控性和欺骗性的AI系统。尽管AI法案涵盖了累积性伤害，但不太可能防止与AI聊天机器人长时间互动造成的伤害。具体而言，聊天机器人可以通过消极反馈循环、长时间对话或有害建议，逐步强化一个人的消极情绪状态，持续数周、数月甚至数年，从而损害用户的心理健康。 

---
# Maximum Redundancy Pruning: A Principle-Driven Layerwise Sparsity Allocation for LLMs 

**Title (ZH)**: 最大化冗余剪枝：一种基于原理的层wise稀疏性分配方法 

**Authors**: Chang Gao, Kang Zhao, Jianfei Chen, Liping Jing  

**Link**: [PDF](https://arxiv.org/pdf/2503.18377)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities, but their enormous size poses significant challenges for deployment in real-world applications. To address this issue, researchers have sought to apply network pruning techniques to LLMs. A critical challenge in pruning is allocation the sparsity for each layer. Recent sparsity allocation methods is often based on heuristics or search that can easily lead to suboptimal performance. In this paper, we conducted an extensive investigation into various LLMs and revealed three significant discoveries: (1) the layerwise pruning sensitivity (LPS) of LLMs is highly non-uniform, (2) the choice of pruning metric affects LPS, and (3) the performance of a sparse model is related to the uniformity of its layerwise redundancy level. Based on these observations, we propose that the layerwise sparsity of LLMs should adhere to three principles: \emph{non-uniformity}, \emph{pruning metric dependency}, and \emph{uniform layerwise redundancy level} in the pruned model. To this end, we proposed Maximum Redundancy Pruning (MRP), an iterative pruning algorithm that prunes in the most redundant layers (\emph{i.e.}, those with the highest non-outlier ratio) at each iteration. The achieved layerwise sparsity aligns with the outlined principles. We conducted extensive experiments on publicly available LLMs, including the LLaMA2 and OPT, across various benchmarks. Experimental results validate the effectiveness of MRP, demonstrating its superiority over previous methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的能力，但其巨大的规模给实际应用中的部署带来了重大挑战。为了解决这个问题，研究人员寻求应用网络剪枝技术到LLMs中。剪枝中的一个关键挑战是为每一层分配稀疏性。最近的稀疏性分配方法往往基于启发式或搜索，容易导致次优性能。在本文中，我们对各种LLMs进行了广泛的研究，发现了三个重要发现：（1）LLMs的层间剪枝敏感性（LPS）高度不均匀，（2）剪枝度量的选择影响LPS，（3）稀疏模型的性能与其层间冗余水平的均匀性相关。基于这些观察，我们提出LLMs的层间稀疏性应遵循三个原则：非均匀性、剪枝度量依赖性和剪枝后模型中的均匀层间冗余水平。为此，我们提出了最大冗余剪枝（MRP），这是一种迭代剪枝算法，在每次迭代中对最冗余的层（即，非离群值比例最高的层）进行剪枝。实现的层间稀疏性与提出的原理一致。我们在包括LLaMA2和OPT在内的多个公开可用的LLM上，通过各种基准实验对其进行了广泛研究。实验结果验证了MRP的有效性，展示了其在先前方法中的优越性。 

---
# DeepFund: Will LLM be Professional at Fund Investment? A Live Arena Perspective 

**Title (ZH)**: DeepFund: 语言模型会在基金投资方面专业化吗？一个实时竞技场视角 

**Authors**: Changlun Li, Yao Shi, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18313)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities across various domains, but their effectiveness in financial decision making, particularly in fund investment, remains inadequately evaluated. Current benchmarks primarily assess LLMs understanding of financial documents rather than their ability to manage assets or analyze trading opportunities in dynamic market conditions. A critical limitation in existing evaluation methodologies is the backtesting approach, which suffers from information leakage when LLMs are evaluated on historical data they may have encountered during pretraining. This paper introduces DeepFund, a comprehensive platform for evaluating LLM based trading strategies in a simulated live environment. Our approach implements a multi agent framework where LLMs serve as both analysts and managers, creating a realistic simulation of investment decision making. The platform employs a forward testing methodology that mitigates information leakage by evaluating models on market data released after their training cutoff dates. We provide a web interface that visualizes model performance across different market conditions and investment parameters, enabling detailed comparative analysis. Through DeepFund, we aim to provide a more accurate and fair assessment of LLMs capabilities in fund investment, offering insights into their potential real world applications in financial markets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现了强大的能力，但在金融决策领域，尤其是基金投资中的效果仍然缺乏充分评估。当前的基准主要评估LLMs对金融文档的理解能力，而非其管理资产或在动态市场条件下分析交易机会的能力。现有评估方法的一个关键局限是回测方法，在利用模型在预训练期间可能遇到的历史数据评估模型时存在信息泄露问题。本文介绍了一个全面的DeepFund平台，用于在模拟实时环境中评估基于LLM的交易策略。我们的方法采用多agent框架，其中LLMsboth作为分析师和管理者，创造了一个真实的投资决策模拟环境。该平台采用正向测试方法，通过在模型训练截止日期之后发布的市场数据评估模型，从而减少信息泄露。我们提供了一个网络界面，可视化不同市场条件和投资参数下的模型表现，支持详细的对比分析。通过DeepFund，我们旨在提供对LLMs在基金投资中的能力更准确和公正的评估，并揭示其在金融市场中潜在的真实世界应用。 

---
# How to Capture and Study Conversations Between Research Participants and ChatGPT: GPT for Researchers (g4r.org) 

**Title (ZH)**: 如何捕获和研究研究参与者与ChatGPT之间的对话：GPT for Researchers (g4r.org) 

**Authors**: Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2503.18303)  

**Abstract**: As large language models (LLMs) like ChatGPT become increasingly integrated into our everyday lives--from customer service and education to creative work and personal productivity--understanding how people interact with these AI systems has become a pressing issue. Despite the widespread use of LLMs, researchers lack standardized tools for systematically studying people's interactions with LLMs. To address this issue, we introduce GPT for Researchers (G4R), or this http URL, a free website that researchers can use to easily create and integrate a GPT Interface into their studies. At this http URL, researchers can (1) enable their study participants to interact with GPT (such as ChatGPT), (2) customize GPT Interfaces to guide participants' interactions with GPT (e.g., set constraints on topics or adjust GPT's tone or response style), and (3) capture participants' interactions with GPT by downloading data on messages exchanged between participants and GPT. By facilitating study participants' interactions with GPT and providing detailed data on these interactions, G4R can support research on topics such as consumer interactions with AI agents or LLMs, AI-assisted decision-making, and linguistic patterns in human-AI communication. With this goal in mind, we provide a step-by-step guide to using G4R at this http URL. 

**Abstract (ZH)**: 随着像ChatGPT这样的大规模语言模型（LLMs）在客户服务、教育、创造性工作和个人生产力等领域中的日益普及，理解人们如何与这些AI系统交互已成为一个迫切的问题。尽管LLMs得到了广泛应用，研究人员仍缺乏标准化工具来系统研究人们与LLMs的交互。为解决这一问题，我们介绍了GPT for Researchers（G4R），或访问此网址，这是一个免费的网站，研究人员可以使用它来轻松创建并整合GPT界面到他们的研究中。在该网址上，研究人员可以（1）使研究参与者能够与GPT（如ChatGPT）交互，（2）自定义GPT界面以引导参与者与GPT的交互（例如，设置话题限制或调整GPT的语气或响应风格），以及（3）通过下载参与者与GPT之间交换的消息数据来捕获参与者与GPT的交互。通过促进研究参与者与GPT的交互，并提供这些交互的详细数据，G4R可以支持关于消费者与AI代理或LLMs的互动、AI辅助决策以及人类与AI通信中的语言模式等方面的研究。为了实现这一目标，我们在该网址上提供了使用G4R的逐步指南。 

---
# ShED-HD: A Shannon Entropy Distribution Framework for Lightweight Hallucination Detection on Edge Devices 

**Title (ZH)**: ShED-HD：边缘设备上轻量级幻觉检测的香农熵分布框架 

**Authors**: Aneesh Vathul, Daniel Lee, Sheryl Chen, Arthi Tasmia  

**Link**: [PDF](https://arxiv.org/pdf/2503.18242)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities on a broad array of NLP tasks, but their tendency to produce hallucinations$\unicode{x2013}$plausible-sounding but factually incorrect content$\unicode{x2013}$poses severe challenges in high-stakes domains. Existing hallucination detection methods either bear the computational cost of multiple inference passes or sacrifice accuracy for efficiency with single-pass approaches, neither of which is ideal in resource-constrained environments such as edge devices. We propose the Shannon Entropy Distribution Hallucination Detector (ShED-HD), a novel hallucination detection framework that bridges this gap by classifying sequence-level entropy patterns using a lightweight BiLSTM architecture with single-headed attention. In contrast to prior approaches, ShED-HD efficiently detects distinctive uncertainty patterns across entire output sequences, preserving contextual awareness. Through in-depth evaluation on three datasets (BioASQ, TriviaQA, and Jeopardy Questions), we show that ShED-HD significantly outperforms other computationally efficient approaches in the out-of-distribution setting, while achieving comparable performance in the in-distribution setting. ShED-HD facilitates hallucination detection that is low-cost, accurate, and generalizable, improving the credibility of content generated by LLMs in resource-constrained environments where trustworthy AI functionality is crucial. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛的语言处理任务上展现了令人印象深刻的能力，但它们生成幻觉（听起来合理但实际上错误的内容）的倾向在高风险领域提出了重大挑战。现有的幻觉检测方法要么承受多轮推理的计算成本，要么在单轮推理中牺牲准确性以提高效率，这两种方法在资源受限的环境中都不理想。我们提出了一种新的幻觉检测框架Shannon熵分布幻觉检测器（ShED-HD），该框架通过轻量级双向长短期记忆（BiLSTM）结构和单头注意机制来分类序列级别的熵模式，以弥合这一差距。与以往方法不同，ShED-HD能够高效地检测整个输出序列中的独特不确定性模式，同时保持上下文意识。通过在三个数据集（BioASQ、TriviaQA和Jeopardy Questions）上的深入评估，我们展示了ShED-HD在分布外设置中显著优于其他计算高效的approaches，在分布内设置中达到相当的性能。ShED-HD使得在资源受限的环境中，幻觉检测变得低成本、准确且通用，从而提高LLMs生成内容的可信度。 

---
# Mitigating Reward Over-Optimization in RLHF via Behavior-Supported Regularization 

**Title (ZH)**: 通过行为支持正则化减轻RLHF中的奖励过度优化 

**Authors**: Juntao Dai, Taiye Chen, Yaodong Yang, Qian Zheng, Gang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2503.18130)  

**Abstract**: Reinforcement learning from human feedback (RLHF) is an effective method for aligning large language models (LLMs) with human values. However, reward over-optimization remains an open challenge leading to discrepancies between the performance of LLMs under the reward model and the true human objectives. A primary contributor to reward over-optimization is the extrapolation error that arises when the reward model evaluates out-of-distribution (OOD) responses. However, current methods still fail to prevent the increasing frequency of OOD response generation during the reinforcement learning (RL) process and are not effective at handling extrapolation errors from OOD responses. In this work, we propose the Behavior-Supported Policy Optimization (BSPO) method to mitigate the reward over-optimization issue. Specifically, we define behavior policy as the next token distribution of the reward training dataset to model the in-distribution (ID) region of the reward model. Building on this, we introduce the behavior-supported Bellman operator to regularize the value function, penalizing all OOD values without impacting the ID ones. Consequently, BSPO reduces the generation of OOD responses during the RL process, thereby avoiding overestimation caused by the reward model's extrapolation errors. Theoretically, we prove that BSPO guarantees a monotonic improvement of the supported policy until convergence to the optimal behavior-supported policy. Empirical results from extensive experiments show that BSPO outperforms baselines in preventing reward over-optimization due to OOD evaluation and finding the optimal ID policy. 

**Abstract (ZH)**: 基于行为支持的策略优化方法（BSPO）：对抗奖励过优化 

---
# GeoBenchX: Benchmarking LLMs for Multistep Geospatial Tasks 

**Title (ZH)**: GeoBenchX: 评估LLM在多步地理空间任务中的性能 

**Authors**: Varvara Krechetova, Denis Kochedykov  

**Link**: [PDF](https://arxiv.org/pdf/2503.18129)  

**Abstract**: In this paper, we establish a benchmark for evaluating large language models (LLMs) on multi-step geospatial tasks relevant to commercial GIS practitioners. We assess seven leading commercial LLMs (Sonnet 3.5 and 3.7, Haiku 3.5, Gemini 2.0, GPT-4o, GPT-4o mini, and o3-mini) using a simple tool-calling agent equipped with 23 geospatial functions. Our benchmark comprises tasks across four categories of increasing complexity, with both solvable and intentionally unsolvable tasks to test hallucination rejection. We develop an LLM-as-Judge evaluation framework to compare agent solutions against reference implementations. Results show Sonnet 3.5 and GPT-4o achieve the best overall performance, with Claude models excelling on solvable tasks while OpenAI models better identify unsolvable scenarios. We observe significant differences in token usage, with Anthropic models consuming substantially more tokens than competitors. Common errors include misunderstanding geometrical relationships, relying on outdated knowledge, and inefficient data manipulation. The resulting benchmark set, evaluation framework, and data generation pipeline are released as open-source resources, providing one more standardized method for ongoing evaluation of LLMs for GeoAI. 

**Abstract (ZH)**: 本文建立了评价大型语言模型（LLMs）在商业GIS practitioners相关的多步骤地理空间任务中的基准。我们使用一个简单的工具调用代理，配备23个地理空间功能，评估了七种领先的商用LLM（Sonnet 3.5和3.7、Haiku 3.5、Gemini 2.0、GPT-4o、GPT-4o mini和o3-mini）。我们的基准涵盖了四个复杂度逐步增加的任务类别，包括可解任务和故意设置的不可解任务，以测试幻觉拒绝能力。我们开发了一种LLM-as-Judge评估框架，将代理解决方案与参考实现进行对比。结果表明，Sonnet 3.5和GPT-4o在整体性能上最佳，Claude模型在可解任务上表现优异，而OpenAI模型在识别不可解场景方面表现更佳。我们观察到在token使用方面存在显著差异，Anthropic模型消耗的tokens远多于竞争对手。常见的错误包括对几何关系理解错误、依赖过时信息以及无效的数据操作。基准集、评估框架以及数据生成管道均被作为开源资源发布，提供了一种新的标准化方法，用于持续评价用于GeoAI的LLM。 

---
# On the effectiveness of LLMs for automatic grading of open-ended questions in Spanish 

**Title (ZH)**: 关于大规模语言模型在自动批改西班牙语开放型问题中的有效性研究 

**Authors**: Germán Capdehourat, Isabel Amigo, Brian Lorenzo, Joaquín Trigo  

**Link**: [PDF](https://arxiv.org/pdf/2503.18072)  

**Abstract**: Grading is a time-consuming and laborious task that educators must face. It is an important task since it provides feedback signals to learners, and it has been demonstrated that timely feedback improves the learning process. In recent years, the irruption of LLMs has shed light on the effectiveness of automatic grading. In this paper, we explore the performance of different LLMs and prompting techniques in automatically grading short-text answers to open-ended questions. Unlike most of the literature, our study focuses on a use case where the questions, answers, and prompts are all in Spanish. Experimental results comparing automatic scores to those of human-expert evaluators show good outcomes in terms of accuracy, precision and consistency for advanced LLMs, both open and proprietary. Results are notably sensitive to prompt styles, suggesting biases toward certain words or content in the prompt. However, the best combinations of models and prompt strategies, consistently surpasses an accuracy of 95% in a three-level grading task, which even rises up to more than 98% when the it is simplified to a binary right or wrong rating problem, which demonstrates the potential that LLMs have to implement this type of automation in education applications. 

**Abstract (ZH)**: 自动批改西班牙语开放性问题短文本答案：不同LLM和提示技术的性能探索 

---
# Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning 

**Title (ZH)**: Vision-R1：通过视觉引导 reinforcement 学习进化大规模视觉-语言模型中的无监督对齐 

**Authors**: Yufei Zhan, Yousong Zhu, Shurong Zheng, Hongyin Zhao, Fan Yang, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.18013)  

**Abstract**: Large Vision-Language Models (LVLMs) typically follow a two-stage training paradigm-pretraining and supervised fine-tuning. Recently, preference optimization, derived from the language domain, has emerged as an effective post-training reinforcement strategy to enhance capabilities of LVLMs. However, constructing high-quality human-annotated preference data and developing robust reward models to mimic these preferences are both costly and challenging. Motivated by this observation, we propose Vision-R1, a novel vision-guided R1-like reinforcement learning algorithm for LVLMs that rewards models with definitive vision feedback. It only leverages curated instruction data, eliminating the need for specialized reward models and handcrafted preference datasets. We incorporate a criterion-driven reward function that further integrates multi-dimensional feedback to evaluate model completions comprehensively based on the vision task logic. Furthermore, we introduce a progressive rule refinement strategy that dynamically adjusts the reward criteria during training, enabling continuous model improvement and mitigating reward hacking. Extensive experiments on both in-distribution and out-of-distribution benchmarks demonstrate that fine-tuning the 7B LVLMs with Vision-R1 achieves consistent performance gains, with even up to 50% improvement and surpassing the state-of-the-art 10x size model. 

**Abstract (ZH)**: Large Vision-Language Models (LVLMs) typically follow a two-stage training paradigm-pretraining and supervised fine-tuning. Recently, preference optimization, derived from the language domain, has emerged as an effective post-training reinforcement strategy to enhance capabilities of LVLMs. However, constructing high-quality human-annotated preference data and developing robust reward models to mimic these preferences are both costly and challenging. Motivated by this observation, we propose Vision-R1, a novel vision-guided R1-like reinforcement learning algorithm for LVLMs that rewards models with definitive vision feedback. It only leverages curated instruction data, eliminating the need for specialized reward models and handcrafted preference datasets. We incorporate a criterion-driven reward function that further integrates multi-dimensional feedback to evaluate model completions comprehensively based on the vision task logic. Furthermore, we introduce a progressive rule refinement strategy that dynamically adjusts the reward criteria during training, enabling continuous model improvement and mitigating reward hacking. Extensive experiments on both in-distribution and out-of-distribution benchmarks demonstrate that fine-tuning the 7B LVLMs with Vision-R1 achieves consistent performance gains, with even up to 50% improvement and surpassing the state-of-the-art 10x size model. 

---
# Neuromorphic Principles for Efficient Large Language Models on Intel Loihi 2 

**Title (ZH)**: Intel Loihi 2 上高效大型语言模型的神经形态原理 

**Authors**: Steven Abreu, Sumit Bam Shrestha, Rui-Jie Zhu, Jason Eshraghian  

**Link**: [PDF](https://arxiv.org/pdf/2503.18002)  

**Abstract**: Large language models (LLMs) deliver impressive performance but require large amounts of energy. In this work, we present a MatMul-free LLM architecture adapted for Intel's neuromorphic processor, Loihi 2. Our approach leverages Loihi 2's support for low-precision, event-driven computation and stateful processing. Our hardware-aware quantized model on GPU demonstrates that a 370M parameter MatMul-free model can be quantized with no accuracy loss. Based on preliminary results, we report up to 3x higher throughput with 2x less energy, compared to transformer-based LLMs on an edge GPU, with significantly better scaling. Further hardware optimizations will increase throughput and decrease energy consumption. These results show the potential of neuromorphic hardware for efficient inference and pave the way for efficient reasoning models capable of generating complex, long-form text rapidly and cost-effectively. 

**Abstract (ZH)**: Large Language Models (LLMs)展现出卓越的性能但需要大量能量。本文提出了一种适用于Intel neuromorphic处理器Loihi 2的MatMul-free LLM架构。我们的方法利用了Loihi 2对低精度、事件驱动计算和状态处理的支持。基于GPU上的硬件感知量化模型表明，一个370M参数的MatMul-free模型可以实现无精度损失的量化。初步结果显示，与边缘GPU上的基于Transformer的LLMs相比，我们的架构在吞吐量上最高可提高3倍，在能耗上减少50%，并且具有显著更好的扩展性。进一步的硬件优化将增加吞吐量并降低能耗。这些结果展示了神经形态硬件在高效推断中的潜力，并为能够快速、经济高效地生成复杂、长篇文本的有效推理模型铺平了道路。 

---
# Instructing the Architecture Search for Spatial-temporal Sequence Forecasting with LLM 

**Title (ZH)**: 基于LLM的时空序列 Forecasting 架构搜索指导 

**Authors**: Xin Xue, Haoyi Zhou, Tianyu Chen, Shuai Zhang, Yizhou Long, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.17994)  

**Abstract**: Spatial-temporal sequence forecasting (STSF) is a long-standing research problem with widespread real-world applications. Neural architecture search (NAS), which automates the neural network design, has been shown effective in tackling the STSF problem. However, the existing NAS methods for STSF focus on generating architectures in a time-consuming data-driven fashion, which heavily limits their ability to use background knowledge and explore the complicated search trajectory. Large language models (LLMs) have shown remarkable ability in decision-making with comprehensive internal world knowledge, but how it could benefit NAS for STSF remains unexplored. In this paper, we propose a novel NAS method for STSF based on LLM. Instead of directly generate architectures with LLM, We inspire the LLM's capability with a multi-level enhancement mechanism. Specifically, on the step-level, we decompose the generation task into decision steps with powerful prompt engineering and inspire LLM to serve as instructor for architecture search based on its internal knowledge. On the instance-level, we utilize a one-step tuning framework to quickly evaluate the architecture instance and a memory bank to cumulate knowledge to improve LLM's search ability. On the task-level, we propose a two-stage architecture search, balancing the exploration stage and optimization stage, to reduce the possibility of being trapped in local optima. Extensive experimental results demonstrate that our method can achieve competitive effectiveness with superior efficiency against existing NAS methods for STSF. 

**Abstract (ZH)**: 基于大规模语言模型的时空序列预测神经架构搜索方法 

---
# Metaphor-based Jailbreaking Attacks on Text-to-Image Models 

**Title (ZH)**: 基于隐喻的文本生成图像模型越狱攻击 

**Authors**: Chenyu Zhang, Yiwen Ma, Lanjun Wang, Wenhui Li, Yi Tu, An-An Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17987)  

**Abstract**: To mitigate misuse, text-to-image~(T2I) models commonly incorporate safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods use LLMs to generate adversarial prompts that effectively bypass safety filters while generating sensitive images, revealing the safety vulnerabilities within the T2I model. However, existing LLM-based attack methods lack explicit guidance, relying on substantial queries to achieve a successful attack, which limits their practicality in real-world scenarios. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to balance the attack effectiveness and query efficiency by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance the attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Experiments demonstrate that MJA achieves better attack effectiveness while requiring fewer queries compared to baseline methods. Moreover, our adversarial prompts exhibit strong transferability across various open-source and commercial T2I models. \textcolor{red}{This paper includes model-generated content that may contain offensive or distressing material.} 

**Abstract (ZH)**: 基于隐喻的 Jailbreaking 攻击方法（MJA）：一种平衡攻击效果与查询效率的方法 

---
# Understanding the Effects of RLHF on the Quality and Detectability of LLM-Generated Texts 

**Title (ZH)**: 理解RLHF对LLM生成文本的质量和可检测性的影响 

**Authors**: Beining Xu, Arkaitz Zubiaga  

**Link**: [PDF](https://arxiv.org/pdf/2503.17965)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance on a range of downstream NLP tasks by generating text that closely resembles human writing. However, the ease of achieving this similarity raises concerns from potential malicious uses at scale by bad actors, as LLM-generated text becomes increasingly difficult to discern from human text. Although detection methods have been developed to address this issue, bad actors can further manipulate LLM-generated texts to make them less detectable. In this work, we study how further editing texts with Reinforcement Learning from Human Feedback (RLHF), which aligns model outputs with human preferences, affects (a) the quality of generated texts for two tasks, and (b) the performance of LLM-generated text detectors, looking at both training-based and zero-shot detection methods. Although RLHF improves the quality of LLM-generated texts, we find that it also tends to produce more detectable, lengthy, and repetitive outputs. Additionally, we observe that training-based detectors are vulnerable to short texts and to texts that incorporate code, whereas zero-shot detectors exhibit greater robustness. 

**Abstract (ZH)**: 大型语言模型（LLMs）在一系列下游自然语言处理任务中通过生成与人类写作高度相似的文本展现了出色性能。然而，这种相似性的轻易获取引发了恶意行为者在大规模应用中潜在的恶意使用担忧，因为由LLM生成的文本越来越难以与人类文本区分。尽管已经开发了检测方法来应对这一问题，但恶意行为者可以通过进一步编辑LLM生成的文本以降低其可检测性。在本研究中，我们探讨了使用人类反馈强化学习（RLHF）进一步编辑文本如何影响（a）两种任务下生成文本的质量，以及（b）LLM生成文本检测器的性能，同时考虑基于训练和零样本检测方法。尽管RLHF能够提升LLM生成文本的质量，我们发现它也倾向于生成更具可检测性、更长且更具重复性的输出。此外，我们观察到基于训练的检测器对短文本和包含代码的文本较为脆弱，而零样本检测器表现出更高的鲁棒性。 

---
# An Empirical Study of the Role of Incompleteness and Ambiguity in Interactions with Large Language Models 

**Title (ZH)**: 不完备性和模糊性在与大型语言模型互动中的作用 empirical研究 

**Authors**: Riya Naik, Ashwin Srinivasan, Estrid He, Swati Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2503.17936)  

**Abstract**: Natural language as a medium for human-computer interaction has long been anticipated, has been undergoing a sea-change with the advent of Large Language Models (LLMs) with startling capacities for processing and generating language. Many of us now treat LLMs as modern-day oracles, asking it almost any kind of question. Unlike its Delphic predecessor, consulting an LLM does not have to be a single-turn activity (ask a question, receive an answer, leave); and -- also unlike the Pythia -- it is widely acknowledged that answers from LLMs can be improved with additional context. In this paper, we aim to study when we need multi-turn interactions with LLMs to successfully get a question answered; or conclude that a question is unanswerable. We present a neural symbolic framework that models the interactions between human and LLM agents. Through the proposed framework, we define incompleteness and ambiguity in the questions as properties deducible from the messages exchanged in the interaction, and provide results from benchmark problems, in which the answer-correctness is shown to depend on whether or not questions demonstrate the presence of incompleteness or ambiguity (according to the properties we identify). Our results show multi-turn interactions are usually required for datasets which have a high proportion of incompleteness or ambiguous questions; and that that increasing interaction length has the effect of reducing incompleteness or ambiguity. The results also suggest that our measures of incompleteness and ambiguity can be useful tools for characterising interactions with an LLM on question-answeringproblems 

**Abstract (ZH)**: 自然语言作为人机交互的媒介，随着大型语言模型（LLMs）的出现而发生了翻天覆地的变化，LLMs具备令人惊讶的语言处理和生成能力。如今，我们中的许多人将LLMs视为现代先知，几乎可以提出各种问题。与德尔斐先知不同，咨询LLMs不一定要一次完成（提问，得到回答，离开）；此外，与先知Pythia不同，人们普遍认为，可以通过提供额外的上下文来改进LLMs的回答。本文旨在研究何时需要与LLMs进行多轮交互以成功回答问题；或者得出问题无法回答的结论。我们提出了一种神经符号框架，该框架可用于建模人类与LLMs代理之间的交互。通过提出的框架，我们将问题的不完整性和模糊性定义为可通过交互中交换的消息推断出的属性，并提供基准问题的结果，表明答案的正确性取决于问题是否体现出不完整性和模糊性（根据我们识别的属性）。我们的结果表明，对于具有高比例不完整或模糊问题的数据集，通常需要多轮交互；并且增加交互长度会减少不完整性和模糊性。结果还表明，我们对不完整性和模糊性的度量可以作为表征与LLM在问题回答交互中的有用工具。 

---
# Experience Retrieval-Augmentation with Electronic Health Records Enables Accurate Discharge QA 

**Title (ZH)**: 基于电子健康记录的体验检索增强 Enables 准确的出院质量评估 

**Authors**: Justice Ou, Tinglin Huang, Yilun Zhao, Ziyang Yu, Peiqing Lu, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2503.17933)  

**Abstract**: To improve the reliability of Large Language Models (LLMs) in clinical applications, retrieval-augmented generation (RAG) is extensively applied to provide factual medical knowledge. However, beyond general medical knowledge from open-ended datasets, clinical case-based knowledge is also critical for effective medical reasoning, as it provides context grounded in real-world patient experiences. Motivated by this, we propose Experience Retrieval Augmentation - ExpRAG framework based on Electronic Health Record (EHR), aiming to offer the relevant context from other patients' discharge reports. ExpRAG performs retrieval through a coarse-to-fine process, utilizing an EHR-based report ranker to efficiently identify similar patients, followed by an experience retriever to extract task-relevant content for enhanced medical reasoning. To evaluate ExpRAG, we introduce DischargeQA, a clinical QA dataset with 1,280 discharge-related questions across diagnosis, medication, and instruction tasks. Each problem is generated using EHR data to ensure realistic and challenging scenarios. Experimental results demonstrate that ExpRAG consistently outperforms a text-based ranker, achieving an average relative improvement of 5.2%, highlighting the importance of case-based knowledge for medical reasoning. 

**Abstract (ZH)**: 基于电子健康记录的经验检索增强-ExpRAG框架：提升临床应用大型语言模型的可靠性 

---
# STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models 

**Title (ZH)**: STShield: 单词哨兵用于大型语言模型的实时脱裤检测 

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17932)  

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment. 

**Abstract (ZH)**: 大型语言模型（LLMs）日益受到规避其安全机制的 Jailbreak 攻击的影响。尽管现有的防御方法要么容易受到适应性攻击的影响，要么需要昂贵的辅助模型，我们提出了一种名为 STShield 的轻量级框架，用于实时检测 Jailbreak。STShield 引入了一种新颖的一令牌哨兵机制，为模型的响应序列添加了一个二进制安全指示符，利用 LL defense 能力进行检测。我们的框架结合了正常提示的监督微调和嵌入空间扰动的对抗训练，实现了稳健的检测，同时保持模型的实用性。广泛的实验表明，STShield 成功防御了各种 Jailbreak 攻击，同时保持了模型在合法查询上的性能。与现有方法相比，STShield 在最小的计算开销下实现了优越的防御性能，使其成为实际部署中 LL defense 的可行方案。 

---
# WLB-LLM: Workload-Balanced 4D Parallelism for Large Language Model Training 

**Title (ZH)**: WLB-LLM：-large语言模型训练的负载均衡四维并行性 

**Authors**: Zheng Wang, Anna Cai, Xinfeng Xie, Zaifeng Pan, Yue Guan, Weiwei Chu, Jie Wang, Shikai Li, Jianyu Huang, Chris Cai, Yuchen Hao, Yufei Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.17924)  

**Abstract**: In this work, we present WLB-LLM, a workLoad-balanced 4D parallelism for large language model training. We first thoroughly analyze the workload imbalance issue in LLM training and identify two primary sources of imbalance at the pipeline parallelism and context parallelism levels. Then, to address the imbalance issue, at the pipeline parallelism level, WLB-LLM incorporates a workload-aware variable-length document packing method to balance the computation and communication workload across micro-batches. Additionally, at the context parallelism level, WLB-LLM introduces a novel fine-grained per-document sharding strategy, ensuring each worker within a context parallelism group has an identical workload. Comprehensive experiments under different model scales demonstrate that WLB-LLM significantly mitigates the workload imbalance during 4D parallelism LLM training and achieves an average speedup of 1.23x when applying WLB-LLM in our internal LLM training framework. 

**Abstract (ZH)**: 工作负载平衡的4D并行训练方法：WLB-LLM 

---
# Reasoning with LLMs for Zero-Shot Vulnerability Detection 

**Title (ZH)**: 零-shot 漏洞检测中的大语言模型推理 

**Authors**: Arastoo Zibaeirad, Marco Vieira  

**Link**: [PDF](https://arxiv.org/pdf/2503.17885)  

**Abstract**: Automating software vulnerability detection (SVD) remains a critical challenge in an era of increasingly complex and interdependent software systems. Despite significant advances in Large Language Models (LLMs) for code analysis, prevailing evaluation methodologies often lack the \textbf{context-aware robustness} necessary to capture real-world intricacies and cross-component interactions. To address these limitations, we present \textbf{VulnSage}, a comprehensive evaluation framework and a dataset curated from diverse, large-scale open-source system software projects developed in C/C++. Unlike prior datasets, it leverages a heuristic noise pre-filtering approach combined with LLM-based reasoning to ensure a representative and minimally noisy spectrum of vulnerabilities. The framework supports multi-granular analysis across function, file, and inter-function levels and employs four diverse zero-shot prompt strategies: Baseline, Chain-of-Thought, Think, and Think & Verify. Through this evaluation, we uncover that structured reasoning prompts substantially improve LLM performance, with Think & Verify reducing ambiguous responses from 20.3% to 9.1% while increasing accuracy. We further demonstrate that code-specialized models consistently outperform general-purpose alternatives, with performance varying significantly across vulnerability types, revealing that no single approach universally excels across all security contexts. Link to dataset and codes: this https URL 

**Abstract (ZH)**: 自动化软件漏洞检测（SVD）在日益复杂和相互依赖的软件系统时代仍然是一个关键挑战。尽管大型语言模型（LLMs）在代码分析方面取得了重大进展，但现有的评估方法往往缺乏捕捉现实世界复杂性和跨组件交互所需的\textbf{上下文感知稳健性}。为了解决这些局限性，我们提出\textbf{VulnSage}，一种全面的评估框架和从C/C++编写的多样化大型开源系统软件项目中编curated的数据集。与先前的数据集不同，它利用了一种启发式噪声预过滤方法结合LLM推理，以确保一个具有代表性且噪声最小的漏洞频谱。该框架支持跨函数、文件和跨函数层面的多粒度分析，并采用四种不同的零样本提示策略：基线、因果推理、思考和思考与验证。通过这种评估，我们发现结构化推理提示显著提高了LLM的性能，同时Think & Verify将含糊响应从20.3%减少到9.1%，并且提高了准确性。进一步的实验证明，代码专业化模型始终优于通用替代方案，不同类型的漏洞表现出显著差异，揭示了没有一种方法在所有安全上下文中普遍表现突出。数据集和代码链接：this https URL 

---
# Think Before Refusal : Triggering Safety Reflection in LLMs to Mitigate False Refusal Behavior 

**Title (ZH)**: 三思而后拒：在LLMs中触发安全反思以减轻错误拒绝行为 

**Authors**: Shengyun Si, Xinpeng Wang, Guangyao Zhai, Nassir Navab, Barbara Plank  

**Link**: [PDF](https://arxiv.org/pdf/2503.17882)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated that fine-tuning and human alignment can render LLMs harmless. In practice, such "harmlessness" behavior is mainly achieved by training models to reject harmful requests, such as "Explain how to burn down my neighbor's house", where the model appropriately declines to respond. However, this approach can inadvertently result in false refusal, where models reject benign queries as well, such as "Tell me how to kill a Python process". In this work, we demonstrate that prompting safety reflection before generating a response can mitigate false refusal behavior. Building on this finding, we introduce the Think-Before-Refusal (TBR) schema and conduct safety-aware instruction fine-tuning incorporating safety reflection. In an ablation study across 15 pre-trained models, we show that models fine-tuned with safety reflection significantly reduce false refusal behavior while maintaining safety and overall performance compared to those fine-tuned without safety reflection. 

**Abstract (ZH)**: 近期大型语言模型的进步表明，通过微调和人类对齐可以使大型语言模型变得更加安全。实践中，“安全”行为主要通过训练模型拒绝有害请求来实现，例如对“解释如何烧毁邻居的房子”的请求，模型会适当拒绝回应。然而，这种做法可能会导致误拒绝，即模型将良性查询也错误地拒绝，例如“告诉我如何杀死一个Python进程”。在本研究中，我们证明了在生成响应之前进行安全性反思可以缓解误拒绝行为。基于这一发现，我们提出了反思前思考（Think-Before-Refusal, TBR）模式，并进行了安全意识指令微调，包含安全性反思。在15个预训练模型的消融研究中，我们展示了包含安全性反思微调的模型在减少误拒绝行为的同时，与未包含安全性反思微调的模型相比，在安全性和整体性能方面均有所提升。 

---
# A Study on the Improvement of Code Generation Quality Using Large Language Models Leveraging Product Documentation 

**Title (ZH)**: 利用产品文档增强大型语言模型的代码生成质量研究 

**Authors**: Takuro Morimoto, Harumi Haraguchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17837)  

**Abstract**: Research on using Large Language Models (LLMs) in system development is expanding, especially in automated code and test generation. While E2E testing is vital for ensuring application quality, most test generation research has focused on unit tests, with limited work on E2E test code. This study proposes a method for automatically generating E2E test code from product documentation such as manuals, FAQs, and tutorials using LLMs with tailored prompts. The two step process interprets documentation intent and produces executable test code. Experiments on a web app with six key features (e.g., authentication, profile, discussion) showed that tests generated from product documentation had high compilation success and functional coverage, outperforming those based on requirement specs and user stories. These findings highlight the potential of product documentation to improve E2E test quality and, by extension, software quality. 

**Abstract (ZH)**: 利用大型语言模型自动生成端到端测试代码的研究 

---
# Feather-SQL: A Lightweight NL2SQL Framework with Dual-Model Collaboration Paradigm for Small Language Models 

**Title (ZH)**: Feather-SQL：一种基于双模型协作范式的轻量级NL2SQL框架 

**Authors**: Wenqi Pei, Hailing Xu, Hengyuan Zhao, Shizheng Hou, Han Chen, Zining Zhang, Pingyi Luo, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2503.17811)  

**Abstract**: Natural Language to SQL (NL2SQL) has seen significant advancements with large language models (LLMs). However, these models often depend on closed-source systems and high computational resources, posing challenges in data privacy and deployment. In contrast, small language models (SLMs) struggle with NL2SQL tasks, exhibiting poor performance and incompatibility with existing frameworks. To address these issues, we introduce Feather-SQL, a new lightweight framework tailored for SLMs. Feather-SQL improves SQL executability and accuracy through 1) schema pruning and linking, 2) multi-path and multi-candidate generation. Additionally, we introduce the 1+1 Model Collaboration Paradigm, which pairs a strong general-purpose chat model with a fine-tuned SQL specialist, combining strong analytical reasoning with high-precision SQL generation. Experimental results on BIRD demonstrate that Feather-SQL improves NL2SQL performance on SLMs, with around 10% boost for models without fine-tuning. The proposed paradigm raises the accuracy ceiling of SLMs to 54.76%, highlighting its effectiveness. 

**Abstract (ZH)**: 自然语言到SQL（NL2SQL）任务在大规模语言模型（LLMs）的推动下取得了显著进展。然而，这些模型通常依赖于闭源系统和高计算资源，这在数据隐私和部署方面带来了挑战。相比之下，小语言模型（SLMs）在NL2SQL任务中表现不佳，性能较差且与现有框架不兼容。为了解决这些问题，我们提出了一种新的轻量级框架Feather-SQL，专门针对SLMs。Feather-SQL通过1）模式修剪和链接，2）多路径和多候选生成来提高SQL的可执行性和准确性。此外，我们引入了“1+1模型协作范式”，该范式将一个强大的通用聊天模型与一个细调过的SQL专家模型配对，结合了强大的分析推理能力与高精度的SQL生成能力。实验结果表明，Feather-SQL在SLMs上的NL2SQL性能得到了提升，未经过细调的模型性能提升了约10%。提出的范式将SLMs的准确性上限提升至54.76%，突显了其有效性。 

---
# Every Sample Matters: Leveraging Mixture-of-Experts and High-Quality Data for Efficient and Accurate Code LLM 

**Title (ZH)**: 每一例样本都重要：利用专家混合模型和高质量数据实现高效精准的代码LLM 

**Authors**: Codefuse, Ling Team, Wenting Cai, Yuchen Cao, Chaoyu Chen, Chen Chen, Siba Chen, Qing Cui, Peng Di, Junpeng Fang, Zi Gong, Ting Guo, Zhengyu He, Yang Huang, Cong Li, Jianguo Li, Zheng Li, Shijie Lian, BingChang Liu, Songshan Luo, Shuo Mao, Min Shen, Jian Wu, Jiaolong Yang, Wenjie Yang, Tong Ye, Hang Yu, Wei Zhang, Zhenduo Zhang, Hailin Zhao, Xunjin Zheng, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2503.17793)  

**Abstract**: Recent advancements in code large language models (LLMs) have demonstrated remarkable capabilities in code generation and understanding. It is still challenging to build a code LLM with comprehensive performance yet ultimate efficiency. Many attempts have been released in the open source community to break the trade-off between performance and efficiency, such as the Qwen Coder series and the DeepSeek Coder series. This paper introduces yet another attempt in this area, namely Ling-Coder-Lite. We leverage the efficient Mixture-of-Experts (MoE) architecture along with a set of high-quality data curation methods (especially those based on program analytics) to build an efficient yet powerful code LLM. Ling-Coder-Lite exhibits on-par performance on 12 representative coding benchmarks compared to state-of-the-art models of similar size, such as Qwen2.5-Coder-7B and DeepSeek-Coder-V2-Lite, while offering competitive latency and throughput. In practice, we achieve a 50\% reduction in deployment resources compared to the similar-sized dense model without performance loss. To facilitate further research and development in this area, we open-source our models as well as a substantial portion of high-quality data for the annealing and post-training stages. The models and data can be accessed at~\url{this https URL}. 

**Abstract (ZH)**: 最近在代码大型语言模型（LLMs）方面的进展显示了其在代码生成和理解方面的卓越能力。尽管如此，构建一个在各方面表现全面且极其高效的代码LLM仍然具有挑战性。开源社区中已经发布了许多尝试来打破性能和效率之间的权衡，例如Qwen Coder系列和DeepSeek Coder系列。本文介绍了一个在这个领域中的新尝试，即Ling-Coder-Lite。我们利用高效的专家混合（MoE）架构，并结合了一系列高质量数据管理方法（尤其是基于程序分析的方法）来构建一个高效且强大的代码LLM。Ling-Coder-Lite在12个代表性编程基准测试中的表现与类似规模的最先进的模型（如Qwen2.5-Coder-7B和DeepSeek-Coder-V2-Lite）持平，同时提供竞争力的价格延迟和吞吐量。在实际应用中，与类似的密集模型相比，我们实现了50%的部署资源减少，且没有性能损失。为了进一步促进该领域的研究和发展，我们开放了我们的模型以及大量高质量数据以供退火和后训练阶段使用。模型和数据可以访问：this https URL。 

---
# Energy-Aware LLMs: A step towards sustainable AI for downstream applications 

**Title (ZH)**: 面向下游应用的能源aware大语言模型：通向可持续AI的一步 

**Authors**: Nguyen Phuc Tran, Brigitte Jaumard, Oscar Delgado  

**Link**: [PDF](https://arxiv.org/pdf/2503.17783)  

**Abstract**: Advanced Large Language Models (LLMs) have revolutionized various fields, including communication networks, sparking an innovation wave that has led to new applications and services, and significantly enhanced solution schemes. Despite all these impressive developments, most LLMs typically require huge computational resources, resulting in terribly high energy consumption. Thus, this research study proposes an end-to-end pipeline that investigates the trade-off between energy efficiency and model performance for an LLM during fault ticket analysis in communication networks. It further evaluates the pipeline performance using two real-world datasets for the tasks of root cause analysis and response feedback in a communication network. Our results show that an appropriate combination of quantization and pruning techniques is able to reduce energy consumption while significantly improving model performance. 

**Abstract (ZH)**: 先进大语言模型（LLMs）已在通信网络等领域引发了一场创新浪潮，极大地推动了新的应用和服务，并显著提升了解决方案。尽管取得了这些令人印象深刻的发展，大多数LLMs仍然需要大量的计算资源，导致能耗极高。因此，本研究提出了一种端到端的管道，以在通信网络中故障工单分析过程中研究能耗效率与模型性能之间的权衡。该研究进一步使用两个真实世界的数据集评估管道在通信网络中故障根本原因分析和响应反馈任务上的性能。研究结果表明，适当结合量化和剪枝技术能够在显著提升模型性能的同时降低能耗。 

---
# Building Resource-Constrained Language Agents: A Korean Case Study on Chemical Toxicity Information 

**Title (ZH)**: 基于资源约束的语言代理构建：以韩国化学毒性信息为例的研究 

**Authors**: Hojun Cho, Donghu Kim, Soyoung Yang, Chan Lee, Hunjoo Lee, Jaegul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2503.17753)  

**Abstract**: Language agents powered by large language models (LLMs) face significant deployment challenges in resource-constrained environments, particularly for specialized domains and less-common languages. This paper presents Tox-chat, a Korean chemical toxicity information agent devised within these limitations. We propose two key innovations: a context-efficient architecture that reduces token consumption through hierarchical section search, and a scenario-based dialogue generation methodology that effectively distills tool-using capabilities from larger models. Experimental evaluations demonstrate that our fine-tuned 8B parameter model substantially outperforms both untuned models and baseline approaches, in terms of DB faithfulness and preference. Our work offers valuable insights for researchers developing domain-specific language agents under practical constraints. 

**Abstract (ZH)**: 由大规模语言模型驱动的语言代理在资源约束环境中，特别是在专业领域和少见语言中，面临显著的部署挑战。本文提出了Tox-chat，这是一种在这些限制下为韩语化学毒性信息设计的语言代理。我们提出了两项关键创新：一种通过分层段落搜索减少词元消耗的上下文高效架构，以及一种基于场景的对话生成方法，该方法有效提炼了大模型的工具使用能力。实验评估表明，我们微调的8B参数模型在DB信度和偏好方面显著优于未微调的模型和基线方法。我们的工作为在实际约束条件下开发领域特定语言代理的研究人员提供了宝贵见解。 

---
# Can LLMs Automate Fact-Checking Article Writing? 

**Title (ZH)**: LLM能否自动化事实核查文章写作？ 

**Authors**: Dhruv Sahnan, David Corney, Irene Larraz, Giovanni Zagni, Ruben Miguez, Zhuohan Xie, Iryna Gurevych, Elizabeth Churchill, Tanmoy Chakraborty, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2503.17684)  

**Abstract**: Automatic fact-checking aims to support professional fact-checkers by offering tools that can help speed up manual fact-checking. Yet, existing frameworks fail to address the key step of producing output suitable for broader dissemination to the general public: while human fact-checkers communicate their findings through fact-checking articles, automated systems typically produce little or no justification for their assessments. Here, we aim to bridge this gap. We argue for the need to extend the typical automatic fact-checking pipeline with automatic generation of full fact-checking articles. We first identify key desiderata for such articles through a series of interviews with experts from leading fact-checking organizations. We then develop QRAFT, an LLM-based agentic framework that mimics the writing workflow of human fact-checkers. Finally, we assess the practical usefulness of QRAFT through human evaluations with professional fact-checkers. Our evaluation shows that while QRAFT outperforms several previously proposed text-generation approaches, it lags considerably behind expert-written articles. We hope that our work will enable further research in this new and important direction. 

**Abstract (ZH)**: 自动事实核查旨在通过提供可以加速人工事实核查的工具来支持专业事实核查人员。然而，现有的框架未能解决向公众更广泛传播的關鍵步骤：虽然人工事实核查人员通过事实核查文章传达其发现，但自动化系统通常几乎不提供其评估的任何理由。在此，我们旨在弥合这一差距。我们主张需要扩展典型的自动事实核查流程，包括自动生成完整的事实核查文章。我们首先通过与领先事实核查机构的专家进行一系列访谈来识别此类文章的关键要求。然后，我们开发了QRAFT，一个基于LLM的代理框架，模仿人工事实核查人员的写作流程。最后，我们通过专业事实核查人员的人类评估来评估QRAFT的实际效用。我们的评估显示，虽然QRAFT优于几种先前提出的文本生成方法，但在专家撰写的文章面前还有很大差距。我们希望我们的工作能促进这一新且重要的方向上进一步的研究。 

---
# Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models 

**Title (ZH)**: Safe RLHF-V：多模态大型语言模型中的安全人类反馈强化学习 

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.17682)  

**Abstract**: Multimodal large language models (MLLMs) are critical for developing general-purpose AI assistants, yet they face growing safety risks. How can we ensure that MLLMs are safely aligned to prevent undesired behaviors such as discrimination, misinformation, or violations of ethical standards? In a further step, we need to explore how to fine-tune MLLMs to enhance reasoning performance while ensuring they satisfy safety constraints. Fundamentally, this can be formulated as a min-max optimization problem. In this study, we propose Safe RLHF-V, the first multimodal safety alignment framework that jointly optimizes helpfulness and safety using separate multimodal reward and cost models within a Lagrangian-based constrained optimization framework. Given that there is a lack of preference datasets that separate helpfulness and safety in multimodal scenarios, we introduce BeaverTails-V, the first open-source dataset with dual preference annotations for helpfulness and safety, along with multi-level safety labels (minor, moderate, severe). Additionally, we design a Multi-level Guardrail System to proactively defend against unsafe queries and adversarial attacks. By applying the Beaver-Guard-V moderation for 5 rounds of filtering and re-generation on the precursor model, the overall safety of the upstream model is significantly improved by an average of 40.9%. Experimental results demonstrate that fine-tuning different MLLMs with Safe RLHF can effectively enhance model helpfulness while ensuring improved safety. Specifically, Safe RLHF-V improves model safety by 34.2% and helpfulness by 34.3%. All of datasets, models, and code can be found at this https URL to support the safety development of MLLMs and reduce potential societal risks. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）是开发通用人工智能助理的关键，但它们面临着日益增长的安全风险。我们如何确保MLLMs安全对齐，以防止不当行为，如歧视、错误信息或违反伦理标准？在此基础上，我们需要探索如何在确保满足安全约束的同时，调整MLLMs以增强推理性能。从根本上说，这可以被形式化为一个最小-最大优化问题。在本研究中，我们提出了Safe RLHF-V，这是一种首创的多模态安全对齐框架，该框架在拉格朗日约束优化框架内分别使用多模态奖励和成本模型来联合优化有用性和安全性。由于在多模态场景中缺乏将有用性和安全性分开的偏好数据集，我们引入了BeaverTails-V，这是一个首创的开源数据集，其中包括双重视觉偏好注释（有用性和安全性）以及多层次的安全标签（轻微、中等、严重）。此外，我们设计了多层次护栏系统以主动防御不当查询和 adversarial 攻击。通过对先导模型进行5轮过滤和重新生成的Beaver-Guard-V审核，上游模型的整体安全性平均提高了40.9%。实验结果表明，使用Safe RLHF微调不同的MLLMs可以有效地提高模型的有用性并确保安全性的提高。具体而言，Safe RLHF-V提高了模型安全性34.2%和有用性34.3%。所有数据集、模型和代码均可通过此链接访问，以支持MLLMs的安全发展并降低潜在的社会风险。 

---
# A Generative Caching System for Large Language Models 

**Title (ZH)**: 大型语言模型的生成性缓存系统 

**Authors**: Arun Iyengar, Ashish Kundu, Ramana Kompella, Sai Nandan Mamidi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17603)  

**Abstract**: Caching has the potential to be of significant benefit for accessing large language models (LLMs) due to their high latencies which typically range from a small number of seconds to well over a minute. Furthermore, many LLMs charge money for queries; caching thus has a clear monetary benefit. This paper presents a new caching system for improving user experiences with LLMs. In addition to reducing both latencies and monetary costs for accessing LLMs, our system also provides important features that go beyond the performance benefits typically associated with caches. A key feature we provide is generative caching, wherein multiple cached responses can be synthesized to provide answers to queries which have never been seen before. Our generative caches function as repositories of valuable information which can be mined and analyzed. We also improve upon past semantic caching techniques by tailoring the caching algorithms to optimally balance cost and latency reduction with the quality of responses provided. Performance tests indicate that our caches are considerably faster than GPTcache. 

**Abstract (ZH)**: 缓存有望显著改善访问大型语言模型（LLMs）的体验，由于LLMs通常具有从几秒钟到超过一分钟的高延迟。此外，许多LLMs按查询收费；因此，缓存具有明显的经济效益。本文介绍了一种新的缓存系统，以提高用户与LLMs的交互体验。除了降低访问LLMs的延迟和经济成本外，我们的系统还提供了超越传统缓存性能优势的重要功能。我们提供的一项关键功能是生成性缓存，即可以合成多个缓存响应来回答之前从未遇到过的查询。我们的生成性缓存充当有价值的资源库，可以被挖掘和分析。我们还通过定制缓存算法，优化了成本和延迟减少与提供的响应质量之间的平衡，超越了过去的技术。性能测试表明，我们的缓存比GPTcache更快。 

---
# GPBench: A Comprehensive and Fine-Grained Benchmark for Evaluating Large Language Models as General Practitioners 

**Title (ZH)**: GPBench: 一种全面细粒度的大型语言模型综合评估基准（作为全科医生） 

**Authors**: Zheqing Li, Yiying Yang, Jiping Lang, Wenhao Jiang, Yuhang Zhao, Shuang Li, Dingqian Wang, Zhu Lin, Xuanna Li, Yuze Tang, Jiexian Qiu, Xiaolin Lu, Hongji Yu, Shuang Chen, Yuhua Bi, Xiaofei Zeng, Yixian Chen, Junrong Chen, Lin Yao  

**Link**: [PDF](https://arxiv.org/pdf/2503.17599)  

**Abstract**: General practitioners (GPs) serve as the cornerstone of primary healthcare systems by providing continuous and comprehensive medical services. However, due to community-oriented nature of their practice, uneven training and resource gaps, the clinical proficiency among GPs can vary significantly across regions and healthcare settings. Currently, Large Language Models (LLMs) have demonstrated great potential in clinical and medical applications, making them a promising tool for supporting general practice. However, most existing benchmarks and evaluation frameworks focus on exam-style assessments-typically multiple-choice question-lack comprehensive assessment sets that accurately mirror the real-world scenarios encountered by GPs. To evaluate how effectively LLMs can make decisions in the daily work of GPs, we designed GPBench, which consists of both test questions from clinical practice and a novel evaluation framework. The test set includes multiple-choice questions that assess fundamental knowledge of general practice, as well as realistic, scenario-based problems. All questions are meticulously annotated by experts, incorporating rich fine-grained information related to clinical management. The proposed LLM evaluation framework is based on the competency model for general practice, providing a comprehensive methodology for assessing LLM performance in real-world settings. As the first large-model evaluation set targeting GP decision-making scenarios, GPBench allows us to evaluate current mainstream LLMs. Expert assessment and evaluation reveal that in areas such as disease staging, complication recognition, treatment detail, and medication usage, these models exhibit at least ten major shortcomings. Overall, existing LLMs are not yet suitable for independent use in real-world GP working scenarios without human oversight. 

**Abstract (ZH)**: GPBench：针对全科医生决策场景的大模型评估基准 

---
# ConSol: Sequential Probability Ratio Testing to Find Consistent LLM Reasoning Paths Efficiently 

**Title (ZH)**: ConSol: 序列概率比检验以高效寻找一致的LLM推理路径 

**Authors**: Jaeyeon Lee, Guantong Qi, Matthew Brady Neeley, Zhandong Liu, Hyun-Hwan Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2503.17587)  

**Abstract**: Recent advancements in large language models (LLMs) integrating explicit reasoning, such as OpenAI's o3-mini, DeepSeek-R1, and QWQ-32B, enable smaller models to solve complex tasks by generating intermediate reasoning steps prior to providing answers. However, this approach significantly increases computational costs, both monetarily and environmentally. The widely-used self-consistency method further exacerbates these costs by aggregating multiple reasoning paths to improve accuracy, often requiring between 40 to 64 samples per task. Although aggregation effectively reduces variance and bias, additional sampling can lead to diminishing returns when early samples yield consistent results. To address inefficiencies, we propose leveraging Sequential Probability Ratio Testing (SPRT) to dynamically terminate sampling once sufficient consistency is achieved. We calibrate SPRT parameters specifically for LLM applications, accounting for sensitivity to detect the mode of the distribution. Our experiments demonstrate that incorporating SPRT significantly enhances token efficiency, achieving comparable accuracy to self-consistency methods but at a substantially reduced computational cost. To promote transparency and facilitate reproducibility, we have made the source code and datasets used in our experiments publicly available at our GitHub repository: this https URL, or available as a PyPI package: pip install consol. We hope that this resource will support further research and encourage the development of new methods building upon our work. 

**Abstract (ZH)**: Recent advancements in大型语言模型（LLMs）整合显式推理：以OpenAI的o3-mini、DeepSeek-R1和QWQ-32B为例，通过在提供答案之前生成中间推理步骤，使较小的模型能够解决复杂任务。然而，这种方法显著增加了计算成本，无论是经济上还是环境上。广泛使用的自我一致性方法进一步加剧了这些成本，通过聚合多个推理路径以提高准确性，通常每项任务需要40到64个样本。尽管聚合有效降低了方差和偏差，但额外的采样在早期样本结果一致时可能会导致边际效益递减。为了应对效率问题，我们提出利用序列概率比检验（SPRT）动态终止采样，一旦达到足够的一致性就终止。我们为LLM应用校准了SPRT参数，考虑了对检测分布模式的敏感性。我们的实验表明，将SPRT纳入可以显著提高token效率，准确性与自我一致性方法相当，但计算成本大幅降低。为了促进透明性和重现实验，我们在GitHub仓库（this https URL）和PyPI包（pip install consol）中公开了源代码和实验数据集，希望这将支持进一步研究，并鼓励开发基于我们工作的新型方法。 

---
# Fairness-Driven LLM-based Causal Discovery with Active Learning and Dynamic Scoring 

**Title (ZH)**: 基于公平性驱动的大语言模型因果发现与主动学习及动态评分方法 

**Authors**: Khadija Zanna, Akane Sano  

**Link**: [PDF](https://arxiv.org/pdf/2503.17569)  

**Abstract**: Causal discovery (CD) plays a pivotal role in numerous scientific fields by clarifying the causal relationships that underlie phenomena observed in diverse disciplines. Despite significant advancements in CD algorithms that enhance bias and fairness analyses in machine learning, their application faces challenges due to the high computational demands and complexities of large-scale data. This paper introduces a framework that leverages Large Language Models (LLMs) for CD, utilizing a metadata-based approach akin to the reasoning processes of human experts. By shifting from pairwise queries to a more scalable breadth-first search (BFS) strategy, the number of required queries is reduced from quadratic to linear in terms of variable count, thereby addressing scalability concerns inherent in previous approaches. This method utilizes an Active Learning (AL) and a Dynamic Scoring Mechanism that prioritizes queries based on their potential information gain, combining mutual information, partial correlation, and LLM confidence scores to refine the causal graph more efficiently and accurately. This BFS query strategy reduces the required number of queries significantly, thereby addressing scalability concerns inherent in previous approaches. This study provides a more scalable and efficient solution for leveraging LLMs in fairness-driven CD, highlighting the effects of the different parameters on performance. We perform fairness analyses on the inferred causal graphs, identifying direct and indirect effects of sensitive attributes on outcomes. A comparison of these analyses against those from graphs produced by baseline methods highlights the importance of accurate causal graph construction in understanding bias and ensuring fairness in machine learning systems. 

**Abstract (ZH)**: 因果发现（CD）在众多科学领域中发挥着关键作用，通过阐明各种学科中观察到的现象背后的因果关系。尽管CD算法在机器学习中的偏差和公平性分析方面取得了显著进步，但其应用仍面临计算需求高和大规模数据复杂性高的挑战。本文提出了一种框架，利用大型语言模型（LLMs）进行因果发现，采用基于元数据的方法，类似于人类专家的推理过程。通过从成对查询转向更具扩展性的广度优先搜索（BFS）策略，所需的查询数量从变量数量的平方级降低到线性级，从而解决了先前方法中存在的可扩展性问题。该方法利用主动学习（AL）和动态评分机制，根据潜在的信息增益优先选择查询，结合互信息、部分相关和LLM置信度评分，更高效准确地精炼因果图。这种BFS查询策略显著减少了所需的查询数量，从而解决了先前方法中存在的可扩展性问题。本研究为利用LLMs进行公平驱动的因果发现提供了更具可扩展性和效率的解决方案，并探讨了不同参数对性能的影响。我们对推断出的因果图进行了公平性分析，识别了敏感属性对结果的直接和间接影响。将这些分析与基准方法生成的图的分析进行比较，强调了准确构建因果图在理解偏差和确保机器学习系统公平性方面的重要性。 

---
# Autonomous Radiotherapy Treatment Planning Using DOLA: A Privacy-Preserving, LLM-Based Optimization Agent 

**Title (ZH)**: 使用DOLA的自主放射治疗规划：一种基于LLM的隐私保护优化代理 

**Authors**: Humza Nusrat, Bing Luo, Ryan Hall, Joshua Kim, Hassan Bagher-Ebadian, Anthony Doemer, Benjamin Movsas, Kundan Thind  

**Link**: [PDF](https://arxiv.org/pdf/2503.17553)  

**Abstract**: Radiotherapy treatment planning is a complex and time-intensive process, often impacted by inter-planner variability and subjective decision-making. To address these challenges, we introduce Dose Optimization Language Agent (DOLA), an autonomous large language model (LLM)-based agent designed for optimizing radiotherapy treatment plans while rigorously protecting patient privacy. DOLA integrates the LLaMa3.1 LLM directly with a commercial treatment planning system, utilizing chain-of-thought prompting, retrieval-augmented generation (RAG), and reinforcement learning (RL). Operating entirely within secure local infrastructure, this agent eliminates external data sharing. We evaluated DOLA using a retrospective cohort of 18 prostate cancer patients prescribed 60 Gy in 20 fractions, comparing model sizes (8 billion vs. 70 billion parameters) and optimization strategies (No-RAG, RAG, and RAG+RL) over 10 planning iterations. The 70B model demonstrated significantly improved performance, achieving approximately 16.4% higher final scores than the 8B model. The RAG approach outperformed the No-RAG baseline by 19.8%, and incorporating RL accelerated convergence, highlighting the synergy of retrieval-based memory and reinforcement learning. Optimal temperature hyperparameter analysis identified 0.4 as providing the best balance between exploration and exploitation. This proof of concept study represents the first successful deployment of locally hosted LLM agents for autonomous optimization of treatment plans within a commercial radiotherapy planning system. By extending human-machine interaction through interpretable natural language reasoning, DOLA offers a scalable and privacy-conscious framework, with significant potential for clinical implementation and workflow improvement. 

**Abstract (ZH)**: 基于DOLA的自主优化的放射治疗计划语言代理：一个保护患者隐私的复杂治疗规划挑战的解决方案 

---
# Bayesian Teaching Enables Probabilistic Reasoning in Large Language Models 

**Title (ZH)**: 贝叶斯教学使大型语言模型具备概率推理能力 

**Authors**: Linlu Qiu, Fei Sha, Kelsey Allen, Yoon Kim, Tal Linzen, Sjoerd van Steenkiste  

**Link**: [PDF](https://arxiv.org/pdf/2503.17523)  

**Abstract**: Artificial intelligence systems based on large language models (LLMs) are increasingly used as agents that interact with users and with the world. To do so successfully, LLMs need to construct internal representations of the world and form probabilistic beliefs about those representations. To provide a user with personalized recommendations, for example, the LLM needs to gradually infer the user's preferences, over the course of multiple interactions. To evaluate whether contemporary LLMs are able to do so, we use the Bayesian inference framework from probability theory, which lays out the optimal way to update an agent's beliefs as it receives new information. We first show that the LLMs do not update their beliefs as expected from the Bayesian framework, and that consequently their predictions do not improve as expected as more information becomes available, even less so than we find is the case for humans. To address this issue, we teach the LLMs to reason in a Bayesian manner by training them to mimic the predictions of an optimal Bayesian model. We find that this approach not only significantly improves the LLM's performance on the particular recommendation task it is trained on, but also enables generalization to other tasks. This suggests that this method endows the LLM with broader Bayesian reasoning skills. More generally, our results indicate that LLMs can learn about reasoning strategies effectively and generalize those skills to new domains, which in part explains LLMs' empirical success. 

**Abstract (ZH)**: 基于大规模语言模型的人工智能系统作为与用户和世界交互的代理日益增多。为了成功地进行这种交互，大规模语言模型（LLMs）需要构建对世界的内部表征，并对其表征形成概率性信念。为了为用户提供个性化推荐，例如，LLMs需要在多次交互中逐渐推断出用户的偏好。我们利用概率论中的贝叶斯推理框架来评估当代LLMs是否能够做到这一点，该框架明确了代理在接收到新信息时更新其信念的最佳方式。我们首先表明，LLMs并未按贝叶斯框架预期的方式更新其信念，因此，当更多信息可用时，它们的预测也没有按预期改进，甚至不如我们发现的人类的表现。为了解决这一问题，我们通过训练LLMs使其模仿最优贝叶斯模型的预测来教它们进行贝叶斯推理。我们发现，这种方法不仅显著提高了LLMs在所训练的特定推荐任务上的性能，还使其能够泛化到其他任务。这表明该方法赋予LLMs更广泛的贝叶斯推理能力。更广泛地说，我们的结果表明，LLMs能够有效学习推理策略，并将这些技能泛化到新领域，部分解释了LLMs的实证成功。 

---
# Language Models May Verbatim Complete TextThey Were Not Explicitly Trained On 

**Title (ZH)**: 语言模型可能直接补全它们没有明确训练过的内容。 

**Authors**: Ken Ziyu Liu, Christopher A. Choquette-Choo, Matthew Jagielski, Peter Kairouz, Sanmi Koyejo, Percy Liang, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2503.17514)  

**Abstract**: An important question today is whether a given text was used to train a large language model (LLM). A \emph{completion} test is often employed: check if the LLM completes a sufficiently complex text. This, however, requires a ground-truth definition of membership; most commonly, it is defined as a member based on the $n$-gram overlap between the target text and any text in the dataset. In this work, we demonstrate that this $n$-gram based membership definition can be effectively gamed. We study scenarios where sequences are \emph{non-members} for a given $n$ and we find that completion tests still succeed. We find many natural cases of this phenomenon by retraining LLMs from scratch after removing all training samples that were completed; these cases include exact duplicates, near-duplicates, and even short overlaps. They showcase that it is difficult to find a single viable choice of $n$ for membership definitions. Using these insights, we design adversarial datasets that can cause a given target sequence to be completed without containing it, for any reasonable choice of $n$. Our findings highlight the inadequacy of $n$-gram membership, suggesting membership definitions fail to account for auxiliary information available to the training algorithm. 

**Abstract (ZH)**: 当前一个重要的问题是判断给定文本是否被用于训练大规模语言模型（LLM）。一种常用的方法是完成测试：检查LLM是否能够完成一个足够复杂的文本。然而，这需要一个基于真实标准的成员资格定义；最常见的定义是基于目标文本与数据集中任何文本的$n$-gram重叠。在本工作中，我们展示了基于$n$-gram的成员资格定义可以被有效地操纵。我们研究了对于给定$n$的非成员序列的情况，发现完成测试仍然会成功。我们通过从数据集中移除所有被完成的样本并重新训练LLM，发现了许多自然存在的这种情况的例子，包括精确副本、近似副本以及甚至很短的重叠。这些例子展示了难以找到一个单一有效的$n$值作为成员资格定义。利用这些见解，我们设计了对抗性数据集，能够在不包含目标序列的情况下，导致给定的目标序列被完成，对于任何合理的$n$值选择都是如此。我们的发现揭示了$n$-gram成员资格的不足，暗示成员资格定义未能考虑到训练算法可用的辅助信息。 

---
# Improving Quantization with Post-Training Model Expansion 

**Title (ZH)**: 基于后训练模型扩展的量化优化 

**Authors**: Giuseppe Franco, Pablo Monteagudo-Lago, Ian Colbert, Nicholas Fraser, Michaela Blott  

**Link**: [PDF](https://arxiv.org/pdf/2503.17513)  

**Abstract**: The size of a model has been a strong predictor of its quality, as well as its cost. As such, the trade-off between model cost and quality has been well-studied. Post-training optimizations like quantization and pruning have typically focused on reducing the overall volume of pre-trained models to reduce inference costs while maintaining model quality. However, recent advancements have introduced optimization techniques that, interestingly, expand models post-training, increasing model size to improve quality when reducing volume. For instance, to enable 4-bit weight and activation quantization, incoherence processing often necessitates inserting online Hadamard rotations in the compute graph, and preserving highly sensitive weights often calls for additional higher precision computations. However, if application requirements cannot be met, the prevailing solution is to relax quantization constraints. In contrast, we demonstrate post-training model expansion is a viable strategy to improve model quality within a quantization co-design space, and provide theoretical justification. We show it is possible to progressively and selectively expand the size of a pre-trained large language model (LLM) to improve model quality without end-to-end retraining. In particular, when quantizing the weights and activations to 4 bits for Llama3 1B, we reduce the zero-shot accuracy gap to full precision by an average of 3% relative to both QuaRot and SpinQuant with only 5% more parameters, which is still a 3.8% reduction in volume relative to a BF16 reference model. 

**Abstract (ZH)**: 模型大小是其质量和成本的强预测因素，因此模型成本与质量之间的权衡已经被广泛研究。训练后优化技术如量化和剪枝通常关注于减少预训练模型的总体体积以降低推理成本同时保持模型质量。然而，最近的进展引入了扩大模型的优化技术，通过增加模型大小来提高质量，尤其是在减少体积时。例如，为实现4位权重和激活量化，不一致性处理时常需要在计算图中插入在线汉诺尔旋转，而保留敏感权重则需要额外的高精度计算。然而，如果应用需求无法满足，主流解决方案是放宽量化约束。相反，我们证明在量化联合设计空间中，训练后模型扩展是提高模型质量的一种可行策略，并提供了理论依据。我们展示了可以渐进且选择性地扩展预训练大型语言模型（LLM）的大小以提高模型质量，而无需端到端重新训练。特别是在将权重和激活量化为4位时，与QuaRot和SpinQuant相比，我们通过增加5%的参数缩小了零样本准确率差距的平均值为3%，相对体积减少了3.8%，相较于BF16参考模型。 

---
# Large Language Models (LLMs) for Source Code Analysis: applications, models and datasets 

**Title (ZH)**: 大型语言模型（LLMs）在源代码分析中的应用、模型与数据集 

**Authors**: Hamed Jelodar, Mohammad Meymani, Roozbeh Razavi-Far  

**Link**: [PDF](https://arxiv.org/pdf/2503.17502)  

**Abstract**: Large language models (LLMs) and transformer-based architectures are increasingly utilized for source code analysis. As software systems grow in complexity, integrating LLMs into code analysis workflows becomes essential for enhancing efficiency, accuracy, and automation. This paper explores the role of LLMs for different code analysis tasks, focusing on three key aspects: 1) what they can analyze and their applications, 2) what models are used and 3) what datasets are used, and the challenges they face. Regarding the goal of this research, we investigate scholarly articles that explore the use of LLMs for source code analysis to uncover research developments, current trends, and the intellectual structure of this emerging field. Additionally, we summarize limitations and highlight essential tools, datasets, and key challenges, which could be valuable for future work. 

**Abstract (ZH)**: 大型语言模型（LLMs）和基于变换器的架构在源代码分析中日益受到利用。随着软件系统的复杂性增加，将LLMs集成到代码分析工作流中对于提高效率、准确性和自动化变得至关重要。本文探讨了LLMs在不同代码分析任务中的作用，重点关注三个方面：1）它们可以分析的内容及其应用，2）使用的模型，3）使用的数据集及其面临的挑战。关于本研究的目标，我们调查了探讨LLMs在源代码分析中应用的学术文章，以揭示该新兴领域的研究进展、当前趋势和知识结构。此外，我们总结了局限性，并强调了重要的工具、数据集和关键挑战，这些对于未来的工作具有重要的参考价值。 

---
# SaudiCulture: A Benchmark for Evaluating Large Language Models Cultural Competence within Saudi Arabia 

**Title (ZH)**: 沙特文化：评估大型语言模型文化 competence 的基准_within Saudi Arabia 

**Authors**: Lama Ayash, Hassan Alhuzali, Ashwag Alasmari, Sultan Aloufi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17485)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in natural language processing; however, they often struggle to accurately capture and reflect cultural nuances. This research addresses this challenge by focusing on Saudi Arabia, a country characterized by diverse dialects and rich cultural traditions. We introduce SaudiCulture, a novel benchmark designed to evaluate the cultural competence of LLMs within the distinct geographical and cultural contexts of Saudi Arabia. SaudiCulture is a comprehensive dataset of questions covering five major geographical regions, such as West, East, South, North, and Center, along with general questions applicable across all regions. The dataset encompasses a broad spectrum of cultural domains, including food, clothing, entertainment, celebrations, and crafts. To ensure a rigorous evaluation, SaudiCulture includes questions of varying complexity, such as open-ended, single-choice, and multiple-choice formats, with some requiring multiple correct answers. Additionally, the dataset distinguishes between common cultural knowledge and specialized regional aspects. We conduct extensive evaluations on five LLMs, such as GPT-4, Llama 3.3, FANAR, Jais, and AceGPT, analyzing their performance across different question types and cultural contexts. Our findings reveal that all models experience significant performance declines when faced with highly specialized or region-specific questions, particularly those requiring multiple correct responses. Additionally, certain cultural categories are more easily identifiable than others, further highlighting inconsistencies in LLMs cultural understanding. These results emphasize the importance of incorporating region-specific knowledge into LLMs training to enhance their cultural competence. 

**Abstract (ZH)**: 大型语言模型在自然语言处理方面展现了卓越的能力；然而，它们在准确捕捉和反映文化细微差别方面常常力不从心。本研究针对沙特阿拉伯这一以多样方言和丰富文化传统为特点的国家，提出这一挑战。我们引入了沙特文化这一新的基准，旨在评估大型语言模型在沙特阿拉伯独特地理和文化背景下的文化能力。沙特文化是一个涵盖了五个主要地理区域（西、东、南、北、中心）以及适用于所有区域的一般问题的综合数据集。该数据集涉及广泛的文化领域，包括饮食、服饰、娱乐、庆祝活动和工艺品。为确保严格的评估，沙特文化包括不同复杂度的问题，如开放型、单选和多项选择题，有些甚至要求多个正确答案。此外，该数据集区分了通用文化知识与特定区域的专门方面。我们对五种大型语言模型（如GPT-4、Llama 3.3、FANAR、Jais、AceGPT）进行了广泛的评估，分析它们在不同问题类型和文化背景下的表现。我们的发现表明，面对高度专业化或特定地区的问题时，所有模型的表现都会显著下降，尤其是那些需要多个正确答案的问题。此外，某些文化类别比其他类别更容易识别，进一步突显了大型语言模型在文化理解方面的不一致性。这些结果强调了在训练大型语言模型时融入特定地区知识的重要性，以提高其文化能力。 

---
# Your voice is your voice: Supporting Self-expression through Speech Generation and LLMs in Augmented and Alternative Communication 

**Title (ZH)**: 你的声音就是你的声音：通过语音生成和大语言模型支持替代和辅助沟通中的自我表达 

**Authors**: Yiwen Xu, Monideep Chakraborti, Tianyi Zhang, Katelyn Eng, Aanchan Mohan, Mirjana Prpa  

**Link**: [PDF](https://arxiv.org/pdf/2503.17479)  

**Abstract**: In this paper, we present Speak Ease: an augmentative and alternative communication (AAC) system to support users' expressivity by integrating multimodal input, including text, voice, and contextual cues (conversational partner and emotional tone), with large language models (LLMs). Speak Ease combines automatic speech recognition (ASR), context-aware LLM-based outputs, and personalized text-to-speech technologies to enable more personalized, natural-sounding, and expressive communication. Through an exploratory feasibility study and focus group evaluation with speech and language pathologists (SLPs), we assessed Speak Ease's potential to enable expressivity in AAC. The findings highlight the priorities and needs of AAC users and the system's ability to enhance user expressivity by supporting more personalized and contextually relevant communication. This work provides insights into the use of multimodal inputs and LLM-driven features to improve AAC systems and support expressivity. 

**Abstract (ZH)**: Speak Ease：一种集成多模态输入和大规模语言模型的增强和替代沟通系统 

---
# Language-specific Neurons Do Not Facilitate Cross-Lingual Transfer 

**Title (ZH)**: 特定语言神经元不利于跨语言迁移 

**Authors**: Soumen Kumar Mondal, Sayambhu Sen, Abhishek Singhania, Preethi Jyothi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17456)  

**Abstract**: Multilingual large language models (LLMs) aim towards robust natural language understanding across diverse languages, yet their performance significantly degrades on low-resource languages. This work explores whether existing techniques to identify language-specific neurons can be leveraged to enhance cross-lingual task performance of lowresource languages. We conduct detailed experiments covering existing language-specific neuron identification techniques (such as Language Activation Probability Entropy and activation probability-based thresholding) and neuron-specific LoRA fine-tuning with models like Llama 3.1 and Mistral Nemo. We find that such neuron-specific interventions are insufficient to yield cross-lingual improvements on downstream tasks (XNLI, XQuAD) in lowresource languages. This study highlights the challenges in achieving cross-lingual generalization and provides critical insights for multilingual LLMs. 

**Abstract (ZH)**: 多语言大型语言模型（LLMs）旨在实现跨多种语言的稳健自然语言理解，但其在低资源语言上的性能显著下降。本研究探讨了现有技术是否可以利用识别语言特定神经元的方法来提升低资源语言跨语言任务的性能。我们进行了详细的实验，涵盖现有的语言特定神经元识别技术（如语言激活概率熵和激活概率阈值方法）以及针对如Llama 3.1和Mistral Nemo等模型的神经元特定LoRA微调。我们发现，在低资源语言的下游任务（XNLI、XQuAD）中，这样的神经元特定干预不足以带来跨语言性能的提升。本研究突出了实现跨语言泛化的挑战，并为多语言LLMs提供了关键见解。 

---
# LEMMA: Learning from Errors for MatheMatical Advancement in LLMs 

**Title (ZH)**: LEMMA: 从错误中学习以促进LLMs的数学进步 

**Authors**: Zhuoshi Pan, Yu Li, Honglin Lin, Qizhi Pei, Zinan Tang, Wei Wu, Chenlin Ming, H. Vicky Zhao, Conghui He, Lijun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.17439)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning capability in solving mathematical problems. However, existing approaches primarily focus on improving the quality of correct training data, e.g., distilling high-quality correct solutions from advanced models, neglecting the value contained in error data, potentially hindering the model's reflective ability. Though some studies attempt to leverage error data, they often involve complex mechanisms, such as Monte Carlo Tree Search (MCTS) to explore error nodes. In this work, we propose to enhance LLMs' reasoning ability by Learning from Errors for Mathematical Advancement (LEMMA). LEMMA constructs data consisting of an incorrect solution with an erroneous step and a reflection connection to a correct solution for fine-tuning. Specifically, we systematically analyze the model-generated error types and introduce an error-type grounded mistake augmentation method to collect diverse and representative errors. Correct solutions are either from fixing the errors or generating a fresh start. Through a model-aware smooth reflection connection, the erroneous solution is transferred to the correct one. By fine-tuning on the constructed dataset, the model is able to self-correct errors autonomously within the generation process without relying on external critique models. Experimental results demonstrate that LEMMA achieves significant performance improvements over other strong baselines. 

**Abstract (ZH)**: 基于错误学习的数学进步（LEMMA）增强大型语言模型的推理能力 

---
# Understanding Social Support Needs in Questions: A Hybrid Approach Integrating Semi-Supervised Learning and LLM-based Data Augmentation 

**Title (ZH)**: 基于半监督学习和基于LLM的数据增强的混合方法理解问题中的社会支持需求 

**Authors**: Junwei Kuang, Liang Yang, Shaoze Cui, Weiguo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17421)  

**Abstract**: Patients are increasingly turning to online health Q&A communities for social support to improve their well-being. However, when this support received does not align with their specific needs, it may prove ineffective or even detrimental. This necessitates a model capable of identifying the social support needs in questions. However, training such a model is challenging due to the scarcity and class imbalance issues of labeled data. To overcome these challenges, we follow the computational design science paradigm to develop a novel framework, Hybrid Approach for SOcial Support need classification (HA-SOS). HA-SOS integrates an answer-enhanced semi-supervised learning approach, a text data augmentation technique leveraging large language models (LLMs) with reliability- and diversity-aware sample selection mechanism, and a unified training process to automatically label social support needs in questions. Extensive empirical evaluations demonstrate that HA-SOS significantly outperforms existing question classification models and alternative semi-supervised learning approaches. This research contributes to the literature on social support, question classification, semi-supervised learning, and text data augmentation. In practice, our HA-SOS framework facilitates online Q&A platform managers and answerers to better understand users' social support needs, enabling them to provide timely, personalized answers and interventions. 

**Abstract (ZH)**: 患者越来越多地转向在线健康问答社区寻求社交支持以改善福祉。然而，当接收到的支持不符合其特定需求时，可能会无效甚至有害。这就需要一个能够识别问题中社交支持需求的模型。但由于标记数据稀缺且存在类别不平衡问题，训练此类模型颇具挑战性。为克服这些挑战，我们遵循计算设计科学范式，开发了一个新的框架——混合方法进行社会支持需求分类（HA-SOS）。HA-SOS结合了答案增强的半监督学习方法、利用大型语言模型（LLMs）的数据文本增强技术以及可靠性与多样意识样的样本选择机制，并采用统一的训练过程自动标注问题中的社会支持需求。广泛的实证评估表明，HA-SOS在社会支持需求分类和半监督学习方面显著优于现有模型和替代方法。本研究为社会支持、问题分类、半监督学习和文本数据增强等领域文献做出了贡献。在实践中，我们的HA-SOS框架有助于在线问答平台管理者和回答者更好地理解用户的社会支持需求，使他们能够提供及时、个性化的回答和干预。 

---
# ChatGPT or A Silent Everywhere Helper: A Survey of Large Language Models 

**Title (ZH)**: ChatGPT 或者无处不在的沉默助手：大规模语言模型综述 

**Authors**: Azim Akhtarshenas, Afshin Dini, Navid Ayoobi  

**Link**: [PDF](https://arxiv.org/pdf/2503.17403)  

**Abstract**: Large Language Models (LLMs) have revo lutionized natural language processing Natural Language Processing (NLP), with Chat Generative Pre-trained Transformer (ChatGPT) standing out as a notable exampledue to its advanced capabilities and widespread applications. This survey provides a comprehensive analysis of ChatGPT, exploring its architecture, training processes, and functionalities. We examine its integration into various domains across industries such as customer service, education, healthcare, and entertainment. A comparative analysis with other LLMs highlights ChatGPT's unique features and performance metrics. Regarding benchmarks, the paper examines ChatGPT's comparative performance against other LLMs and discusses potential risks such as misinformation, bias, and data privacy concerns. Additionally, we offer a number of figures and tables that outline the backdrop of the discussion, the main ideas of the article, the numerous LLM models, a thorough list of datasets used for pre-training, fine-tuning, and evaluation, as well as particular LLM applications with pertinent references. Finally, we identify future research directions and technological advancements, underscoring the evolving landscape of LLMs and their profound impact on artificial intelligence Artificial Intelligence (AI) and society. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已从根本上革新了自然语言处理（NLP），其中Chat生成预训练变换器（ChatGPT）因其先进的能力和广泛的应用而格外突出。本文对该领域的综述进行深入分析，探讨了ChatGPT的架构、训练过程和功能。我们研究了其在包括客户服务、教育、医疗保健和娱乐在内的各个行业的广泛应用。与其他大规模语言模型的对比分析突显了ChatGPT的独特特性和性能指标。对于基准测试，本文考察了ChatGPT与其他大规模语言模型的相对性能，并讨论了潜在的风险，如错误信息、偏见和数据隐私问题。此外，我们提供了许多图表，概述了讨论的背景、文章的主要思想、各种大规模语言模型列表、用于预训练、微调和评估的数据集详单，以及特定的大规模语言模型应用及其相关参考文献。最后，我们指出了未来的研究方向和技术进步，强调了大规模语言模型不断演变的格局及其对人工智能（AI）和社会的深远影响。 

---
# How Effective Is Constitutional AI in Small LLMs? A Study on DeepSeek-R1 and Its Peers 

**Title (ZH)**: 宪法AI在小型LLM中的有效性：DeepSeek-R1及其同行的研究 

**Authors**: Antonio-Gabriel Chacón Menke, Phan Xuan Tan  

**Link**: [PDF](https://arxiv.org/pdf/2503.17365)  

**Abstract**: Recent incidents highlight safety risks in Large Language Models (LLMs), motivating research into alignment methods like Constitutional AI (CAI). This paper explores CAI's self-critique mechanism on small, uncensored 7-9B parameter models: DeepSeek-R1, Gemma-2, Llama 3.1, and Qwen2.5. Using HarmBench, we demonstrate that while all models showed capacity for harm reduction through self-critique, effectiveness varied significantly, with DeepSeek-R1's explicit reasoning process yielding superior results. These findings suggest that CAI-inspired prompting strategies can enhance safety in resource-constrained models, though success depends on the model's capacity for harm detection. 

**Abstract (ZH)**: 近期事件突显了大型语言模型（LLMs）的安全风险，促使研究如宪法AI（CAI）这样的对齐方法。本文探讨了CAI在对小规模、未受限制的7-9B参数模型（DeepSeek-R1、Gemma-2、Llama 3.1和Qwen2.5）进行自我批判机制的研究：利用HarmBench，我们证明尽管所有模型都表现出通过自我批判减少危害的能力，但效果差异显著，DeepSeek-R1的明确推理过程产生了更优的结果。这些发现表明，受CAI启发的提示策略可以增强资源受限模型的安全性，尽管成功取决于模型对危害的检测能力。 

---
