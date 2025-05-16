# Towards a Deeper Understanding of Reasoning Capabilities in Large Language Models 

**Title (ZH)**: 向对大型语言模型推理能力的深入理解迈进 

**Authors**: Annie Wong, Thomas Bäck, Aske Plaat, Niki van Stein, Anna V. Kononova  

**Link**: [PDF](https://arxiv.org/pdf/2505.10543)  

**Abstract**: While large language models demonstrate impressive performance on static benchmarks, the true potential of large language models as self-learning and reasoning agents in dynamic environments remains unclear. This study systematically evaluates the efficacy of self-reflection, heuristic mutation, and planning as prompting techniques to test the adaptive capabilities of agents. We conduct experiments with various open-source language models in dynamic environments and find that larger models generally outperform smaller ones, but that strategic prompting can close this performance gap. Second, a too-long prompt can negatively impact smaller models on basic reactive tasks, while larger models show more robust behaviour. Third, advanced prompting techniques primarily benefit smaller models on complex games, but offer less improvement for already high-performing large language models. Yet, we find that advanced reasoning methods yield highly variable outcomes: while capable of significantly improving performance when reasoning and decision-making align, they also introduce instability and can lead to big performance drops. Compared to human performance, our findings reveal little evidence of true emergent reasoning. Instead, large language model performance exhibits persistent limitations in crucial areas such as planning, reasoning, and spatial coordination, suggesting that current-generation large language models still suffer fundamental shortcomings that may not be fully overcome through self-reflective prompting alone. Reasoning is a multi-faceted task, and while reasoning methods like Chain of thought improves multi-step reasoning on math word problems, our findings using dynamic benchmarks highlight important shortcomings in general reasoning capabilities, indicating a need to move beyond static benchmarks to capture the complexity of reasoning. 

**Abstract (ZH)**: 大语言模型在动态环境中的自学习与推理潜力探究：自省、启发式变异与规划的效用系统评估 

---
# Empirically evaluating commonsense intelligence in large language models with large-scale human judgments 

**Title (ZH)**: 大规模人工判断 empirically 评估大型语言模型的常识智能 

**Authors**: Tuan Dung Nguyen, Duncan J. Watts, Mark E. Whiting  

**Link**: [PDF](https://arxiv.org/pdf/2505.10309)  

**Abstract**: Commonsense intelligence in machines is often assessed by static benchmarks that compare a model's output against human-prescribed correct labels. An important, albeit implicit, assumption of these labels is that they accurately capture what any human would think, effectively treating human common sense as homogeneous. However, recent empirical work has shown that humans vary enormously in what they consider commonsensical; thus what appears self-evident to one benchmark designer may not be so to another. Here, we propose a novel method for evaluating common sense in artificial intelligence (AI), specifically in large language models (LLMs), that incorporates empirically observed heterogeneity among humans by measuring the correspondence between a model's judgment and that of a human population. We first find that, when treated as independent survey respondents, most LLMs remain below the human median in their individual commonsense competence. Second, when used as simulators of a hypothetical population, LLMs correlate with real humans only modestly in the extent to which they agree on the same set of statements. In both cases, smaller, open-weight models are surprisingly more competitive than larger, proprietary frontier models. Our evaluation framework, which ties commonsense intelligence to its cultural basis, contributes to the growing call for adapting AI models to human collectivities that possess different, often incompatible, social stocks of knowledge. 

**Abstract (ZH)**: 机器中的常识智能通常通过静态基准来评估，这些基准将模型的输出与人类规定的正确标签进行比较。这些标签隐含地假设它们准确捕捉了任何人类的思维方式，从而将人类的常识视为同质的。然而，最近的实证研究显示，人类在认为什么是常识方面存在巨大差异；因此，对一个基准设计者来说显而易见的常识可能对另一个设计者来说并非如此。在这里，我们提出了一种评估人工智能（AI）中的常识的新方法，特别是在大型语言模型（LLMs）中的方法，该方法通过测量模型判断与人类群体的一致性来纳入观察到的人类异质性。我们首先发现，当被视为独立的调查受访者时，大多数LLMs在其个体常识能力方面仍然低于人类中位数。其次，当作为假设群体的模拟器时，LLMs在同意一组声明的程度上与真实人类的相关性仅中等。在两种情况下，较小的、开放权重模型比较大的、专有的前沿模型出人意料地更具竞争力。我们评分框架将常识智能与其文化基础联系起来，有助于适应具备不同且往往不兼容的社会知识储备的人类群体的AI模型的呼声。 

---
# Leveraging Graph Retrieval-Augmented Generation to Support Learners' Understanding of Knowledge Concepts in MOOCs 

**Title (ZH)**: 利用图检索增强生成支持MOOC learners的知识概念理解 

**Authors**: Mohamed Abdelmagied, Mohamed Amine Chatti, Shoeb Joarder, Qurat Ul Ain, Rawaa Alatrash  

**Link**: [PDF](https://arxiv.org/pdf/2505.10074)  

**Abstract**: Massive Open Online Courses (MOOCs) lack direct interaction between learners and instructors, making it challenging for learners to understand new knowledge concepts. Recently, learners have increasingly used Large Language Models (LLMs) to support them in acquiring new knowledge. However, LLMs are prone to hallucinations which limits their reliability. Retrieval-Augmented Generation (RAG) addresses this issue by retrieving relevant documents before generating a response. However, the application of RAG across different MOOCs is limited by unstructured learning material. Furthermore, current RAG systems do not actively guide learners toward their learning needs. To address these challenges, we propose a Graph RAG pipeline that leverages Educational Knowledge Graphs (EduKGs) and Personal Knowledge Graphs (PKGs) to guide learners to understand knowledge concepts in the MOOC platform CourseMapper. Specifically, we implement (1) a PKG-based Question Generation method to recommend personalized questions for learners in context, and (2) an EduKG-based Question Answering method that leverages the relationships between knowledge concepts in the EduKG to answer learner selected questions. To evaluate both methods, we conducted a study with 3 expert instructors on 3 different MOOCs in the MOOC platform CourseMapper. The results of the evaluation show the potential of Graph RAG to empower learners to understand new knowledge concepts in a personalized learning experience. 

**Abstract (ZH)**: 大规模开放在线课程（MOOCs）缺乏学员与讲师之间的直接互动，导致学员理解新知识概念存在挑战。最近，学员越来越多地使用大型语言模型（LLMs）来支持他们获取新知识。然而，LLMs容易产生幻觉，这限制了其可靠性。检索增强生成（RAG）通过在生成响应前检索相关文档来解决这一问题。然而，RAG在不同MOOC中的应用受到非结构化学习材料的限制。此外，当前的RAG系统没有主动引导学员满足他们的学习需求。为应对这些挑战，我们提出了一个基于图的RAG流水线，该流水线利用教育知识图（EduKG）和个人知识图（PKG）来引导学员在CourseMapper MOOC平台上理解知识概念。具体而言，我们实现了（1）基于PKG的问题生成方法，为学员提供个性化的问题建议，以及（2）基于EduKG的问题回答方法，利用EduKG中知识概念之间的关系来解答学员选定的问题。为了评估这两种方法，我们在MOOC平台CourseMapper上选择了3位专家教师对3门不同的MOOC进行了研究。评估结果表明，基于图的RAG有潜力为学员提供一种个性化的学习体验，帮助他们理解新知识概念。 

---
# Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents 

**Title (ZH)**: 预操作：多步规划与推理提高LLM代理的执行能力 

**Authors**: Mrinal Rawat, Ambuje Gupta, Rushil Goomer, Alessandro Di Bari, Neha Gupta, Roberto Pieraccini  

**Link**: [PDF](https://arxiv.org/pdf/2505.09970)  

**Abstract**: The ReAct (Reasoning + Action) capability in large language models (LLMs) has become the foundation of modern agentic systems. Recent LLMs, such as DeepSeek-R1 and OpenAI o1/o3, exemplify this by emphasizing reasoning through the generation of ample intermediate tokens, which help build a strong premise before producing the final output tokens. In this paper, we introduce Pre-Act, a novel approach that enhances the agent's performance by creating a multi-step execution plan along with the detailed reasoning for the given user input. This plan incrementally incorporates previous steps and tool outputs, refining itself after each step execution until the final response is obtained. Our approach is applicable to both conversational and non-conversational agents. To measure the performance of task-oriented agents comprehensively, we propose a two-level evaluation framework: (1) turn level and (2) end-to-end. Our turn-level evaluation, averaged across five models, shows that our approach, Pre-Act, outperforms ReAct by 70% in Action Recall on the Almita dataset. While this approach is effective for larger models, smaller models crucial for practical applications, where latency and cost are key constraints, often struggle with complex reasoning tasks required for agentic systems. To address this limitation, we fine-tune relatively small models such as Llama 3.1 (8B & 70B) using the proposed Pre-Act approach. Our experiments show that the fine-tuned 70B model outperforms GPT-4, achieving a 69.5% improvement in action accuracy (turn-level) and a 28% improvement in goal completion rate (end-to-end) on the Almita (out-of-domain) dataset. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的ReAct（推理+行动）能力已成为现代自主系统的基础。最近的LLMs，如DeepSeek-R1和OpenAI o1/o3，通过生成大量的中间tokens强调推理，这有助于在生成最终输出tokens之前建立一个强大的前提。本文提出了Pre-Act，一种新颖的方法，通过为给定用户输入创建多步执行计划及其详细推理来增强代理的性能。该计划逐步整合了之前的步骤和工具输出，在每一步执行后逐步完善，直至最终响应生成。我们的方法适用于both会话型和非会话型代理。为全面衡量任务导向代理的表现，我们提出了一个两级评估框架：（1）回合级和（2）端到端。通过五个模型的平均结果，我们的Pre-Act方法在Almita数据集上的Action Recall上比ReAct高出70%。虽然这种方法对大型模型有效，但对实战应用中至关重要的小型模型，由于延迟和成本的关键限制，往往难以完成需要自主系统进行的复杂推理任务。为解决这一局限，我们使用提出的Pre-Act方法对相对较小的模型Llama 3.1（8B和70B）进行微调。实验结果表明，微调后的70B模型在Almita（跨域）数据集上的动作准确性提高了69.5%，在端到端目标完成率上提高了28%。 

---
# Neural Thermodynamic Laws for Large Language Model Training 

**Title (ZH)**: 大型语言模型训练的神经热力学定律 

**Authors**: Ziming Liu, Yizhou Liu, Jeff Gore, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2505.10559)  

**Abstract**: Beyond neural scaling laws, little is known about the laws underlying large language models (LLMs). We introduce Neural Thermodynamic Laws (NTL) -- a new framework that offers fresh insights into LLM training dynamics. On the theoretical side, we demonstrate that key thermodynamic quantities (e.g., temperature, entropy, heat capacity, thermal conduction) and classical thermodynamic principles (e.g., the three laws of thermodynamics and the equipartition theorem) naturally emerge under river-valley loss landscape assumptions. On the practical side, this scientific perspective yields intuitive guidelines for designing learning rate schedules. 

**Abstract (ZH)**: 超越神经网络缩放律，关于大型语言模型的内在规律知之甚少。我们引入了神经热力学定律（NTL）——一种新的框架，为大型语言模型的训练动态提供了新的见解。从理论角度来看，我们在河谷损失景观假设下证明了关键的热力学量（如温度、熵、比热、热传导）和经典热力学原理（如热力学三大定律和等概原理）自然地出现。从实践角度来看，这种科学视角提供了设计学习率调度的直观指南。 

---
# Multi-Token Prediction Needs Registers 

**Title (ZH)**: 多令牌预测需要寄存器 

**Authors**: Anastasios Gerontopoulos, Spyros Gidaris, Nikos Komodakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.10518)  

**Abstract**: Multi-token prediction has emerged as a promising objective for improving language model pretraining, but its benefits have not consistently generalized to other settings such as fine-tuning. In this paper, we propose MuToR, a simple and effective approach to multi-token prediction that interleaves learnable register tokens into the input sequence, each tasked with predicting future targets. Compared to existing methods, MuToR offers several key advantages: it introduces only a negligible number of additional parameters, requires no architectural changes--ensuring compatibility with off-the-shelf pretrained language models--and remains aligned with the next-token pretraining objective, making it especially well-suited for supervised fine-tuning. Moreover, it naturally supports scalable prediction horizons. We demonstrate the effectiveness and versatility of MuToR across a range of use cases, including supervised fine-tuning, parameter-efficient fine-tuning (PEFT), and pretraining, on challenging generative tasks in both language and vision domains. Our code will be available at: this https URL. 

**Abstract (ZH)**: 基于多令牌预测的MuToR方法：一种简单有效的预训练新方法 

---
# Superposition Yields Robust Neural Scaling 

**Title (ZH)**: 叠加效应赋予神经网络规模化稳健性 

**Authors**: Yizhou liu, Ziming Liu, Jeff Gore  

**Link**: [PDF](https://arxiv.org/pdf/2505.10465)  

**Abstract**: The success of today's large language models (LLMs) depends on the observation that larger models perform better. However, the origin of this neural scaling law -- the finding that loss decreases as a power law with model size -- remains unclear. Starting from two empirical principles -- that LLMs represent more things than the model dimensions (widths) they have (i.e., representations are superposed), and that words or concepts in language occur with varying frequencies -- we constructed a toy model to study the loss scaling with model size. We found that when superposition is weak, meaning only the most frequent features are represented without interference, the scaling of loss with model size depends on the underlying feature frequency; if feature frequencies follow a power law, so does the loss. In contrast, under strong superposition, where all features are represented but overlap with each other, the loss becomes inversely proportional to the model dimension across a wide range of feature frequency distributions. This robust scaling behavior is explained geometrically: when many more vectors are packed into a lower dimensional space, the interference (squared overlaps) between vectors scales inversely with that dimension. We then analyzed four families of open-sourced LLMs and found that they exhibit strong superposition and quantitatively match the predictions of our toy model. The Chinchilla scaling law turned out to also agree with our results. We conclude that representation superposition is an important mechanism underlying the observed neural scaling laws. We anticipate that these insights will inspire new training strategies and model architectures to achieve better performance with less computation and fewer parameters. 

**Abstract (ZH)**: 今天大型语言模型的成功取决于观测到的较大模型性能更好的现象，但这一神经网络缩放定律（即损失随着模型大小以幂律形式减少的发现）的起源仍不清楚。基于两个经验原则——语言模型表示的东西比它们所具有的模型维度（宽度）多（即表示是叠加的），以及语言中的词或概念以不同的频率出现，我们构建了一个玩具模型来研究损失随模型大小的缩放。我们发现，当叠加较弱时，即只有最频繁的特征被表示而没有干扰，损失随模型大小的缩放取决于底层特征频率；如果特征频率呈幂律分布，损失也是如此。相反，在叠加较强时，即所有特征都被表示但互相重叠，损失在整个特征频率分布范围内与模型维度成反比。这种稳健的缩放行为从几何上得到了解释：当将更多向量压缩到低维空间时，向量之间的干扰（平方重叠）与该维度成反比。然后，我们分析了四个开源语言模型家族，并发现它们表现出强烈的叠加，并且定量化地符合我们玩具模型的预测。Chinchilla缩放定律也与我们的结果一致。我们得出结论，表示的叠加是观察到的神经网络缩放定律背后的重要机制。我们预计这些见解将启发新的训练策略和模型架构，以实现更好的性能和更少的计算量和参数。 

---
# Are Large Language Models Robust in Understanding Code Against Semantics-Preserving Mutations? 

**Title (ZH)**: 大规模语言模型在理解代码时对抗语义保留的变异具有鲁棒性吗？ 

**Authors**: Pedro Orvalho, Marta Kwiatkowska  

**Link**: [PDF](https://arxiv.org/pdf/2505.10443)  

**Abstract**: Understanding the reasoning and robustness of Large Language Models (LLMs) is critical for their reliable use in programming tasks. While recent studies have assessed LLMs' ability to predict program outputs, most focus solely on the accuracy of those predictions, without evaluating the reasoning behind them. Moreover, it has been observed on mathematical reasoning tasks that LLMs can arrive at correct answers through flawed logic, raising concerns about similar issues in code understanding.
In this work, we evaluate whether state-of-the-art LLMs with up to 8B parameters can reason about Python programs or are simply guessing. We apply five semantics-preserving code mutations: renaming variables, mirroring comparison expressions, swapping if-else branches, converting for loops to while, and loop unrolling. These mutations maintain program semantics while altering its syntax. We evaluated six LLMs and performed a human expert analysis using LiveCodeBench to assess whether the correct predictions are based on sound reasoning. We also evaluated prediction stability across different code mutations on LiveCodeBench and CruxEval. Our findings show that some LLMs, such as Llama3.2, produce correct predictions based on flawed reasoning in up to 61% of cases. Furthermore, LLMs often change predictions in response to our code mutations, indicating limited robustness in their semantic understanding. 

**Abstract (ZH)**: 理解大型语言模型（LLMs）的推理和鲁棒性对于它们在编程任务中的可靠使用至关重要。尽管 recent 研究评估了 LLMs 预测程序输出的能力，但大多数研究仅关注这些预测的准确性，而未评估其背后的推理过程。此外，在数学推理任务中观察到，LLM 可能通过错误的逻辑得出正确答案，这引起了人们对代码理解中类似问题的担忧。

在本工作中，我们评估最先进的具有 8B 参数的 LLM 是否能够合理地推理关于 Python 程序，而不仅仅是猜测。我们应用了五种语义保持的代码变异：重命名变量、镜像比较表达式、交换 if-else 分支、将 for 循环转换为 while 循环以及循环展开。这些变异保持程序语义同时改变其语法。我们评估了六种 LLM，并使用 LiveCodeBench 进行人工专家分析，以评估正确的预测是否基于合理的推理。我们还在 LiveCodeBench 和 CruxEval 上评估了不同代码变异预测的稳定性。我们的研究发现，某些 LLM，如 Llama3.2，在多达 61% 的情况下基于错误的推理生成正确的预测。此外，LLM 对我们代码变异的预测响应变化，表明它们在语义理解上的鲁棒性有限。 

---
# Rethinking Repetition Problems of LLMs in Code Generation 

**Title (ZH)**: 重新思考代码生成中LLMs的重复问题 

**Authors**: Yihong Dong, Yuchen Liu, Xue Jiang, Zhi Jin, Ge Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.10402)  

**Abstract**: With the advent of neural language models, the performance of code generation has been significantly boosted. However, the problem of repetitions during the generation process continues to linger. Previous work has primarily focused on content repetition, which is merely a fraction of the broader repetition problem in code generation. A more prevalent and challenging problem is structural repetition. In structural repetition, the repeated code appears in various patterns but possesses a fixed structure, which can be inherently reflected in grammar. In this paper, we formally define structural repetition and propose an efficient decoding approach called RPG, which stands for Repetition Penalization based on Grammar, to alleviate the repetition problems in code generation for LLMs. Specifically, RPG first leverages grammar rules to identify repetition problems during code generation, and then strategically decays the likelihood of critical tokens that contribute to repetitions, thereby mitigating them in code generation. To facilitate this study, we construct a new dataset CodeRepetEval to comprehensively evaluate approaches for mitigating the repetition problems in code generation. Extensive experimental results demonstrate that RPG substantially outperforms the best-performing baselines on CodeRepetEval dataset as well as HumanEval and MBPP benchmarks, effectively reducing repetitions and enhancing the quality of generated code. 

**Abstract (ZH)**: 随着神经语言模型的发展，代码生成的性能得到了显著提升。然而，生成过程中的重复问题仍然存在。以往研究主要关注内容重复，这只是代码生成中更广泛重复问题的一小部分。一个更为普遍且具有挑战性的问题是结构重复。在结构重复中，重复的代码以各种模式出现，但具有固定的结构，这种结构可以内在地反映在语法规则中。本文正式定义结构重复，并提出了一种基于语法的重复惩罚解码方法RPG（Repetition Penalization based on Grammar）来缓解大语言模型在代码生成中的重复问题。具体而言，RPG 首先利用语法规则在代码生成过程中识别重复问题，然后战略性地降低导致重复的关键标记的出现概率，从而减轻代码生成中的重复现象。为了促进这项研究，我们构建了一个新的数据集CodeRepetEval，以全面评估缓解代码生成中重复问题的方法。广泛的实验结果表明，RPG 在CodeRepetEval 数据集以及HumanEval 和 MBPP 基准测试中均显著优于基准方法，有效减少了重复，并提高了生成代码的质量。 

---
# Are Sparse Autoencoders Useful for Java Function Bug Detection? 

**Title (ZH)**: 稀疏自编码器对Java函数bug检测有用吗？ 

**Authors**: Rui Melo, Claudia Mamede, Andre Catarino, Rui Abreu, Henrique Lopes Cardoso  

**Link**: [PDF](https://arxiv.org/pdf/2505.10375)  

**Abstract**: Software vulnerabilities such as buffer overflows and SQL injections are a major source of security breaches. Traditional methods for vulnerability detection remain essential but are limited by high false positive rates, scalability issues, and reliance on manual effort. These constraints have driven interest in AI-based approaches to automated vulnerability detection and secure code generation. While Large Language Models (LLMs) have opened new avenues for classification tasks, their complexity and opacity pose challenges for interpretability and deployment. Sparse Autoencoder offer a promising solution to this problem. We explore whether SAEs can serve as a lightweight, interpretable alternative for bug detection in Java functions. We evaluate the effectiveness of SAEs when applied to representations from GPT-2 Small and Gemma 2B, examining their capacity to highlight buggy behaviour without fine-tuning the underlying LLMs. We found that SAE-derived features enable bug detection with an F1 score of up to 89%, consistently outperforming fine-tuned transformer encoder baselines. Our work provides the first empirical evidence that SAEs can be used to detect software bugs directly from the internal representations of pretrained LLMs, without any fine-tuning or task-specific supervision. 

**Abstract (ZH)**: 基于稀疏自编码器的Java函数漏洞检测研究 

---
# AutoPentest: Enhancing Vulnerability Management With Autonomous LLM Agents 

**Title (ZH)**: AutoPentest：通过自主LLM代理增强漏洞管理 

**Authors**: Julius Henke  

**Link**: [PDF](https://arxiv.org/pdf/2505.10321)  

**Abstract**: A recent area of increasing research is the use of Large Language Models (LLMs) in penetration testing, which promises to reduce costs and thus allow for higher frequency. We conduct a review of related work, identifying best practices and common evaluation issues. We then present AutoPentest, an application for performing black-box penetration tests with a high degree of autonomy. AutoPentest is based on the LLM GPT-4o from OpenAI and the LLM agent framework LangChain. It can perform complex multi-step tasks, augmented by external tools and knowledge bases. We conduct a study on three capture-the-flag style Hack The Box (HTB) machines, comparing our implementation AutoPentest with the baseline approach of manually using the ChatGPT-4o user interface. Both approaches are able to complete 15-25 % of the subtasks on the HTB machines, with AutoPentest slightly outperforming ChatGPT. We measure a total cost of \$96.20 US when using AutoPentest across all experiments, while a one-month subscription to ChatGPT Plus costs \$20. The results show that further implementation efforts and the use of more powerful LLMs released in the future are likely to make this a viable part of vulnerability management. 

**Abstract (ZH)**: 近年来，大规模语言模型在渗透测试中的应用成为一个研究热点，有望降低成本，从而提高测试频率。我们对相关研究进行了综述，总结了最佳实践和常见的评估问题。随后，我们介绍了一个名为AutoPentest的应用程序，该程序能够在较高自主程度下执行黑盒渗透测试。AutoPentest基于OpenAI的GPT-4o和LangChain的大规模语言模型代理框架，能够执行复杂的多步任务，并借助外部工具和知识库。我们通过对比Hack The Box (HTB)平台上的三种夺旗风格机器，在AutoPentest与手动使用ChatGPT-4o用户界面的基线方法之间进行了研究。两种方法均能完成15-25%的子任务，AutoPentest略胜一筹。使用AutoPentest进行所有实验的总成本为96.20美元，而一个月的ChatGPT Plus订阅费用为20美元。研究结果表明，进一步的实现努力和未来更强大大规模语言模型的应用预计将使这一方法成为漏洞管理的一个可行部分。 

---
# J1: Incentivizing Thinking in LLM-as-a-Judge via Reinforcement Learning 

**Title (ZH)**: 基于强化学习激励LLM作为法官进行思考 

**Authors**: Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, Swarnadeep Saha  

**Link**: [PDF](https://arxiv.org/pdf/2505.10320)  

**Abstract**: The progress of AI is bottlenecked by the quality of evaluation, and powerful LLM-as-a-Judge models have proved to be a core solution. Improved judgment ability is enabled by stronger chain-of-thought reasoning, motivating the need to find the best recipes for training such models to think. In this work we introduce J1, a reinforcement learning approach to training such models. Our method converts both verifiable and non-verifiable prompts to judgment tasks with verifiable rewards that incentivize thinking and mitigate judgment bias. In particular, our approach outperforms all other existing 8B or 70B models when trained at those sizes, including models distilled from DeepSeek-R1. J1 also outperforms o1-mini, and even R1 on some benchmarks, despite training a smaller model. We provide analysis and ablations comparing Pairwise-J1 vs Pointwise-J1 models, offline vs online training recipes, reward strategies, seed prompts, and variations in thought length and content. We find that our models make better judgments by learning to outline evaluation criteria, comparing against self-generated reference answers, and re-evaluating the correctness of model responses. 

**Abstract (ZH)**: AI进展受评估质量瓶颈制约，强大的LLM-as-a-Judge模型已被证明是核心解决方案。增强的判断能力得益于更强的链式思维推理，促使我们需要找到训练此类模型思考的最佳方法。在本工作中，我们引入了J1，一种强化学习方法来训练此类模型。我们的方法将验证性和非验证性提示转化为具有验证性奖励的判断任务，激励思考并减轻判断偏见。特别是，当训练到这些规模时，我们的方法在所有现有的8B或70B模型中表现最佳，包括从DeepSeek-R1蒸馏而来的模型。J1在某些基准测试中甚至优于o1-mini和R1，尽管训练了一个较小的模型。我们提供了关于Pairwise-J1模型与Pointwise-J1模型、离线与在线训练食谱、奖励策略、种子提示以及思维长度和内容变化的分析和消融研究。我们发现，通过学习制定评估标准、与自动生成的参考答案进行比较以及重新评估模型响应的正确性，我们的模型能够做出更好的判断。 

---
# The Evolving Landscape of Generative Large Language Models and Traditional Natural Language Processing in Medicine 

**Title (ZH)**: 生成式大型语言模型与传统自然语言处理在医学领域的演变 landscape 

**Authors**: Rui Yang, Huitao Li, Matthew Yu Heng Wong, Yuhe Ke, Xin Li, Kunyu Yu, Jingchi Liao, Jonathan Chong Kai Liew, Sabarinath Vinod Nair, Jasmine Chiat Ling Ong, Irene Li, Douglas Teodoro, Chuan Hong, Daniel Shu Wei Ting, Nan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.10261)  

**Abstract**: Natural language processing (NLP) has been traditionally applied to medicine, and generative large language models (LLMs) have become prominent recently. However, the differences between them across different medical tasks remain underexplored. We analyzed 19,123 studies, finding that generative LLMs demonstrate advantages in open-ended tasks, while traditional NLP dominates in information extraction and analysis tasks. As these technologies advance, ethical use of them is essential to ensure their potential in medical applications. 

**Abstract (ZH)**: 自然语言处理（NLP）在医学领域的应用传统上占据主导地位，而生成型大型语言模型（LLMs） recently崭露头角。然而，它们在不同医学任务中的差异仍需进一步探索。我们分析了19,123项研究，发现生成型LLMs在开放型任务中表现出优势，而传统NLP在信息提取和分析任务中占主导地位。随着这些技术的进步，确保它们在医学应用中的潜力使用它们是至关重要的，伦理使用尤为关键。 

---
# Comparing LLM Text Annotation Skills: A Study on Human Rights Violations in Social Media Data 

**Title (ZH)**: 比较大型语言模型的文本注释能力：社交媒体数据中人权侵犯行为的标注研究 

**Authors**: Poli Apollinaire Nemkova, Solomon Ubani, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.10260)  

**Abstract**: In the era of increasingly sophisticated natural language processing (NLP) systems, large language models (LLMs) have demonstrated remarkable potential for diverse applications, including tasks requiring nuanced textual understanding and contextual reasoning. This study investigates the capabilities of multiple state-of-the-art LLMs - GPT-3.5, GPT-4, LLAMA3, Mistral 7B, and Claude-2 - for zero-shot and few-shot annotation of a complex textual dataset comprising social media posts in Russian and Ukrainian. Specifically, the focus is on the binary classification task of identifying references to human rights violations within the dataset.
To evaluate the effectiveness of these models, their annotations are compared against a gold standard set of human double-annotated labels across 1000 samples. The analysis includes assessing annotation performance under different prompting conditions, with prompts provided in both English and Russian. Additionally, the study explores the unique patterns of errors and disagreements exhibited by each model, offering insights into their strengths, limitations, and cross-linguistic adaptability.
By juxtaposing LLM outputs with human annotations, this research contributes to understanding the reliability and applicability of LLMs for sensitive, domain-specific tasks in multilingual contexts. It also sheds light on how language models handle inherently subjective and context-dependent judgments, a critical consideration for their deployment in real-world scenarios. 

**Abstract (ZH)**: 在自然语言处理系统日益复杂的时代，大型语言模型（LLMs）在多种应用场景中展现了显著潜力，包括需要细腻文本理解和情境推理的任务。本研究调查了多种先进大型语言模型——GPT-3.5、GPT-4、LLAMA3、Mistral 7B和Claude-2——在复杂文本数据集上的零样本和少样本注释能力，该数据集包含俄语和乌克兰语的社交媒体帖子。具体而言，研究重点在于识别数据集中关于人权侵犯的参考信息的二分类任务。

为了评估这些模型的效果，将模型的注释与其在1000个样本上的人类双注标注的黄金标准进行比较。分析包括在不同提示条件下评估注释性能，提示以英语和俄语提供。此外，研究还探讨了每个模型独有的错误和分歧模式，提供了对其优点、局限性和跨语言适应性的见解。

通过将LLM输出与人类注释进行对比，本研究为理解LLMs在多语言背景下进行敏感的主题特定任务的可靠性和适用性做出了贡献。研究还揭示了语言模型如何处理固有的主观性和情境依赖性判断，这对于它们在实际场景中的部署至关重要。 

---
# Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M 

**Title (ZH)**: LLMs是否 Memorize 推荐数据集？对 MovieLens-1M 的初步研究 

**Authors**: Dario Di Palma, Felice Antonio Merra, Maurizio Sfilio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.10212)  

**Abstract**: Large Language Models (LLMs) have become increasingly central to recommendation scenarios due to their remarkable natural language understanding and generation capabilities. Although significant research has explored the use of LLMs for various recommendation tasks, little effort has been dedicated to verifying whether they have memorized public recommendation dataset as part of their training data. This is undesirable because memorization reduces the generalizability of research findings, as benchmarking on memorized datasets does not guarantee generalization to unseen datasets. Furthermore, memorization can amplify biases, for example, some popular items may be recommended more frequently than others.
In this work, we investigate whether LLMs have memorized public recommendation datasets. Specifically, we examine two model families (GPT and Llama) across multiple sizes, focusing on one of the most widely used dataset in recommender systems: MovieLens-1M. First, we define dataset memorization as the extent to which item attributes, user profiles, and user-item interactions can be retrieved by prompting the LLMs. Second, we analyze the impact of memorization on recommendation performance. Lastly, we examine whether memorization varies across model families and model sizes. Our results reveal that all models exhibit some degree of memorization of MovieLens-1M, and that recommendation performance is related to the extent of memorization. We have made all the code publicly available at: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）在推荐场景中因其出色的自然语言理解与生成能力而变得日益重要。尽管已经进行了大量研究探索LLMs在各种推荐任务中的应用，但很少有研究关注它们是否在其训练数据中记住了公开的推荐数据集。这种情况是不理想的，因为记忆性的存在会降低研究发现的普适性，因为在记住了的数据集上的基准测试并不能保证在未见数据集上的泛化。此外，记忆性会放大偏差，例如，一些流行项目比其他项目更容易被推荐。

在本工作中，我们探究LLMs是否记住了公开的推荐数据集。具体来说，我们考察了两个模型家族（GPT和Llama）的不同规模，并重点关注推荐系统中最广泛使用的数据集之一：MovieLens-1M。首先，我们将数据集记忆性定义为通过提示LLMs可以检索到的项目属性、用户特征和用户-项目交互的程度。其次，我们分析记忆性对推荐性能的影响。最后，我们考察不同模型家族和模型规模下的记忆性差异。我们的结果显示，所有模型在不同程度上都记住了MovieLens-1M数据集，并且推荐性能与记忆性的程度相关。我们已将所有代码公开发布在：this https URL 

---
# Dark LLMs: The Growing Threat of Unaligned AI Models 

**Title (ZH)**: 暗LSTM：未对齐AI模型日益增长的威胁 

**Authors**: Michael Fire, Yitzhak Elbazis, Adi Wasenstein, Lior Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2505.10066)  

**Abstract**: Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated. 

**Abstract (ZH)**: 大型语言模型（LLMs）迅速重塑现代生活，推动医疗、教育等多个领域的进步。然而，与这些模型的强大能力并存的是一大威胁：这些模型对“越狱”攻击的高度易感性。大型语言模型（LLMs）的核心脆弱性源于其学习的数据本身。只要训练数据中包含未经筛选的问题内容或“暗网”内容，模型就有可能习得不良模式或弱点，允许用户规避其预期的安全控制。我们的研究揭示了故意设计缺乏伦理限制或通过越狱技术修改的“暗网”LLMs所带来的日益严重的威胁。在我们的研究中，我们发现了通用的越狱攻击，有效地接管了多个最新的先进模型，使这些模型几乎可以回答任何问题，并在请求时产生有害的输出。我们的攻击思路在七个多月前已在线发布，但测试的许多LLMs仍对此类攻击易感。尽管我们作出了负责任的信息披露努力，但主要LLM提供商的回应往往不足，凸显了行业在AI安全方面存在的令人担忧的差距。随着模型训练变得更加便捷和便宜，开源LLM的普及进一步加剧了普遍滥用的风险。若不采取果断干预措施，LLMs可能会继续平民化对危险知识的访问，带来的风险可能超出预期。 

---
# Analysing Safety Risks in LLMs Fine-Tuned with Pseudo-Malicious Cyber Security Data 

**Title (ZH)**: 分析使用伪恶意网络安全部门数据微调的大语言模型的安全风险 

**Authors**: Adel ElZemity, Budi Arief, Shujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.09974)  

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. We present a systematic evaluation of safety risks in fine-tuned LLMs for cyber security applications. Using the OWASP Top 10 for LLM Applications framework, we assessed seven open-source LLMs: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. Our evaluation shows that fine-tuning reduces safety resilience across all tested LLMs (e.g., the safety score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). We propose and evaluate a safety alignment approach that carefully rewords instruction-response pairs to include explicit safety precautions and ethical considerations. This approach demonstrates that it is possible to maintain or even improve model safety while preserving technical utility, offering a practical path forward for developing safer fine-tuning methodologies. This work offers a systematic evaluation for safety risks in LLMs, enabling safer adoption of generative AI in sensitive domains, and contributing towards the development of secure, trustworthy, and ethically aligned LLMs. 

**Abstract (ZH)**: 大型语言模型在网络安全应用中的集成既带来了显著机会，也引入了关键风险和安全顾虑，包括个人数据泄露和新型恶意软件的自动化生成。我们系统评估了针对网络安全应用细调的大型语言模型的安全风险。基于OWASP Top 10框架，我们评估了七个开源的大语言模型：Phi 3 Mini 3.8B、Mistral 7B、Qwen 2.5 7B、Llama 3 8B、Llama 3.1 8B、Gemma 2 9B和Llama 2 70B。评估结果显示，细调降低了所有受测大语言模型的安全韧性（例如，对提示注入攻击的安全评分为0.95降至0.15）。我们提出并评估了一种安全对齐方法，该方法仔细修改指令-响应对，明确包含安全预防措施和伦理考虑。该方法证明了在保持或甚至提高模型安全性的同时保留技术实用性是可能的，为开发更安全的细调方法提供了实际途径。本研究提供了大语言模型安全风险的系统评估，有助于在敏感领域更安全地采用生成型人工智能，并为开发安全、值得信赖且伦理对齐的大语言模型做出贡献。 

---
# Personalizing Large Language Models using Retrieval Augmented Generation and Knowledge Graph 

**Title (ZH)**: 使用检索增强生成和知识图谱个性化大型语言模型 

**Authors**: Deeksha Prahlad, Chanhee Lee, Dongha Kim, Hokeun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.09945)  

**Abstract**: The advent of large language models (LLMs) has allowed numerous applications, including the generation of queried responses, to be leveraged in chatbots and other conversational assistants. Being trained on a plethora of data, LLMs often undergo high levels of over-fitting, resulting in the generation of extra and incorrect data, thus causing hallucinations in output generation. One of the root causes of such problems is the lack of timely, factual, and personalized information fed to the LLM. In this paper, we propose an approach to address these problems by introducing retrieval augmented generation (RAG) using knowledge graphs (KGs) to assist the LLM in personalized response generation tailored to the users. KGs have the advantage of storing continuously updated factual information in a structured way. While our KGs can be used for a variety of frequently updated personal data, such as calendar, contact, and location data, we focus on calendar data in this paper. Our experimental results show that our approach works significantly better in understanding personal information and generating accurate responses compared to the baseline LLMs using personal data as text inputs, with a moderate reduction in response time. 

**Abstract (ZH)**: 大型语言模型（LLMs）的出现使得查询响应等众多应用能够在聊天机器人和其他对话助手中得到利用。由于在大量数据上进行训练，LLMs 往往会发生高度的过拟合，导致生成多余的错误数据，从而在输出生成中引发幻觉。这些问题的一个根本原因是向LLM提供的及时、准确且个性化的信息不足。在本文中，我们提出了一种通过引入基于知识图谱（KGs）的检索增强生成（RAG）方法来解决这些问题，并利用知识图谱帮助LLM生成针对用户个性化的响应。知识图谱的优势在于以结构化的方式存储不断更新的事实信息。虽然我们的知识图谱可以用于各种需要频繁更新的个人数据，如日历、联系人和位置数据，但在本文中我们专注于日历数据。我们的实验结果表明，与使用个人数据作为文本输入的基础型LLM相比，我们的方法在理解和生成准确响应方面明显更具优势，并且响应时间仅有适度增加。 

---
# Reinforced Interactive Continual Learning via Real-time Noisy Human Feedback 

**Title (ZH)**: 基于实时 noisy 人类反馈的强化互动连续学习 

**Authors**: Yutao Yang, Jie Zhou, Junsong Li, Qianjun Pan, Bihao Zhan, Qin Chen, Xipeng Qiu, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.09925)  

**Abstract**: This paper introduces an interactive continual learning paradigm where AI models dynamically learn new skills from real-time human feedback while retaining prior knowledge. This paradigm distinctively addresses two major limitations of traditional continual learning: (1) dynamic model updates using streaming, real-time human-annotated data, rather than static datasets with fixed labels, and (2) the assumption of clean labels, by explicitly handling the noisy feedback common in real-world interactions. To tackle these problems, we propose RiCL, a Reinforced interactive Continual Learning framework leveraging Large Language Models (LLMs) to learn new skills effectively from dynamic feedback. RiCL incorporates three key components: a temporal consistency-aware purifier to automatically discern clean from noisy samples in data streams; an interaction-aware direct preference optimization strategy to align model behavior with human intent by reconciling AI-generated and human-provided feedback; and a noise-resistant contrastive learning module that captures robust representations by exploiting inherent data relationships, thus avoiding reliance on potentially unreliable labels. Extensive experiments on two benchmark datasets (FewRel and TACRED), contaminated with realistic noise patterns, demonstrate that our RiCL approach substantially outperforms existing combinations of state-of-the-art online continual learning and noisy-label learning methods. 

**Abstract (ZH)**: 一种基于实时人类反馈的交互式连续学习范式：RiCL框架及其应用 

---
# Comparing Exploration-Exploitation Strategies of LLMs and Humans: Insights from Standard Multi-armed Bandit Tasks 

**Title (ZH)**: 比较大规模语言模型和人类的探索-利用策略：来自标准多臂老虎机任务的见解 

**Authors**: Ziyuan Zhang, Darcy Wang, Ningyuan Chen, Rodrigo Mansur, Vahid Sarhangian  

**Link**: [PDF](https://arxiv.org/pdf/2505.09901)  

**Abstract**: Large language models (LLMs) are increasingly used to simulate or automate human behavior in complex sequential decision-making tasks. A natural question is then whether LLMs exhibit similar decision-making behavior to humans, and can achieve comparable (or superior) performance. In this work, we focus on the exploration-exploitation (E&E) tradeoff, a fundamental aspect of dynamic decision-making under uncertainty. We employ canonical multi-armed bandit (MAB) tasks introduced in the cognitive science and psychiatry literature to conduct a comparative study of the E&E strategies of LLMs, humans, and MAB algorithms. We use interpretable choice models to capture the E&E strategies of the agents and investigate how explicit reasoning, through both prompting strategies and reasoning-enhanced models, shapes LLM decision-making. We find that reasoning shifts LLMs toward more human-like behavior, characterized by a mix of random and directed exploration. In simple stationary tasks, reasoning-enabled LLMs exhibit similar levels of random and directed exploration compared to humans. However, in more complex, non-stationary environments, LLMs struggle to match human adaptability, particularly in effective directed exploration, despite achieving similar regret in certain scenarios. Our findings highlight both the promise and limits of LLMs as simulators of human behavior and tools for automated decision-making and point to potential areas of improvements. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂序列决策任务中越来越多地用于模拟或自动化人类行为。一个自然的问题是LLMs是否会展现出与人类相似的决策行为，并能达到相当（或更优）的表现。在本文中，我们关注探索-利用（E&E）权衡，这是在不确定性条件下动态决策的一个基本方面。我们采用认知科学和精神病学文献中引入的经典多臂 bandit（MAB）任务，对LLMs、人类和MAB算法的E&E策略进行比较研究。我们利用可解释的选择模型来捕捉代理的E&E策略，并研究通过提示策略和增强推理模型的显式推理如何影响LLMs的决策。我们发现，推理使LLMs倾向于更具人类特征的行为，表现为随机探索和定向探索的结合。在简单的稳定任务中，具有推理能力的LLMs在随机探索和定向探索方面与人类表现出类似的水平。然而，在更复杂且非稳定环境下，尽管某些情况下能够达到类似的遗憾度，LLMs在有效定向探索方面仍难以匹配人类的适应性。我们的研究结果既突显了LLMs作为人类行为模拟器和自动化决策工具的潜力，也指出了改进的潜在领域。 

---
# Do Large Language Models Know Conflict? Investigating Parametric vs. Non-Parametric Knowledge of LLMs for Conflict Forecasting 

**Title (ZH)**: 大规模语言模型了解冲突吗？探究LLMs在冲突预测中的参数性与非参数性知识差异 

**Authors**: Apollinaire Poli Nemkova, Sarath Chandra Lingareddy, Sagnik Ray Choudhury, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2505.09852)  

**Abstract**: Large Language Models (LLMs) have shown impressive performance across natural language tasks, but their ability to forecast violent conflict remains underexplored. We investigate whether LLMs possess meaningful parametric knowledge-encoded in their pretrained weights-to predict conflict escalation and fatalities without external data. This is critical for early warning systems, humanitarian planning, and policy-making. We compare this parametric knowledge with non-parametric capabilities, where LLMs access structured and unstructured context from conflict datasets (e.g., ACLED, GDELT) and recent news reports via Retrieval-Augmented Generation (RAG). Incorporating external information could enhance model performance by providing up-to-date context otherwise missing from pretrained weights. Our two-part evaluation framework spans 2020-2024 across conflict-prone regions in the Horn of Africa and the Middle East. In the parametric setting, LLMs predict conflict trends and fatalities relying only on pretrained knowledge. In the non-parametric setting, models receive summaries of recent conflict events, indicators, and geopolitical developments. We compare predicted conflict trend labels (e.g., Escalate, Stable Conflict, De-escalate, Peace) and fatalities against historical data. Our findings highlight the strengths and limitations of LLMs for conflict forecasting and the benefits of augmenting them with structured external knowledge. 

**Abstract (ZH)**: 大型语言模型在自然语言任务中表现出色，但其预测暴力冲突的能力尚未得到充分探索。我们研究LLMs是否在其预训练权重中蕴含了有意义的参数化知识，能够预测冲突升级和伤亡情况，无需外部数据。这对于早期预警系统、人道主义规划和政策制定至关重要。我们对比了这种参数化知识与非参数化能力，其中LLMs通过检索增强生成（RAG）访问冲突数据集（如ACLED、GDELT）和近期新闻报告的结构化和非结构化上下文。整合外部信息能够通过提供预训练权重中缺乏的最新上下文来增强模型性能。我们的两部分评价框架覆盖2020-2024年，研究区域包括非洲之角和中东的冲突高发地区。在参数化设置中，LLMs仅依赖预训练知识预测冲突趋势和伤亡情况。在非参数化设置中，模型接收近期冲突事件、指标和地缘政治发展的摘要。我们对比预测的冲突趋势标签（如升级、稳定冲突、降级、和平）和伤亡情况与历史数据。我们的研究成果突显了LLMs在冲突预测中的优势和局限性，并强调了与结构化外部知识相结合的好处。 

---
# Evaluating Large Language Models for the Generation of Unit Tests with Equivalence Partitions and Boundary Values 

**Title (ZH)**: 基于等价类和边界值的单元测试生成的大语言模型评估 

**Authors**: Martín Rodríguez, Gustavo Rossi, Alejandro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2505.09830)  

**Abstract**: The design and implementation of unit tests is a complex task many programmers neglect. This research evaluates the potential of Large Language Models (LLMs) in automatically generating test cases, comparing them with manual tests. An optimized prompt was developed, that integrates code and requirements, covering critical cases such as equivalence partitions and boundary values. The strengths and weaknesses of LLMs versus trained programmers were compared through quantitative metrics and manual qualitative analysis. The results show that the effectiveness of LLMs depends on well-designed prompts, robust implementation, and precise requirements. Although flexible and promising, LLMs still require human supervision. This work highlights the importance of manual qualitative analysis as an essential complement to automation in unit test evaluation. 

**Abstract (ZH)**: 大型语言模型在自动生成测试案例中的设计与实现及其与手工测试的比较研究 

---
# Exploring the generalization of LLM truth directions on conversational formats 

**Title (ZH)**: 探索大模型在对话格式中事实方向的一致性泛化能力 

**Authors**: Timour Ichmoukhamedov, David Martens  

**Link**: [PDF](https://arxiv.org/pdf/2505.09807)  

**Abstract**: Several recent works argue that LLMs have a universal truth direction where true and false statements are linearly separable in the activation space of the model. It has been demonstrated that linear probes trained on a single hidden state of the model already generalize across a range of topics and might even be used for lie detection in LLM conversations. In this work we explore how this truth direction generalizes between various conversational formats. We find good generalization between short conversations that end on a lie, but poor generalization to longer formats where the lie appears earlier in the input prompt. We propose a solution that significantly improves this type of generalization by adding a fixed key phrase at the end of each conversation. Our results highlight the challenges towards reliable LLM lie detectors that generalize to new settings. 

**Abstract (ZH)**: 几种近期的研究认为，大规模语言模型（LLM）具有一个通用的真实方向，在模型的激活空间中，真话和假话陈述是可以线性区分的。已有研究表明，基于模型单个隐藏状态训练的线性探测器已经能够在多种主题上泛化，并且甚至可以用于检测LLM对话中的谎言。在本文中，我们探讨了这种真实方向在不同对话格式之间的泛化情况。我们发现，在短对话中较好地泛化，其中对话以谎言结束，但在谎言出现在输入提示较早位置的更长对话格式中泛化效果较差。我们提出了一种解决方案，通过在每个对话的结尾添加一个固定的关键短语，显著改进了这种类型的泛化。我们的结果强调了可靠的大规模语言模型谎言检测器在新环境中泛化的挑战。 

---
# Contextual Phenotyping of Pediatric Sepsis Cohort Using Large Language Models 

**Title (ZH)**: 使用大型语言模型对儿童败血症队列进行语境表型分析 

**Authors**: Aditya Nagori, Ayush Gautam, Matthew O. Wiens, Vuong Nguyen, Nathan Kenya Mugisha, Jerome Kabakyenga, Niranjan Kissoon, John Mark Ansermino, Rishikesan Kamaleswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.09805)  

**Abstract**: Clustering patient subgroups is essential for personalized care and efficient resource use. Traditional clustering methods struggle with high-dimensional, heterogeneous healthcare data and lack contextual understanding. This study evaluates Large Language Model (LLM) based clustering against classical methods using a pediatric sepsis dataset from a low-income country (LIC), containing 2,686 records with 28 numerical and 119 categorical variables. Patient records were serialized into text with and without a clustering objective. Embeddings were generated using quantized LLAMA 3.1 8B, DeepSeek-R1-Distill-Llama-8B with low-rank adaptation(LoRA), and Stella-En-400M-V5 models. K-means clustering was applied to these embeddings. Classical comparisons included K-Medoids clustering on UMAP and FAMD-reduced mixed data. Silhouette scores and statistical tests evaluated cluster quality and distinctiveness. Stella-En-400M-V5 achieved the highest Silhouette Score (0.86). LLAMA 3.1 8B with the clustering objective performed better with higher number of clusters, identifying subgroups with distinct nutritional, clinical, and socioeconomic profiles. LLM-based methods outperformed classical techniques by capturing richer context and prioritizing key features. These results highlight potential of LLMs for contextual phenotyping and informed decision-making in resource-limited settings. 

**Abstract (ZH)**: 基于大规模语言模型的聚类方法对于个性化护理和资源有效利用至关重要：低收入国家儿童败血症数据的聚类分析 

---
# Trustless Autonomy: Understanding Motivations, Benefits and Governance Dilemma in Self-Sovereign Decentralized AI Agents 

**Title (ZH)**: 无信任自主性：自我主权去中心化AI代理的动机、益处与治理困境探究 

**Authors**: Botao Amber Hu, Yuhan Liu, Helena Rong  

**Link**: [PDF](https://arxiv.org/pdf/2505.09757)  

**Abstract**: The recent trend of self-sovereign Decentralized AI Agents (DeAgents) combines Large Language Model (LLM)-based AI agents with decentralization technologies such as blockchain smart contracts and trusted execution environments (TEEs). These tamper-resistant trustless substrates allow agents to achieve self-sovereignty through ownership of cryptowallet private keys and control of digital assets and social media accounts. DeAgent eliminates centralized control and reduces human intervention, addressing key trust concerns inherent in centralized AI systems. However, given ongoing challenges in LLM reliability such as hallucinations, this creates paradoxical tension between trustlessness and unreliable autonomy. This study addresses this empirical research gap through interviews with DeAgents stakeholders-experts, founders, and developers-to examine their motivations, benefits, and governance dilemmas. The findings will guide future DeAgents system and protocol design and inform discussions about governance in sociotechnical AI systems in the future agentic web. 

**Abstract (ZH)**: 近期自主权去中心化AI代理（DeAgents）的趋势将基于大型语言模型（LLM）的AI代理与区块链智能合约和可信执行环境（TEEs）等去中心化技术相结合。这些防篡改的信任最小化基础结构使代理能够通过控制加密钱包私钥、数字资产和社会媒体账户实现自主权。DeAgent消除了集中控制并减少了人类干预，解决了集中式AI系统中固有的关键信任问题。然而，鉴于大型语言模型可靠性方面的持续挑战，如幻觉现象，这在信任最小化与不可靠自主性之间创造了悖论性的张力。本研究通过访谈DeAgent利益相关者——专家、创始人和开发者——来探讨他们的动机、优势和治理难题，填补了这一实证研究缺口。研究发现将指导未来DeAgents系统和协议的设计，并为未来基于社会技术的AI系统的治理讨论提供参考。 

---
# Achieving Tokenizer Flexibility in Language Models through Heuristic Adaptation and Supertoken Learning 

**Title (ZH)**: 通过启发式适应和超词学习实现语言模型中的分词灵活性 

**Authors**: Shaurya Sharthak, Vinayak Pahalwan, Adithya Kamath, Adarsh Shirawalmath  

**Link**: [PDF](https://arxiv.org/pdf/2505.09738)  

**Abstract**: Pretrained language models (LLMs) are often constrained by their fixed tokenization schemes, leading to inefficiencies and performance limitations, particularly for multilingual or specialized applications. This tokenizer lock-in presents significant challenges. standard methods to overcome this often require prohibitive computational resources. Although tokenizer replacement with heuristic initialization aims to reduce this burden, existing methods often require exhaustive residual fine-tuning and still may not fully preserve semantic nuances or adequately address the underlying compression inefficiencies. Our framework introduces two innovations: first, Tokenadapt, a model-agnostic tokenizer transplantation method, and second, novel pre-tokenization learning for multi-word Supertokens to enhance compression and reduce fragmentation. Tokenadapt initializes new unique token embeddings via a hybrid heuristic that combines two methods: a local estimate based on subword decomposition using the old tokenizer, and a global estimate utilizing the top-k semantically similar tokens from the original vocabulary. This methodology aims to preserve semantics while significantly minimizing retraining requirements. Empirical investigations validate both contributions: the transplantation heuristic successfully initializes unique tokens, markedly outperforming conventional baselines and sophisticated methods including Transtokenizer and ReTok, while our Supertokens achieve notable compression gains. Our zero-shot perplexity results demonstrate that the TokenAdapt hybrid initialization consistently yields lower perplexity ratios compared to both ReTok and TransTokenizer baselines across different base models and newly trained target tokenizers. TokenAdapt typically reduced the overall perplexity ratio significantly compared to ReTok, yielding at least a 2-fold improvement in these aggregate scores. 

**Abstract (ZH)**: 预训练语言模型（LLMs）往往受限于固定的分词方案，导致效率低下和性能限制，特别是在多语言或专业应用中。这种分词锁定带来了显著挑战。克服这一限制的标准方法通常需要昂贵的计算资源。虽然利用启发式初始化进行分词替换旨在减轻这一负担，但现有方法往往仍然需要耗尽式的残差微调，而无法完全保留语义细微差别或充分解决潜在的压缩效率问题。我们的框架引入了两项创新：首先，Tokenadapt，一种模型无关的分词移植方法；其次，多词Supertokens的新型预分词学习，以增强压缩并减少碎片化。Tokenadapt通过结合两种方法的混合启发式初始化新独特词嵌入：基于旧分词器进行子词分解的局部估计，以及利用原始词汇表中最相似的top-k语义词的全局估计。该方法旨在保留语义同时显著减少重训需求。实证研究验证了两项贡献：分词移植启发式方法成功初始化独特词，并显著优于传统基线和先进的Transtokenizer、ReTok等方法，而我们的Supertokens实现了显著的压缩增益。TokenAdapt的零样本困惑度结果表明，该混合初始化方法在不同基础模型和新训练的目标分词器上，始终比ReTok和TransTokenizer基线模型具有更低的困惑度比率。与ReTok相比，TokenAdapt通常显著降低了整体困惑度比率，这些综合评分提高了至少两倍。 

---
# An AI-Powered Research Assistant in the Lab: A Practical Guide for Text Analysis Through Iterative Collaboration with LLMs 

**Title (ZH)**: 实验室中的AI赋能研究助理：与大规模语言模型迭代协作的文本分析实用指南 

**Authors**: Gino Carmona-Díaz, William Jiménez-Leal, María Alejandra Grisales, Chandra Sripada, Santiago Amaya, Michael Inzlicht, Juan Pablo Bermúdez  

**Link**: [PDF](https://arxiv.org/pdf/2505.09724)  

**Abstract**: Analyzing texts such as open-ended responses, headlines, or social media posts is a time- and labor-intensive process highly susceptible to bias. LLMs are promising tools for text analysis, using either a predefined (top-down) or a data-driven (bottom-up) taxonomy, without sacrificing quality. Here we present a step-by-step tutorial to efficiently develop, test, and apply taxonomies for analyzing unstructured data through an iterative and collaborative process between researchers and LLMs. Using personal goals provided by participants as an example, we demonstrate how to write prompts to review datasets and generate a taxonomy of life domains, evaluate and refine the taxonomy through prompt and direct modifications, test the taxonomy and assess intercoder agreements, and apply the taxonomy to categorize an entire dataset with high intercoder reliability. We discuss the possibilities and limitations of using LLMs for text analysis. 

**Abstract (ZH)**: 分析开放响应、标题或社交媒体帖子等文本是一个耗时且劳动密集型的过程，极易产生偏差。大规模语言模型是进行文本分析的有前途的工具，可以在预定义（自上而下）或数据驱动（自下而上）分类法中使用，同时不牺牲质量。在这里，我们提供了一种逐步教程，通过研究人员与大规模语言模型的迭代协作过程，高效地开发、测试和应用分析非结构化数据的分类法。使用参与者提供的个人目标为例，我们展示了如何编写提示审查数据集并生成生活领域分类法，通过提示和直接修改评估和优化分类法，测试分类法并评估编码员间一致性，并将分类法应用于高编码员一致性地分类整个数据集。我们讨论了使用大规模语言模型进行文本分析的可能性和局限性。 

---
# System Prompt Optimization with Meta-Learning 

**Title (ZH)**: 基于元学习的系统提示优化 

**Authors**: Yumin Choi, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09666)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities, with optimizing their input prompts playing a pivotal role in maximizing their performance. However, while LLM prompts consist of both the task-agnostic system prompts and task-specific user prompts, existing work on prompt optimization has focused on user prompts specific to individual queries or tasks, and largely overlooked the system prompt that is, once optimized, applicable across different tasks and domains. Motivated by this, we introduce the novel problem of bilevel system prompt optimization, whose objective is to design system prompts that are robust to diverse user prompts and transferable to unseen tasks. To tackle this problem, we then propose a meta-learning framework, which meta-learns the system prompt by optimizing it over various user prompts across multiple datasets, while simultaneously updating the user prompts in an iterative manner to ensure synergy between them. We conduct experiments on 14 unseen datasets spanning 5 different domains, on which we show that our approach produces system prompts that generalize effectively to diverse user prompts. Also, our findings reveal that the optimized system prompt enables rapid adaptation even to unseen tasks, requiring fewer optimization steps for test-time user prompts while achieving improved performance. 

**Abstract (ZH)**: 大型语言模型的双层系统提示优化：设计鲁棒且可转移的系统提示 

---
# Unlocking Location Intelligence: A Survey from Deep Learning to The LLM Era 

**Title (ZH)**: 解锁位置智能：从深度学习到大语言模型时代的综述 

**Authors**: Xixuan Hao, Yutian Jiang, Xingchen Zou, Jiabo Liu, Yifang Yin, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.09651)  

**Abstract**: Location Intelligence (LI), the science of transforming location-centric geospatial data into actionable knowledge, has become a cornerstone of modern spatial decision-making. The rapid evolution of Geospatial Representation Learning is fundamentally reshaping LI development through two successive technological revolutions: the deep learning breakthrough and the emerging large language model (LLM) paradigm. While deep neural networks (DNNs) have demonstrated remarkable success in automated feature extraction from structured geospatial data (e.g., satellite imagery, GPS trajectories), the recent integration of LLMs introduces transformative capabilities for cross-modal geospatial reasoning and unstructured geo-textual data processing. This survey presents a comprehensive review of geospatial representation learning across both technological eras, organizing them into a structured taxonomy based on the complete pipeline comprising: (1) data perspective, (2) methodological perspective and (3) application perspective. We also highlight current advancements, discuss existing limitations, and propose potential future research directions in the LLM era. This work offers a thorough exploration of the field and providing a roadmap for further innovation in LI. The summary of the up-to-date paper list can be found in this https URL and will undergo continuous updates. 

**Abstract (ZH)**: 地理空间表示学习：从深度学习突破到新兴大语言模型时代的位置智能技术革命综述 

---
