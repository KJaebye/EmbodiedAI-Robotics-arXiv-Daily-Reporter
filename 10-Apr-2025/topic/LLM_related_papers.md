# Review of Case-Based Reasoning for LLM Agents: Theoretical Foundations, Architectural Components, and Cognitive Integration 

**Title (ZH)**: 基于案例推理的LLM代理回顾：理论基础、架构组件以及认知整合 

**Authors**: Kostas Hatalis, Despina Christou, Vyshnavi Kondapalli  

**Link**: [PDF](https://arxiv.org/pdf/2504.06943)  

**Abstract**: Agents powered by Large Language Models (LLMs) have recently demonstrated impressive capabilities in various tasks. Still, they face limitations in tasks requiring specific, structured knowledge, flexibility, or accountable decision-making. While agents are capable of perceiving their environments, forming inferences, planning, and executing actions towards goals, they often face issues such as hallucinations and lack of contextual memory across interactions. This paper explores how Case-Based Reasoning (CBR), a strategy that solves new problems by referencing past experiences, can be integrated into LLM agent frameworks. This integration allows LLMs to leverage explicit knowledge, enhancing their effectiveness. We systematically review the theoretical foundations of these enhanced agents, identify critical framework components, and formulate a mathematical model for the CBR processes of case retrieval, adaptation, and learning. We also evaluate CBR-enhanced agents against other methods like Chain-of-Thought reasoning and standard Retrieval-Augmented Generation, analyzing their relative strengths. Moreover, we explore how leveraging CBR's cognitive dimensions (including self-reflection, introspection, and curiosity) via goal-driven autonomy mechanisms can further enhance the LLM agent capabilities. Contributing to the ongoing research on neuro-symbolic hybrid systems, this work posits CBR as a viable technique for enhancing the reasoning skills and cognitive aspects of autonomous LLM agents. 

**Abstract (ZH)**: 基于大型语言模型的智能体：通过案例推理增强的能力探索 

---
# FamilyTool: A Multi-hop Personalized Tool Use Benchmark 

**Title (ZH)**: FamilyTool: 一种多跳个性化工具使用基准 

**Authors**: Yuxin Wang, Yiran Guo, Yining Zheng, Zhangyue Yin, Shuo Chen, Jie Yang, Jiajun Chen, Xuanjing Huang, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06766)  

**Abstract**: The integration of tool learning with Large Language Models (LLMs) has expanded their capabilities in handling complex tasks by leveraging external tools. However, existing benchmarks for tool learning inadequately address critical real-world personalized scenarios, particularly those requiring multi-hop reasoning and inductive knowledge adaptation in dynamic environments. To bridge this gap, we introduce FamilyTool, a novel benchmark grounded in a family-based knowledge graph (KG) that simulates personalized, multi-hop tool use scenarios. FamilyTool challenges LLMs with queries spanning 1 to 3 relational hops (e.g., inferring familial connections and preferences) and incorporates an inductive KG setting where models must adapt to unseen user preferences and relationships without re-training, a common limitation in prior approaches that compromises generalization. We further propose KGETool: a simple KG-augmented evaluation pipeline to systematically assess LLMs' tool use ability in these settings. Experiments reveal significant performance gaps in state-of-the-art LLMs, with accuracy dropping sharply as hop complexity increases and inductive scenarios exposing severe generalization deficits. These findings underscore the limitations of current LLMs in handling personalized, evolving real-world contexts and highlight the urgent need for advancements in tool-learning frameworks. FamilyTool serves as a critical resource for evaluating and advancing LLM agents' reasoning, adaptability, and scalability in complex, dynamic environments. Code and dataset are available at Github. 

**Abstract (ZH)**: 工具学习与大型语言模型的集成扩展了它们处理复杂任务的能力，通过利用外部工具。然而，现有的工具学习基准在解决关键的实际个性化场景方面存在不足，特别是在需要多跳推理和动态环境中归纳知识适应的能力上。为弥补这一差距，我们介绍了基于家庭知识图谱（KG）的FamilyTool，这是一种新的基准方法，模拟了个性化、多跳工具使用场景。FamilyTool 用涉及1到3个关系跳数的查询（例如，推断家庭关系和偏好）挑战大型语言模型，并引入了一个归纳KG设置，要求模型在无需重新训练的情况下适应未见过的用户偏好和关系，这克服了先前方法中的一个常见限制，即影响泛化能力。我们还提出了KGETool：一种简单的知识图谱增强评估流水线，系统评估大型语言模型在这些环境下的工具使用能力。实验表明，最先进的大型语言模型在这些设置下的表现存在显著差异，随着跳数复杂性的增加，准确率急剧下降，而归纳场景暴露出严重的泛化缺陷。这些发现强调了当前大型语言模型处理个性化、不断演变的实际环境的局限性，并突显了工具学习框架亟待改进的迫切需求。FamilyTool 作为评估和促进大型语言模型代理在复杂动态环境中的推理、适应能力和可扩展性的关键资源。相关代码和数据集可在GitHub上获取。 

---
# Right Prediction, Wrong Reasoning: Uncovering LLM Misalignment in RA Disease Diagnosis 

**Title (ZH)**: 正确的预测，错误的推理：揭示RA疾病诊断中LLM的偏向性 

**Authors**: Umakanta Maharana, Sarthak Verma, Avarna Agarwal, Prakashini Mruthyunjaya, Dwarikanath Mahapatra, Sakir Ahmed, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2504.06581)  

**Abstract**: Large language models (LLMs) offer a promising pre-screening tool, improving early disease detection and providing enhanced healthcare access for underprivileged communities. The early diagnosis of various diseases continues to be a significant challenge in healthcare, primarily due to the nonspecific nature of early symptoms, the shortage of expert medical practitioners, and the need for prolonged clinical evaluations, all of which can delay treatment and adversely affect patient outcomes. With impressive accuracy in prediction across a range of diseases, LLMs have the potential to revolutionize clinical pre-screening and decision-making for various medical conditions. In this work, we study the diagnostic capability of LLMs for Rheumatoid Arthritis (RA) with real world patients data. Patient data was collected alongside diagnoses from medical experts, and the performance of LLMs was evaluated in comparison to expert diagnoses for RA disease prediction. We notice an interesting pattern in disease diagnosis and find an unexpected \textit{misalignment between prediction and explanation}. We conduct a series of multi-round analyses using different LLM agents. The best-performing model accurately predicts rheumatoid arthritis (RA) diseases approximately 95\% of the time. However, when medical experts evaluated the reasoning generated by the model, they found that nearly 68\% of the reasoning was incorrect. This study highlights a clear misalignment between LLMs high prediction accuracy and its flawed reasoning, raising important questions about relying on LLM explanations in clinical settings. \textbf{LLMs provide incorrect reasoning to arrive at the correct answer for RA disease diagnosis.} 

**Abstract (ZH)**: 大型语言模型（LLMs）提供了一种有前景的预筛查工具，有助于早期疾病检测并为贫困社区提供增强的医疗访问权限。在医疗保健领域，各种疾病的早期诊断仍然是一项重大挑战，主要原因在于早期症状的非特异性、专家医疗 practitioners 的短缺以及需要长时间的临床评估，所有这些都可能导致治疗延迟并影响患者结果。凭借在多种疾病预测中表现出的卓越准确性，LLMs 有潜力革新不同医疗条件的临床预筛查和决策过程。在本文中，我们研究了LLMs在现实患者数据中对类风湿性关节炎（RA）的诊断能力。患者数据与医疗专家的诊断结果一并收集，并将LLMs的表现与专家对RA疾病预测的诊断进行了比较。我们注意到一种有趣的疾病诊断模式，并发现了一个意想不到的“预测与解释之间的不一致”。我们使用不同的LLM代理进行了多轮分析。表现最佳的模型大约95%的时间准确预测了类风湿性关节炎（RA）疾病。然而，当医疗专家评估模型生成的推理时，他们发现近68%的推理是不正确的。本研究突显了LLMs高预测准确性与其推理缺陷之间的明确不一致，引发了在临床环境中依赖LLM解释的重要问题。**LLMs提供错误的推理以得出正确的RA疾病诊断答案。** 

---
# Missing Premise exacerbates Overthinking: Are Reasoning Models losing Critical Thinking Skill? 

**Title (ZH)**: 缺省前提加剧过度思考：推理模型丧失批判性思维能力了吗？ 

**Authors**: Chenrui Fan, Ming Li, Lichao Sun, Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06514)  

**Abstract**: We find that the response length of reasoning LLMs, whether trained by reinforcement learning or supervised learning, drastically increases for ill-posed questions with missing premises (MiP), ending up with redundant and ineffective thinking. This newly introduced scenario exacerbates the general overthinking issue to a large extent, which we name as the MiP-Overthinking. Such failures are against the ``test-time scaling law'' but have been widely observed on multiple datasets we curated with MiP, indicating the harm of cheap overthinking and a lack of critical thinking. Surprisingly, LLMs not specifically trained for reasoning exhibit much better performance on the MiP scenario, producing much shorter responses that quickly identify ill-posed queries. This implies a critical flaw of the current training recipe for reasoning LLMs, which does not encourage efficient thinking adequately, leading to the abuse of thinking patterns. To further investigate the reasons behind such failures, we conduct fine-grained analyses of the reasoning length, overthinking patterns, and location of critical thinking on different types of LLMs. Moreover, our extended ablation study reveals that the overthinking is contagious through the distillation of reasoning models' responses. These results improve the understanding of overthinking and shed novel insights into mitigating the problem. 

**Abstract (ZH)**: 我们发现，无论是通过强化学习还是监督学习训练的推理大语言模型，在缺失前提条件（MiP）的 poorly 提出的问题上的响应长度显著增加，最终导致无效且多余的思考。这一新引入的场景在很大程度上加剧了普遍存在的过度思考问题，我们将其命名为 MiP-过度思考。这些失败违背了“测试时缩放定律”，但在我们收集的多个包含 MiP 的数据集上广泛观察到，这表明了廉价的过度思考和缺乏批判性思考的危害。令人惊讶的是，未特别针对推理训练的大语言模型在 MiP 场景中表现出更佳性能，产生了更短的响应并迅速识别出 poorly 提出的查询。这暗示了目前推理大语言模型的训练方法存在关键缺陷，未能充分鼓励有效的思考，导致思考模式的滥用。为了进一步探讨这些失败的原因，我们对不同类型的 LLM 的推理长度、过度思考模式以及关键思考位置进行了精细分析。此外，我们的扩展消融研究揭示了过度思考可以通过推理模型响应的蒸馏传染。这些结果加深了我们对过度思考的理解，并提供了缓解该问题的新见解。 

---
# Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning 

**Title (ZH)**: 塑造子空间：受限全面微调在持续学习中的应用 

**Authors**: Nikhil Shivakumar Nayak, Krishnateja Killamsetty, Ligong Han, Abhishek Bhandwaldar, Prateek Chanda, Kai Xu, Hao Wang, Aldo Pareja, Oleg Silkin, Mustafa Eyceoz, Akash Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2504.07097)  

**Abstract**: Continual learning in large language models (LLMs) is prone to catastrophic forgetting, where adapting to new tasks significantly degrades performance on previously learned ones. Existing methods typically rely on low-rank, parameter-efficient updates that limit the model's expressivity and introduce additional parameters per task, leading to scalability issues. To address these limitations, we propose a novel continual full fine-tuning approach leveraging adaptive singular value decomposition (SVD). Our method dynamically identifies task-specific low-rank parameter subspaces and constrains updates to be orthogonal to critical directions associated with prior tasks, thus effectively minimizing interference without additional parameter overhead or storing previous task gradients. We evaluate our approach extensively on standard continual learning benchmarks using both encoder-decoder (T5-Large) and decoder-only (LLaMA-2 7B) models, spanning diverse tasks including classification, generation, and reasoning. Empirically, our method achieves state-of-the-art results, up to 7% higher average accuracy than recent baselines like O-LoRA, and notably maintains the model's general linguistic capabilities, instruction-following accuracy, and safety throughout the continual learning process by reducing forgetting to near-negligible levels. Our adaptive SVD framework effectively balances model plasticity and knowledge retention, providing a practical, theoretically grounded, and computationally scalable solution for continual learning scenarios in large language models. 

**Abstract (ZH)**: 大型语言模型中的持续学习易发生灾难性遗忘，其中适应新的任务会显著恶化之前学习任务的表现。现有方法通常依赖低秩、参数高效的更新，这限制了模型的表达能力和造成每个任务增加额外参数的问题，导致可扩展性问题。为解决这些限制，我们提出了一种新的基于自适应奇异值分解（SVD）的持续完整微调方法。该方法动态地识别任务特定的低秩参数子空间，并将更新约束为与先前任务关键方向正交，从而有效地减少干扰，同时无需额外的参数开销或存储先前任务的梯度。我们使用包括编码器-解码器（T5-Large）和仅解码器（LLaMA-2 7B）模型，广泛地在标准持续学习基准上评估了该方法，涵盖分类、生成和推理等多种任务。实验结果表明，该方法实现了最先进的结果，平均准确率比最近的基线（如O-LoRA）高出7%，并且在整个持续学习过程中，有效减少了遗忘程度，保持了模型的一般语言能力、指令跟随准确性和安全性。自适应SVD框架有效平衡了模型的可塑性和知识保留，为大型语言模型中的持续学习场景提供了实用、理论依据充分且计算高效的解决方案。 

---
# KG-LLM-Bench: A Scalable Benchmark for Evaluating LLM Reasoning on Textualized Knowledge Graphs 

**Title (ZH)**: KG-LLM-Bench：一种评估大规模语言模型在文本化知识图上推理能力的可扩展基准 

**Authors**: Elan Markowitz, Krupa Galiya, Greg Ver Steeg, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2504.07087)  

**Abstract**: Knowledge graphs have emerged as a popular method for injecting up-to-date, factual knowledge into large language models (LLMs). This is typically achieved by converting the knowledge graph into text that the LLM can process in context. While multiple methods of encoding knowledge graphs have been proposed, the impact of this textualization process on LLM performance remains under-explored. We introduce KG-LLM-Bench, a comprehensive and extensible benchmark spanning five knowledge graph understanding tasks, and evaluate how different encoding strategies affect performance across various base models. Our extensive experiments with seven language models and five textualization strategies provide insights for optimizing LLM performance on KG reasoning tasks. 

**Abstract (ZH)**: 知识图谱已 emergence 作为一种流行的方法，用于向大规模语言模型（LLMs）注入最新的事实性知识。这通常通过将知识图谱转换为 LLM 可以在上下文中处理的文本来实现。尽管已经提出了多种知识图谱编码方法，但这一文本化过程对 LLM 性能的影响仍未得到充分探索。我们引入了 KG-LLM-Bench，这是一个涵盖五个知识图谱理解任务的全面且可扩展的基准测试，并评估了不同编码策略在各种基础模型上的性能影响。我们的 extensive 实验使用了七种语言模型和五种文本化策略，为优化 LLM 在 KG 推理任务上的性能提供了见解。 

---
# Self-Steering Language Models 

**Title (ZH)**: 自引导语言模型 

**Authors**: Gabriel Grand, Joshua B. Tenenbaum, Vikash K. Mansinghka, Alexander K. Lew, Jacob Andreas  

**Link**: [PDF](https://arxiv.org/pdf/2504.07081)  

**Abstract**: While test-time reasoning enables language models to tackle complex tasks, searching or planning in natural language can be slow, costly, and error-prone. But even when LMs struggle to emulate the precise reasoning steps needed to solve a problem, they often excel at describing its abstract structure--both how to verify solutions and how to search for them. This paper introduces DisCIPL, a method for "self-steering" LMs where a Planner model generates a task-specific inference program that is executed by a population of Follower models. Our approach equips LMs with the ability to write recursive search procedures that guide LM inference, enabling new forms of verifiable and efficient reasoning. When instantiated with a small Follower (e.g., Llama-3.2-1B), DisCIPL matches (and sometimes outperforms) much larger models, including GPT-4o and o1, on challenging constrained generation tasks. In decoupling planning from execution, our work opens up a design space of highly-parallelized Monte Carlo inference strategies that outperform standard best-of-N sampling, require no finetuning, and can be implemented automatically by existing LMs. 

**Abstract (ZH)**: 虽然测试时推理使语言模型能够应对复杂任务，但在自然语言中进行搜索或规划可能会变得缓慢、昂贵且容易出错。但在语言模型难以模拟解决一个问题所需的精确推理步骤时，它们往往在描述其抽象结构方面表现出色，包括如何验证解决方案和如何搜索解决方案。本文介绍了DisCIPL方法，该方法实现了“自我引导”的语言模型，其中规划模型生成一个针对特定任务的推理程序，由一组跟随模型执行。我们的方法赋予语言模型编写递归搜索过程的能力，以引导模型的推理，从而实现新的可验证和高效的推理形式。当使用小型跟随者（例如，Llama-3.2-1B）实例化时，DisCIPL在具有挑战性的受限生成任务上与更大规模的模型（包括GPT-4o和o1）相当甚至表现更优。通过将规划与执行解耦，我们的工作开辟了高性能蒙特卡洛推理策略的设计空间，而这些策略能优于标准的-best-of-N采样策略，无需微调，且可以通过现有的语言模型自动实现。 

---
# DeduCE: Deductive Consistency as a Framework to Evaluate LLM Reasoning 

**Title (ZH)**: DeduCE: 通过检验推理一致性评估大型语言模型的框架 

**Authors**: Atharva Pandey, Kshitij Dubey, Rahul Sharma, Amit Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2504.07080)  

**Abstract**: Despite great performance on Olympiad-level reasoning problems, frontier large language models can still struggle on high school math when presented with novel problems outside standard benchmarks. Going beyond final accuracy, we propose a deductive consistency metric to analyze chain-of-thought output from language models (LMs).Formally, deductive reasoning involves two subtasks: understanding a set of input premises and inferring the conclusions that follow from them. The proposed metric studies LMs' performance on these subtasks, with the goal of explaining LMs' reasoning errors on novel problems: how well do LMs understand input premises with increasing context lengths, and how well can they infer conclusions over multiple reasoning hops? Since existing benchmarks may be memorized, we develop a pipeline to evaluate LMs' deductive consistency on novel, perturbed versions of benchmark problems. On novel grade school math problems (GSM-8k), we find that LMs are fairly robust to increasing number of input premises, but suffer significant accuracy decay as the number of reasoning hops is increased. Interestingly, these errors are masked in the original benchmark as all models achieve near 100% accuracy. As we increase the number of solution steps using a synthetic dataset, prediction over multiple hops still remains the major source of error compared to understanding input premises. Other factors, such as shifts in language style or natural propagation of early errors do not explain the trends. Our analysis provides a new view to characterize LM reasoning -- as computations over a window of input premises and reasoning hops -- that can provide unified evaluation across problem domains. 

**Abstract (ZH)**: 尽管在奥林匹克水平的推理问题上表现出色，前沿的大语言模型在面临标准基准之外的新型高中数学问题时仍然会遇到困难。我们提出了一种演绎一致性度量来分析语言模型（LMs）的思维链输出，超越最终的准确性，旨在研究LMs在演绎推理子任务上的表现，解释其在新型问题上的推理错误：随着上下文长度的增加，LMs对输入前提的理解程度如何？随着推理跳数的增加，它们推导结论的能力又如何？由于现有基准可能存在记忆效应，我们开发了一个管道来评估LMs在新型、扰动过的基准问题上的演绎一致性。在新型小学数学问题（GSM-8k）上，我们发现LMs对输入前提数量的增加具有相当的鲁棒性，但在推理跳数增加时准确性显著下降。有趣的是，在原始基准中，这些错误被掩盖了，因为所有模型几乎达到了100%的准确率。随着使用合成数据集增加解题步骤，多跳预测仍然是比理解输入前提的主要错误来源。其他因素，如语言风格的转变或早期错误的自然传播，并不能解释这些趋势。我们的分析提供了一种新的视角来刻画LM的推理——作为输入前提和推理跳数窗口上的计算——这可以在不同问题领域提供统一的评估。 

---
# HalluciNot: Hallucination Detection Through Context and Common Knowledge Verification 

**Title (ZH)**: HalluciNot: 通过上下文和常识验证的幻觉检测 

**Authors**: Bibek Paudel, Alexander Lyzhov, Preetam Joshi, Puneet Anand  

**Link**: [PDF](https://arxiv.org/pdf/2504.07069)  

**Abstract**: This paper introduces a comprehensive system for detecting hallucinations in large language model (LLM) outputs in enterprise settings. We present a novel taxonomy of LLM responses specific to hallucination in enterprise applications, categorizing them into context-based, common knowledge, enterprise-specific, and innocuous statements. Our hallucination detection model HDM-2 validates LLM responses with respect to both context and generally known facts (common knowledge). It provides both hallucination scores and word-level annotations, enabling precise identification of problematic content. To evaluate it on context-based and common-knowledge hallucinations, we introduce a new dataset HDMBench. Experimental results demonstrate that HDM-2 out-performs existing approaches across RagTruth, TruthfulQA, and HDMBench datasets. This work addresses the specific challenges of enterprise deployment, including computational efficiency, domain specialization, and fine-grained error identification. Our evaluation dataset, model weights, and inference code are publicly available. 

**Abstract (ZH)**: 本文引入了一个全面的企业环境中文本生成模型幻觉检测系统。我们提出了一种针对企业应用中幻觉的新型分类体系，将其分为基于上下文的、常识性的、企业特定的和无害的声明。我们的幻觉检测模型HDM-2根据上下文和一般公认的事实对文本生成模型的响应进行验证，提供了幻觉评分和单词级别注释，有助于精确识别问题内容。为了评估其在基于上下文和常识性幻觉上的表现，我们引入了一个新的数据集HDMBench。实验结果表明，HDM-2在RagTruth、TruthfulQA和HDMBench数据集上的表现优于现有方法。本文解决了企业部署特有的挑战，包括计算效率、领域专门化和精细错误识别。我们的评估数据集、模型权重和推理代码均公开提供。 

---
# Integrating Cognitive Processing Signals into Language Models: A Review of Advances, Applications and Future Directions 

**Title (ZH)**: 将认知处理信号整合进语言模型：进展、应用及未来方向 

**Authors**: Angela Lopez-Cardona, Sebastian Idesis, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2504.06843)  

**Abstract**: Recently, the integration of cognitive neuroscience in Natural Language Processing (NLP) has gained significant attention. This article provides a critical and timely overview of recent advancements in leveraging cognitive signals, particularly Eye-tracking (ET) signals, to enhance Language Models (LMs) and Multimodal Large Language Models (MLLMs). By incorporating user-centric cognitive signals, these approaches address key challenges, including data scarcity and the environmental costs of training large-scale models. Cognitive signals enable efficient data augmentation, faster convergence, and improved human alignment. The review emphasises the potential of ET data in tasks like Visual Question Answering (VQA) and mitigating hallucinations in MLLMs, and concludes by discussing emerging challenges and research trends. 

**Abstract (ZH)**: 最近，认知神经科学在自然语言处理（NLP）中的集成引起了广泛关注。本文提供了一篇及时且批判性的综述，概述了通过利用认知信号，特别是眼动追踪（ET）信号，来增强语言模型（LMs）和多模态大规模语言模型（MLLMs）的近期进展。通过纳入用户中心的认知信号，这些方法解决了数据稀缺性和大规模模型训练的环境成本等关键挑战。认知信号使得高效的数据增强、更快的收敛和更好的人类对齐成为可能。综述强调了ET数据在视觉问答（VQA）任务和减轻MLLMs幻觉方面的潜力，并讨论了新兴挑战和研究趋势。 

---
# Zero-Shot Image-Based Large Language Model Approach to Road Pavement Monitoring 

**Title (ZH)**: 基于零样本图像的大型语言模型道路铺装监测方法 

**Authors**: Shuoshuo Xu, Kai Zhao, James Loney, Zili Li, Andrea Visentin  

**Link**: [PDF](https://arxiv.org/pdf/2504.06785)  

**Abstract**: Effective and rapid evaluation of pavement surface condition is critical for prioritizing maintenance, ensuring transportation safety, and minimizing vehicle wear and tear. While conventional manual inspections suffer from subjectivity, existing machine learning-based methods are constrained by their reliance on large and high-quality labeled datasets, which require significant resources and limit adaptability across varied road conditions. The revolutionary advancements in Large Language Models (LLMs) present significant potential for overcoming these challenges. In this study, we propose an innovative automated zero-shot learning approach that leverages the image recognition and natural language understanding capabilities of LLMs to assess road conditions effectively. Multiple LLM-based assessment models were developed, employing prompt engineering strategies aligned with the Pavement Surface Condition Index (PSCI) standards. These models' accuracy and reliability were evaluated against official PSCI results, with an optimized model ultimately selected. Extensive tests benchmarked the optimized model against evaluations from various levels experts using Google Street View road images. The results reveal that the LLM-based approach can effectively assess road conditions, with the optimized model -employing comprehensive and structured prompt engineering strategies -outperforming simpler configurations by achieving high accuracy and consistency, even surpassing expert evaluations. Moreover, successfully applying the optimized model to Google Street View images demonstrates its potential for future city-scale deployments. These findings highlight the transformative potential of LLMs in automating road damage evaluations and underscore the pivotal role of detailed prompt engineering in achieving reliable assessments. 

**Abstract (ZH)**: 基于大语言模型的零样本学习路面状况评估方法研究 

---
# Bridging the Gap Between Preference Alignment and Machine Unlearning 

**Title (ZH)**: 偏好对齐与机器遗忘之间的差距桥梁 

**Authors**: Xiaohua Feng, Yuyuan Li, Huwei Ji, Jiaming Zhang, Li Zhang, Tianyu Du, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06659)  

**Abstract**: Despite advances in Preference Alignment (PA) for Large Language Models (LLMs), mainstream methods like Reinforcement Learning with Human Feedback (RLHF) face notable challenges. These approaches require high-quality datasets of positive preference examples, which are costly to obtain and computationally intensive due to training instability, limiting their use in low-resource scenarios. LLM unlearning technique presents a promising alternative, by directly removing the influence of negative examples. However, current research has primarily focused on empirical validation, lacking systematic quantitative analysis. To bridge this gap, we propose a framework to explore the relationship between PA and LLM unlearning. Specifically, we introduce a bi-level optimization-based method to quantify the impact of unlearning specific negative examples on PA performance. Our analysis reveals that not all negative examples contribute equally to alignment improvement when unlearned, and the effect varies significantly across examples. Building on this insight, we pose a crucial question: how can we optimally select and weight negative examples for unlearning to maximize PA performance? To answer this, we propose a framework called Unlearning to Align (U2A), which leverages bi-level optimization to efficiently select and unlearn examples for optimal PA performance. We validate the proposed method through extensive experiments, with results confirming its effectiveness. 

**Abstract (ZH)**: 尽管在大型语言模型（LLMs）的偏好对齐（PA）方面取得了进展，主流方法如强化学习带人类反馈（RLHF）仍面临显著挑战。这些方法需要高质量的正偏好示例数据集，获取成本高且由于训练不稳定而计算密集，限制了其在低资源场景中的应用。LLM去学习技术为一种有前景的替代方案，可以直接去除负面示例的影响。然而，当前研究主要集中在实证验证上，缺乏系统的定量分析。为弥合这一差距，我们提出了一种框架来探讨偏好对齐与LLM去学习之间的关系。具体而言，我们引入了一种基于双层优化的方法来量化移除特定负面示例对偏好对齐性能的影响。我们的分析表明，并非所有负面示例在去除时对对齐改进的贡献都是均等的，且不同示例的效果差异显著。基于这一洞见，我们提出了一个关键问题：如何通过优化选择和加权负面示例来最大化偏好对齐性能？为回答这一问题，我们提出了一种名为U2A（Unlearning to Align）的框架，利用双层优化有效地选择和去除示例以实现最优的偏好对齐性能。我们通过广泛的实验验证了所提出的方法，结果证实了其有效性。 

---
# A Neuro-inspired Interpretation of Unlearning in Large Language Models through Sample-level Unlearning Difficulty 

**Title (ZH)**: 大型语言模型中基于样本层面遗忘难度的神经启发式遗忘解释 

**Authors**: Xiaohua Feng, Yuyuan Li, Chengye Wang, Junlin Liu, Li Zhang, Chaochao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.06658)  

**Abstract**: Driven by privacy protection laws and regulations, unlearning in Large Language Models (LLMs) is gaining increasing attention. However, current research often neglects the interpretability of the unlearning process, particularly concerning sample-level unlearning difficulty. Existing studies typically assume a uniform unlearning difficulty across samples. This simplification risks attributing the performance of unlearning algorithms to sample selection rather than the algorithm's design, potentially steering the development of LLM unlearning in the wrong direction. Thus, we investigate the relationship between LLM unlearning and sample characteristics, with a focus on unlearning difficulty. Drawing inspiration from neuroscience, we propose a Memory Removal Difficulty ($\mathrm{MRD}$) metric to quantify sample-level unlearning difficulty. Using $\mathrm{MRD}$, we analyze the characteristics of hard-to-unlearn versus easy-to-unlearn samples. Furthermore, we propose an $\mathrm{MRD}$-based weighted sampling method to optimize existing unlearning algorithms, which prioritizes easily forgettable samples, thereby improving unlearning efficiency and effectiveness. We validate the proposed metric and method using public benchmarks and datasets, with results confirming its effectiveness. 

**Abstract (ZH)**: 受隐私保护法律法规驱动，大规模语言模型的去学习问题正逐渐引起关注。然而，当前研究往往忽视了去学习过程的可解释性，特别是针对样本级别的去学习难度。现有研究通常假设样本的去学习难度一致。这一简化可能导致将去学习算法的性能归因于样本选择，而非算法设计，从而可能误导大规模语言模型去学习的发展方向。因此，我们探讨了大规模语言模型的去学习与其样本特征之间的关系，重点关注去学习难度。受神经科学的启发，我们提出了一种记忆移除难度（$\mathrm{MRD}$）度量标准来量化样本级别的去学习难度。利用$\mathrm{MRD}$，我们分析了难以去学习的样本与容易去学习的样本的特征。此外，我们提出了一种基于$\mathrm{MRD}$的加权采样方法，以优化现有的去学习算法，优先考虑容易忘记的样本，从而提高去学习的效率和效果。我们使用公开的基准和数据集验证了所提出的度量标准和方法，结果证实了其有效性。 

---
# Automated Business Process Analysis: An LLM-Based Approach to Value Assessment 

**Title (ZH)**: 基于大语言模型的自动化业务流程分析：价值评估方法 

**Authors**: William De Michele, Abel Armas Cervantes, Lea Frermann  

**Link**: [PDF](https://arxiv.org/pdf/2504.06600)  

**Abstract**: Business processes are fundamental to organizational operations, yet their optimization remains challenging due to the timeconsuming nature of manual process analysis. Our paper harnesses Large Language Models (LLMs) to automate value-added analysis, a qualitative process analysis technique that aims to identify steps in the process that do not deliver value. To date, this technique is predominantly manual, time-consuming, and subjective. Our method offers a more principled approach which operates in two phases: first, decomposing high-level activities into detailed steps to enable granular analysis, and second, performing a value-added analysis to classify each step according to Lean principles. This approach enables systematic identification of waste while maintaining the semantic understanding necessary for qualitative analysis. We develop our approach using 50 business process models, for which we collect and publish manual ground-truth labels. Our evaluation, comparing zero-shot baselines with more structured prompts reveals (a) a consistent benefit of structured prompting and (b) promising performance for both tasks. We discuss the potential for LLMs to augment human expertise in qualitative process analysis while reducing the time and subjectivity inherent in manual approaches. 

**Abstract (ZH)**: 利用大型语言模型自动进行增值分析以优化业务流程：一种基于精益原则的两阶段方法 

---
# Lugha-Llama: Adapting Large Language Models for African Languages 

**Title (ZH)**: Lugha-Llama：适应非洲语言的大规模语言模型 

**Authors**: Happy Buzaaba, Alexander Wettig, David Ifeoluwa Adelani, Christiane Fellbaum  

**Link**: [PDF](https://arxiv.org/pdf/2504.06536)  

**Abstract**: Large language models (LLMs) have achieved impressive results in a wide range of natural language applications. However, they often struggle to recognize low-resource languages, in particular African languages, which are not well represented in large training corpora. In this paper, we consider how to adapt LLMs to low-resource African languages. We find that combining curated data from African languages with high-quality English educational texts results in a training mix that substantially improves the model's performance on these languages. On the challenging IrokoBench dataset, our models consistently achieve the best performance amongst similarly sized baselines, particularly on knowledge-intensive multiple-choice questions (AfriMMLU). Additionally, on the cross-lingual question answering benchmark AfriQA, our models outperform the base model by over 10%. To better understand the role of English data during training, we translate a subset of 200M tokens into Swahili language and perform an analysis which reveals that the content of these data is primarily responsible for the strong performance. We release our models and data to encourage future research on African languages. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种自然语言应用中取得了令人瞩目的成果。然而，它们往往难以识别低资源语言，特别是非洲语言，这些语言在大规模训练语料库中代表性不足。本文探讨了如何将LLMs适应低资源非洲语言。我们发现，将非洲语言的精选数据与高质量的英语教育文本结合，形成了一种训练混合数据，显著提高了模型在这些语言上的表现。在具有挑战性的IrokoBench数据集上，我们的模型在相同规模的基础模型中始终取得最佳性能，特别是在知识密集型多项选择题（AfriMMLU）方面。此外，在跨语言问答基准AfriQA上，我们的模型比基础模型高出超过10%。为了更好地理解训练过程中英语数据的作用，我们将其部分2亿个词元翻译成斯瓦希里语，并进行了一项分析，结果表明这些数据的内容主要负责了这种强劲的表现。我们发布了我们的模型和数据，以鼓励对非洲语言未来的研究。 

---
# Can you Finetune your Binoculars? Embedding Text Watermarks into the Weights of Large Language Models 

**Title (ZH)**: 你可以微调你的双筒望远镜吗？将文本水印嵌入大型语言模型的权重中 

**Authors**: Fay Elhassan, Niccolò Ajroldi, Antonio Orvieto, Jonas Geiping  

**Link**: [PDF](https://arxiv.org/pdf/2504.06446)  

**Abstract**: The indistinguishability of AI-generated content from human text raises challenges in transparency and accountability. While several methods exist to watermark models behind APIs, embedding watermark strategies directly into model weights that are later reflected in the outputs of the model is challenging. In this study we propose a strategy to finetune a pair of low-rank adapters of a model, one serving as the text-generating model, and the other as the detector, so that a subtle watermark is embedded into the text generated by the first model and simultaneously optimized for detectability by the second. In this way, the watermarking strategy is fully learned end-to-end. This process imposes an optimization challenge, as balancing watermark robustness, naturalness, and task performance requires trade-offs. We discuss strategies on how to optimize this min-max objective and present results showing the effect of this modification to instruction finetuning. 

**Abstract (ZH)**: AI生成内容与人类文本难以区分增加了透明度和问责制的挑战：一种端到端学习的细调策略 

---
# Don't Let It Hallucinate: Premise Verification via Retrieval-Augmented Logical Reasoning 

**Title (ZH)**: 不要让它幻觉：基于检索增强逻辑推理的前提验证 

**Authors**: Yuehan Qin, Shawn Li, Yi Nian, Xinyan Velocity Yu, Yue Zhao, Xuezhe Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.06438)  

**Abstract**: Large language models (LLMs) have shown substantial capacity for generating fluent, contextually appropriate responses. However, they can produce hallucinated outputs, especially when a user query includes one or more false premises-claims that contradict established facts. Such premises can mislead LLMs into offering fabricated or misleading details. Existing approaches include pretraining, fine-tuning, and inference-time techniques that often rely on access to logits or address hallucinations after they occur. These methods tend to be computationally expensive, require extensive training data, or lack proactive mechanisms to prevent hallucination before generation, limiting their efficiency in real-time applications. We propose a retrieval-based framework that identifies and addresses false premises before generation. Our method first transforms a user's query into a logical representation, then applies retrieval-augmented generation (RAG) to assess the validity of each premise using factual sources. Finally, we incorporate the verification results into the LLM's prompt to maintain factual consistency in the final output. Experiments show that this approach effectively reduces hallucinations, improves factual accuracy, and does not require access to model logits or large-scale fine-tuning. 

**Abstract (ZH)**: 大规模语言模型（LLMs）展示了生成流畅且上下文相关响应的显著能力。然而，当用户查询包含一个或多个虚假前提（与已确立的事实相矛盾的断言）时，它们可能会生成虚构或误导性的输出。现有方法包括预训练、微调以及推理时的技术，这些方法通常依赖于对logits的访问，或者在生成后处理幻觉。这些方法往往计算成本高、需要大量的训练数据，或者缺乏在生成前预防幻觉的主动机制，限制了它们在实时应用中的效率。我们提出了一种检索为基础的框架，在生成之前识别并处理虚假前提。该方法首先将用户的查询转化为逻辑表示，然后运用检索增强生成（RAG）来使用事实来源评估每个前提的有效性。最后，我们将验证结果融入到LLM的提示中，以确保最终输出的符合事实。实验表明，这种方法能够有效减少幻觉，提高事实准确性，并且不需要访问模型logits或大规模微调。 

---
# Language-Dependent Political Bias in AI: A Study of ChatGPT and Gemini 

**Title (ZH)**: 依赖语言的政治偏见在AI中：ChatGPT和Gemini的研究 

**Authors**: Dogus Yuksel, Mehmet Cem Catalbas, Bora Oc  

**Link**: [PDF](https://arxiv.org/pdf/2504.06436)  

**Abstract**: As leading examples of large language models, ChatGPT and Gemini claim to provide accurate and unbiased information, emphasizing their commitment to political neutrality and avoidance of personal bias. This research investigates the political tendency of large language models and the existence of differentiation according to the query language. For this purpose, ChatGPT and Gemini were subjected to a political axis test using 14 different languages. The findings of the study suggest that these large language models do exhibit political tendencies, with both models demonstrating liberal and leftist biases. A comparative analysis revealed that Gemini exhibited a more pronounced liberal and left-wing tendency compared to ChatGPT. The study also found that these political biases varied depending on the language used for inquiry. The study delves into the factors that constitute political tendencies and linguistic differentiation, exploring differences in the sources and scope of educational data, structural and grammatical features of languages, cultural and political contexts, and the model's response to linguistic features. From this standpoint, and an ethical perspective, it is proposed that artificial intelligence tools should refrain from asserting a lack of political tendencies and neutrality, instead striving for political neutrality and executing user queries by incorporating these tendencies. 

**Abstract (ZH)**: 作为大型语言模型的领先范例，ChatGPT和Gemini声称提供准确且无偏见的信息，强调其政治中立和避免个人偏见的承诺。本研究调查了大型语言模型的政治倾向及其查询语言根据政治轴的分化情况。为此，使用14种不同的语言对ChatGPT和Gemini进行了政治轴测试。研究结果表明，这些大型语言模型确实表现出政治倾向，两者都表现出自由派和左倾偏见。对比分析显示，Gemini相比ChatGPT更表现出明显的自由派和左倾倾向。研究还发现，这些政治偏见在使用不同语言进行查询时有所差异。本研究深入探讨了构成政治倾向和语言分化的因素，探索了教育资源和范围、语言的结构和语法特征、文化及政治背景以及模型对语言特征的响应差异。从这一视角及伦理角度来看，本研究建议人工智能工具应避免声称缺乏政治倾向和中立性，而是追求政治中立并综合考虑这些倾向执行用户查询。 

---
# A Geometric-Aware Perspective and Beyond: Hybrid Quantum-Classical Machine Learning Methods 

**Title (ZH)**: 具几何感知视角与超越：混合量子-经典机器学习方法 

**Authors**: Azadeh Alavia, Hossein Akhoundib, Fatemeh Kouchmeshkib, Mojtaba Mahmoodianc, Sanduni Jayasinghec, Yongli Rena, Abdolrahman Alavi  

**Link**: [PDF](https://arxiv.org/pdf/2504.06328)  

**Abstract**: Geometric Machine Learning (GML) has shown that respecting non-Euclidean geometry in data spaces can significantly improve performance over naive Euclidean assumptions. In parallel, Quantum Machine Learning (QML) has emerged as a promising paradigm that leverages superposition, entanglement, and interference within quantum state manifolds for learning tasks. This paper offers a unifying perspective by casting QML as a specialized yet more expressive branch of GML. We argue that quantum states, whether pure or mixed, reside on curved manifolds (e.g., projective Hilbert spaces or density-operator manifolds), mirroring how covariance matrices inhabit the manifold of symmetric positive definite (SPD) matrices or how image sets occupy Grassmann manifolds. However, QML also benefits from purely quantum properties, such as entanglement-induced curvature, that can yield richer kernel structures and more nuanced data embeddings.
We illustrate these ideas with published and newly discussed results, including hybrid classical -quantum pipelines for diabetic foot ulcer classification and structural health monitoring. Despite near-term hardware limitations that constrain purely quantum solutions, hybrid architectures already demonstrate tangible benefits by combining classical manifold-based feature extraction with quantum embeddings. We present a detailed mathematical treatment of the geometrical underpinnings of quantum states, emphasizing parallels to classical Riemannian geometry and manifold-based optimization. Finally, we outline open research challenges and future directions, including Quantum Large Language Models (LLMs), quantum reinforcement learning, and emerging hardware approaches, demonstrating how synergizing GML and QML principles can unlock the next generation of machine intelligence. 

**Abstract (ZH)**: 几何机器学习（GML）表明，在数据空间中尊重非欧几里得几何可以显著提高性能，超越了简单的欧几里得假设。与此同时，量子机器学习（QML）作为一种有前景的范式出现了，它利用量子态流形中的叠加、纠缠和干涉来进行学习任务。本文从统一的角度将QML视为GML的一个专门但更具有表现力的分支。我们argue量子态，无论是纯态还是混合态，都位于弯曲流形上（例如，投影希洛特空间或密度算子流形），这类似于协方差矩阵存在于对称正定（SPD）矩阵的流形上，或者图像集占据格拉斯曼流形。然而，QML还受益于纯粹的量子属性，例如由纠缠引起的曲率，这些属性可以产生更丰富的核结构和更细腻的数据嵌入。

我们通过一些已发表和新讨论的结果，如糖尿病足溃疡分类和结构健康监测的混合经典-量子管道来阐述这些观点。尽管短期内硬件限制阻碍了纯量子解决方案的发展，但混合架构已经通过结合基于流形的经典特征提取与量子嵌入显示了实际优势。我们详述了量子态的几何基础，强调与经典黎曼几何及基于流形的优化的类比关系。最后，我们概述了开放的研究挑战和未来方向，包括量子大型语言模型（LLMs）、量子强化学习和新兴硬件方法，并展示了如何结合GML和QML原则来解锁下一代机器智能。 

---
# From Stability to Inconsistency: A Study of Moral Preferences in LLMs 

**Title (ZH)**: 从稳定到不一致性：LLM中道德偏好的研究 

**Authors**: Monika Jotautaite, Mary Phuong, Chatrik Singh Mangat, Maria Angelica Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2504.06324)  

**Abstract**: As large language models (LLMs) increasingly integrate into our daily lives, it becomes crucial to understand their implicit biases and moral tendencies. To address this, we introduce a Moral Foundations LLM dataset (MFD-LLM) grounded in Moral Foundations Theory, which conceptualizes human morality through six core foundations. We propose a novel evaluation method that captures the full spectrum of LLMs' revealed moral preferences by answering a range of real-world moral dilemmas. Our findings reveal that state-of-the-art models have remarkably homogeneous value preferences, yet demonstrate a lack of consistency. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）日益融入我们的日常生活中，理解其隐含偏见和道德倾向变得至关重要。为此，我们引入了一个基于道德基础理论（Moral Foundations Theory）的道德基础LLM数据集（MFD-LLM），该理论通过六个核心基础来概念化人类的道德观。我们提出了一种新颖的评估方法，通过回答一系列现实生活中的道德难题来捕捉LLMs展现出来的完整道德偏好谱系。我们的研究发现，最先进的模型在价值观偏好上表现出显著的同质性，但缺乏一致性。 

---
# Mosaic: Composite Projection Pruning for Resource-efficient LLMs 

**Title (ZH)**: 拼图：面向资源高效的大语言模型的复合投影剪枝 

**Authors**: Bailey J. Eccles, Leon Wong, Blesson Varghese  

**Link**: [PDF](https://arxiv.org/pdf/2504.06323)  

**Abstract**: Extensive compute and memory requirements limit the deployment of large language models (LLMs) on any hardware. Compression methods, such as pruning, can reduce model size, which in turn reduces resource requirements. State-of-the-art pruning is based on coarse-grained methods. They are time-consuming and inherently remove critical model parameters, adversely impacting the quality of the pruned model. This paper introduces projection pruning, a novel fine-grained method for pruning LLMs. In addition, LLM projection pruning is enhanced by a new approach we refer to as composite projection pruning - the synergistic combination of unstructured pruning that retains accuracy and structured pruning that reduces model size. We develop Mosaic, a novel system to create and deploy pruned LLMs using composite projection pruning. Mosaic is evaluated using a range of performance and quality metrics on multiple hardware platforms, LLMs, and datasets. Mosaic is 7.19x faster in producing models than existing approaches. Mosaic models achieve up to 84.2% lower perplexity and 31.4% higher accuracy than models obtained from coarse-grained pruning. Up to 67% faster inference and 68% lower GPU memory use is noted for Mosaic models. 

**Abstract (ZH)**: 大规模语言模型的广泛计算和内存需求限制了其在任何硬件上的部署。压缩方法，如剪枝，可以减小模型大小，从而减少资源需求。最先进的剪枝方法基于粗粒度的方法。它们耗时且不可避免地会移除关键的模型参数，负面影响了剪枝模型的质量。本文介绍了一种名为投影剪枝的新颖细粒度方法，用于剪枝大规模语言模型。此外，通过一种我们称之为复合投影剪枝的全新方法——无结构剪枝保留准确性和有结构剪枝减少模型大小的协同组合，增强了大规模语言模型的投影剪枝。我们开发了Mosaic，一种新型系统，使用复合投影剪枝创建和部署剪枝的大规模语言模型。Mosaic在多种硬件平台、大规模语言模型和数据集上使用范围广泛的性能和质量指标进行了评估。Mosaic在生成模型方面的速度比现有方法快7.19倍。Mosaic模型的困惑度降低了最多84.2%，准确率提高了31.4%，高于粗粒度剪枝获得的模型。Mosaic模型的推理速度提高了最多67%，GPU内存使用量降低了68%。 

---
# Accelerating LLM Inference Throughput via Asynchronous KV Cache Prefetching 

**Title (ZH)**: 通过异步键值缓存预取加速LLM推理 throughput 

**Authors**: Yanhao Dong, Yubo Miao, Weinan Li, Xiao Zheng, Chao Wang, Feng Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06319)  

**Abstract**: Large Language Models (LLMs) exhibit pronounced memory-bound characteristics during inference due to High Bandwidth Memory (HBM) bandwidth constraints. In this paper, we propose an L2 Cache-oriented asynchronous KV Cache prefetching method to break through the memory bandwidth bottleneck in LLM inference through computation-load overlap. By strategically scheduling idle memory bandwidth during active computation windows, our method proactively prefetches required KV Cache into GPU L2 cache, enabling high-speed L2 cache hits for subsequent accesses and effectively hiding HBM access latency within computational cycles. Extensive experiments on NVIDIA H20 GPUs demonstrate that the proposed method achieves 2.15x improvement in attention kernel efficiency and up to 1.97x end-to-end throughput enhancement, surpassing state-of-the-art baseline FlashAttention-3. Notably, our solution maintains orthogonality to existing optimization techniques and can be integrated with current inference frameworks, providing a scalable latency-hiding solution for next-generation LLM inference engines. 

**Abstract (ZH)**: Large Language Models的推理过程中由于高带宽内存(HBM)带宽约束表现出显著的记忆绑定特性。本文提出了一种面向L2缓存的异步键值缓存预取方法，通过计算负载重叠突破LLM推理中的内存带宽瓶颈。通过在活跃计算窗口期间战略性调度闲置的内存带宽，本方法主动将所需的键值缓存预取到GPU L2缓存中，从而在后续访问时实现高速L2缓存命中，并有效隐藏HBM访问延迟到计算周期内。实验结果表明，本方法在NVIDIA H20 GPU上实现了注意力核效率2.15倍的提升，并最多提升了1.97倍的端到端吞吐量，超越了最新的基准FlashAttention-3。值得注意的是，本解决方案与现有的优化技术保持正交，并且可以与当前的推理框架集成，为下一代LLM推理引擎提供可扩展的延迟隐藏解决方案。 

---
# Rethinking RoPE: A Mathematical Blueprint for N-dimensional Positional Encoding 

**Title (ZH)**: 重新思考RoPE：N维位置编码的数学蓝图 

**Authors**: Haiping Liu, Hongpeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2504.06308)  

**Abstract**: Rotary Position Embedding (RoPE) is widely adopted in Transformers due to its ability to encode relative positions with high efficiency and extrapolation capability. However, existing RoPE variants lack a unified theoretical foundation, especially in higher dimensions. In this paper, we propose a systematic mathematical framework for RoPE grounded in Lie group and Lie algebra theory. We identify two core properties of RoPE, named relativity and reversibility, and derive general constraints and constructions for valid RoPE in 1D, 2D, and N-dimensional (ND). We prove that RoPE must lie in the basis of a maximal abelian subalgebra (MASA) of the special orthogonal Lie algebra, and show that standard RoPE corresponds to the maximal toral subalgebra. Furthermore, we propose to model inter-dimensional interactions by learning an orthogonal basis transformation. Our framework unifies and explains existing RoPE designs, while enabling principled extensions to new modalities and tasks. 

**Abstract (ZH)**: 基于李群和李代数理论的旋转位置嵌入系统化数学框架 

---
# Optimizing Large Language Models: Metrics, Energy Efficiency, and Case Study Insights 

**Title (ZH)**: 优化大型语言模型：评价指标、能源效率及案例研究洞察 

**Authors**: Tahniat Khan, Soroor Motie, Sedef Akinli Kocak, Shaina Raza  

**Link**: [PDF](https://arxiv.org/pdf/2504.06307)  

**Abstract**: The rapid adoption of large language models (LLMs) has led to significant energy consumption and carbon emissions, posing a critical challenge to the sustainability of generative AI technologies. This paper explores the integration of energy-efficient optimization techniques in the deployment of LLMs to address these environmental concerns. We present a case study and framework that demonstrate how strategic quantization and local inference techniques can substantially lower the carbon footprints of LLMs without compromising their operational effectiveness. Experimental results reveal that these methods can reduce energy consumption and carbon emissions by up to 45\% post quantization, making them particularly suitable for resource-constrained environments. The findings provide actionable insights for achieving sustainability in AI while maintaining high levels of accuracy and responsiveness. 

**Abstract (ZH)**: 大规模语言模型的快速 Adopt 采用导致了显著的能源消耗和碳排放，对生成型 AI 技术的可持续性构成了关键挑战。本文探讨了在部署大规模语言模型时集成高效的能源优化技术，以应对这些环境问题。我们展示了一项案例研究和框架，说明如何通过战略性量化和本地推理技术大幅降低大规模语言模型的碳足迹，而不影响其操作有效性。实验结果表明，这些方法在量化之后可以降低高达 45% 的能源消耗和碳排放，特别适合资源受限的环境。研究结果提供了在保证高准确性和响应性的基础上实现 AI 可持续性的可行见解。 

---
# On the Effectiveness and Generalization of Race Representations for Debiasing High-Stakes Decisions 

**Title (ZH)**: 高风险决策中种族表示的纠偏有效性与泛化能力研究 

**Authors**: Dang Nguyen, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.06303)  

**Abstract**: Understanding and mitigating biases is critical for the adoption of large language models (LLMs) in high-stakes decision-making. We introduce Admissions and Hiring, decision tasks with hypothetical applicant profiles where a person's race can be inferred from their name, as simplified test beds for racial bias. We show that Gemma 2B Instruct and LLaMA 3.2 3B Instruct exhibit strong biases. Gemma grants admission to 26% more White than Black applicants, and LLaMA hires 60% more Asian than White applicants. We demonstrate that these biases are resistant to prompt engineering: multiple prompting strategies all fail to promote fairness. In contrast, using distributed alignment search, we can identify "race subspaces" within model activations and intervene on them to debias model decisions. Averaging the representation across all races within the subspaces reduces Gemma's bias by 37-57%. Finally, we examine the generalizability of Gemma's race subspaces, and find limited evidence for generalization, where changing the prompt format can affect the race representation. Our work suggests mechanistic approaches may provide a promising venue for improving the fairness of LLMs, but a universal race representation remains elusive. 

**Abstract (ZH)**: 理解并减轻偏差对于在高风险决策中采用大规模语言模型（LLMs）至关重要。我们介绍了 Admission 和 Hiring，具有假设申请人背景的任务，在这些任务中可以从姓名推断出一个人的种族，作为种族偏差的简化测试平台。我们显示，Gemma 2B Instruct 和 LLaMA 3.2 3B Instruct 显示出强烈的偏差。Gemma 向白人申请者授予入学资格的比例比向非洲裔美国人申请者高 26%，而 LLaMA 对亚裔申请者的雇佣率比对白人申请者的高 60%。我们证明这些偏差对提示工程具有抵抗力：多种提示策略均未能促进公正性。相比之下，通过分布式对齐搜索，我们可以在模型激活中识别出“种族子空间”，并对它们进行干预以消除模型决策中的偏差。在子空间内跨所有种族平均表示将 Gemma 的偏差降低 37-57%。最后，我们检查了 Gemma 的种族子空间的一般适用性，发现变化提示格式会影响种族表示的有限证据。我们的工作表明，机制性方法可能为改进 LLM 的公平性提供有前景的途径，但普遍适用的种族表示仍然难以实现。 

---
# Dynamic Evaluation Framework for Personalized and Trustworthy Agents: A Multi-Session Approach to Preference Adaptability 

**Title (ZH)**: 个性化和可信代理的动态评估框架：一种会话多阶段偏好适应方法 

**Authors**: Chirag Shah, Hideo Joho, Kirandeep Kaur, Preetam Prabhu Srikar Dammu  

**Link**: [PDF](https://arxiv.org/pdf/2504.06277)  

**Abstract**: Recent advancements in generative AI have significantly increased interest in personalized agents. With increased personalization, there is also a greater need for being able to trust decision-making and action taking capabilities of these agents. However, the evaluation methods for these agents remain outdated and inadequate, often failing to capture the dynamic and evolving nature of user interactions. In this conceptual article, we argue for a paradigm shift in evaluating personalized and adaptive agents. We propose a comprehensive novel framework that models user personas with unique attributes and preferences. In this framework, agents interact with these simulated users through structured interviews to gather their preferences and offer customized recommendations. These recommendations are then assessed dynamically using simulations driven by Large Language Models (LLMs), enabling an adaptive and iterative evaluation process. Our flexible framework is designed to support a variety of agents and applications, ensuring a comprehensive and versatile evaluation of recommendation strategies that focus on proactive, personalized, and trustworthy aspects. 

**Abstract (ZH)**: 近期生成式人工智能的发展显著增加了对个性化代理的兴趣。随着个性化程度的提高，人们也需要更信任这些代理的决策能力和行动能力。然而，这些代理的评估方法仍然过时且不足，往往未能捕捉用户交互的动态和演变特性。在本文中，我们提出了评估个性化和适应性代理范式的转变。我们提出了一种综合性的新框架，该框架基于具有独特属性和偏好的用户画像。在这种框架中，代理通过结构化的访谈与这些模拟用户交互，以收集用户的偏好并提供定制化建议。随后，这些建议通过大型语言模型（LLMs）驱动的模拟动态评估，实现适应性和迭代性的评估过程。我们的灵活框架旨在支持各种代理和应用，确保对注重前瞻性、个性化和可信度的推荐策略进行全面而多样的评估。 

---
# ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources 

**Title (ZH)**: ER-RAG: 基于ER统一异构数据源建模增强RAG 

**Authors**: Yikuan Xia, Jiazun Chen, Yirui Zhan, Suifeng Zhao, Weipeng Jiang, Chaorui Zhang, Wei Han, Bo Bai, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.06271)  

**Abstract**: Large language models (LLMs) excel in question-answering (QA) tasks, and retrieval-augmented generation (RAG) enhances their precision by incorporating external evidence from diverse sources like web pages, databases, and knowledge graphs. However, current RAG methods rely on agent-specific strategies for individual data sources, posing challenges low-resource or black-box environments and complicates operations when evidence is fragmented across sources. To address these limitations, we propose ER-RAG, a framework that unifies evidence integration across heterogeneous data sources using the Entity-Relationship (ER) model. ER-RAG standardizes entity retrieval and relationship querying through ER-based APIs with GET and JOIN operations. It employs a two-stage generation process: first, a preference optimization module selects optimal sources; second, another module constructs API chains based on source schemas. This unified approach allows efficient fine-tuning and seamless integration across diverse data sources. ER-RAG demonstrated its effectiveness by winning all three tracks of the 2024 KDDCup CRAG Challenge, achieving performance on par with commercial RAG pipelines using an 8B LLM backbone. It outperformed hybrid competitors by 3.1% in LLM score and accelerated retrieval by 5.5X. 

**Abstract (ZH)**: 大型语言模型（LLMs）在问答（QA）任务中表现出色，检索增强生成（RAG）通过整合来自网页、数据库和知识图谱等多样来源的外部证据来提高其精确度。然而，当前的RAG方法依赖于针对个别数据源的特定策略，这在低资源或黑盒环境中提出了挑战，并且当证据分散在多个来源时会复杂化操作。为了解决这些限制，我们提出了一种ER-RAG框架，该框架使用实体-关系（ER）模型统一异构数据源的证据集成。ER-RAG通过基于ER的APIs和GET、JOIN操作标准化实体检索和关系查询。它采用两阶段生成过程：首先，偏好优化模块选择最佳来源；其次，另一个模块基于数据源模式构建API链。这种统一的方法允许在多种数据源之间高效调整和无缝集成。ER-RAG通过赢得2024 KDDCup CRAG挑战的所有三个赛道，展示了其有效性，使用8B LLM骨干时，其性能与商用RAG流水线相当。与混合竞争对手相比，其LLM得分为3.1%的提升，并加速了检索速度5.5倍。 

---
# StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization 

**Title (ZH)**: StealthRank: 通过隐蔽的提示优化进行LLM排名操纵 

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2504.05804)  

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems. 

**Abstract (ZH)**: 大型语言模型（LLMs）集成到信息检索系统中引入了新的攻击面，尤其是针对对抗性排名操纵。我们提出了一种名为StealthRank的新型对抗性排名攻击方法，该方法在保持文本流畅性和隐蔽性的同时操纵LLM驱动的产品推荐系统。与现有方法往往引入可检测的异常不同，StealthRank采用能量优化框架结合拉格朗日动态机制生成StealthRank提示（SRPs），即嵌入在产品描述中的对抗性文本序列，这些序列微妙而有效地影响LLM的排名机制。我们跨多个LLM评估了StealthRank，展示了其隐蔽提升目标产品排名的能力，同时避免了容易被检测的明确操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面均优于最先进的对抗性排名基准，突显了LLM驱动的推荐系统中的关键漏洞。 

---
# Leveraging LLMs for User Stories in AI Systems: UStAI Dataset 

**Title (ZH)**: 利用大语言模型在AI系统中的用户故事：UStAI数据集 

**Authors**: Asma Yamani, Malak Baslyman, Moataz Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2504.00513)  

**Abstract**: AI systems are gaining widespread adoption across various sectors and domains. Creating high-quality AI system requirements is crucial for aligning the AI system with business goals and consumer values and for social responsibility. However, with the uncertain nature of AI systems and the heavy reliance on sensitive data, more research is needed to address the elicitation and analysis of AI systems requirements. With the proprietary nature of many AI systems, there is a lack of open-source requirements artifacts and technical requirements documents for AI systems, limiting broader research and investigation. With Large Language Models (LLMs) emerging as a promising alternative to human-generated text, this paper investigates the potential use of LLMs to generate user stories for AI systems based on abstracts from scholarly papers. We conducted an empirical evaluation using three LLMs and generated $1260$ user stories from $42$ abstracts from $26$ domains. We assess their quality using the Quality User Story (QUS) framework. Moreover, we identify relevant non-functional requirements (NFRs) and ethical principles. Our analysis demonstrates that the investigated LLMs can generate user stories inspired by the needs of various stakeholders, offering a promising approach for generating user stories for research purposes and for aiding in the early requirements elicitation phase of AI systems. We have compiled and curated a collection of stories generated by various LLMs into a dataset (UStAI), which is now publicly available for use. 

**Abstract (ZH)**: 基于学术论文摘要的大规模语言模型生成AI系统用户故事的研究 

---
# CMAT: A Multi-Agent Collaboration Tuning Framework for Enhancing Small Language Models 

**Title (ZH)**: CMAT：增强小型语言模型的多代理协作调优框架 

**Authors**: Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, JingSong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2404.01663)  

**Abstract**: Open large language models (LLMs) have significantly advanced the field of natural language processing, showcasing impressive performance across various this http URL the significant advancements in LLMs, their effective operation still relies heavily on human input to accurately guide the dialogue flow, with agent tuning being a crucial optimization technique that involves human adjustments to the model for better response to such this http URL this dependency, our work introduces the TinyAgent model, trained on a meticulously curated high-quality dataset. We also present the Collaborative Multi-Agent Tuning (CMAT) framework, an innovative system designed to augment language agent capabilities through adaptive weight updates based on environmental feedback. This framework fosters collaborative learning and real-time adaptation among multiple intelligent agents, enhancing their context-awareness and long-term memory. In this research, we propose a new communication agent framework that integrates multi-agent systems with environmental feedback mechanisms, offering a scalable method to explore cooperative behaviors. Notably, our TinyAgent-7B model exhibits performance on par with GPT-3.5, despite having fewer parameters, signifying a substantial improvement in the efficiency and effectiveness of LLMs. 

**Abstract (ZH)**: 开放型大型语言模型（LLMs）在自然语言处理领域取得了显著进展，展现出在多种任务上的 impressive 表现。尽管在LLMs方面取得了重大进展，其有效运行仍然很大程度上依赖于人类输入来准确引导对话流程，代理调优是一种关键的优化技术，涉及对模型进行人工调整以更好地应对此类任务。为减少这种依赖，我们提出了TinyAgent模型，该模型基于精心筛选的高质量数据集进行训练。我们还提出了协作多代理调优（CMAT）框架，这是一种创新系统，通过基于环境反馈的自适应权重更新来增强语言代理的能力。该框架促进了多个智能代理之间的协作学习和实时适应，增强它们的上下文感知能力和长期记忆。在本研究中，我们提出了一种新的通信代理框架，将多代理系统与环境反馈机制相结合，提供了一种可扩展的方法来探索合作行为。值得注意的是，我们的TinyAgent-7B模型在参数量较少的情况下，性能与GPT-3.5相当，这表明在LLMs的效率和有效性方面取得了显著改进。 

---
