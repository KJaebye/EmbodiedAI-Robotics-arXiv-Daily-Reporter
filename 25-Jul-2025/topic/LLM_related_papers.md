# Revisiting LLM Reasoning via Information Bottleneck 

**Title (ZH)**: 重访通过信息瓶颈的大型语言模型推理 

**Authors**: Shiye Lei, Zhihao Cheng, Kai Jia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18391)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable progress in reasoning capabilities through reinforcement learning with verifiable rewards (RLVR). By leveraging simple rule-based rewards, RL effectively incentivizes LLMs to produce extended chain-of-thought (CoT) reasoning trajectories, progressively guiding them toward correct answers. However, existing approaches remain largely heuristic and intuition-driven, limiting the development of principled methodologies. In this paper, we present a theoretical characterization of LLM reasoning grounded in information bottleneck (IB) principle, introducing IB-aware reasoning optimization (IBRO), a framework that encourages reasoning trajectories to be both informative about the final correct answer and generalizable across diverse prompts. We derive a practical token-level surrogate objective and propose an efficient approximation, resulting in the lightweight IB regularization method. This technique integrates seamlessly into existing RL-based post-training frameworks without additional computational overhead, requiring only a one-line code modification. Empirically, we validate IB regularization across multiple mathematical reasoning benchmarks and RL algorithms, demonstrating consistent improvements in LLM reasoning performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过验证性奖励强化学习（RLVR）在推理能力方面 recently取得了显著进展。通过利用简单的规则基线奖励，RL 有效激励 LLMs 生成扩展的链式思考（CoT）推理轨迹，逐步引导其走向正确答案。然而，现有方法仍然主要依赖启发式方法和直觉驱动，限制了原理性方法的发展。在本文中，我们基于信息瓶颈（IB）原理对 LLM 推理进行了理论刻画，提出了信息瓶颈感知推理优化（IBRO）框架，该框架鼓励推理轨迹既关于最终正确答案具有信息性，又在多种提示下具有泛化性。我们推导出一个实用的 token 级别替代目标，并提出了一种高效的近似方法，从而得到轻量级的信息瓶颈正则化方法。该技术无缝集成到现有的 RL 基于训练后的方法中，无需额外的计算开销，仅需一行代码修改。经验上，我们在多个数学推理基准和 RL 算法上验证了信息瓶颈正则化方法，展示了在 LLM 推理性能上的一致性改进。 

---
# Reasoning Beyond the Obvious: Evaluating Divergent and Convergent Thinking in LLMs for Financial Scenarios 

**Title (ZH)**: 超越表面思考：评估LLMs在金融场景中发散思维和收敛思维的能力 

**Authors**: Zhuang Qiang Bok, Watson Wei Khong Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.18368)  

**Abstract**: Most reasoning benchmarks for LLMs emphasize factual accuracy or step-by-step logic. In finance, however, professionals must not only converge on optimal decisions but also generate creative, plausible futures under uncertainty. We introduce ConDiFi, a benchmark that jointly evaluates divergent and convergent thinking in LLMs for financial tasks.
ConDiFi features 607 macro-financial prompts for divergent reasoning and 990 multi-hop adversarial MCQs for convergent reasoning. Using this benchmark, we evaluated 14 leading models and uncovered striking differences. Despite high fluency, GPT-4o underperforms on Novelty and Actionability. In contrast, models like DeepSeek-R1 and Cohere Command R+ rank among the top for generating actionable, insights suitable for investment decisions. ConDiFi provides a new perspective to assess reasoning capabilities essential to safe and strategic deployment of LLMs in finance. 

**Abstract (ZH)**: ConDiFi：一种联合评估LLM发散与收敛思维能力的金融基准 

---
# Decoupling Knowledge and Reasoning in LLMs: An Exploration Using Cognitive Dual-System Theory 

**Title (ZH)**: 分离知识与推理在大语言模型中的作用：基于认知双系统理论的探索 

**Authors**: Mutian Yang, Jiandong Gao, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18178)  

**Abstract**: While large language models (LLMs) leverage both knowledge and reasoning during inference, the capacity to distinguish between them plays a pivotal role in model analysis, interpretability, and development. Inspired by dual-system cognitive theory, we propose a cognition attribution framework to decouple the contribution of knowledge and reasoning. In particular, the cognition of LLMs is decomposed into two distinct yet complementary phases: knowledge retrieval (Phase 1) and reasoning adjustment (Phase 2). To separate these phases, LLMs are prompted to generate answers under two different cognitive modes, fast thinking and slow thinking, respectively. The performance under different cognitive modes is analyzed to quantify the contribution of knowledge and reasoning. This architecture is employed to 15 LLMs across 3 datasets. Results reveal: (1) reasoning adjustment is domain-specific, benefiting reasoning-intensive domains (e.g., mathematics, physics, and chemistry) and potentially imparing knowledge-intensive domains. (2) Parameter scaling improves both knowledge and reasoning, with knowledge improvements being more pronounced. Additionally, parameter scaling make LLMs reasoning significantly more prudent, while moderately more intelligent. (3) Knowledge primarily resides in lower network layers, while reasoning operates in higher layers. Our framework not only helps understand LLMs from a "decoupling" perspective, but also provides new insights into existing research, including scaling laws, hierarchical knowledge editing, and limitations of small-model reasoning. 

**Abstract (ZH)**: 大语言模型知识与推理的认知归因框架：基于双系统认知理论的分解方法 

---
# E.A.R.T.H.: Structuring Creative Evolution through Model Error in Generative AI 

**Title (ZH)**: E.A.R.T.H.: 通过模型误差结构化生成式AI的创意进化 

**Authors**: Yusen Peng, Shuhua Mao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18004)  

**Abstract**: How can AI move beyond imitation toward genuine creativity? This paper proposes the E.A.R.T.H. framework, a five-stage generative pipeline that transforms model-generated errors into creative assets through Error generation, Amplification, Refine selection, Transform, and Harness feedback. Drawing on cognitive science and generative modeling, we posit that "creative potential hides in failure" and operationalize this via structured prompts, semantic scoring, and human-in-the-loop evaluation. Implemented using LLaMA-2-7B-Chat, SBERT, BERTScore, CLIP, BLIP-2, and Stable Diffusion, the pipeline employs a composite reward function based on novelty, surprise, and relevance. At the Refine stage, creativity scores increase by 52.5% (1.179 to 1.898, t = -5.56, p < 0.001), with final outputs reaching 2.010 - a 70.4% improvement. Refined slogans are 48.4% shorter, 40.7% more novel, with only a 4.0% drop in relevance. Cross-modal tests show strong slogan-to-image alignment (CLIPScore: 0.249; BERTScore F1: 0.816). In human evaluations, 60% of outputs scored >= 4.0, with metaphorical slogans (avg. 4.09) outperforming literal ones (3.99). Feedback highlights stylistic precision and emotional resonance. These results demonstrate that error-centered, feedback-driven generation enhances creativity, offering a scalable path toward self-evolving, human-aligned creative AI. 

**Abstract (ZH)**: AI如何从模仿迈向真正的创造力？本文提出了E.A.R.T.H.框架，一个五阶段生成管道，通过错误生成、增强、 refine选择、转换和利用反馈将模型生成的错误转化为创意资产。通过结合认知科学和生成建模，我们提出“创造力潜藏在失败之中”，并通过结构化提示、语义评分和人机交互评估进行操作化。该管道使用LLaMA-2-7B-Chat、SBERT、BERTScore、CLIP、BLIP-2和Stable Diffusion实现，并基于新颖性、惊讶性和相关性构建了复合奖励函数。在refine阶段，创意分数提高了52.5%（从1.179增加到1.898，t=-5.56，p<0.001），最终输出达到2.010，提升了70.4%。经过refine的口号缩短了48.4%，更加新颖，相关性仅下降了4.0%。跨模态测试显示口号与图像的强对齐（CLIPScore: 0.249；BERTScore F1: 0.816）。在人类评估中，60%的输出得分不低于4.0，具有隐喻的口号（平均4.09）优于字面的口号（3.99）。反馈强调了风格的精准和情感的共鸣。这些结果表明，以错误为中心、基于反馈的生成能够提升创造力，为自我进化的、与人类目标一致的创造型AI指明了一条可扩展的道路。 

---
# SMARTAPS: Tool-augmented LLMs for Operations Management 

**Title (ZH)**: SMARTAPS: 工具增强的大语言模型在运营管理中的应用 

**Authors**: Timothy Tin Long Yu, Mahdi Mostajabdaveh, Jabo Serge Byusa, Rindra Ramamonjison, Giuseppe Carenini, Kun Mao, Zirui Zhou, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17927)  

**Abstract**: Large language models (LLMs) present intriguing opportunities to enhance user interaction with traditional algorithms and tools in real-world applications. An advanced planning system (APS) is a sophisticated software that leverages optimization to help operations planners create, interpret, and modify an operational plan. While highly beneficial, many customers are priced out of using an APS due to the ongoing costs of consultants responsible for customization and maintenance. To address the need for a more accessible APS expressed by supply chain planners, we present SmartAPS, a conversational system built on a tool-augmented LLM. Our system provides operations planners with an intuitive natural language chat interface, allowing them to query information, perform counterfactual reasoning, receive recommendations, and execute scenario analysis to better manage their operation. A short video demonstrating the system has been released: this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）为增强传统算法和工具在实际应用中与用户的互动提供了令人intriguing兴奋的机会。一种先进的计划系统（APS）是一种利用优化来帮助运营规划者创建、解释和修改运营计划的高级软件。虽然功能强大，但由于负责定制和维护的咨询师的持续成本，许多客户无法使用APS。为应对供应链规划者对更易于访问的APS的需求，我们提出了基于工具增强的大语言模型构建的SmartAPS，一种会话系统。该系统为运营规划者提供了一个直观的自然语言聊天界面，使他们能够查询信息、进行反事实推理、接收建议并执行情景分析，以更好地管理其运营。有关该系统的short简短视频已经发布：this https URL。 

---
# AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs 

**Title (ZH)**: AQuilt: 将逻辑与自我检查融入低成本、高相关性数据合成以供专业大语言模型使用 

**Authors**: Xiaopeng Ke, Hexuan Deng, Xuebo Liu, Jun Rao, Zhenxi Song, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18584)  

**Abstract**: Despite the impressive performance of large language models (LLMs) in general domains, they often underperform in specialized domains. Existing approaches typically rely on data synthesis methods and yield promising results by using unlabeled data to capture domain-specific features. However, these methods either incur high computational costs or suffer from performance limitations, while also demonstrating insufficient generalization across different tasks. To address these challenges, we propose AQuilt, a framework for constructing instruction-tuning data for any specialized domains from corresponding unlabeled data, including Answer, Question, Unlabeled data, Inspection, Logic, and Task type. By incorporating logic and inspection, we encourage reasoning processes and self-inspection to enhance model performance. Moreover, customizable task instructions enable high-quality data generation for any task. As a result, we construct a dataset of 703k examples to train a powerful data synthesis model. Experiments show that AQuilt is comparable to DeepSeek-V3 while utilizing just 17% of the production cost. Further analysis demonstrates that our generated data exhibits higher relevance to downstream tasks. Source code, models, and scripts are available at this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在通用领域表现出色，但在专门领域往往表现不佳。现有方法通常依赖数据合成方法，并通过使用未标记数据来捕获领域特定特征，从而取得令人鼓舞的结果。然而，这些方法要么计算成本高昂，要么性能有限，同时也表现出在不同任务上泛化不足的问题。为应对这些挑战，我们提出AQuilt框架，用于从相应的未标记数据中构建任何专门领域的指令调优数据，包括Answer、Question、Unlabeled数据、Inspection、Logic和Task类型。通过整合逻辑和检查，我们鼓励推理过程和自我检查以提升模型性能。此外，可定制的任务指令能够为任何任务生成高质量的数据。因此，我们构建了一个包含703,000个示例的数据集来训练一个强大的数据合成模型。实验结果显示，AQuilt与DeepSeek-V3性能相当，但仅使用17%的生产成本。进一步的分析表明，我们生成的数据与下游任务的相关性更高。源代码、模型和脚本可在此链接访问。 

---
# HARLF: Hierarchical Reinforcement Learning and Lightweight LLM-Driven Sentiment Integration for Financial Portfolio Optimization 

**Title (ZH)**: HARLF: 分层强化学习和轻量级大语言模型驱动的情感集成在金融投资组合优化中的应用 

**Authors**: Benjamin Coriat, Eric Benhamou  

**Link**: [PDF](https://arxiv.org/pdf/2507.18560)  

**Abstract**: This paper presents a novel hierarchical framework for portfolio optimization, integrating lightweight Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) to combine sentiment signals from financial news with traditional market indicators. Our three-tier architecture employs base RL agents to process hybrid data, meta-agents to aggregate their decisions, and a super-agent to merge decisions based on market data and sentiment analysis. Evaluated on data from 2018 to 2024, after training on 2000-2017, the framework achieves a 26% annualized return and a Sharpe ratio of 1.2, outperforming equal-weighted and S&P 500 benchmarks. Key contributions include scalable cross-modal integration, a hierarchical RL structure for enhanced stability, and open-source reproducibility. 

**Abstract (ZH)**: 基于轻量级大型语言模型和深度强化学习的多层次投资组合优化框架 

---
# Automated Code Review Using Large Language Models with Symbolic Reasoning 

**Title (ZH)**: 使用符号推理的大语言模型驱动的自动化代码审查 

**Authors**: Busra Icoz, Goksel Biricik  

**Link**: [PDF](https://arxiv.org/pdf/2507.18476)  

**Abstract**: Code review is one of the key processes in the software development lifecycle and is essential to maintain code quality. However, manual code review is subjective and time consuming. Given its rule-based nature, code review is well suited for automation. In recent years, significant efforts have been made to automate this process with the help of artificial intelligence. Recent developments in Large Language Models (LLMs) have also emerged as a promising tool in this area, but these models often lack the logical reasoning capabilities needed to fully understand and evaluate code. To overcome this limitation, this study proposes a hybrid approach that integrates symbolic reasoning techniques with LLMs to automate the code review process. We tested our approach using the CodexGlue dataset, comparing several models, including CodeT5, CodeBERT, and GraphCodeBERT, to assess the effectiveness of combining symbolic reasoning and prompting techniques with LLMs. Our results show that this approach improves the accuracy and efficiency of automated code review. 

**Abstract (ZH)**: 代码审查是软件开发生命周期中的关键过程，对于维护代码质量至关重要。然而，手工代码审查具有主观性和耗时性。鉴于其基于规则的特性，代码审查非常适合自动化。近年来，借助人工智能的帮助，显著努力已经投入于自动化这一过程。大型语言模型的最新发展也为这一领域带来了前景广阔的工具，但这些模型往往缺乏充分理解并评估代码所需的逻辑推理能力。为克服这一局限，本研究提出了一种将符号推理技术与大型语言模型相结合的混合方法，以自动化代码审查过程。我们使用CodexGlue数据集测试了该方法，比较了包括CodeT5、CodeBERT和GraphCodeBERT在内的多种模型，评估将符号推理和提示技术与大型语言模型结合的有效性。研究结果表明，该方法提高了自动化代码审查的准确性和效率。 

---
# Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving 

**Title (ZH)**: Sandwich：分离预填充解码编译以实现高效CPU大语言模型服务 

**Authors**: Juntao Zhao, Jiuru Li, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18454)  

**Abstract**: Utilizing CPUs to serve large language models (LLMs) is a resource-friendly alternative to GPU serving. Existing CPU-based solutions ignore workload differences between the prefill and the decode phases of LLM inference, applying a static per-NUMA (Non-Uniform Memory Access) node model partition and utilizing vendor libraries for operator-level execution, which is suboptimal. We propose Sandwich, a hardware-centric CPU-based LLM serving engine that uses different execution plans for the prefill and decode phases and optimizes them separately.
We evaluate Sandwich across diverse baselines and datasets on five CPU platforms, including x86 with AVX-2 and AVX-512, as well as ARM with NEON. Sandwich achieves an average 2.01x throughput improvement and 90% satisfactory time-to-first-token (TTFT) and time-per-output-token (TPOT) latencies with up to 3.40x lower requirements in single sequence serving, and significant improvement in Goodput in continuous-batching serving. The GEMM kernels generated by Sandwich outperform representative vendor kernels and other dynamic shape solutions, achieving performance comparable to static compilers with three orders of magnitude less kernel tuning costs. 

**Abstract (ZH)**: 利用CPU服务于大型语言模型（LLMs）是一种资源友好的替代GPU服务的选择。现有的基于CPU的解决方案忽略了LLM推理中填充和解码阶段的工作负载差异，采用静态的非统一内存访问（NUMA）节点模型划分，并利用供应商库进行操作级执行，这不尽如人意。我们提出了Sandwich，一种以硬件为中心的基于CPU的LLM服务引擎，为填充和解码阶段使用不同的执行计划并分别进行优化。 

---
# Restoring Rhythm: Punctuation Restoration Using Transformer Models for Bangla, a Low-Resource Language 

**Title (ZH)**: 恢复节奏：使用变压器模型的孟加拉语标点恢复 

**Authors**: Md Obyedullahil Mamun, Md Adyelullahil Mamun, Arif Ahmad, Md. Imran Hossain Emu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18448)  

**Abstract**: Punctuation restoration enhances the readability of text and is critical for post-processing tasks in Automatic Speech Recognition (ASR), especially for low-resource languages like Bangla. In this study, we explore the application of transformer-based models, specifically XLM-RoBERTa-large, to automatically restore punctuation in unpunctuated Bangla text. We focus on predicting four punctuation marks: period, comma, question mark, and exclamation mark across diverse text domains. To address the scarcity of annotated resources, we constructed a large, varied training corpus and applied data augmentation techniques. Our best-performing model, trained with an augmentation factor of alpha = 0.20%, achieves an accuracy of 97.1% on the News test set, 91.2% on the Reference set, and 90.2% on the ASR set.
Results show strong generalization to reference and ASR transcripts, demonstrating the model's effectiveness in real-world, noisy scenarios. This work establishes a strong baseline for Bangla punctuation restoration and contributes publicly available datasets and code to support future research in low-resource NLP. 

**Abstract (ZH)**: 基于变压器模型的标点符号恢复提升孟加拉语文本可读性，并在自动语音识别后处理任务中至关重要，尤其是在低资源语言如孟加拉语领域。在本研究中，我们探讨了使用XLM-RoBERTa-large等基于变压器的模型自动恢复未标点孟加拉语文本中标点符号的应用。我们重点关注在多种文本领域预测四种标点符号：句号、逗号、问号和感叹号。为了解决标注资源稀缺的问题，我们构建了一个大型且多样化的训练语料库，并应用了数据增强技术。在增强因子α=0.20%的情况下，我们的最佳模型在News测试集上实现了97.1%的准确率，在Reference集上实现了91.2%的准确率，在ASR集上实现了90.2%的准确率。研究结果表明，该模型在参考和ASR转录中具有良好的泛化能力，证明了其在现实世界嘈杂场景中的有效性。本研究为孟加拉语标点符号恢复建立了强有力的基准，并提供了公开可用的数据集和代码，以支持未来低资源自然语言处理领域的研究。 

---
# AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data 

**Title (ZH)**: AraTable：评估LLMs在阿拉伯表格数据推理和理解方面的表现 

**Authors**: Rana Alshaikh, Israa Alghanmi, Shelan Jeawak  

**Link**: [PDF](https://arxiv.org/pdf/2507.18442)  

**Abstract**: The cognitive and reasoning abilities of large language models (LLMs) have enabled remarkable progress in natural language processing. However, their performance in interpreting structured data, especially in tabular formats, remains limited. Although benchmarks for English tabular data are widely available, Arabic is still underrepresented because of the limited availability of public resources and its unique language features. To address this gap, we present AraTable, a novel and comprehensive benchmark designed to evaluate the reasoning and understanding capabilities of LLMs when applied to Arabic tabular data. AraTable consists of various evaluation tasks, such as direct question answering, fact verification, and complex reasoning, involving a wide range of Arabic tabular sources. Our methodology follows a hybrid pipeline, where initial content is generated by LLMs and subsequently filtered and verified by human experts to ensure high dataset quality. Initial analyses using AraTable show that, while LLMs perform adequately on simpler tabular tasks such as direct question answering, they continue to face significant cognitive challenges when tasks require deeper reasoning and fact verification. This indicates that there are substantial opportunities for future work to improve performance on complex tabular reasoning tasks. We also propose a fully automated evaluation framework that uses a self-deliberation mechanism and achieves performance nearly identical to that of human judges. This research provides a valuable, publicly available resource and evaluation framework that can help accelerate the development of foundational models for processing and analysing Arabic structured data. 

**Abstract (ZH)**: 大型语言模型（LLMs）的认知与推理能力促进了自然语言处理的显著进步。然而，在解释结构化数据，尤其是表格格式数据方面的表现仍然有限。尽管英语文本表格基准数据广泛可用，阿拉伯语仍因公共资源有限和语言特点独特而相对欠缺。为弥补这一差距，我们提出了AraTable，一个新颖而全面的基准测试，旨在评估LLMs在阿拉伯语表格数据上的推理与理解能力。AraTable包含了多种评估任务，如直接问答、事实验证和复杂推理，涵盖广泛阿拉伯语表格来源。我们的方法采用混合管道，初始内容由LLMs生成，再由人类专家过滤和验证以确保数据集质量。初步分析表明，虽然LLMs在简单的表格任务，如直接问答上表现良好，但在需要更深推理和事实验证的任务中仍面临显著的认知挑战。这表明，未来工作在复杂表格推理任务上有很大的改进空间。我们还提出了一种完全自动化的评估框架，通过自动推理机制实现了与人类评委几乎相同的性能。本研究提供了有价值的公共可用资源和评估框架，有助于加快处理和分析阿拉伯语结构化数据的基础模型开发。 

---
# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy 

**Title (ZH)**: CLEAR: 通过LLM作为法官简化错误分析 

**Authors**: Asaf Yehudai, Lilach Eden, Yotam Perlitz, Roy Bar-Haim, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2507.18392)  

**Abstract**: The evaluation of Large Language Models (LLMs) increasingly relies on other LLMs acting as judges. However, current evaluation paradigms typically yield a single score or ranking, answering which model is better but not why. While essential for benchmarking, these top-level scores obscure the specific, actionable reasons behind a model's performance. To bridge this gap, we introduce CLEAR, an interactive, open-source package for LLM-based error analysis. CLEAR first generates per-instance textual feedback, then it creates a set of system-level error issues, and quantifies the prevalence of each identified issue. Our package also provides users with an interactive dashboard that allows for a comprehensive error analysis through aggregate visualizations, applies interactive filters to isolate specific issues or score ranges, and drills down to the individual instances that exemplify a particular behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks, and showcase its utility through a user case study. 

**Abstract (ZH)**: 大型语言模型（LLMs）的评估越来越多地依赖于其他LLMs作为评判者。然而，当前的评估范式通常仅给出单一得分或排名，回答了哪个模型更好但没有解释原因。虽然对于基准测试至关重要，这些高层面的得分掩盖了模型表现背后的具体可操作原因。为了解决这一差距，我们引入了CLEAR，一个基于LLM的错误分析交互式开源包。CLEAR首先生成针对每个实例的文本反馈，然后创建一系列系统级错误问题集，并量化每个识别问题的发生频率。我们的包还为用户提供了一个交互式仪表板，允许通过汇总可视化进行全面的错误分析，应用交互式滤镜以隔离特定问题或分数范围，并钻取到体现特定行为模式的个体实例。我们展示了CLEAR在RAG和Math基准测试中的分析，并通过用户案例研究突显其实用性。 

---
# LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models 

**Title (ZH)**: LoRA-Leak: 面向LoRA微调语言模型的成员推理攻击 

**Authors**: Delong Ran, Xinlei He, Tianshuo Cong, Anyu Wang, Qi Li, Xiaoyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18302)  

**Abstract**: Language Models (LMs) typically adhere to a "pre-training and fine-tuning" paradigm, where a universal pre-trained model can be fine-tuned to cater to various specialized domains. Low-Rank Adaptation (LoRA) has gained the most widespread use in LM fine-tuning due to its lightweight computational cost and remarkable performance. Because the proportion of parameters tuned by LoRA is relatively small, there might be a misleading impression that the LoRA fine-tuning data is invulnerable to Membership Inference Attacks (MIAs). However, we identify that utilizing the pre-trained model can induce more information leakage, which is neglected by existing MIAs. Therefore, we introduce LoRA-Leak, a holistic evaluation framework for MIAs against the fine-tuning datasets of LMs. LoRA-Leak incorporates fifteen membership inference attacks, including ten existing MIAs, and five improved MIAs that leverage the pre-trained model as a reference. In experiments, we apply LoRA-Leak to three advanced LMs across three popular natural language processing tasks, demonstrating that LoRA-based fine-tuned LMs are still vulnerable to MIAs (e.g., 0.775 AUC under conservative fine-tuning settings). We also applied LoRA-Leak to different fine-tuning settings to understand the resulting privacy risks. We further explore four defenses and find that only dropout and excluding specific LM layers during fine-tuning effectively mitigate MIA risks while maintaining utility. We highlight that under the "pre-training and fine-tuning" paradigm, the existence of the pre-trained model makes MIA a more severe risk for LoRA-based LMs. We hope that our findings can provide guidance on data privacy protection for specialized LM providers. 

**Abstract (ZH)**: LoRA-Leak：面向语言模型细调数据的全方位成员推理攻击评估框架 

---
# GenAI for Automotive Software Development: From Requirements to Wheels 

**Title (ZH)**: GenAI在汽车软件开发中的应用：从需求到成品车 

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Krzysztof Lebioda, Andre Schamschurko, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2507.18223)  

**Abstract**: This paper introduces a GenAI-empowered approach to automated development of automotive software, with emphasis on autonomous and Advanced Driver Assistance Systems (ADAS) capabilities. The process starts with requirements as input, while the main generated outputs are test scenario code for simulation environment, together with implementation of desired ADAS capabilities targeting hardware platform of the vehicle connected to testbench. Moreover, we introduce additional steps for requirements consistency checking leveraging Model-Driven Engineering (MDE). In the proposed workflow, Large Language Models (LLMs) are used for model-based summarization of requirements (Ecore metamodel, XMI model instance and OCL constraint creation), test scenario generation, simulation code (Python) and target platform code generation (C++). Additionally, Retrieval Augmented Generation (RAG) is adopted to enhance test scenario generation from autonomous driving regulations-related documents. Our approach aims shorter compliance and re-engineering cycles, as well as reduced development and testing time when it comes to ADAS-related capabilities. 

**Abstract (ZH)**: 基于GenAI的汽车软件自动化开发方法：强调自主和高级驾驶辅助系统（ADAS）能力 

---
# Information Security Based on LLM Approaches: A Review 

**Title (ZH)**: 基于大语言模型方法的信息安全：一个综述 

**Authors**: Chang Gong, Zhongwen Li, Xiaoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18215)  

**Abstract**: Information security is facing increasingly severe challenges, and traditional protection means are difficult to cope with complex and changing threats. In recent years, as an emerging intelligent technology, large language models (LLMs) have shown a broad application prospect in the field of information security. In this paper, we focus on the key role of LLM in information security, systematically review its application progress in malicious behavior prediction, network threat analysis, system vulnerability detection, malicious code identification, and cryptographic algorithm optimization, and explore its potential in enhancing security protection performance. Based on neural networks and Transformer architecture, this paper analyzes the technical basis of large language models and their advantages in natural language processing tasks. It is shown that the introduction of large language modeling helps to improve the detection accuracy and reduce the false alarm rate of security systems. Finally, this paper summarizes the current application results and points out that it still faces challenges in model transparency, interpretability, and scene adaptability, among other issues. It is necessary to explore further the optimization of the model structure and the improvement of the generalization ability to realize a more intelligent and accurate information security protection system. 

**Abstract (ZH)**: 信息安全部门正面临日益严峻的挑战，传统保护手段难以应对复杂多变的威胁。近年来，作为新兴的智能技术，大规模语言模型（LLMs）在信息安全领域展现了广泛的应用前景。本文聚焦大规模语言模型在信息安全中的关键作用，系统回顾其在恶意行为预测、网络安全威胁分析、系统漏洞检测、恶意代码识别以及加密算法优化等方面的应用进展，并探讨其在提升安全防护性能方面的潜力。基于神经网络和Transformer架构，本文分析了大规模语言模型的技术基础及其在自然语言处理任务中的优势。研究表明，引入大规模语言建模有助于提高安全系统的检测准确性和降低误报率。最后，本文总结了当前的应用成果，并指出模型透明性、可解释性和应用场景适应性等方面仍面临挑战，需要进一步探索模型结构的优化和泛化能力的提升，以实现更加智能和准确的信息安全保护系统。 

---
# Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection 

**Title (ZH)**: 使用GMTP保护RAG流水线：基于梯度的掩蔽词概率方法检测中毒文档 

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.18202)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings. 

**Abstract (ZH)**: 基于梯度的掩码令牌概率（GMTP）：检测和过滤 adversarially 制作的文档以增强 Retrieval-Augmented Generation 的安全性 

---
# SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models 

**Title (ZH)**: SCOPE: 随机和反偏置选项放置以评估大型语言模型 

**Authors**: Wonjun Jeong, Dongseok Kim, Taegkeun Whangbo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18182)  

**Abstract**: Large Language Models (LLMs) can achieve inflated scores on multiple-choice tasks by exploiting inherent biases in option positions or labels, rather than demonstrating genuine understanding. This study introduces SCOPE, an evaluation framework designed to measure and mitigate such selection bias in a dataset-independent manner. By repeatedly invoking a null prompt that lacks semantic content, SCOPE estimates each model's unique position-bias distribution. It then redistributes the answer slot according to the inverse-bias distribution, thereby equalizing the lucky-rate, the probability of selecting the correct answer by chance. Furthermore, it prevents semantically similar distractors from being placed adjacent to the answer, thereby blocking near-miss guesses based on superficial proximity cues. Across multiple benchmark experiments, SCOPE consistently outperformed existing debiasing methods in terms of stable performance improvements and showed clearer confidence distributions over correct options. This framework thus offers a new standard for enhancing the fairness and reliability of LLM evaluations. 

**Abstract (ZH)**: 大型语言模型（LLMs）可以通过利用选项位置或标签中的固有偏差来实现选择题中的高分，而不是展示真正的理解能力。本研究介绍了一种名为SCOPE的评估框架，旨在以数据集无关的方式测量和缓解这种选择偏差。通过反复调用一个没有语义内容的空提示，SCOPE估计每个模型的独特位置偏差分布。然后，它根据逆偏差分布重新分配答案槽，从而平等化随机正确率，即纯属偶然选择正确答案的概率。此外，它防止语义相似的干扰项紧邻正确答案放置，从而阻止基于表面接近线索的接近正确猜测。在多个基准实验中，SCOPE在稳定性能提升方面始终优于现有去偏见方法，并且在正确选项上展示了更清晰的信心分布。因此，该框架为提高LLM评估的公平性和可靠性提供了一个新的标准。 

---
# Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models 

**Title (ZH)**: 遵循均值：检测文本嵌入模型中的粘性词 token 

**Authors**: Kexin Chen, Dongxia Wang, Yi Liu, Haonan Zhang, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18171)  

**Abstract**: Despite the widespread use of Transformer-based text embedding models in NLP tasks, surprising 'sticky tokens' can undermine the reliability of embeddings. These tokens, when repeatedly inserted into sentences, pull sentence similarity toward a certain value, disrupting the normal distribution of embedding distances and degrading downstream performance. In this paper, we systematically investigate such anomalous tokens, formally defining them and introducing an efficient detection method, Sticky Token Detector (STD), based on sentence and token filtering. Applying STD to 40 checkpoints across 14 model families, we discover a total of 868 sticky tokens. Our analysis reveals that these tokens often originate from special or unused entries in the vocabulary, as well as fragmented subwords from multilingual corpora. Notably, their presence does not strictly correlate with model size or vocabulary size. We further evaluate how sticky tokens affect downstream tasks like clustering and retrieval, observing significant performance drops of up to 50%. Through attention-layer analysis, we show that sticky tokens disproportionately dominate the model's internal representations, raising concerns about tokenization robustness. Our findings show the need for better tokenization strategies and model design to mitigate the impact of sticky tokens in future text embedding applications. 

**Abstract (ZH)**: 基于Transformer的文本嵌入模型中意外的“粘性令牌”对嵌入可靠性的挑战：系统研究及检测方法 

---
# When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label 

**Title (ZH)**: 图上噪音标签与类别不平衡共存时：一种结合LLM和伪标签的图增强方法 

**Authors**: Riting Xia, Rucong Wang, Yulin Liu, Anchen Li, Xueyan Liu, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18153)  

**Abstract**: Class-imbalanced graph node classification is a practical yet underexplored research problem. Although recent studies have attempted to address this issue, they typically assume clean and reliable labels when processing class-imbalanced graphs. This assumption often violates the nature of real-world graphs, where labels frequently contain noise. Given this gap, this paper systematically investigates robust node classification for class-imbalanced graphs with noisy labels. We propose GraphALP, a novel Graph Augmentation framework based on Large language models (LLMs) and Pseudo-labeling techniques. Specifically, we design an LLM-based oversampling method to generate synthetic minority nodes, producing label-accurate minority nodes to alleviate class imbalance. Based on the class-balanced graphs, we develop a dynamically weighted pseudo-labeling method to obtain high-confidence pseudo labels to reduce label noise ratio. Additionally, we implement a secondary LLM-guided oversampling mechanism to mitigate potential class distribution skew caused by pseudo labels. Experimental results show that GraphALP achieves superior performance over state-of-the-art methods on class-imbalanced graphs with noisy labels. 

**Abstract (ZH)**: 带有噪声标签的类别不平衡图节点分类是一个实际但尚未充分探索的研究问题。 

---
# HIVMedQA: Benchmarking large language models for HIV medical decision support 

**Title (ZH)**: HIVMedQA：评估大型语言模型在HIV医疗决策支持中的性能 

**Authors**: Gonzalo Cardenal Antolin, Jacques Fellay, Bashkim Jaha, Roger Kouyos, Niko Beerenwinkel, Diane Duroux  

**Link**: [PDF](https://arxiv.org/pdf/2507.18143)  

**Abstract**: Large language models (LLMs) are emerging as valuable tools to support clinicians in routine decision-making. HIV management is a compelling use case due to its complexity, including diverse treatment options, comorbidities, and adherence challenges. However, integrating LLMs into clinical practice raises concerns about accuracy, potential harm, and clinician acceptance. Despite their promise, AI applications in HIV care remain underexplored, and LLM benchmarking studies are scarce. This study evaluates the current capabilities of LLMs in HIV management, highlighting their strengths and limitations. We introduce HIVMedQA, a benchmark designed to assess open-ended medical question answering in HIV care. The dataset consists of curated, clinically relevant questions developed with input from an infectious disease physician. We evaluated seven general-purpose and three medically specialized LLMs, applying prompt engineering to enhance performance. Our evaluation framework incorporates both lexical similarity and an LLM-as-a-judge approach, extended to better reflect clinical relevance. We assessed performance across key dimensions: question comprehension, reasoning, knowledge recall, bias, potential harm, and factual accuracy. Results show that Gemini 2.5 Pro consistently outperformed other models across most dimensions. Notably, two of the top three models were proprietary. Performance declined as question complexity increased. Medically fine-tuned models did not always outperform general-purpose ones, and larger model size was not a reliable predictor of performance. Reasoning and comprehension were more challenging than factual recall, and cognitive biases such as recency and status quo were observed. These findings underscore the need for targeted development and evaluation to ensure safe, effective LLM integration in clinical care. 

**Abstract (ZH)**: 大型语言模型（LLMs）在支持临床决策中的应用：以HIV管理为例的基准评估与探讨 

---
# GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness 

**Title (ZH)**: GOAT-SLM：一种awareness Paralinguistic和Speaker特征的口语语言模型 

**Authors**: Hongjie Chen, Zehan Li, Yaodong Song, Wenming Deng, Yitong Yao, Yuxin Zhang, Hang Lv, Xuechao Zhu, Jian Kang, Jie Lian, Jie Li, Chao Wang, Shuangyong Song, Yongxiang Li, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2507.18119)  

**Abstract**: Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems. 

**Abstract (ZH)**: Recent Advances in End-to-End Spoken Language Models with Paralinguistic and Speaker Characteristic Awareness 

---
# Group Sequence Policy Optimization 

**Title (ZH)**: 组序列策略优化 

**Authors**: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.18071)  

**Abstract**: This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models. 

**Abstract (ZH)**: 此论文介绍了Group Sequence Policy Optimization (GSPO)算法，这是一种用于训练大规模语言模型的稳定、高效且高性能的强化学习算法。与之前采用令牌级别重要性比例的算法不同，GSPO基于序列似然性定义重要性比例，并在序列级别进行剪裁、奖励和优化。我们证明了相比于GRPO算法，GSPO在训练效率和性能上表现出卓越的优势，尤其能够稳定Mixture-of-Experts (MoE) RL训练，并且有潜力简化RL基础设施的设计。这些特点已经促成最新Qwen3模型的显著改进。 

---
# TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios 

**Title (ZH)**: TELEVAL：一个为中文交互场景设计的语音语言模型动态基准 

**Authors**: Zehan Li, Hongjie Chen, Yuxin Zhang, Jing Zhou, Xuening Wang, Hang Lv, Mengjie Du, Yaodong Song, Jie Lian, Jian Kang, Jie Li, Yongxiang Li, Zhongjiang He, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18061)  

**Abstract**: Spoken language models (SLMs) have seen rapid progress in recent years, along with the development of numerous benchmarks for evaluating their performance. However, most existing benchmarks primarily focus on evaluating whether SLMs can perform complex tasks comparable to those tackled by large language models (LLMs), often failing to align with how users naturally interact in real-world conversational scenarios. In this paper, we propose TELEVAL, a dynamic benchmark specifically designed to evaluate SLMs' effectiveness as conversational agents in realistic Chinese interactive settings. TELEVAL defines three evaluation dimensions: Explicit Semantics, Paralinguistic and Implicit Semantics, and System Abilities. It adopts a dialogue format consistent with real-world usage and evaluates text and audio outputs separately. TELEVAL particularly focuses on the model's ability to extract implicit cues from user speech and respond appropriately without additional instructions. Our experiments demonstrate that despite recent progress, existing SLMs still have considerable room for improvement in natural conversational tasks. We hope that TELEVAL can serve as a user-centered evaluation framework that directly reflects the user experience and contributes to the development of more capable dialogue-oriented SLMs. 

**Abstract (ZH)**: TELEVAL：专为现实中文交互场景设计的对话代理评估基准 

---
# Synthetic Data Generation for Phrase Break Prediction with Large Language Model 

**Title (ZH)**: 大规模语言模型驱动的短语断点预测合成数据生成 

**Authors**: Hoyeon Lee, Sejung Son, Ye-Eun Kang, Jong-Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18044)  

**Abstract**: Current approaches to phrase break prediction address crucial prosodic aspects of text-to-speech systems but heavily rely on vast human annotations from audio or text, incurring significant manual effort and cost. Inherent variability in the speech domain, driven by phonetic factors, further complicates acquiring consistent, high-quality data. Recently, large language models (LLMs) have shown success in addressing data challenges in NLP by generating tailored synthetic data while reducing manual annotation needs. Motivated by this, we explore leveraging LLM to generate synthetic phrase break annotations, addressing the challenges of both manual annotation and speech-related tasks by comparing with traditional annotations and assessing effectiveness across multiple languages. Our findings suggest that LLM-based synthetic data generation effectively mitigates data challenges in phrase break prediction and highlights the potential of LLMs as a viable solution for the speech domain. 

**Abstract (ZH)**: 基于大语言模型的合成短语断言注释在短语断言预测中的应用 

---
# GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs 

**Title (ZH)**: GrAInS: 基于梯度的归因在推理时 steering 大型语言模型和视觉语言模型 

**Authors**: Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2507.18043)  

**Abstract**: Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities. 

**Abstract (ZH)**: 基于推理时修正的方法提供了一种轻量级替代方案，用于在测试时通过修改内部激活而不更新模型权重来调整大规模语言模型（LLMs）和多模态视觉-语言模型（VLMs）。然而，现有大多数方法依赖于固定的整体干预向量，忽视了单个输入词元的因果影响，并未能充分利用模型logits中的有用梯度，特别是在视觉和文本输入贡献不均的多模态设置中。为了解决这些局限性，我们引入了GrAInS，这是一种既适用于语言模型也适用于多模态视觉-语言模型及其任务的基于推理时修正的方法。GrAInS 使用对比梯度归因方法（Integrated Gradients）来识别按其对偏好输出和非偏好输出贡献度的正向和负向影响确定的最具影响力的top-k词元。然后，利用这些词元构建方向修正向量，以捕捉从不良行为到良好行为的语义转变。在推理时，GrAInS 在变压器层中根据词元级别的归因信号调整隐藏激活，并进行归一化以保持表征的规模。这使人们能够在不重新训练或添加辅助监督的情况下，对模型行为实现细粒度、可解释和模块化的控制。实证研究表明，GrAInS 在准确度、减少幻觉率和增强对齐胜率方面，均优于微调和现有修正基线，同时保持了模型的流畅性和通用能力。 

---
# NeuralDB: Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database 

**Title (ZH)**: NeuralDB：将知识编辑扩展至100,000条事实的LLM神经KV数据库 

**Authors**: Weizhi Fei, Hao Shi, Jing Xu, Jingchen Peng, Jiazheng Li, Jingzhao Zhang, Bo Bai, Wei Han, Zhenyuan Chen, Xueyan Niu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18028)  

**Abstract**: Efficiently editing knowledge stored in large language models (LLMs) enables model updates without large-scale training. One possible solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of facts. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits. In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module, % In particular, our gated module only operates when inference involves the edited facts, effectively preserving the general abilities of LLMs. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFacts datasets, using GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB not only excels in editing efficacy, generalization, specificity, fluency, and consistency, but also preserves overall performance across six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50x} more than in prior work). 

**Abstract (ZH)**: 高效编辑大型语言模型中存储的知识 enables 模型更新而无需大规模训练的一个可能解决方案是 Locate-and-Edit (L\&E)，允许同时修改大量事实。然而，在扩展到数千个编辑时，这种编辑可能会损害大型语言模型的一般能力，甚至导致遗忘编辑的事实。在本文中，我们将现有的线性 L\&E 方法建模为查询一个键值（KV）数据库。从这一视角出发，我们提出了 NeuralDB，这是一种编辑框架，明确地将编辑的事实表示为一个配备了非线性门控检索模块的神经 KV 数据库，特别是我们的门控模块仅在涉及编辑的事实进行推理时才操作，有效地保留了大型语言模型的一般能力。在 ZsRE 和 CounterFacts 数据集上对 10,000 个事实进行了编辑的全面实验使用了 GPT2-XL、GPT-J（6B）和 Llama-3（8B）。实验结果表明，NeuralDB 不仅在编辑效能、泛化能力、特异性、流畅性和一致性方面表现出色，还在六个代表性文本理解和生成任务中保持了整体性能。进一步的实验表明，即使扩展到 100,000 个事实（比先前工作多 50 倍），NeuralDB 仍然保持其有效性。 

---
# GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures 

**Title (ZH)**: GRR-CoCa: 利用大规模语言模型机制构建多模态模型架构 

**Authors**: Jake R. Patock, Nicole Catherine Lewis, Kevin McCoy, Christina Gomez, Canling Chen, Lorenzo Luzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18009)  

**Abstract**: State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains. 

**Abstract (ZH)**: 基于最新进展的对比caption生成模型GRR-CoCa：改进的多模态模型在文本解码器和视觉变换器中的应用 

---
# Decoding Instructional Dialogue: Human-AI Collaborative Analysis of Teacher Use of AI Tool at Scale 

**Title (ZH)**: 解码教学对话：大规模分析教师使用AI工具的人机协作研究 

**Authors**: Alex Liu, Lief Esbenshade, Shawon Sarkar, Victor Tian, Zachary Zhang, Kevin He, Min Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.17985)  

**Abstract**: The integration of large language models (LLMs) into educational tools has the potential to substantially impact how teachers plan instruction, support diverse learners, and engage in professional reflection. Yet little is known about how educators actually use these tools in practice and how their interactions with AI can be meaningfully studied at scale. This paper presents a human-AI collaborative methodology for large-scale qualitative analysis of over 140,000 educator-AI messages drawn from a generative AI platform used by K-12 teachers. Through a four-phase coding pipeline, we combined inductive theme discovery, codebook development, structured annotation, and model benchmarking to examine patterns of educator engagement and evaluate the performance of LLMs in qualitative coding tasks. We developed a hierarchical codebook aligned with established teacher evaluation frameworks, capturing educators' instructional goals, contextual needs, and pedagogical strategies. Our findings demonstrate that LLMs, particularly Claude 3.5 Haiku, can reliably support theme identification, extend human recognition in complex scenarios, and outperform open-weight models in both accuracy and structural reliability. The analysis also reveals substantive patterns in how educators inquire AI to enhance instructional practices (79.7 percent of total conversations), create or adapt content (76.1 percent), support assessment and feedback loop (46.9 percent), attend to student needs for tailored instruction (43.3 percent), and assist other professional responsibilities (34.2 percent), highlighting emerging AI-related competencies that have direct implications for teacher preparation and professional development. This study offers a scalable, transparent model for AI-augmented qualitative research and provides foundational insights into the evolving role of generative AI in educational practice. 

**Abstract (ZH)**: 大型语言模型（LLMs）与教育工具的整合有可能显著影响教师的课程规划、支持多元学习者以及专业反思的方式。然而，关于教育工作者在实际中如何使用这些工具以及如何大规模研究其与AI的交互知之甚少。本文提出了一种人机协作方法论，用于对来自K-12教师使用的生成型AI平台的超过140,000条教育者-AI消息进行大规模定性分析。通过四阶段编码管道，我们结合了归纳主题发现、编码手册开发、结构化注释和模型基准测试，以考察教育者参与的模式并评估LLMs在定性编码任务中的表现。我们开发了一个与公认的教师评价框架相契合的分级编码手册，捕捉了教育者的教学目标、情境需求和教学策略。我们的研究结果表明，特别是Claude 3.5 Haiku，LLMs能够可靠地支持主题识别，扩展人在复杂情境下的识别能力，并在准确性与结构性可靠性方面优于未加权模型。分析还揭示了教育者如何通过AI增强教学实践（占总对话的79.7%）、创编或改编内容（76.1%）、支持评估和反馈循环（46.9%）、关注个性化教学以满足学生需求（43.3%），以及协助其他专业职责（34.2%）等实质性模式，强调了新兴的与AI相关的专业能力，这些能力对教师培养和专业发展具有直接影响。本研究提供了一种可扩展且透明的模型，用于增强型AI辅助的定性研究，并提供了关于生成型AI在教育实践中的演变角色的基础性见解。 

---
# Are LLM Belief Updates Consistent with Bayes' Theorem? 

**Title (ZH)**: LLM信念更新是否符合贝叶斯定理？ 

**Authors**: Sohaib Imran, Ihor Kendiukhov, Matthew Broerman, Aditya Thomas, Riccardo Campanella, Rob Lamb, Peter M. Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2507.17951)  

**Abstract**: Do larger and more capable language models learn to update their "beliefs" about propositions more consistently with Bayes' theorem when presented with evidence in-context? To test this, we formulate a Bayesian Coherence Coefficient (BCC) metric and generate a dataset with which to measure the BCC. We measure BCC for multiple pre-trained-only language models across five model families, comparing against the number of model parameters, the amount of training data, and model scores on common benchmarks. Our results provide evidence for our hypothesis that larger and more capable pre-trained language models assign credences that are more coherent with Bayes' theorem. These results have important implications for our understanding and governance of LLMs. 

**Abstract (ZH)**: 当面对上下文证据时，更大的且更为 capable 的语言模型是否能够更一致地根据 Bayes 定理更新其“信念”？为了验证这一点，我们制定了一个贝叶斯一致性系数（BCC）指标，并生成了一个数据集来测量 BCC。我们测量了五大家族多个仅预训练的语言模型的 BCC，并将其与模型参数量、训练数据量以及模型在常见基准上的得分进行比较。我们的结果证明了我们的假设：更大的且更为 capable 的预训练语言模型赋予的信念与 Bayes 定理更为一致。这些结果对于理解和治理大语言模型具有重要意义。 

---
# Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text 

**Title (ZH)**: 评估AI文本检测器、few-shot提示和Chain-of-Thought提示性能：基于DeepSeek生成的文本 

**Authors**: Hulayyil Alshammari, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17944)  

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%). 

**Abstract (ZH)**: 大规模语言模型（LLMs）迅速改变了书面材料的创作方式。LLMs引发了关于写作完整性的质疑，从而促进了人工智能（AI）检测技术的发展。对抗性攻击，如标准和人性化改写，削弱了检测器检测机器生成文本的能力。先前的研究主要集中在ChatGPT和其他知名的语言模型上，并且在检测器的准确性方面显示出差异。然而，关于最近发布的DeepSeek，文献中明显缺乏研究。因此，在本文中，我们调查了六个通用的AI检测工具——AI Text Classifier、Content Detector AI、Copyleaks、QuillBot、GPT-2和GPTZero，看它们能否一致地识别由DeepSeek生成的文本。这些检测器受到了上述对抗性攻击的挑战。我们还将DeepSeek作为检测器，通过少量提示和链式思考（CoT）来进行分类。我们收集了49个来自LLM时代前的人类撰写的问答对，并使用DeepSeek-v3生成了匹配的答案，生成了49个机器生成的样本。然后，我们应用了改写和人性化等对抗性技术，增加了196个额外样本。这些样本被用来挑战检测器的鲁棒性，并评估其准确率的影响。尽管QuillBot和Copyleaks在原始和改写后的DeepSeek文本上表现出接近完美的性能，但其他检测器，特别是AI Text Classifier和GPT-2，显示出不一致的结果。最有效的攻击是人性化，这将Copyleaks、QuillBot和GPTZero的准确性分别降低到71%、58%和52%。少量提示和CoT提示显示了较高的准确性，五个样本中只有一个被错误分类（AI召回率96%，人类召回率100%）。 

---
# Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking 

**Title (ZH)**: 基于LLM的排名中缓解位置偏见的自适应重复方法 

**Authors**: Ali Vardasbi, Gustavo Penha, Claudia Hauff, Hugues Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2507.17788)  

**Abstract**: When using LLMs to rank items based on given criteria, or evaluate answers, the order of candidate items can influence the model's final decision. This sensitivity to item positioning in a LLM's prompt is known as position bias. Prior research shows that this bias exists even in large models, though its severity varies across models and tasks. In addition to position bias, LLMs also exhibit varying degrees of low repetition consistency, where repeating the LLM call with the same candidate ordering can lead to different rankings. To address both inconsistencies, a common approach is to prompt the model multiple times with different candidate orderings and aggregate the results via majority voting. However, this repetition strategy, significantly increases computational costs. Extending prior findings, we observe that both the direction -- favoring either the earlier or later candidate in the prompt -- and magnitude of position bias across instances vary substantially, even within a single dataset. This observation highlights the need for a per-instance mitigation strategy. To this end, we introduce a dynamic early-stopping method that adaptively determines the number of repetitions required for each instance. Evaluating our approach across three LLMs of varying sizes and on two tasks, namely re-ranking and alignment, we demonstrate that transitioning to a dynamic repetition strategy reduces the number of LLM calls by an average of 81%, while preserving the accuracy. Furthermore, we propose a confidence-based adaptation to our early-stopping method, reducing LLM calls by an average of 87% compared to static repetition, with only a slight accuracy trade-off relative to our original early-stopping method. 

**Abstract (ZH)**: 使用LLM根据给定标准对项目进行排名或评估答案时，候选项项目的顺序会影响模型的最终决策。这种对LLM提示中候选项位置的敏感性称为位置偏置。先前的研究表明，这种偏置即使在大规模模型中也存在，但其严重程度因模型和任务而异。除了位置偏置外，LLM在重复一致性方面也表现出不同程度的低重复性，即使用相同的候选项顺序重复调用LLM可能导致不同的排名。为了应对这两种不一致性，一种常见方法是在不同候选项顺序下多次提示模型，并通过多数投票聚合结果。然而，这种重复策略显著增加了计算成本。在此基础上，我们观察到位置偏置的方向和强度在不同实例之间变化显著，即使在同一数据集中也是如此。这一观察强调了需要实例特定缓解策略的必要性。为此，我们引入了一种动态提前停止方法，该方法自适应地确定每个实例所需的重复次数。在三种不同规模的LLM上评估我们的方法，并应用于排名和对齐两项任务中，我们证明了切换到动态重复策略可以将LLM调用次数平均减少81%，同时保持准确性。此外，我们提出了一种基于置信度的提前停止方法的适应性改进，与静态重复相比，平均可将LLM调用次数减少87%，且相对于原始的提前停止方法仅略有准确性上的权衡。 

---
# Hyperbolic Deep Learning for Foundation Models: A Survey 

**Title (ZH)**: hyperbolic深度学习在基础模型中的应用：一个综述 

**Authors**: Neil He, Hiren Madhu, Ngoc Bui, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2507.17787)  

**Abstract**: Foundation models pre-trained on massive datasets, including large language models (LLMs), vision-language models (VLMs), and large multimodal models, have demonstrated remarkable success in diverse downstream tasks. However, recent studies have shown fundamental limitations of these models: (1) limited representational capacity, (2) lower adaptability, and (3) diminishing scalability. These shortcomings raise a critical question: is Euclidean geometry truly the optimal inductive bias for all foundation models, or could incorporating alternative geometric spaces enable models to better align with the intrinsic structure of real-world data and improve reasoning processes? Hyperbolic spaces, a class of non-Euclidean manifolds characterized by exponential volume growth with respect to distance, offer a mathematically grounded solution. These spaces enable low-distortion embeddings of hierarchical structures (e.g., trees, taxonomies) and power-law distributions with substantially fewer dimensions compared to Euclidean counterparts. Recent advances have leveraged these properties to enhance foundation models, including improving LLMs' complex reasoning ability, VLMs' zero-shot generalization, and cross-modal semantic alignment, while maintaining parameter efficiency. This paper provides a comprehensive review of hyperbolic neural networks and their recent development for foundation models. We further outline key challenges and research directions to advance the field. 

**Abstract (ZH)**: 基于大规模数据预训练的基础模型，包括大规模语言模型（LLMs）、多模态视觉-语言模型（VLMs）和大规模多模态模型，在多样化的下游任务中表现出显著的成功。然而，近期研究表明这些模型存在根本性的限制：（1）有限的表征能力，（2）较低的适应性，以及（3）减弱的可扩展性。这些不足引发了关键问题：欧几里得几何是否真的适用于所有基础模型的最佳归纳偏置，或者通过引入替代几何空间能否使模型更好地与现实世界数据的内在结构对齐并提高推理过程？双曲空间作为一类非欧几里得流形，在距离增长方面具有指数体积增长的特点，为这一问题提供了数学基础的解决方案。双曲空间能够以远少于欧几里得空间的维度低失真嵌入分层结构（如树、分类体系）及幂律分布，近期进展利用了这些特性来提升基础模型的表现，包括增强大规模语言模型的复杂推理能力、多模态视觉-语言模型的零样本泛化能力和跨模态语义对齐，同时保持了参数效率。本文提供了双曲神经网络及其在基础模型中最新进展的全面综述，并进一步概述了关键挑战和研究方向以推进该领域。 

---
# Human-AI Co-Creation: A Framework for Collaborative Design in Intelligent Systems 

**Title (ZH)**: 人类与人工智能协作创作：智能系统中合作设计的框架 

**Authors**: Zhangqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17774)  

**Abstract**: As artificial intelligence (AI) continues to evolve from a back-end computational tool into an interactive, generative collaborator, its integration into early-stage design processes demands a rethinking of traditional workflows in human-centered design. This paper explores the emergent paradigm of human-AI co-creation, where AI is not merely used for automation or efficiency gains, but actively participates in ideation, visual conceptualization, and decision-making. Specifically, we investigate the use of large language models (LLMs) like GPT-4 and multimodal diffusion models such as Stable Diffusion as creative agents that engage designers in iterative cycles of proposal, critique, and revision. 

**Abstract (ZH)**: 随着人工智能（AI）从后台计算工具演变为互动式的生成性合作者，其在早期设计过程中的集成需要重新思考以人为本的设计传统工作流程。本文探讨了人机共生创造的新兴范式，在这种范式中，AI 不仅用于自动化或效率提升，还主动参与创意生成、视觉概念化和决策制定。具体而言，我们研究了使用大型语言模型（如 GPT-4）和多模态扩散模型（如 Stable Diffusion）作为创意代理，与设计师进行迭代的提案、评价和修订循环。 

---
# Exploring Communication Strategies for Collaborative LLM Agents in Mathematical Problem-Solving 

**Title (ZH)**: 探索数学问题解决中协作型LLM代理的通信策略 

**Authors**: Liang Zhang, Xiaoming Zhai, Jionghao Lin, Jionghao Lin, Jennifer Kleiman, Diego Zapata-Rivera, Carol Forsyth, Yang Jiang, Xiangen Hu, Arthur C. Graesser  

**Link**: [PDF](https://arxiv.org/pdf/2507.17753)  

**Abstract**: Large Language Model (LLM) agents are increasingly utilized in AI-aided education to support tutoring and learning. Effective communication strategies among LLM agents improve collaborative problem-solving efficiency and facilitate cost-effective adoption in education. However, little research has systematically evaluated the impact of different communication strategies on agents' problem-solving. Our study examines four communication modes, \textit{teacher-student interaction}, \textit{peer-to-peer collaboration}, \textit{reciprocal peer teaching}, and \textit{critical debate}, in a dual-agent, chat-based mathematical problem-solving environment using the OpenAI GPT-4o model. Evaluated on the MATH dataset, our results show that dual-agent setups outperform single agents, with \textit{peer-to-peer collaboration} achieving the highest accuracy. Dialogue acts like statements, acknowledgment, and hints play a key role in collaborative problem-solving. While multi-agent frameworks enhance computational tasks, effective communication strategies are essential for tackling complex problems in AI education. 

**Abstract (ZH)**: 大型语言模型（LLM）代理在AI辅助教育中的应用日益增多，用于支持辅导和学习。不同的沟通策略在LLM代理间的有效沟通提高了协作问题解决的效率，并促进了教育中的成本效益采用。然而，关于不同沟通策略对代理问题解决影响的研究较少。本研究在使用OpenAI GPT-4o模型的双代理、基于聊天的数学问题解决环境中，探讨了四种沟通模式：教师-学生互动、同伴间协作、互惠同伴教学和批判性辩论的影响。以MATH数据集为评价标准，研究表明，双代理设置优于单代理，同伴间协作在准确性上最高。对话行为如陈述、确认和提示在协作问题解决中发挥着关键作用。多代理框架虽然增强了计算任务，但有效的沟通策略对于解决AI教育中的复杂问题至关重要。 

---
