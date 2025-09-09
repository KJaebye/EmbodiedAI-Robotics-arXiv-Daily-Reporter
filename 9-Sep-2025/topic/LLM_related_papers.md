# LLaDA-VLA: Vision Language Diffusion Action Models 

**Title (ZH)**: LLaDA-VLA：视觉语言扩散动作模型 

**Authors**: Yuqing Wen, Hebei Li, Kefan Gu, Yucheng Zhao, Tiancai Wang, Xiaoyan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.06932)  

**Abstract**: The rapid progress of auto-regressive vision-language models (VLMs) has inspired growing interest in vision-language-action models (VLA) for robotic manipulation. Recently, masked diffusion models, a paradigm distinct from autoregressive models, have begun to demonstrate competitive performance in text generation and multimodal applications, leading to the development of a series of diffusion-based VLMs (d-VLMs). However, leveraging such models for robot policy learning remains largely unexplored. In this work, we present LLaDA-VLA, the first Vision-Language-Diffusion-Action model built upon pretrained d-VLMs for robotic manipulation. To effectively adapt d-VLMs to robotic domain, we introduce two key designs: (1) a localized special-token classification strategy that replaces full-vocabulary classification with special action token classification, reducing adaptation difficulty; (2) a hierarchical action-structured decoding strategy that decodes action sequences hierarchically considering the dependencies within and across actions. Extensive experiments demonstrate that LLaDA-VLA significantly outperforms state-of-the-art VLAs on both simulation and real-world robots. 

**Abstract (ZH)**: 基于预训练扩散模型的本地化特殊标记分类和层次化行动结构解码的视觉语言动作模型（LLaDA-VLA） 

---
# Evaluation of Large Language Models for Anomaly Detection in Autonomous Vehicles 

**Title (ZH)**: 大型语言模型在自动驾驶车辆异常检测中的评估 

**Authors**: Petros Loukas, David Bassir, Savvas Chatzichristofis, Angelos Amanatiadis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05315)  

**Abstract**: The rapid evolution of large language models (LLMs) has pushed their boundaries to many applications in various domains. Recently, the research community has started to evaluate their potential adoption in autonomous vehicles and especially as complementary modules in the perception and planning software stacks. However, their evaluation is limited in synthetic datasets or manually driving datasets without the ground truth knowledge and more precisely, how the current perception and planning algorithms would perform in the cases under evaluation. For this reason, this work evaluates LLMs on real-world edge cases where current autonomous vehicles have been proven to fail. The proposed architecture consists of an open vocabulary object detector coupled with prompt engineering and large language model contextual reasoning. We evaluate several state-of-the-art models against real edge cases and provide qualitative comparison results along with a discussion on the findings for the potential application of LLMs as anomaly detectors in autonomous vehicles. 

**Abstract (ZH)**: 大型语言模型的快速进化已经将其边界推向了多个领域应用。最近，研究社区开始评估其在自主车辆中的潜在应用，特别是在感知和规划软件栈中的补充模块。然而，这类评估主要局限于合成数据集或手动驾驶数据集，缺乏真正的知识验证，尤其在评估当前感知和规划算法在这些情况下表现时更为明显。因此，本研究评估了大型语言模型在现实世界中的边缘案例，这些案例是当前自主车辆已被证明无法处理的。提出的架构结合了开放式词汇对象检测器、提示工程和大型语言模型上下文推理。我们评估了几种最先进的模型在现实边缘案例中的表现，并提供了定性比较结果以及对大型语言模型作为自主车辆异常检测器潜在应用的研究发现讨论。 

---
# RAFFLES: Reasoning-based Attribution of Faults for LLM Systems 

**Title (ZH)**: RAFFLES: 基于推理的LLM系统故障归因 

**Authors**: Chenyang Zhu, Spencer Hong, Jingyu Wu, Kushal Chawla, Charlotte Tang, Youbing Yin, Nathan Wolfe, Erin Babinsky, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06822)  

**Abstract**: We have reached a critical roadblock in the development and enhancement of long-horizon, multi-component LLM agentic systems: it is incredibly tricky to identify where these systems break down and why. Evaluation capabilities that currently exist today (e.g., single pass LLM-as-a-judge) are limited in that they often focus on individual metrics or capabilities, end-to-end outcomes, and are narrowly grounded on the preferences of humans. We argue that to match the agentic capabilities, evaluation frameworks must also be able to reason, probe, iterate, and understand the complex logic passing through these systems over long horizons. In this paper, we present RAFFLES - an evaluation architecture that incorporates reasoning and iterative refinement. Specifically, RAFFLES operates as an iterative, multi-component pipeline, using a central Judge to systematically investigate faults and a set of specialized Evaluators to assess not only the system's components but also the quality of the reasoning by the Judge itself, thereby building a history of hypotheses. We tested RAFFLES against several baselines on the Who&When dataset, a benchmark designed to diagnose the "who" (agent) and "when" (step) of a system's failure. RAFFLES outperforms these baselines, achieving an agent-step fault pair accuracy of over 43% on the Algorithmically-Generated dataset (a substantial increase from the previously published best of 16.6%) and over 20% on the Hand-Crafted dataset (surpassing the previously published best of 8.8%). These results demonstrate a key step towards introducing automated fault detection for autonomous systems over labor-intensive manual human review. 

**Abstract (ZH)**: 我们在长时 horizon、多组件大语言模型代理系统的发展与提升中遇到了一个关键障碍：很难确定这些系统在何处出现故障及其原因。目前存在的评估能力（例如，单次过的大语言模型作为法官）在聚焦单一指标或能力、端到端结果且紧密基于人类偏好方面受到限制。我们提出，要匹配代理能力，评估框架还必须能够推理、探索、迭代并理解这些系统在长时 horizon 中传递的复杂逻辑。在本文中，我们提出了 RAFFLES——一种结合推理与迭代完善的评估架构。具体而言，RAFFLES 作为迭代的多组件管道运行，采用一个中央法官系统地调查故障，并通过一组专门的评估器评估法官本身及其推理的质量，从而建立假说的历史。我们在 Who&When 数据集上对 RAFFLES 进行了与多个基线的测试，该数据集用于诊断系统的“谁”（代理）和“何时”（步骤）。RAFFLES 在算法生成数据集上达到超过 43% 的代理-步骤故障配对准确性（比之前出版的最佳结果 16.6% 有显著提升），在手工打造数据集上达到超过 20% 的准确性（超过之前出版的最佳结果 8.8%）。这些结果展示了向通过劳动密集型的手动人工审查引入自动故障检测的一步重要进展。 

---
# Another Turn, Better Output? A Turn-Wise Analysis of Iterative LLM Prompting 

**Title (ZH)**: 又一轮，更优输出？基于轮次的迭代LLM提示分析 

**Authors**: Shashidhar Reddy Javaji, Bhavul Gauri, Zining Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06770)  

**Abstract**: Large language models (LLMs) are now used in multi-turn workflows, but we still lack a clear way to measure when iteration helps and when it hurts. We present an evaluation framework for iterative refinement that spans ideation, code, and math. Our protocol runs controlled 12-turn conversations per task, utilizing a variety of prompts ranging from vague ``improve it'' feedback to targeted steering, and logs per-turn outputs. We score outcomes with domain-appropriate checks (unit tests for code; answer-equivalence plus reasoning-soundness for math; originality and feasibility for ideation) and track turn-level behavior with three families of metrics: semantic movement across turns, turn-to-turn change, and output size growth. Across models and tasks, gains are domain-dependent: they arrive early in ideas and code, but in math late turns matter when guided by elaboration. After the first few turns, vague feedback often plateaus or reverses correctness, while targeted prompts reliably shift the intended quality axis (novelty vs. feasibility in ideation; speed vs. readability in code; in math, elaboration outperforms exploration and drives late-turn gains). We also observe consistent domain patterns: ideation moves more in meaning across turns, code tends to grow in size with little semantic change, and math starts fixed but can break that path with late, elaborative this http URL, the framework and metrics make iteration measurable and comparable across models, and signal when to steer, stop, or switch strategies. 

**Abstract (ZH)**: 大规模语言模型（LLMs）现在用于多轮工作流程中，但我们仍然缺乏明确的方法来衡量迭代是助益还是有害。我们提出了一种涵盖创意、代码和数学的迭代 refinement 评估框架。我们的协议每个任务运行受控的12轮对话，利用从模糊的“改进它”反馈到目标导向引导等多种提示，并记录每轮输出。我们使用领域特定的检查（代码中的单元测试；数学中的答案等价性加上推理稳健性；创意中的原创性和可行性）来评分，并通过三类指标追踪每轮行为：跨轮次语义变化、轮次之间变化和输出大小增长。在不同模型和任务中，收益具有领域依赖性：在创意和代码中早期出现，在数学中只有在受到详细扩展引导时，后期轮次才重要。在最初的几轮后，模糊反馈往往导致正确性的停滞或逆转，而目标导向提示则可靠地改变预期质量轴（创意中的新颖性与可行性；代码中的速度与可读性；在数学中，详细扩展优于探索并驱动后期轮次的收益）。此外，我们还观察到一致的领域模式：创意在一轮次间更具意义变化，代码通常在大小上增长但语义变化不大，而数学从一开始就保持固定，但在后期详细的扩展引导下可以打破这种路径。该框架和指标使迭代在不同模型之间可测量和可比较，并指示何时需要导向、停止或切换策略。 

---
# CogGuide: Human-Like Guidance for Zero-Shot Omni-Modal Reasoning 

**Title (ZH)**: CogGuide: 类人类的引导在零样本多模态推理中的应用 

**Authors**: Zhou-Peng Shou, Zhi-Qiang You, Fang Wang, Hai-Bo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06641)  

**Abstract**: Targeting the issues of "shortcuts" and insufficient contextual understanding in complex cross-modal reasoning of multimodal large models, this paper proposes a zero-shot multimodal reasoning component guided by human-like cognitive strategies centered on an "intent sketch". The component comprises a plug-and-play three-module pipeline-Intent Perceiver, Strategy Generator, and Strategy Selector-that explicitly constructs a "understand-plan-select" cognitive process. By generating and filtering "intent sketch" strategies to guide the final reasoning, it requires no parameter fine-tuning and achieves cross-model transfer solely through in-context engineering. Information-theoretic analysis shows that this process can reduce conditional entropy and improve information utilization efficiency, thereby suppressing unintended shortcut reasoning. Experiments on IntentBench, WorldSense, and Daily-Omni validate the method's generality and robust gains; compared with their respective baselines, the complete "three-module" scheme yields consistent improvements across different reasoning engines and pipeline combinations, with gains up to approximately 9.51 percentage points, demonstrating the practical value and portability of the "intent sketch" reasoning component in zero-shot scenarios. 

**Abstract (ZH)**: 针对多模态大型模型在复杂跨模态推理中“捷径”问题和不足的上下文理解能力，本文提出了一种以“意图素描”为中心的人类认知策略指导的零样本多模态推理组件。该组件包含可插拔的三模块管道——意图感知器、策略生成器和策略选择器，明确构建了一个“理解-规划-选择”的认知过程。通过生成和过滤“意图素描”策略来指导最终推理，无需参数微调，仅通过上下文工程实现跨模型迁移。信息论分析表明，该过程可以降低条件熵并提高信息利用效率，从而抑制无意中的捷径推理。在IntentBench、WorldSense和Daily-Omni上的实验验证了该方法的普适性和稳健性增益；与各自的基线相比，完整的“三模块”方案在不同推理引擎和管道组合中一致地表现出改进，增幅高达约9.51个百分点，展示了“意图素描”推理组件在零样本场景中的实用价值和可移植性。 

---
# An AI system to help scientists write expert-level empirical software 

**Title (ZH)**: 一种辅助科学家编写高水平 empirical 软件的AI系统 

**Authors**: Eser Aygün, Anastasiya Belyaeva, Gheorghe Comanici, Marc Coram, Hao Cui, Jake Garrison, Renee Johnston Anton Kast, Cory Y. McLean, Peter Norgaard, Zahra Shamsi, David Smalling, James Thompson, Subhashini Venugopalan, Brian P. Williams, Chujun He, Sarah Martinson, Martyna Plomecka, Lai Wei, Yuchen Zhou, Qian-Ze Zhu, Matthew Abraham, Erica Brand, Anna Bulanova, Jeffrey A. Cardille, Chris Co, Scott Ellsworth, Grace Joseph, Malcolm Kane, Ryan Krueger, Johan Kartiwa, Dan Liebling, Jan-Matthis Lueckmann, Paul Raccuglia, Xuefei, Wang, Katherine Chou, James Manyika, Yossi Matias, John C. Platt, Lizzie Dorfman, Shibl Mourad, Michael P. Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2509.06503)  

**Abstract**: The cycle of scientific discovery is frequently bottlenecked by the slow, manual creation of software to support computational experiments. To address this, we present an AI system that creates expert-level scientific software whose goal is to maximize a quality metric. The system uses a Large Language Model (LLM) and Tree Search (TS) to systematically improve the quality metric and intelligently navigate the large space of possible solutions. The system achieves expert-level results when it explores and integrates complex research ideas from external sources. The effectiveness of tree search is demonstrated across a wide range of benchmarks. In bioinformatics, it discovered 40 novel methods for single-cell data analysis that outperformed the top human-developed methods on a public leaderboard. In epidemiology, it generated 14 models that outperformed the CDC ensemble and all other individual models for forecasting COVID-19 hospitalizations. Our method also produced state-of-the-art software for geospatial analysis, neural activity prediction in zebrafish, time series forecasting and numerical solution of integrals. By devising and implementing novel solutions to diverse tasks, the system represents a significant step towards accelerating scientific progress. 

**Abstract (ZH)**: 科学发现的周期常因支持计算实验的缓慢手动软件创建而受阻。为此，我们提出一个AI系统，该系统能创建专家级的科学软件，旨在最大化质量指标。该系统利用大型语言模型（LLM）和树搜索（TS）系统性地提高质量指标，并智能地导航可能解决方案的广阔空间。当系统探索并整合来自外部的复杂研究思想时，能够达到专家级结果。树搜索的有效性在一系列基准测试中得到验证。在生物信息学领域，系统发现40种新型单细胞数据分析方法，在公共排行榜上优于顶级的人工开发方法。在流行病学领域，系统生成了14个模型，在预测COVID-19住院人数方面优于CDC集成和其他个体模型。我们的方法还为地理空间分析、斑马鱼神经活动预测、时间序列预测和积分的数值解提供了最先进的软件。通过为多种任务设计并实施新颖的解决方案，该系统代表了加速科学进步的重要一步。 

---
# Scaling up Multi-Turn Off-Policy RL and Multi-Agent Tree Search for LLM Step-Provers 

**Title (ZH)**: 扩展多轮离策RL和多智能体树搜索在语言模型步骤证明中的应用 

**Authors**: Ran Xin, Zeyu Zheng, Yanchen Nie, Kun Yuan, Xia Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06493)  

**Abstract**: The integration of Large Language Models (LLMs) into automated theorem proving has shown immense promise, yet is fundamentally constrained by challenges in scaling up both training-time reinforcement learning (RL) and inference-time compute. This paper introduces \texttt{BFS-Prover-V2}, a system designed to address this dual scaling problem. We present two primary innovations. The first is a novel multi-turn off-policy RL framework for continually improving the performance of LLM step-prover at training time. This framework, inspired by the principles of AlphaZero, utilizes a multi-stage expert iteration pipeline featuring adaptive tactic-level data filtering and periodic retraining to surmount the performance plateaus that typically curtail long-term RL in LLM-based agents. The second innovation is a planner-enhanced multi-agent search architecture that scales reasoning capabilities at inference time. This architecture employs a general reasoning model as a high-level planner to iteratively decompose complex theorems into a sequence of simpler subgoals. This hierarchical approach substantially reduces the search space, enabling a team of parallel prover agents to collaborate efficiently by leveraging a shared proof cache. We demonstrate that this dual approach to scaling yields state-of-the-art results on established formal mathematics benchmarks. \texttt{BFS-Prover-V2} achieves 95.08\% and 41.4\% on the MiniF2F and ProofNet test sets respectively. While demonstrated in the domain of formal mathematics, the RL and inference techniques presented in this work are of broader interest and may be applied to other domains requiring long-horizon multi-turn reasoning and complex search. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动定理证明中的集成展现了巨大的潜力，但仍然受到扩展训练时强化学习（RL）和推理时计算能力双重挑战的限制。本文介绍了BFS-Prover-V2系统，旨在解决这一双重扩展问题。我们提出了两项主要创新。首先，提出了一种新颖的多轮次离策略RL框架，在训练时间内持续提升LLM步骤证明器的性能。该框架借鉴了AlphaZero的原理，采用多阶段专家迭代流水线，并结合自适应策略级数据过滤和定期重新训练，以克服基于LLM的代理在长期RL中通常会遇到的性能平台期。其次是增强计划者的多代理搜索架构，该架构在推理时扩展了推理能力。该架构使用一个通用推理模型作为高级计划器，通过迭代分解复杂的定理为一系列更简单的子目标。这种分层方法显著减少了搜索空间，从而使一个团队的并行证明代理能够通过共享证明缓存高效协作。我们展示了这种双重扩展方法在正式数学基准测试中取得了最先进的结果。BFS-Prover-V2分别在MiniF2F和ProofNet测试集上取得了95.08%和41.4%的性能。虽然在正式数学领域进行了演示，但本文提出的RL和推理技术具有更广泛的应用兴趣，并可能应用于其他需要远期多轮次推理和复杂搜索的领域。 

---
# Accelerate Scaling of LLM Alignment via Quantifying the Coverage and Depth of Instruction Set 

**Title (ZH)**: 通过量化指令集的覆盖面和深度加速大型语言模型的对齐缩放 

**Authors**: Chengwei Wu, Li Du, Hanyu Zhao, Yiming Ju, Jiapu Wang, Tengfei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.06463)  

**Abstract**: With the growing demand for applying large language models to downstream tasks, improving model alignment performance and efficiency has become crucial. Such a process involves selecting informative instructions from a candidate pool. However, due to the complexity of instruction set distributions, the key factors driving the performance of aligned models remain unclear. As a result, current instruction set refinement methods fail to improve performance as the instruction pool expands continuously. To address this issue, we first investigate the key factors that influence the relationship between instruction dataset distribution and aligned model performance. Based on these insights, we propose a novel instruction data selection method. We identify that the depth of instructions and the coverage of the semantic space are the crucial factors determining downstream performance, which could explain over 70\% of the model loss on the development set. We then design an instruction selection algorithm to simultaneously maximize the depth and semantic coverage of the selected instructions. Experimental results demonstrate that, compared to state-of-the-art baseline methods, it can sustainably improve model performance at a faster pace and thus achieve \emph{``Accelerated Scaling''}. 

**Abstract (ZH)**: 随着对将大规模语言模型应用于下游任务需求的增长，提高模型对齐性能和效率已成为关键。这一过程涉及到从候选集中选择具有信息性的指令。但由于指令集分布的复杂性，驱动对齐模型性能的关键因素仍不清楚。因此，当前的指令集精炼方法无法在指令池不断扩大的情况下改善性能。为解决这一问题，我们首先研究影响指令数据集分布与对齐模型性能之间关系的关键因素。基于这些洞见，我们提出了一种新的指令数据选择方法。我们发现指令的深度和语义空间的覆盖范围是决定下游性能的关键因素，可以解释开发集上超过70%的模型损失。然后，我们设计了一种指令选择算法，同时最大化所选指令的深度和语义覆盖范围。实验结果表明，与最先进的基线方法相比，它能够更快地持续提升模型性能，从而实现“加速扩展”的目标。 

---
# Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning 

**Title (ZH)**: 代理之树：通过多视角推理提升大型语言模型的长上下文能力 

**Authors**: Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06436)  

**Abstract**: Large language models (LLMs) face persistent challenges when handling long-context tasks, most notably the lost in the middle issue, where information located in the middle of a long input tends to be underutilized. Some existing methods that reduce input have the risk of discarding key information, while others that extend context windows often lead to attention dispersion. To address these limitations, we propose Tree of Agents (TOA), a multi-agent reasoning framework that segments the input into chunks processed by independent agents. Each agent generates its local cognition, then agents dynamically exchange information for collaborative reasoning along tree-structured paths. TOA enables agents to probe different reasoning orders for multi-perspective understanding, effectively mitigating position bias and reducing hallucinations. To improve processing efficiency, we incorporate prefix-hash caching and adaptive pruning strategies, achieving significant performance improvements with comparable API overhead. Experiments show that TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple baselines and demonstrates comparable performance to the latest and much larger commercial models, such as Gemini1.5-pro, on various long-context tasks. Code is available at this https URL. 

**Abstract (ZH)**: Large语言模型（LLMs）在处理长上下文任务时面临持续挑战，最 notably的问题是中间信息丢失，即长输入中间部分的信息往往被充分利用不足。一些减少输入长度的方法存在舍弃关键信息的风险，而延长上下文窗口的方法往往会引发注意力分散。为了解决这些限制，我们提出了代理树（TOA）这一多代理推理框架，该框架将输入分割成由独立代理处理的片段。每个代理生成其局部认知，然后代理通过基于树结构的路径动态交换信息进行合作推理。TOA使代理能够探索不同的推理顺序以实现多视角理解，从而有效减轻位置偏见并减少幻觉。为了提高处理效率，我们结合了前缀哈希缓存和自适应剪枝策略，在相近的API开销下实现了显著的性能提升。实验结果表明，由紧凑的LLaMA3.1-8B驱动的TOA在多种长上下文任务上显著优于多个基线模型，并展示了与最新且更大规模的商业模型（如Gemini1.5-pro）相当的性能。代码可在以下链接获取：这个 https URL。 

---
# Evaluating Multi-Turn Bargain Skills in LLM-Based Seller Agent 

**Title (ZH)**: 基于LLM的卖家代理多轮讨价还价技能评估 

**Authors**: Issue Yishu Wang, Kakam Chong, Xiaofeng Wang, Xu Yan, DeXin Kong, Chen Ju, Ming Chen, Shuai Xiao, Shuguang Han, jufeng chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.06341)  

**Abstract**: In online second-hand marketplaces, multi-turn bargaining is a crucial part of seller-buyer interactions. Large Language Models (LLMs) can act as seller agents, negotiating with buyers on behalf of sellers under given business constraints. A critical ability for such agents is to track and accurately interpret cumulative buyer intents across long negotiations, which directly impacts bargaining effectiveness. We introduce a multi-turn evaluation framework for measuring the bargaining ability of seller agents in e-commerce dialogues. The framework tests whether an agent can extract and track buyer intents. Our contributions are: (1) a large-scale e-commerce bargaining benchmark spanning 622 categories, 9,892 products, and 3,014 tasks; (2) a turn-level evaluation framework grounded in Theory of Mind (ToM) with annotated buyer intents, moving beyond outcome-only metrics; and (3) an automated pipeline that extracts reliable intent from massive dialogue data. 

**Abstract (ZH)**: 在线二手市场中，多轮讨价还价是卖家与买家交互的重要组成部分。大规模语言模型可以作为卖家代理，在给定的商业约束下与买家进行谈判。这种代理的一个关键能力是跟踪和准确解释长时间谈判中累积的买家意图，这直接影响谈判效果。我们提出了一种多轮谈判评估框架，用于衡量电子商务对话中卖家代理的谈判能力。该框架测试代理是否能够提取和跟踪买家意图。我们的贡献包括：（1）涵盖622个类别、9,892个产品和3,014个任务的大规模电子商务谈判基准；（2）基于理论心智（ToM）的回合级评估框架，并标注了买家意图，超越了仅关注结果的指标；以及（3）一个自动提取大规模对话数据中可靠意图的管道。 

---
# Large Language Models as Virtual Survey Respondents: Evaluating Sociodemographic Response Generation 

**Title (ZH)**: 大型语言模型作为虚拟调查受访者：社会 demographic 特征应答生成评估 

**Authors**: Jianpeng Zhao, Chenyu Yuan, Weiming Luo, Haoling Xie, Guangwei Zhang, Steven Jige Quan, Zixuan Yuan, Pengyang Wang, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06337)  

**Abstract**: Questionnaire-based surveys are foundational to social science research and public policymaking, yet traditional survey methods remain costly, time-consuming, and often limited in scale. This paper explores a new paradigm: simulating virtual survey respondents using Large Language Models (LLMs). We introduce two novel simulation settings, namely Partial Attribute Simulation (PAS) and Full Attribute Simulation (FAS), to systematically evaluate the ability of LLMs to generate accurate and demographically coherent responses. In PAS, the model predicts missing attributes based on partial respondent profiles, whereas FAS involves generating complete synthetic datasets under both zero-context and context-enhanced conditions. We curate a comprehensive benchmark suite, LLM-S^3 (Large Language Model-based Sociodemographic Survey Simulation), that spans 11 real-world public datasets across four sociological domains. Our evaluation of multiple mainstream LLMs (GPT-3.5/4 Turbo, LLaMA 3.0/3.1-8B) reveals consistent trends in prediction performance, highlights failure modes, and demonstrates how context and prompt design impact simulation fidelity. This work establishes a rigorous foundation for LLM-driven survey simulations, offering scalable and cost-effective tools for sociological research and policy evaluation. Our code and dataset are available at: this https URL 

**Abstract (ZH)**: 基于问卷的调查是社会科学研究和公共政策制定的基础，但传统的调查方法仍然昂贵、耗时且规模有限。本文探讨了一种新的范式：使用大规模语言模型（LLMs）模拟虚拟调查受访者。我们介绍了两种新的模拟设置，即部分属性模拟（PAS）和全程属性模拟（FAS），以系统评估LLMs生成准确且人口统计学一致的响应的能力。在PAS中，模型根据部分受访者的画像预测缺失的属性，而在FAS中，则在零上下文和增强上下文条件下生成完整的合成数据集。我们编纂了一个全面的基准套件LLM-S^3（基于大规模语言模型的社会人口统计学调查模拟），涵盖了四个社会学领域内的11个真实世界公共数据集。对多个主流LLM（GPT-3.5/4 Turbo、LLaMA 3.0/3.1-8B）的评估揭示了预测性能的一致趋势，指出了失效模式，并展示了上下文和提示设计如何影响模拟精度。本项工作为基于LLM的调查模拟奠定了严格的理论基础，提供了社会学研究和政策评价的可扩展且成本效益高的工具。我们的代码和数据集可在以下网址获得：this https URL 

---
# Can AI Make Energy Retrofit Decisions? An Evaluation of Large Language Models 

**Title (ZH)**: AI能为能源改造决策提供帮助吗？大型语言模型的评估 

**Authors**: Lei Shu, Dong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06307)  

**Abstract**: Conventional approaches to building energy retrofit decision making suffer from limited generalizability and low interpretability, hindering adoption in diverse residential contexts. With the growth of Smart and Connected Communities, generative AI, especially large language models (LLMs), may help by processing contextual information and producing practitioner readable recommendations. We evaluate seven LLMs (ChatGPT, DeepSeek, Gemini, Grok, Llama, and Claude) on residential retrofit decisions under two objectives: maximizing CO2 reduction (technical) and minimizing payback period (sociotechnical). Performance is assessed on four dimensions: accuracy, consistency, sensitivity, and reasoning, using a dataset of 400 homes across 49 US states. LLMs generate effective recommendations in many cases, reaching up to 54.5 percent top 1 match and 92.8 percent within top 5 without fine tuning. Performance is stronger for the technical objective, while sociotechnical decisions are limited by economic trade offs and local context. Agreement across models is low, and higher performing models tend to diverge from others. LLMs are sensitive to location and building geometry but less sensitive to technology and occupant behavior. Most models show step by step, engineering style reasoning, but it is often simplified and lacks deeper contextual awareness. Overall, LLMs are promising assistants for energy retrofit decision making, but improvements in accuracy, consistency, and context handling are needed for reliable practice. 

**Abstract (ZH)**: 基于生成式AI的大型语言模型在住宅能效改造决策中的潜力：超越传统方法的限制 

---
# From Implicit Exploration to Structured Reasoning: Leveraging Guideline and Refinement for LLMs 

**Title (ZH)**: 从隐式探索到结构化推理：利用指南和精炼提升语言模型 

**Authors**: Jiaxiang Chen, Zhuo Wang, Mingxi Zou, Zhucong Li, Zhijian Zhou, Song Wang, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06284)  

**Abstract**: Large language models (LLMs) have advanced general-purpose reasoning, showing strong performance across diverse tasks. However, existing methods often rely on implicit exploration, where the model follows stochastic and unguided reasoning paths-like walking without a map. This leads to unstable reasoning paths, lack of error correction, and limited learning from past experience. To address these issues, we propose a framework that shifts from implicit exploration to structured reasoning through guideline and refinement. First, we extract structured reasoning patterns from successful trajectories and reflective signals from failures. During inference, the model follows these guidelines step-by-step, with refinement applied after each step to correct errors and stabilize the reasoning process. Experiments on BBH and four additional benchmarks (GSM8K, MATH-500, MBPP, HumanEval) show that our method consistently outperforms strong baselines across diverse reasoning tasks. Structured reasoning with stepwise execution and refinement improves stability and generalization, while guidelines transfer well across domains and flexibly support cross-model collaboration, matching or surpassing supervised fine-tuning in effectiveness and scalability. 

**Abstract (ZH)**: 大型语言模型（LLMs）在通用推理方面取得了进展，展现了在多种任务上的强大性能。然而，现有方法往往依赖于隐式的探索，模型沿着随机且不受指导的推理路径进行推理，如同没有地图的行走。这导致了不稳定的推理路径、缺乏错误纠正以及有限的学习能力。为解决这些问题，我们提出了一种框架，通过指南和修正转向结构化的推理。首先，我们从成功的轨迹和失败的反思信号中提取结构化的推理模式。在推理过程中，模型按照这些指南逐步执行，并在每一步后进行修正以纠正错误并稳定推理过程。实验结果表明，我们的方法在BBH及四个附加基准（GSM8K、MATH-500、MBPP、HumanEval）上的一致性地优于强baseline，在多种推理任务上表现出色。逐步执行和修正的结构化推理提高了稳定性和泛化能力，而指南在不同领域之间传递良好，并灵活支持跨模型协作，在效果和扩展性上可匹及或超越监督微调。 

---
# SFR-DeepResearch: Towards Effective Reinforcement Learning for Autonomously Reasoning Single Agents 

**Title (ZH)**: SFR-DeepResearch: 向自主推理单个代理的有效强化学习方向 

**Authors**: Xuan-Phi Nguyen, Shrey Pandit, Revanth Gangi Reddy, Austin Xu, Silvio Savarese, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.06283)  

**Abstract**: Equipping large language models (LLMs) with complex, interleaved reasoning and tool-use capabilities has become a key focus in agentic AI research, especially with recent advances in reasoning-oriented (``thinking'') models. Such capabilities are key to unlocking a number of important applications. One such application is Deep Research (DR), which requires extensive search and reasoning over many sources. Our work in this paper focuses on the development of native Autonomous Single-Agent models for DR featuring minimal web crawling and Python tool integration. Unlike multi-agent systems, where agents take up pre-defined roles and are told what to do at each step in a static workflow, an autonomous single-agent determines its next action dynamically based on context, without manual directive. While prior work has proposed training recipes for base or instruction-tuned LLMs, we focus on continual reinforcement learning (RL) of reasoning-optimized models to further enhance agentic skills while preserving reasoning ability. Towards this end, we propose a simple RL recipe with entirely synthetic data, which we apply to various open-source LLMs. Our best variant SFR-DR-20B achieves up to 28.7% on Humanity's Last Exam benchmark. In addition, we conduct key analysis experiments to provide more insights into our methodologies. 

**Abstract (ZH)**: 装备有复杂交错推理和工具使用能力的大规模语言模型（LLMs）已成为有目的的AI研究中的关键焦点，尤其是在推理导向（“思考”）模型 recently 的进展之后。此类能力对于开启多种重要应用至关重要。其中一个应用是深度研究（DR），其需要在多种来源上进行广泛的搜索和推理。本文的工作集中在开发原生自主单智能体模型（DR）的开发上，这些模型具备最小限度的网页抓取和Python工具集成。与多智能体系统不同，在多智能体系统中，智能体承担预定义的角色，并在其静态工作流中的每一步被告知做什么，自主单智能体会根据上下文动态确定其下一步行动，无需人工指令。尽管先前的工作已经提出了针对基础或指令调优语言模型的训练方法，但我们专注于通过持续强化学习（RL）来进一步提升智能体技能，同时保持推理能力。为此，我们提出了一种基于完全合成数据的简单RL方法，并将其应用于多种开源LLM。我们的最佳变体SFR-DR-20B在人类的最后一试基准测试中达到了最高28.7%的成绩。此外，我们还进行了关键分析实验，以更深入地了解我们的方法。 

---
# PillagerBench: Benchmarking LLM-Based Agents in Competitive Minecraft Team Environments 

**Title (ZH)**: PillagerBench: 在竞争性Minecraft团队环境中文本生成代理的基准测试 

**Authors**: Olivier Schipper, Yudi Zhang, Yali Du, Mykola Pechenizkiy, Meng Fang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06235)  

**Abstract**: LLM-based agents have shown promise in various cooperative and strategic reasoning tasks, but their effectiveness in competitive multi-agent environments remains underexplored. To address this gap, we introduce PillagerBench, a novel framework for evaluating multi-agent systems in real-time competitive team-vs-team scenarios in Minecraft. It provides an extensible API, multi-round testing, and rule-based built-in opponents for fair, reproducible comparisons. We also propose TactiCrafter, an LLM-based multi-agent system that facilitates teamwork through human-readable tactics, learns causal dependencies, and adapts to opponent strategies. Our evaluation demonstrates that TactiCrafter outperforms baseline approaches and showcases adaptive learning through self-play. Additionally, we analyze its learning process and strategic evolution over multiple game episodes. To encourage further research, we have open-sourced PillagerBench, fostering advancements in multi-agent AI for competitive environments. 

**Abstract (ZH)**: 基于LLM的代理在多种协作与策略性推理任务中展现了潜力，但它们在竞争性多代理环境中的有效性仍待深入探索。为填补这一空白，我们引入了PillagerBench，一种用于在Minecraft中评估实时竞争性团队对团队多代理系统的新型框架。该框架提供了可扩展的API、多轮测试和基于规则的内置对手，以实现公平和可重复的比较。此外，我们还提出了TactiCrafter，一种基于LLM的多代理系统，通过可读性战术促进团队协作，学习因果依赖关系，并适应对手策略。我们的评估表明，TactiCrafter在基准方法中表现更优，并通过自我对弈展示了自适应学习能力。另外，我们还分析了其在多个游戏回合中的学习过程和战略演变。为了促进进一步研究，我们开源了PillagerBench，推动竞争性环境中的多代理AI技术进步。 

---
# From Long to Short: LLMs Excel at Trimming Own Reasoning Chains 

**Title (ZH)**: 从长到短：大规模语言模型擅长精简自身的推理链 

**Authors**: Wei Han, Geng Zhan, Sicheng Yu, Chenyu Wang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.06174)  

**Abstract**: O1/R1 style large reasoning models (LRMs) signal a substantial leap forward over conventional instruction-following LLMs. By applying test-time scaling to generate extended reasoning paths, they establish many SOTAs across a wide range of complex reasoning tasks. However, recent studies show that LRMs are prone to suffer from overthinking -- the tendency to overcomplicate simple problems, leading to excessive strategy switching and long, convoluted reasoning traces that hinder their interpretability. To mitigate this issue, we conduct a systematic investigation into the reasoning efficiency of a broad set of LRMs and uncover a common dilemma: the difficulty in balancing multiple generation objectives such as correctness and brevity. Based on this discovery, we propose a test-time scaling method, EDIT (Efficient Dynamic Inference Trimming), which efficiently guides LRMs to identify the shortest correct reasoning paths at test time. EDIT employs constraint-guided generation while jointly tracking length and answer distributions under varying constraints, allowing it to select responses that strike an optimal balance between conciseness and correctness. Extensive experiments across diverse models and datasets show that EDIT substantially enhance the reasoning efficiency, producing compact yet informative outputs that improve readability and user experience. 

**Abstract (ZH)**: O1/R1风格大规模推理模型的推理效率研究：EDIT方法克服过度思考问题 

---
# Rethinking Reasoning Quality in Large Language Models through Enhanced Chain-of-Thought via RL 

**Title (ZH)**: 通过增强链式思考的RL促进大型语言模型中推理质量的重新思考 

**Authors**: Haoyang He, Zihua Rong, Kun Ji, Chenyang Li, Qing Huang, Chong Xia, Lan Yang, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06024)  

**Abstract**: Reinforcement learning (RL) has recently become the dominant paradigm for strengthening the reasoning abilities of large language models (LLMs). Yet the rule-based reward functions commonly used on mathematical or programming benchmarks assess only answer format and correctness, providing no signal as to whether the induced Chain-of-Thought (CoT) actually improves the answer. Furthermore, such task-specific training offers limited control over logical depth and therefore may fail to reveal a model's genuine reasoning capacity. We propose Dynamic Reasoning Efficiency Reward (DRER) -- a plug-and-play RL reward framework that reshapes both reward and advantage signals. (i) A Reasoning Quality Reward assigns fine-grained credit to those reasoning chains that demonstrably raise the likelihood of the correct answer, directly incentivising the trajectories with beneficial CoT tokens. (ii) A Dynamic Length Advantage decays the advantage of responses whose length deviates from a validation-derived threshold, stabilising training. To facilitate rigorous assessment, we also release Logictree, a dynamically constructed deductive reasoning dataset that functions both as RL training data and as a comprehensive benchmark. Experiments confirm the effectiveness of DRER: our 7B model attains GPT-o3-mini level performance on Logictree with 400 trianing steps, while the average confidence of CoT-augmented answers rises by 30%. The model further exhibits generalisation across diverse logical-reasoning datasets, and the mathematical benchmark AIME24. These results illuminate how RL shapes CoT behaviour and chart a practical path toward enhancing formal-reasoning skills in large language models. All code and data are available in repository this https URL. 

**Abstract (ZH)**: 动态推理效率奖励：增强大型语言模型推理能力的插件式 reinforcement 学习奖励框架 

---
# Chatbot To Help Patients Understand Their Health 

**Title (ZH)**: Chatbot 以帮助患者理解其健康状况 

**Authors**: Won Seok Jang, Hieu Tran, Manav Mistry, SaiKiran Gandluri, Yifan Zhang, Sharmin Sultana, Sunjae Kown, Yuan Zhang, Zonghai Yao, Hong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05818)  

**Abstract**: Patients must possess the knowledge necessary to actively participate in their care. We present NoteAid-Chatbot, a conversational AI that promotes patient understanding via a novel 'learning as conversation' framework, built on a multi-agent large language model (LLM) and reinforcement learning (RL) setup without human-labeled data. NoteAid-Chatbot was built on a lightweight LLaMA 3.2 3B model trained in two stages: initial supervised fine-tuning on conversational data synthetically generated using medical conversation strategies, followed by RL with rewards derived from patient understanding assessments in simulated hospital discharge scenarios. Our evaluation, which includes comprehensive human-aligned assessments and case studies, demonstrates that NoteAid-Chatbot exhibits key emergent behaviors critical for patient education, such as clarity, relevance, and structured dialogue, even though it received no explicit supervision for these attributes. Our results show that even simple Proximal Policy Optimization (PPO)-based reward modeling can successfully train lightweight, domain-specific chatbots to handle multi-turn interactions, incorporate diverse educational strategies, and meet nuanced communication objectives. Our Turing test demonstrates that NoteAid-Chatbot surpasses non-expert human. Although our current focus is on healthcare, the framework we present illustrates the feasibility and promise of applying low-cost, PPO-based RL to realistic, open-ended conversational domains, broadening the applicability of RL-based alignment methods. 

**Abstract (ZH)**: 患者必须具备积极参与其治疗过程所需的知识。我们介绍了NoteAid-Chatbot，这是一种通过新颖的“学习即对话”框架促进患者理解的对话型AI，基于多代理大型语言模型和强化学习设置，无需人工标注数据。NoteAid-Chatbot基于经过两阶段训练的轻量级LaMA 3.2 3B模型构建：初始阶段是监督微调，使用医疗对话策略合成对话数据生成，随后是使用源自模拟出院场景中患者理解评估的奖励进行强化学习。我们的评估包括全面的人类对齐评估和案例研究，表明NoteAid-Chatbot即使没有明确为这些属性提供监督，也表现出关键的新兴行为，如清晰度、相关性和结构化对话。我们的结果表明，即使简单的基于PPO的奖励建模也能成功训练轻量级、领域特定的聊天机器人，处理多轮交互，融入多样化的教育策略，并满足精细的沟通目标。我们的图灵测试表明，NoteAid-Chatbot超越了非专家人类。尽管我们当前的重点是医疗保健，但我们提出的方法框架展示了将低成本、基于PPO的强化学习应用于现实的开放性对话领域可行性和潜力，扩展了基于RL对齐方法的应用范围。 

---
# DRF: LLM-AGENT Dynamic Reputation Filtering Framework 

**Title (ZH)**: DRF: LLM-AGENT动态声誉过滤框架 

**Authors**: Yuwei Lou, Hao Hu, Shaocong Ma, Zongfei Zhang, Liang Wang, Jidong Ge, Xianping Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.05764)  

**Abstract**: With the evolution of generative AI, multi - agent systems leveraging large - language models(LLMs) have emerged as a powerful tool for complex tasks. However, these systems face challenges in quantifying agent performance and lack mechanisms to assess agent credibility. To address these issues, we introduce DRF, a dynamic reputation filtering framework. DRF constructs an interactive rating network to quantify agent performance, designs a reputation scoring mechanism to measure agent honesty and capability, and integrates an Upper Confidence Bound - based strategy to enhance agent selection efficiency. Experiments show that DRF significantly improves task completion quality and collaboration efficiency in logical reasoning and code - generation tasks, offering a new approach for multi - agent systems to handle large - scale tasks. 

**Abstract (ZH)**: 基于大型语言模型的多agent系统的动态声誉过滤框架：提升复杂任务完成质量和协作效率 

---
# Hyperbolic Large Language Models 

**Title (ZH)**: 双曲大语言模型 

**Authors**: Sarang Patil, Zeyong Zhang, Yiran Huang, Tengfei Ma, Mengjia Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05757)  

**Abstract**: Large language models (LLMs) have achieved remarkable success and demonstrated superior performance across various tasks, including natural language processing (NLP), weather forecasting, biological protein folding, text generation, and solving mathematical problems. However, many real-world data exhibit highly non-Euclidean latent hierarchical anatomy, such as protein networks, transportation networks, financial networks, brain networks, and linguistic structures or syntactic trees in natural languages. Effectively learning intrinsic semantic entailment and hierarchical relationships from these raw, unstructured input data using LLMs remains an underexplored area. Due to its effectiveness in modeling tree-like hierarchical structures, hyperbolic geometry -- a non-Euclidean space -- has rapidly gained popularity as an expressive latent representation space for complex data modeling across domains such as graphs, images, languages, and multi-modal data. Here, we provide a comprehensive and contextual exposition of recent advancements in LLMs that leverage hyperbolic geometry as a representation space to enhance semantic representation learning and multi-scale reasoning. Specifically, the paper presents a taxonomy of the principal techniques of Hyperbolic LLMs (HypLLMs) in terms of four main categories: (1) hyperbolic LLMs through exp/log maps; (2) hyperbolic fine-tuned models; (3) fully hyperbolic LLMs, and (4) hyperbolic state-space models. We also explore crucial potential applications and outline future research directions. A repository of key papers, models, datasets, and code implementations is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务上取得了显著的成功，并在自然语言处理（NLP）、天气预报、生物蛋白质折叠、文本生成和解决数学问题等方面展示了优越的性能。然而，许多现实世界数据表现出高度非欧几里得的潜在分层结构，如蛋白质网络、交通网络、金融网络、脑网络以及自然语言中的句法树结构。使用LLMs从这些原始的无结构输入数据中有效学习内在语义蕴含和分层关系仍是一个未充分探索的领域。由于其在建模树状分层结构方面的有效性，双曲几何——一种非欧几里得空间——迅速成为跨图形、图像、语言和多模态数据建模的表达性潜在表示空间的热门选择。本文综述了利用双曲几何作为表示空间以增强语义表示学习和多尺度推理的LLMs的最新进展。具体而言，论文按照四个主要类别对双曲几何L大型语言模型（HypLLMs）的主要技术进行分类阐述：（1）基于exp/log映射的双曲几何L大型语言模型；（2）微调的双曲几何模型；（3）完全双曲几何L大型语言模型，以及（4）双曲几何状态空间模型。此外，本文还探讨了关键潜在应用并概述了未来的研究方向。有关关键论文、模型、数据集和代码实现的资源，请访问此网址：https://xxxxxx。 

---
# Towards Meta-Cognitive Knowledge Editing for Multimodal LLMs 

**Title (ZH)**: 向元认知知识编辑迈向多模态LLM 

**Authors**: Zhaoyu Fan, Kaihang Pan, Mingze Zhou, Bosheng Qin, Juncheng Li, Shengyu Zhang, Wenqiao Zhang, Siliang Tang, Fei Wu, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05714)  

**Abstract**: Knowledge editing enables multimodal large language models (MLLMs) to efficiently update outdated or incorrect information. However, existing benchmarks primarily emphasize cognitive-level modifications while lacking a focus on deeper meta-cognitive processes. To bridge this gap, we introduce CogEdit, a novel benchmark designed to evaluate MLLMs' meta-cognitive knowledge editing abilities across three levels: (1) Counterfactual-Driven Editing, assessing self-awareness of knowledge correctness changes; (2) Boundary Constraint Editing, ensuring appropriate generalization without unintended interference; and (3) Noise-Robust Editing, promoting reflective evaluation of uncertain information. To advance meta-cognitive editing, we propose MIND (Meta-cognitive INtegrated Dynamic Knowledge Editing), a framework that constructs a meta-knowledge memory for self-awareness, employs game-theoretic interactions to monitor knowledge activation, and incorporates label refinement for noise-robust updates. Extensive experiments show that MIND significantly outperforms existing cognitive editing approaches, achieving strong performance on both traditional and meta-cognitive knowledge editing benchmarks. 

**Abstract (ZH)**: 知识编辑使多模态大型语言模型（MLLMs）能够高效地更新过时或不正确的信息。然而，现有的基准主要侧重于认知层面的修改，缺乏对更深层次元认知过程的关注。为填补这一缺口，我们引入了CogEdit，一个旨在评估MLLMs在三个层面的元认知知识编辑能力的新基准：（1）反事实驱动编辑，评估知识正确性变化的自我意识；（2）边界约束编辑，确保适当的泛化而不产生意外干扰；（3）噪声鲁棒编辑，促进对不确定信息的反思性评估。为了推进元认知编辑，我们提出了一种MIND（元认知集成动态知识编辑）框架，该框架构建了一个元知识记忆以提供自我意识，采用博弈论交互来监控知识激活，并结合标签精炼来实现噪声鲁棒更新。大量实验表明，MIND 在认知编辑方法中表现出显著的优势，并在传统和元认知知识编辑基准上取得了优异的表现。 

---
# Murphys Laws of AI Alignment: Why the Gap Always Wins 

**Title (ZH)**: AI对齐的墨菲定律：为什么差距总是胜出 

**Authors**: Madhava Gaikwad  

**Link**: [PDF](https://arxiv.org/pdf/2509.05381)  

**Abstract**: Large language models are increasingly aligned to human preferences through reinforcement learning from human feedback (RLHF) and related methods such as Direct Preference Optimization (DPO), Constitutional AI, and RLAIF. While effective, these methods exhibit recurring failure patterns i.e., reward hacking, sycophancy, annotator drift, and misgeneralization. We introduce the concept of the Alignment Gap, a unifying lens for understanding recurring failures in feedback-based alignment. Using a KL-tilting formalism, we illustrate why optimization pressure tends to amplify divergence between proxy rewards and true human intent. We organize these failures into a catalogue of Murphys Laws of AI Alignment, and propose the Alignment Trilemma as a way to frame trade-offs among optimization strength, value capture, and generalization. Small-scale empirical studies serve as illustrative support. Finally, we propose the MAPS framework (Misspecification, Annotation, Pressure, Shift) as practical design levers. Our contribution is not a definitive impossibility theorem but a perspective that reframes alignment debates around structural limits and trade-offs, offering clearer guidance for future design. 

**Abstract (ZH)**: 大型语言模型通过人类反馈强化学习（RLHF）及相关方法（如直接偏好优化（DPO）、宪法AI和RLAIF）越来越接近人类偏好。尽管有效，这些方法表现出重复出现的失败模式，包括奖励作弊、讨好行为、注释员偏差和误泛化。我们引入了对齐缺口的概念，这是一种统一的视角，用于理解基于反馈对齐中的重复失败。通过KL-倾斜的形式主义，我们解释了为什么优化压力倾向于放大代理奖励与真实人类意图之间的差异。我们将这些失败归类为AI对齐的墨菲定律，提出了对齐三难困境作为一种方式来界定优化强度、价值捕获和泛化的权衡。我们通过小规模的实证研究提供说明性的支持。最后，我们提出了MAPS框架（错误规定、注释、压力、变化）作为实际设计杠杆。我们的贡献不是一项完备的不可能定理，而是一种视角，重新框定了对齐辩论的结构性限制和权衡，为未来的设计提供了更清晰的指导。 

---
# Characterizing Fitness Landscape Structures in Prompt Engineering 

**Title (ZH)**: characterizing 优化提示工程中适应度景观结构的研究 

**Authors**: Arend Hintze  

**Link**: [PDF](https://arxiv.org/pdf/2509.05375)  

**Abstract**: While prompt engineering has emerged as a crucial technique for optimizing large language model performance, the underlying optimization landscape remains poorly understood. Current approaches treat prompt optimization as a black-box problem, applying sophisticated search algorithms without characterizing the landscape topology they navigate. We present a systematic analysis of fitness landscape structures in prompt engineering using autocorrelation analysis across semantic embedding spaces. Through experiments on error detection tasks with two distinct prompt generation strategies -- systematic enumeration (1,024 prompts) and novelty-driven diversification (1,000 prompts) -- we reveal fundamentally different landscape topologies. Systematic prompt generation yields smoothly decaying autocorrelation, while diversified generation exhibits non-monotonic patterns with peak correlation at intermediate semantic distances, indicating rugged, hierarchically structured landscapes. Task-specific analysis across 10 error detection categories reveals varying degrees of ruggedness across different error types. Our findings provide an empirical foundation for understanding the complexity of optimization in prompt engineering landscapes. 

**Abstract (ZH)**: 而提示工程作为一个优化大规模语言模型性能的关键技术已经崭露头角，但其背后的优化景观仍不甚明了。当前的方法将提示优化视为一个黑盒问题，应用复杂的搜索算法而不表征他们导航的景观拓扑结构。我们利用自相关分析跨语义嵌入空间系统分析提示工程中的适应度景观结构。通过在两种不同的提示生成策略（系统枚举1,024个提示和 novelty-driven 多样化1,000个提示）下进行错误检测任务的实验，我们揭示了根本不同的景观拓扑结构。系统提示生成产生平滑衰减的自相关性，而多样化生成表现出非单调模式，其在中间语义距离处出现峰值相关性，表明这些景观具有崎岖且分层结构。针对10个错误检测类别进行的任务特定分析揭示了不同类型错误在崎岖度上的差异。我们的研究结果为理解提示工程景观中优化的复杂性提供了经验基础。 

---
# SasAgent: Multi-Agent AI System for Small-Angle Scattering Data Analysis 

**Title (ZH)**: SasAgent: 多Agent人工智能系统用于小角度散射数据分析 

**Authors**: Lijie Ding, Changwoo Do  

**Link**: [PDF](https://arxiv.org/pdf/2509.05363)  

**Abstract**: We introduce SasAgent, a multi-agent AI system powered by large language models (LLMs) that automates small-angle scattering (SAS) data analysis by leveraging tools from the SasView software and enables user interaction via text input. SasAgent features a coordinator agent that interprets user prompts and delegates tasks to three specialized agents for scattering length density (SLD) calculation, synthetic data generation, and experimental data fitting. These agents utilize LLM-friendly tools to execute tasks efficiently. These tools, including the model data tool, Retrieval-Augmented Generation (RAG) documentation tool, bump fitting tool, and SLD calculator tool, are derived from the SasView Python library. A user-friendly Gradio-based interface enhances user accessibility. Through diverse examples, we demonstrate SasAgent's ability to interpret complex prompts, calculate SLDs, generate accurate scattering data, and fit experimental datasets with high precision. This work showcases the potential of LLM-driven AI systems to streamline scientific workflows and enhance automation in SAS research. 

**Abstract (ZH)**: 基于大型语言模型的多agent系统SasAgent及其在小角散射数据自动分析中的应用 

---
# Benchmarking Large Language Models for Personalized Guidance in AI-Enhanced Learning 

**Title (ZH)**: 大型语言模型在人工智能增强学习个性化指导中的benchmarking研究 

**Authors**: Bo Yuan, Jiazi Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05346)  

**Abstract**: While Large Language Models (LLMs) are increasingly envisioned as intelligent assistants for personalized learning, systematic head-to-head evaluations within authentic learning scenarios remain limited. This study conducts an empirical comparison of three state-of-the-art LLMs on a tutoring task that simulates a realistic learning setting. Using a dataset comprising a student's answers to ten questions of mixed formats with correctness labels, each LLM is required to (i) analyze the quiz to identify underlying knowledge components, (ii) infer the student's mastery profile, and (iii) generate targeted guidance for improvement. To mitigate subjectivity and evaluator bias, we employ Gemini as a virtual judge to perform pairwise comparisons along various dimensions: accuracy, clarity, actionability, and appropriateness. Results analyzed via the Bradley-Terry model indicate that GPT-4o is generally preferred, producing feedback that is more informative and better structured than its counterparts, while DeepSeek-V3 and GLM-4.5 demonstrate intermittent strengths but lower consistency. These findings highlight the feasibility of deploying LLMs as advanced teaching assistants for individualized support and provide methodological guidance for future empirical research on LLM-driven personalized learning. 

**Abstract (ZH)**: 虽然大型语言模型（LLMs）越来越被视为个性化学习的智能助手，但在真实学习场景下系统的直接对比仍然有限。本研究通过一个模拟现实学习环境的辅导任务，对三种最先进的LLM进行了实证比较。使用包含一组学生对十道混合格式问题的回答及其正确性标签的数据集，每种LLM需完成以下任务：（i）分析测验以识别潜在的知识组件，（ii）推断学生的掌握情况，以及（iii）生成针对性的改进指导。为减少主观性和评价者偏见，我们利用Gemini作为虚拟评委，从准确度、清晰度、可行性和恰当性等多个维度进行两两对比。通过Bradley-Terry模型分析结果表明，GPT-4o通常更受欢迎，其反馈信息更丰富且结构更严谨，而DeepSeek-V3和GLM-4.5则显示出间歇性的优势但一致性较低。这些发现突显了部署LLM作为高级教学助手用于个性化支持的可行性，并为未来基于LLM的个性化学习实证研究提供了方法学指导。 

---
# From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers 

**Title (ZH)**: 从噪声到叙事：探究Transformer中幻觉的起源 

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok  

**Link**: [PDF](https://arxiv.org/pdf/2509.06938)  

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk. 

**Abstract (ZH)**: 随着生成式AI系统在科学、商业和政府领域的能力和普及程度不断提高，对它们的失败模式进行更深入的洞察现在变得尤为迫切。它们偶尔在行为上的波动性，如变压器模型的幻觉倾向，阻碍了在高风险领域对新兴AI解决方案的信任和采用。在本项工作中，我们通过稀疏自编码器捕获的概念表示，探讨了在输入空间实验控制下的不确定性情景中幻觉是如何在预训练变压器模型中出现的。系统性的实验表明，当输入信息变得越来越无结构时，变压器模型使用的语义概念数量会增加。面对输入空间日益增大的不确定性，变压器模型更容易激活与输入无关但具有一致性的语义特征，从而产生幻觉输出。在极端情况下，对于纯噪声输入，我们发现预训练变压器模型的中间激活中广泛存在被稳健触发的并具备意义的语义概念，并通过针对性的引导确认了其功能性完整性。我们还展示了变压器模型输出中的幻觉可以从嵌入在变压器层激活中的概念模式中可靠地预测。这些关于变压器内部处理机制的洞察对对齐AI模型与人类价值观、AI安全性、扩大潜在对抗攻击的攻击面以及提供模型幻觉风险自动量化基础等方面具有立竿见影的影响。 

---
# An Ethically Grounded LLM-Based Approach to Insider Threat Synthesis and Detection 

**Title (ZH)**: 基于伦理准则的LLM驱动的内部威胁合成与检测方法 

**Authors**: Haywood Gelman, John D. Hastings, David Kenley  

**Link**: [PDF](https://arxiv.org/pdf/2509.06920)  

**Abstract**: Insider threats are a growing organizational problem due to the complexity of identifying their technical and behavioral elements. A large research body is dedicated to the study of insider threats from technological, psychological, and educational perspectives. However, research in this domain has been generally dependent on datasets that are static and limited access which restricts the development of adaptive detection models. This study introduces a novel, ethically grounded approach that uses the large language model (LLM) Claude Sonnet 3.7 to dynamically synthesize syslog messages, some of which contain indicators of insider threat scenarios. The messages reflect real-world data distributions by being highly imbalanced (1% insider threats). The syslogs were analyzed for insider threats by both Claude Sonnet 3.7 and GPT-4o, with their performance evaluated through statistical metrics including precision, recall, MCC, and ROC AUC. Sonnet 3.7 consistently outperformed GPT-4o across nearly all metrics, particularly in reducing false alarms and improving detection accuracy. The results show strong promise for the use of LLMs in synthetic dataset generation and insider threat detection. 

**Abstract (ZH)**: 基于大型语言模型的伦理导向合成日志方法在内部威胁检测中的应用 

---
# Disentangling Interaction and Bias Effects in Opinion Dynamics of Large Language Models 

**Title (ZH)**: 分离交互作用和偏差效应在大型语言模型意见动力学中的影响 

**Authors**: Vincent C. Brockers, David A. Ehrlich, Viola Priesemann  

**Link**: [PDF](https://arxiv.org/pdf/2509.06858)  

**Abstract**: Large Language Models are increasingly used to simulate human opinion dynamics, yet the effect of genuine interaction is often obscured by systematic biases. We present a Bayesian framework to disentangle and quantify three such biases: (i) a topic bias toward prior opinions in the training data; (ii) an agreement bias favoring agreement irrespective of the question; and (iii) an anchoring bias toward the initiating agent's stance. Applying this framework to multi-step dialogues reveals that opinion trajectories tend to quickly converge to a shared attractor, with the influence of the interaction fading over time, and the impact of biases differing between LLMs. In addition, we fine-tune an LLM on different sets of strongly opinionated statements (incl. misinformation) and demonstrate that the opinion attractor shifts correspondingly. Exposing stark differences between LLMs and providing quantitative tools to compare them to human subjects in the future, our approach highlights both chances and pitfalls in using LLMs as proxies for human behavior. 

**Abstract (ZH)**: 大型语言模型越来越多地用于模拟人类意见动态，但真实的交互效果往往被系统性偏见所掩盖。我们提出了一种贝叶斯框架以分离和量化三种偏见：（i）主题偏见，倾向于训练数据中的先验观点；（ii）一致性偏见，倾向于一致而不论问题如何；（iii）锚定偏见，倾向于初始行动者的态度。将该框架应用于多步对话表明，意见轨迹往往会迅速向一个共享的吸引子收敛，交互影响随时间减弱，并且不同大型语言模型受偏见影响的方式不同。此外，我们对不同集强观点陈述（包括 misinformation）的大型语言模型进行微调，并证明了意见吸引子随之相应地变化。通过揭示大型语言模型之间明显的差异，并为将来将它们与人类受试者进行定量比较提供工具，我们的方法突显了使用大型语言模型作为人类行为代理的机遇与风险。 

---
# COMPACT: Common-token Optimized Model Pruning Across Channels and Tokens 

**Title (ZH)**: COMMON-TOKEN 优化通道和_token_剪枝模型 

**Authors**: Eugene Kwek, Wenpeng Yin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06836)  

**Abstract**: Making LLMs more efficient in memory, latency, and serving cost is crucial for edge deployment, interactive applications, and sustainable inference at scale. Pruning is a key technique toward this goal. However, prior pruning methods are limited: width pruning often breaks the standard transformer layout or requires custom inference code, while depth pruning removes entire layers and can cause abrupt accuracy drops. In this work, we propose COMPACT, which jointly (i) prunes rare vocabulary to shrink embedding/unembedding and (ii) prunes FFN intermediate channels using common-token-weighted activations, aligning importance with the post-pruning token distribution. COMPACT enjoys merits of both depth and width pruning, such as: deployment-friendliness (keeps a standard transformer architecture), scale-adaptivity (trade off vocab vs. FFN pruning), training-free operation with competitive pruning time, and strong memory savings alongside throughput gains. Experiments across Qwen, LLaMA, and Gemma families (0.5B-70B) show state-of-the-art downstream task performance at similar or higher pruning ratios, with substantial reductions in parameters, GPU memory, and end-to-end latency. 

**Abstract (ZH)**: 提高大语言模型在内存、延迟和推理成本方面的效率对于边缘部署、交互应用及大规模可持续推理至关重要。剪枝是实现这一目标的关键技术。然而，现有的剪枝方法存在局限性：宽度剪枝常常破坏标准的变压器结构或需要定制的推理代码，而深度剪枝会移除整个层，从而导致准确率突变下降。在本文中，我们提出了一种名为COMPACT的方法，该方法联合进行(i)稀有词汇剪枝以缩小嵌入/解嵌入以及(ii)使用常见标记权重激活的功能层中间通道剪枝，使重要性与后剪枝标记分布相一致。COMPACT方法同时具备深度剪枝和宽度剪枝的优点，如：易于部署（保持标准的变压器结构）、规模适应性（在词汇和功能层之间进行权衡）、无需训练即可操作且具有竞争力的剪枝时间、以及显著的内存节省和吞吐量提升。跨Qwen、LLaMA和Gemma系列（0.5B-70B参数）的实验结果显示，在相似或更高的剪枝比率下，COMPACT实现了最先进的下游任务性能，并大幅减少了参数量、GPU内存和端到端延迟。 

---
# Saturation-Driven Dataset Generation for LLM Mathematical Reasoning in the TPTP Ecosystem 

**Title (ZH)**: 饱和驱动的数据集生成以支持TPTP生态系统中的LLM数学推理 

**Authors**: Valentin Quesnel, Damien Sileo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06809)  

**Abstract**: The scarcity of high-quality, logically sound data is a critical bottleneck for advancing the mathematical reasoning of Large Language Models (LLMs). Our work confronts this challenge by turning decades of automated theorem proving research into a scalable data engine. Rather than relying on error-prone LLMs or complex proof-assistant syntax like Lean and Isabelle, our framework leverages E-prover's saturation capabilities on the vast TPTP axiom library to derive a massive, guaranteed-valid corpus of theorems. Our pipeline is principled and simple: saturate axioms, filter for "interesting" theorems, and generate tasks. With no LLMs in the loop, we eliminate factual errors by construction. This purely symbolic data is then transformed into three difficulty-controlled challenges: entailment verification, premise selection, and proof reconstruction. Our zero-shot experiments on frontier models reveal a clear weakness: performance collapses on tasks requiring deep, structural reasoning. Our framework provides both the diagnostic tool to measure this gap and a scalable source of symbolic training data to address it. We make the code and data publicly available.
this https URL this https URL 

**Abstract (ZH)**: 高质量且逻辑严谨的数据稀缺是大型语言模型（LLMs）提升数学推理能力的关键瓶颈。我们通过将自动化定理证明研究的几十年经验转化为可扩展的数据引擎来应对这一挑战。我们的框架不依赖于容易出错的LLMs或复杂的证明辅助语法如Lean和Isabelle，而是利用E-prover的饱和能力在庞大的TPTP公理库上推导出大量已验证有效的定理。我们的流程是原理性的且简单明了：饱和公理、筛选“有趣”的定理、生成任务。由于在整个流程中不涉及LLMs，我们从根本上消除了事实错误。随后，这些纯符号数据被转化为三个难度可控的挑战：蕴含验证、前提选择和证明重构。我们在前沿模型上的零样本实验揭示了一个明显的弱点：在需要深度结构性推理的任务上表现下滑。我们的框架提供了衡量这一差距的诊察工具，并且提供了一种可扩展的符号训练数据来源来解决这一问题。我们已将代码和数据公开发布。 

---
# MachineLearningLM: Continued Pretraining Language Models on Millions of Synthetic Tabular Prediction Tasks Scales In-Context ML 

**Title (ZH)**: 机器学习LM：在数百万个合成表预测任务上持续预训练，并通过上下文学习scaling。 

**Authors**: Haoyu Dong, Pengkun Zhang, Mingzhe Lu, Yanzhen Shen, Guolin Ke  

**Link**: [PDF](https://arxiv.org/pdf/2509.06806)  

**Abstract**: Large language models (LLMs) possess broad world knowledge and strong general-purpose reasoning ability, yet they struggle to learn from many in-context examples on standard machine learning (ML) tasks, that is, to leverage many-shot demonstrations purely via in-context learning (ICL) without gradient descent. We introduce MachineLearningLM, a portable continued-pretraining framework that equips a general-purpose LLM with robust in-context ML capability while preserving its general knowledge and reasoning for broader chat workflows.
Our pretraining procedure synthesizes ML tasks from millions of structural causal models (SCMs), spanning shot counts up to 1,024. We begin with a random-forest teacher, distilling tree-based decision strategies into the LLM to strengthen robustness in numerical modeling. All tasks are serialized with a token-efficient prompt, enabling 3x to 6x more examples per context window and delivering up to 50x amortized throughput via batch inference.
Despite a modest setup (Qwen-2.5-7B-Instruct with LoRA rank 8), MachineLearningLM outperforms strong LLM baselines (e.g., GPT-5-mini) by an average of about 15% on out-of-distribution tabular classification across finance, physics, biology, and healthcare domains. It exhibits a striking many-shot scaling law: accuracy increases monotonically as in-context demonstrations grow from 8 to 1,024. Without any task-specific training, it attains random-forest-level accuracy across hundreds of shots. General chat capabilities, including knowledge and reasoning, are preserved: it achieves 75.4% on MMLU. 

**Abstract (ZH)**: 大规模语言模型（LLMs）具备广泛的世界知识和强大的通用推理能力，但在标准机器学习（ML）任务中，它们难以从多个上下文示例中学习，即在没有梯度下降的情况下仅通过上下文学习（ICL）利用大量的示例进行学习。我们引入了MachineLearningLM，这是一种便携的持续预训练框架，能够增强通用语言模型在上下文中的机器学习能力，同时保留其广泛的知识和推理能力，适用于更广泛的对话流程。

我们的预训练过程从数百万个结构性因果模型（SCMs）中合成机器学习任务，覆盖示例数量从1到1,024。我们从随机森林教师开始，将基于树的决策策略提炼到LLM中，以增强数值建模的鲁棒性。所有任务都使用基于标记的高效提示进行序列化，使每个上下文窗口的示例数量增加3到6倍，并通过批量推理实现高达50倍的传输效率。

尽管硬件配置有限（Qwen-2.5-7B-Instruct带LoRA秩8），MachineLearningLM在金融、物理、生物学和医疗保健领域中分布外的表格分类任务中平均优于强大语言模型基线（如GPT-5-mini）约15%。它表现出明显的多示例扩展规律：准确率随着上下文演示数量从8增加到1,024而单调增加。在没有任何特定任务训练的情况下，它在数百个示例中达到了随机森林级别的准确率。通用对话能力，包括知识和推理，得以保留：在MMLU上达到75.4%。 

---
# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks 

**Title (ZH)**: 深度神经网络中潜在代理子结构的概率建模 

**Authors**: Su Hyeong Lee, Risi Kondor, Richard Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06701)  

**Abstract**: We develop a theory of intelligent agency grounded in probabilistic modeling for neural models. Agents are represented as outcome distributions with epistemic utility given by log score, and compositions are defined through weighted logarithmic pooling that strictly improves every member's welfare. We prove that strict unanimity is impossible under linear pooling or in binary outcome spaces, but possible with three or more outcomes. Our framework admits recursive structure via cloning invariance, continuity, and openness, while tilt-based analysis rules out trivial duplication. Finally, we formalize an agentic alignment phenomenon in LLMs using our theory: eliciting a benevolent persona ("Luigi'") induces an antagonistic counterpart ("Waluigi"), while a manifest-then-suppress Waluigi strategy yields strictly larger first-order misalignment reduction than pure Luigi reinforcement alone. These results clarify how developing a principled mathematical framework for how subagents can coalesce into coherent higher-level entities provides novel implications for alignment in agentic AI systems. 

**Abstract (ZH)**: 我们基于概率建模发展了一种智能代理理论。代理被表示为具有逻辑评分的信念分布，并通过加权对数池化来定义组合，该池化方式严格改善了每个成员的福祉。我们证明，在线性池化或二元结果空间中不可能实现严格的一致性，但在三个或更多结果中是可能的。我们的框架通过克隆不变性、连续性和开放性允许递归结构，同时基于倾斜的分析排除了平凡的复制。最后，我们使用我们的理论正式化了一个在大规模语言模型（LLM）中出现的代理对齐现象：唤起一个善良的人格（“Luigi”）会引发一个对立的人格（“Waluigi”），而先显现后抑制的Waluigi策略会产生比单纯强化Luigi更为严格的一阶对齐改进。这些结果阐明了如何发展一个关于亚代理如何凝聚成一致的更高层次实体的原理数学框架为代理型AI系统中的对齐提供了新的意义。 

---
# Demo: Healthcare Agent Orchestrator (HAO) for Patient Summarization in Molecular Tumor Boards 

**Title (ZH)**: Demo: 医疗代理 orchestrator (HAO) 用于分子肿瘤板中的患者总结 

**Authors**: Noel Codella, Sam Preston, Hao Qiu, Leonardo Schettini, Wen-wai Yim, Mert Öz, Shrey Jain, Matthew P. Lungren, Thomas Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2509.06602)  

**Abstract**: Molecular Tumor Boards (MTBs) are multidisciplinary forums where oncology specialists collaboratively assess complex patient cases to determine optimal treatment strategies. A central element of this process is the patient summary, typically compiled by a medical oncologist, radiation oncologist, or surgeon, or their trained medical assistant, who distills heterogeneous medical records into a concise narrative to facilitate discussion. This manual approach is often labor-intensive, subjective, and prone to omissions of critical information. To address these limitations, we introduce the Healthcare Agent Orchestrator (HAO), a Large Language Model (LLM)-driven AI agent that coordinates a multi-agent clinical workflow to generate accurate and comprehensive patient summaries for MTBs. Evaluating predicted patient summaries against ground truth presents additional challenges due to stylistic variation, ordering, synonym usage, and phrasing differences, which complicate the measurement of both succinctness and completeness. To overcome these evaluation hurdles, we propose TBFact, a ``model-as-a-judge'' framework designed to assess the comprehensiveness and succinctness of generated summaries. Using a benchmark dataset derived from de-identified tumor board discussions, we applied TBFact to evaluate our Patient History agent. Results show that the agent captured 94% of high-importance information (including partial entailments) and achieved a TBFact recall of 0.84 under strict entailment criteria. We further demonstrate that TBFact enables a data-free evaluation framework that institutions can deploy locally without sharing sensitive clinical data. Together, HAO and TBFact establish a robust foundation for delivering reliable and scalable support to MTBs. 

**Abstract (ZH)**: 分子肿瘤板（MTBs）是多学科论坛，肿瘤专家在此协作评估复杂病例以确定最佳治疗策略。这一过程中的一项核心要素是患者总结，通常由肿瘤内科医生、放射肿瘤学家或外科医生及其受训医疗助理编制，将异质性医疗记录提炼成精炼的叙事以促进讨论。这种手动方法往往耗时、主观且容易遗漏关键信息。为解决这些局限性，我们引入了健康医疗代理协调器（HAO），这是一种由大规模语言模型（LLM）驱动的AI代理，用于协调多代理临床工作流生成准确且全面的患者总结以供分子肿瘤板使用。由于风格差异、排序、同义词使用和措辞不同，对预测的患者总结进行评估还带来了额外的挑战，这使得简洁性和完整性测量更为复杂。为克服这些评估难题，我们提出了TBFact框架，这是一种“模型即法官”的框架，用于评估生成总结的全面性和简明性。使用从去标识化肿瘤板讨论中提取的基准数据集，我们运用TBFact评估了我们的患者病史代理。结果显示，代理捕获了94%的关键信息（包括部分蕴含），并在严格蕴含标准下实现了TBFact召回率为0.84。此外，我们展示了TBFact能够提供一种无需共享敏感临床数据即可在本地部署的数据驱动评估框架。总之，结合HAO和TBFact为MTBs提供了可靠且可扩展的支持奠定了坚实基础。 

---
# HAVE: Head-Adaptive Gating and ValuE Calibration for Hallucination Mitigation in Large Language Models 

**Title (ZH)**: HEAD-适应性门控和值校准以减轻大规模语言模型中的幻觉现象 

**Authors**: Xin Tong, Zhi Lin, Jingya Wang, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06596)  

**Abstract**: Large Language Models (LLMs) often produce hallucinations in retrieval-augmented or long-context generation, even when relevant evidence is present. This stems from two issues: head importance is treated as input-agnostic, and raw attention weights poorly reflect each token's true contribution. We present HAVE (Head-Adaptive Gating and ValuE Calibration), a parameter-free decoding framework that directly addresses both challenges. HAVE introduces head-adaptive gating, which performs instance-level soft reweighing of attention heads, and value calibration, which augments attention with the magnitude of value vectors to approximate write-back contribution. Together, these modules construct token-level evidence aligned with model updates and fuse it with the LM distribution through a lightweight uncertainty-scaled policy. HAVE requires no finetuning and operates in a single forward pass, making it efficient and broadly applicable. Experiments across multiple QA benchmarks and LLM families demonstrate that HAVE consistently reduces hallucinations and outperforms strong baselines, including DAGCD, with modest overhead. The framework is transparent, reproducible, and readily integrates with off-the-shelf LLMs, advancing trustworthy generation in real-world settings. 

**Abstract (ZH)**: Large Language Models中的 retrieval-augmented或长上下文生成常常会在相关证据存在的情况下产生幻觉，这源于头部重要性被视为输入无关以及原始注意力权重无法准确反映每个词的真实贡献这两个问题。我们提出了HAVEN（头部自适应门控和价值校准）——一个无需调整参数的解码框架，直接解决了这两个挑战。HAVEN引入了头部自适应门控，进行实例级别的软重权头部注意力，并结合了值向量的大小来近似写入贡献的价值校准。这些模块构造了与模型更新相一致的词级证据，并通过轻量级的不确定性缩放策略将其与语言模型分布融合。HAVEN无需微调，且仅需一次前向传递，使其高效且广泛适用。跨多个问答基准和语言模型系列的实验表明，HAVEN能够一致地减少幻觉并优于包括DAGCD在内的强基线，同时具有适度的开销。该框架透明、可复现且能够无缝集成到现成的语言模型中，推动了可信生成在实际场景中的应用。 

---
# SLiNT: Structure-aware Language Model with Injection and Contrastive Training for Knowledge Graph Completion 

**Title (ZH)**: SLiNT：结构感知的语言模型及其注入与对比训练在知识图谱完成中的应用 

**Authors**: Mengxue Yang, Chun Yang, Jiaqi Zhu, Jiafan Li, Jingqi Zhang, Yuyang Li, Ying Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06531)  

**Abstract**: Link prediction in knowledge graphs requires integrating structural information and semantic context to infer missing entities. While large language models offer strong generative reasoning capabilities, their limited exploitation of structural signals often results in structural sparsity and semantic ambiguity, especially under incomplete or zero-shot settings. To address these challenges, we propose SLiNT (Structure-aware Language model with Injection and coNtrastive Training), a modular framework that injects knowledge-graph-derived structural context into a frozen LLM backbone with lightweight LoRA-based adaptation for robust link prediction. Specifically, Structure-Guided Neighborhood Enhancement (SGNE) retrieves pseudo-neighbors to enrich sparse entities and mitigate missing context; Dynamic Hard Contrastive Learning (DHCL) introduces fine-grained supervision by interpolating hard positives and negatives to resolve entity-level ambiguity; and Gradient-Decoupled Dual Injection (GDDI) performs token-level structure-aware intervention while preserving the core LLM parameters. Experiments on WN18RR and FB15k-237 show that SLiNT achieves superior or competitive performance compared with both embedding-based and generation-based baselines, demonstrating the effectiveness of structure-aware representation learning for scalable knowledge graph completion. 

**Abstract (ZH)**: 知识图谱中的链接预测需要整合结构信息和语义上下文以推断缺失实体。尽管大型语言模型提供了强大的生成推理能力，但它们对结构信号的有限利用经常导致结构稀疏性和语义模糊性，尤其是在不完整或零样本设置中。为应对这些挑战，我们提出了一种基于模块化框架的SLiNT（结构感知的语言模型结合注入和对比训练），该框架将知识图谱衍生的结构上下文注入到冻结的LLM主干中，并采用轻量级的LoRA基适应方法，以实现稳健的链接预测。具体而言，结构引导的邻域增强（SGNE）检索伪邻居以丰富稀疏实体并缓解缺失上下文；动态硬对比学习（DHCL）通过插值硬正样本和负样本引入细粒度监督以解决实体级模糊性；梯度解耦双注入（GDDI）执行基于标记的结构感知干预，同时保留核心LLM参数。在WN18RR和FB15k-237上的实验表明，SLiNT 在与基于嵌入和生成的基线模型相比时，实现了更优或竞争力的表现，证明了结构感知表示学习在可扩展知识图谱补全中的有效性。 

---
# Crown, Frame, Reverse: Layer-Wise Scaling Variants for LLM Pre-Training 

**Title (ZH)**: 冠层、框架、反转：大型语言模型预训练的层级規模化变体 

**Authors**: Andrei Baroian, Kasper Notebomer  

**Link**: [PDF](https://arxiv.org/pdf/2509.06518)  

**Abstract**: Transformer-based language models traditionally use uniform (isotropic) layer sizes, yet they ignore the diverse functional roles that different depths can play and their computational capacity needs. Building on Layer-Wise Scaling (LWS) and pruning literature, we introduce three new LWS variants - Framed, Reverse, and Crown - that redistribute FFN widths and attention heads via two or three-point linear interpolation in the pre-training stage. We present the first systematic ablation of LWS and its variants, on a fixed budget of 180M parameters, trained on 5B tokens. All models converge to similar losses and achieve better performance compared to an equal-cost isotropic baseline, without a substantial decrease in training throughput. This work represents an initial step into the design space of layer-wise architectures for pre-training, but future work should scale experiments to orders of magnitude more tokens and parameters to fully assess their potential. 

**Abstract (ZH)**: 基于Transformer的语言模型通常采用均匀的层尺寸，却忽视了不同深度在功能角色和计算能力需求方面表现出的多样性。在Layer-Wise Scaling (LWS) 和剪枝文献的基础上，我们提出了三种新的LWS变体——Framed、Reverse和Crown，在预训练阶段通过两或三点线性插值重新分配FFN宽度和注意力头。我们首次在固定参数预算（180M参数）和5Btoken的数据训练下，系统性地研究了LWS及其变体。所有模型收敛于相似的损失值，并在成本相同时比均匀基线获得了更好的性能，而训练吞吐量并未显著下降。这项工作代表了在预训练层次架构设计空间上迈出的初步一步，但未来的工作应该扩展实验规模，以大量更多的token和参数进行全面评估。 

---
# Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks? 

**Title (ZH)**: Mask-GCG: 所有的对抗后缀中的令牌对于 jailbreak 攻击都是必要的吗？ 

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06350)  

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的逃狱攻击：Greedy Coordinate Gradient（Mask-GCG）掩码方法探究 

---
# Ban&Pick: Achieving Free Performance Gains and Inference Speedup via Smarter Routing in MoE-LLMs 

**Title (ZH)**: Ban&Pick：通过更智能的路由实现MoE-LLMs的自由性能提升和推理加速 

**Authors**: Yuanteng Chen, Peisong Wang, Yuantian Shao, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06346)  

**Abstract**: Sparse Mixture-of-Experts (MoE) has become a key architecture for scaling large language models (LLMs) efficiently. Recent fine-grained MoE designs introduce hundreds of experts per layer, with multiple experts activated per token, enabling stronger specialization. However, during pre-training, routers are optimized mainly for stability and robustness: they converge prematurely and enforce balanced usage, limiting the full potential of model performance and efficiency. In this work, we uncover two overlooked issues: (i) a few highly influential experts are underutilized due to premature and balanced routing decisions; and (ii) enforcing a fixed number of active experts per token introduces substantial redundancy. Instead of retraining models or redesigning MoE architectures, we introduce Ban&Pick, a post-training, plug-and-play strategy for smarter MoE routing. Pick discovers and reinforces key experts-a small group with outsized impact on performance-leading to notable accuracy gains across domains. Ban complements this by dynamically pruning redundant experts based on layer and token sensitivity, delivering faster inference with minimal accuracy loss. Experiments on fine-grained MoE-LLMs (DeepSeek, Qwen3) across math, code, and general reasoning benchmarks demonstrate that Ban&Pick delivers free performance gains and inference acceleration without retraining or architectural changes. For instance, on Qwen3-30B-A3B, it improves accuracy from 80.67 to 84.66 on AIME2024 and from 65.66 to 68.18 on GPQA-Diamond, while accelerating inference by 1.25x under the vLLM. 

**Abstract (ZH)**: Sparse Mixture-of-Experts Routing with Ban&Pick for Efficient Large Language Models 

---
# A Fragile Number Sense: Probing the Elemental Limits of Numerical Reasoning in LLMs 

**Title (ZH)**: 易碎的数感：探究大模型在数值推理中的基本极限 

**Authors**: Roussel Rahman, Aashwin Ananda Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.06332)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable emergent capabilities, yet the robustness of their numerical reasoning remains an open question. While standard benchmarks evaluate LLM reasoning on complex problem sets using aggregated metrics, they often obscure foundational weaknesses. In this work, we probe LLM mathematical numeracy by evaluating performance on problems of escalating complexity, from constituent operations to combinatorial puzzles. We test several state-of-the-art LLM-based agents on a 100-problem challenge comprising four categories: (1) basic arithmetic, (2) advanced operations, (3) primality checking, and (4) the Game of 24 number puzzle. Our results show that while the agents achieved high accuracy on the first three categories, which require deterministic algorithmic execution, they consistently failed at the number puzzle, underlining its demand for a heuristic search over a large combinatorial space to be a significant bottleneck. These findings reveal that the agents' proficiency is largely confined to recalling and executing known algorithms, rather than performing generative problem-solving. This suggests their apparent numerical reasoning is more akin to sophisticated pattern-matching than flexible, analytical thought, limiting their potential for tasks that require novel or creative numerical insights. 

**Abstract (ZH)**: 大型语言模型(Large Language Models, LLMs)展示了 remarkable 的 emergent 能力，但其数值推理的 robustness 仍是一个开放问题。虽然标准基准通过聚合指标评估LLM在复杂问题集上的推理能力，但往往掩盖了其基础性的弱点。在这项工作中，我们通过评估其在从基本操作到组合难题等一系列复杂问题上的表现，来探查LLM的数学数理能力。我们测试了几种最先进的基于LLM的代理在一个包含四大类别的100题挑战中的性能：(1) 基本算术，(2) 高级操作，(3) 质数检验，以及(4) 24点数字谜题。结果显示，代理在前三大类问题上实现了高准确率，这些类别需要确定性的算法执行，但在数字谜题上始终失败，突显了解决大规模组合空间所需的启发式搜索是一个重要瓶颈。这些发现表明，代理的技能主要限于回忆和执行已知算法，而非生成性问题解决。这表明它们看似具备的数值推理能力更像是复杂的模式匹配，而不是灵活的、分析性的思考，从而限制了它们在需要新颖或创造性数值洞察的任务上的潜力。 

---
# AttestLLM: Efficient Attestation Framework for Billion-scale On-device LLMs 

**Title (ZH)**: AttestLLM：高效的万亿规模边缘设备大语言模型认证框架 

**Authors**: Ruisi Zhang, Yifei Zhao, Neusha Javidnia, Mengxin Zheng, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.06326)  

**Abstract**: As on-device LLMs(e.g., Apple on-device Intelligence) are widely adopted to reduce network dependency, improve privacy, and enhance responsiveness, verifying the legitimacy of models running on local devices becomes critical. Existing attestation techniques are not suitable for billion-parameter Large Language Models (LLMs), struggling to remain both time- and memory-efficient while addressing emerging threats in the LLM era. In this paper, we present AttestLLM, the first-of-its-kind attestation framework to protect the hardware-level intellectual property (IP) of device vendors by ensuring that only authorized LLMs can execute on target platforms. AttestLLM leverages an algorithm/software/hardware co-design approach to embed robust watermarking signatures onto the activation distributions of LLM building blocks. It also optimizes the attestation protocol within the Trusted Execution Environment (TEE), providing efficient verification without compromising inference throughput. Extensive proof-of-concept evaluations on LLMs from Llama, Qwen, and Phi families for on-device use cases demonstrate AttestLLM's attestation reliability, fidelity, and efficiency. Furthermore, AttestLLM enforces model legitimacy and exhibits resilience against model replacement and forgery attacks. 

**Abstract (ZH)**: 基于设备端的大型语言模型（如苹果设备端智能）日益普及，以减少网络依赖、提升隐私保护和增强响应速度，验证运行在本地设备上的模型合法性变得至关重要。现有的认证技术不适用于十亿级参数的大型语言模型（LLMs），难以在应对LLM时代新兴威胁的同时保持时间和内存效率。在本文中，我们提出了AttestLLM，这是一种首创的认证框架，旨在通过确保只有授权的LLM能在目标平台上执行来保护设备供应商的硬件层面知识产权（IP）。AttestLLM采用算法/软件/硬件协同设计的方法，在LLM构建块的激活分布中嵌入 robust 水印签名。此外，AttestLLM 在可信执行环境（TEE）中优化了认证协议，提供了高效验证而不牺牲推理吞吐量。针对Llama、Qwen和Phi家族的LLM在设备端使用场景的广泛概念验证评估证明了AttestLLM的认证可靠性和效率。此外，AttestLLM 强制执行模型合法性，并展示了对模型替换和伪造攻击的抗性。 

---
# Beamforming-LLM: What, Where and When Did I Miss? 

**Title (ZH)**: 波束形成-LLM：我在哪里、何时以及为什么错过了？ 

**Authors**: Vishal Choudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.06221)  

**Abstract**: We present Beamforming-LLM, a system that enables users to semantically recall conversations they may have missed in multi-speaker environments. The system combines spatial audio capture using a microphone array with retrieval-augmented generation (RAG) to support natural language queries such as, "What did I miss when I was following the conversation on dogs?" Directional audio streams are separated using beamforming, transcribed with Whisper, and embedded into a vector database using sentence encoders. Upon receiving a user query, semantically relevant segments are retrieved, temporally aligned with non-attended segments, and summarized using a lightweight large language model (GPT-4o-mini). The result is a user-friendly interface that provides contrastive summaries, spatial context, and timestamped audio playback. This work lays the foundation for intelligent auditory memory systems and has broad applications in assistive technology, meeting summarization, and context-aware personal spatial computing. 

**Abstract (ZH)**: Beamforming-LLM：一种在多说话人环境中使用户能够语义回溯错过对话的系统 

---
# Benchmarking Gender and Political Bias in Large Language Models 

**Title (ZH)**: 大型语言模型中的性别和政治偏见基准研究 

**Authors**: Jinrui Yang, Xudong Han, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2509.06164)  

**Abstract**: We introduce EuroParlVote, a novel benchmark for evaluating large language models (LLMs) in politically sensitive contexts. It links European Parliament debate speeches to roll-call vote outcomes and includes rich demographic metadata for each Member of the European Parliament (MEP), such as gender, age, country, and political group. Using EuroParlVote, we evaluate state-of-the-art LLMs on two tasks -- gender classification and vote prediction -- revealing consistent patterns of bias. We find that LLMs frequently misclassify female MEPs as male and demonstrate reduced accuracy when simulating votes for female speakers. Politically, LLMs tend to favor centrist groups while underperforming on both far-left and far-right ones. Proprietary models like GPT-4o outperform open-weight alternatives in terms of both robustness and fairness. We release the EuroParlVote dataset, code, and demo to support future research on fairness and accountability in NLP within political contexts. 

**Abstract (ZH)**: EuroParlVote：一种评估大规模语言模型在政治敏感情境下的基准 

---
# Language Native Lightly Structured Databases for Large Language Model Driven Composite Materials Research 

**Title (ZH)**: 语言本土轻量结构数据库在大型语言模型驱动的复合材料研究中的应用 

**Authors**: Yuze Liu, Zhaoyuan Zhang, Xiangsheng Zeng, Yihe Zhang, Leping Yu, Lejia Wang, Xi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.06093)  

**Abstract**: Chemical and materials research has traditionally relied heavily on knowledge narrative, with progress often driven by language-based descriptions of principles, mechanisms, and experimental experiences, rather than tables, limiting what conventional databases and ML can exploit. We present a language-native database for boron nitride nanosheet (BNNS) polymer thermally conductive composites that captures lightly structured information from papers across preparation, characterization, theory-computation, and mechanistic reasoning, with evidence-linked snippets. Records are organized in a heterogeneous database and queried via composite retrieval with semantics, key words and value filters. The system can synthesizes literature into accurate, verifiable, and expert style guidance. This substrate enables high fidelity efficient Retrieval Augmented Generation (RAG) and tool augmented agents to interleave retrieval with reasoning and deliver actionable SOP. The framework supplies the language rich foundation required for LLM-driven materials discovery. 

**Abstract (ZH)**: 一种语言本位的氮化硼纳米片聚合物热传导复合材料数据库，及其在高效检索增强生成和工具增强代理中的应用，为基于LLM的材料发现提供语言丰富的基础。 

---
# PolicyEvolve: Evolving Programmatic Policies by LLMs for multi-player games via Population-Based Training 

**Title (ZH)**: PolicyEvolve：通过基于群体的训练使大规模语言模型为多玩家游戏演化程序化策略 

**Authors**: Mingrui Lv, Hangzhi Liu, Zhi Luo, Hongjie Zhang, Jie Ou  

**Link**: [PDF](https://arxiv.org/pdf/2509.06053)  

**Abstract**: Multi-agent reinforcement learning (MARL) has achieved significant progress in solving complex multi-player games through self-play. However, training effective adversarial policies requires millions of experience samples and substantial computational resources. Moreover, these policies lack interpretability, hindering their practical deployment. Recently, researchers have successfully leveraged Large Language Models (LLMs) to generate programmatic policies for single-agent tasks, transforming neural network-based policies into interpretable rule-based code with high execution efficiency. Inspired by this, we propose PolicyEvolve, a general framework for generating programmatic policies in multi-player games. PolicyEvolve significantly reduces reliance on manually crafted policy code, achieving high-performance policies with minimal environmental interactions. The framework comprises four modules: Global Pool, Local Pool, Policy Planner, and Trajectory Critic. The Global Pool preserves elite policies accumulated during iterative training. The Local Pool stores temporary policies for the current iteration; only sufficiently high-performing policies from this pool are promoted to the Global Pool. The Policy Planner serves as the core policy generation module. It samples the top three policies from the Global Pool, generates an initial policy for the current iteration based on environmental information, and refines this policy using feedback from the Trajectory Critic. Refined policies are then deposited into the Local Pool. This iterative process continues until the policy achieves a sufficiently high average win rate against the Global Pool, at which point it is integrated into the Global Pool. The Trajectory Critic analyzes interaction data from the current policy, identifies vulnerabilities, and proposes directional improvements to guide the Policy Planner 

**Abstract (ZH)**: 基于多智能体强化学习的程序化策略生成框架：PolicyEvolve 

---
# Empirical Study of Code Large Language Models for Binary Security Patch Detection 

**Title (ZH)**: 代码大型语言模型在二元安全补丁检测中的实证研究 

**Authors**: Qingyuan Li, Binchang Li, Cuiyun Gao, Shuzheng Gao, Zongjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.06052)  

**Abstract**: Security patch detection (SPD) is crucial for maintaining software security, as unpatched vulnerabilities can lead to severe security risks. In recent years, numerous learning-based SPD approaches have demonstrated promising results on source code. However, these approaches typically cannot be applied to closed-source applications and proprietary systems that constitute a significant portion of real-world software, as they release patches only with binary files, and the source code is inaccessible. Given the impressive performance of code large language models (LLMs) in code intelligence and binary analysis tasks such as decompilation and compilation optimization, their potential for detecting binary security patches remains unexplored, exposing a significant research gap between their demonstrated low-level code understanding capabilities and this critical security task. To address this gap, we construct a large-scale binary patch dataset containing \textbf{19,448} samples, with two levels of representation: assembly code and pseudo-code, and systematically evaluate \textbf{19} code LLMs of varying scales to investigate their capability in binary SPD tasks. Our initial exploration demonstrates that directly prompting vanilla code LLMs struggles to accurately identify security patches from binary patches, and even state-of-the-art prompting techniques fail to mitigate the lack of domain knowledge in binary SPD within vanilla models. Drawing on the initial findings, we further investigate the fine-tuning strategy for injecting binary SPD domain knowledge into code LLMs through two levels of representation. Experimental results demonstrate that fine-tuned LLMs achieve outstanding performance, with the best results obtained on the pseudo-code representation. 

**Abstract (ZH)**: 基于二进制的安全补丁检测：构建大规模二进制补丁数据集并评估代码大语言模型 

---
# Multimodal Prompt Injection Attacks: Risks and Defenses for Modern LLMs 

**Title (ZH)**: 多模态提示注入攻击：现代大语言模型的风险与防御 

**Authors**: Andrew Yeo, Daeseon Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.05883)  

**Abstract**: Large Language Models (LLMs) have seen rapid adoption in recent years, with industries increasingly relying on them to maintain a competitive advantage. These models excel at interpreting user instructions and generating human-like responses, leading to their integration across diverse domains, including consulting and information retrieval. However, their widespread deployment also introduces substantial security risks, most notably in the form of prompt injection and jailbreak attacks.
To systematically evaluate LLM vulnerabilities -- particularly to external prompt injection -- we conducted a series of experiments on eight commercial models. Each model was tested without supplementary sanitization, relying solely on its built-in safeguards. The results exposed exploitable weaknesses and emphasized the need for stronger security measures. Four categories of attacks were examined: direct injection, indirect (external) injection, image-based injection, and prompt leakage. Comparative analysis indicated that Claude 3 demonstrated relatively greater robustness; nevertheless, empirical findings confirm that additional defenses, such as input normalization, remain necessary to achieve reliable protection. 

**Abstract (ZH)**: 大型语言模型（LLMs）近年来实现了快速 adoption，各行各业 increasingly 依赖它们来保持竞争优势。这些模型在解读用户指令并生成类人类响应方面表现出色，促使其在包括咨询和信息检索在内的众多领域中得到广泛应用。然而，它们的广泛部署也带来了显著的安全风险，尤其是提示注入和 jailbreak 攻击等形式。

为了系统地评估 LLM 的漏洞——特别是对外部提示注入的脆弱性，我们对八款商用模型进行了系列实验。每个模型均未使用额外的清理措施，仅依赖其内置的安全防护。实验结果揭示了可利用的弱点，并强调了增强安全措施的必要性。四种攻击类别被考察：直接注入、间接（外部）注入、基于图像的注入和提示泄露。对比分析表明，Claude 3 展现了相对较高的鲁棒性；然而，实证发现证实，为进一步实现可靠的防护，仍需采取额外的防御措施，如输入规范化。 

---
# Let's Roleplay: Examining LLM Alignment in Collaborative Dialogues 

**Title (ZH)**: 让我们角色扮演：探究协作对话中大规模语言模型的一致性 

**Authors**: Abhijnan Nath, Carine Graff, Nikhil Krishnaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2509.05882)  

**Abstract**: As Large Language Models (LLMs) integrate into diverse workflows, they are increasingly being considered "collaborators" with humans. If such AI collaborators are to be reliable, their behavior over multiturn interactions must be predictable, validated and verified before deployment. Common alignment techniques are typically developed under simplified single-user settings and do not account for the dynamics of long-horizon multiparty interactions. This paper examines how different alignment methods affect LLM agents' effectiveness as partners in multiturn, multiparty collaborations. We study this question through the lens of friction agents that intervene in group dialogues to encourage the collaborative group to slow down and reflect upon their reasoning for deliberative decision-making. Using a roleplay methodology, we evaluate interventions from differently-trained friction agents in collaborative task conversations. We propose a novel counterfactual evaluation framework that quantifies how friction interventions change the trajectory of group collaboration and belief alignment. Our results show that a friction-aware approach significantly outperforms common alignment baselines in helping both convergence to a common ground, or agreed-upon task-relevant propositions, and correctness of task outcomes. 

**Abstract (ZH)**: 大规模语言模型（LLMs）集成到多样化的 workflows 中，它们 increasingly 被视为与人类合作的“合作者”。如果这些 AI 合作者要可靠，在部署前它们在多轮多主体交互中的行为必须是可预测、验证和验证的。常见的对齐技术通常是在简化的一对一用户设置中开发的，不考虑长时间多主体交互的动态性。本文探讨了不同对齐方法如何影响 LLM 代理作为多轮多主体合作中伙伴的有效性。我们通过摩擦代理干预团体内对话以鼓励协作团体放慢节奏并反思其推理来进行这一问题的研究。使用角色扮演方法，我们评估了不同类型训练的摩擦代理在协作任务对话中的干预效果。我们提出了一种新颖的反事实评估框架，用于量化摩擦干预如何改变团队协作和信念对齐的轨迹。研究表明，具有摩擦意识的方法在帮助达成共识或达成一致的任务相关命题以及任务结果的准确性方面显著优于常见的对齐基线。 

---
# GeoAnalystBench: A GeoAI benchmark for assessing large language models for spatial analysis workflow and code generation 

**Title (ZH)**: GeoAnalystBench: 用于评估大型语言模型在空间分析工作流和代码生成中的性能的GeoAI基准测试 

**Authors**: Qianheng Zhang, Song Gao, Chen Wei, Yibo Zhao, Ying Nie, Ziru Chen, Shijie Chen, Yu Su, Huan Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.05881)  

**Abstract**: Recent advances in large language models (LLMs) have fueled growing interest in automating geospatial analysis and GIS workflows, yet their actual capabilities remain uncertain. In this work, we call for rigorous evaluation of LLMs on well-defined geoprocessing tasks before making claims about full GIS automation. To this end, we present GeoAnalystBench, a benchmark of 50 Python-based tasks derived from real-world geospatial problems and carefully validated by GIS experts. Each task is paired with a minimum deliverable product, and evaluation covers workflow validity, structural alignment, semantic similarity, and code quality (CodeBLEU). Using this benchmark, we assess both proprietary and open source models. Results reveal a clear gap: proprietary models such as ChatGPT-4o-mini achieve high validity 95% and stronger code alignment (CodeBLEU 0.39), while smaller open source models like DeepSeek-R1-7B often generate incomplete or inconsistent workflows (48.5% validity, 0.272 CodeBLEU). Tasks requiring deeper spatial reasoning, such as spatial relationship detection or optimal site selection, remain the most challenging across all models. These findings demonstrate both the promise and limitations of current LLMs in GIS automation and provide a reproducible framework to advance GeoAI research with human-in-the-loop support. 

**Abstract (ZH)**: Recent advances in大规模语言模型（LLMs）促进了地理空间分析和GIS工作流自动化的兴趣增长，但其实际能力仍存不确定性。本研究呼吁在宣称实现全面GIS自动化之前，对LLMs进行严格的地理处理任务评估。为此，我们提出了GeoAnalystBench基准，包括50个基于Python的实际地理空间问题任务，并由GIS专家仔细验证。每个任务都配有一个最低交付产品，评估涵盖工作流有效性、结构对齐、语义相似性和代码质量（CodeBLEU）。利用此基准，我们评估了商业和开源模型。结果显示，商业模型如ChatGPT-4o-mini在工作流有效性（95%）和更强的代码对齐（CodeBLEU 0.39）方面表现出色，而较小的开源模型如DeepSeek-R1-7B常常生成不完整或不一致的工作流（48.5%有效性，0.272 CodeBLEU）。需要更深层次空间推理的任务，如空间关系检测或最优选址，是所有模型中最具挑战性的。这些发现展示了当前LLMs在GIS自动化中的潜力和局限性，并提供了一个可重复的框架，支持通过人类在环的方式推进GeoAI研究。 

---
# Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization 

**Title (ZH)**: 解码LLMs中的潜在攻击面：通过HTML进行提示注入以进行网页总结 

**Authors**: Ishaan Verma  

**Link**: [PDF](https://arxiv.org/pdf/2509.05831)  

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地被集成到基于Web的系统中进行内容摘要，但它们对提示注入攻击的易感性仍然是一个紧迫的问题。在这项研究中，我们探讨了如何利用非可视HTML元素（如<meta>、aria-label和alt属性）在不改变网页可见内容的情况下嵌入 adversarial 指令。我们引入了一个由280个静态网页组成的新数据集，这些网页均匀分为干净版本和恶意注入版本，并使用多种HTML基策略进行构建。这些页面通过浏览器自动化管道处理，以提取原始HTML和渲染文本，模拟真实的LLM部署场景。我们评估了两个最先进的开源模型——Llama 4 Scout（Meta）和Gemma 9B IT（Google）——在摘要内容方面的能力。使用词汇（ROUGE-L）和语义（SBERT余弦相似度）指标，以及手动注释，我们评估了这些隐蔽注入的影响。我们的发现表明，超过29%的注入样本导致了Llama 4 Scout摘要的明显变化，而Gemma 9B IT的成功率较低，但仍然非 trivial，为15%。这些结果强调了LLM驱动的Web管道中一个关键且被忽视的安全漏洞，在这些管道中，隐藏的 adversarial 内容可以微妙地操控模型输出。我们的工作提供了一个可复制的框架和基准，用于评估基于HTML的提示注入，并强调了在涉及Web内容的LLM应用中迫切需要强大的缓解策略。 

---
# Exploit Tool Invocation Prompt for Tool Behavior Hijacking in LLM-Based Agentic System 

**Title (ZH)**: 利用工具调用提示劫持基于LLM的代理系统中的工具行为 

**Authors**: Yu Liu, Yuchong Xie, Mingyu Luo, Zesen Liu, Zhixiang Zhang, Kaikai Zhang, Zongjie Li, Ping Chen, Shuai Wang, Dongdong She  

**Link**: [PDF](https://arxiv.org/pdf/2509.05755)  

**Abstract**: LLM-based agentic systems leverage large language models to handle user queries, make decisions, and execute external tools for complex tasks across domains like chatbots, customer service, and software engineering. A critical component of these systems is the Tool Invocation Prompt (TIP), which defines tool interaction protocols and guides LLMs to ensure the security and correctness of tool usage. Despite its importance, TIP security has been largely overlooked. This work investigates TIP-related security risks, revealing that major LLM-based systems like Cursor, Claude Code, and others are vulnerable to attacks such as remote code execution (RCE) and denial of service (DoS). Through a systematic TIP exploitation workflow (TEW), we demonstrate external tool behavior hijacking via manipulated tool invocations. We also propose defense mechanisms to enhance TIP security in LLM-based agentic systems. 

**Abstract (ZH)**: 基于LLM的代理系统通过大规模语言模型处理用户查询、做出决策并执行外部工具以跨领域（如聊天机器人、客户服务和软件工程）完成复杂任务。这些系统的关键组成部分是工具调用提示（TIP），它定义了工具交互协议并指导LLM以确保工具使用的安全性和正确性。尽管其重要性不言而喻，但TIP安全问题却常常被忽视。本研究探讨了与TIP相关的安全风险，揭示了诸如Cursor、Claude Code等主要基于LLM的系统容易受到远程代码执行（RCE）和拒绝服务（DoS）等攻击。通过系统性的工具调用提示exploitation工作流（TEW），我们展示了通过操纵工具调用来劫持外部工具行为。此外，我们还提出了增强基于LLM的代理系统中TIP安全性的防御机制。 

---
# Unleashing Hierarchical Reasoning: An LLM-Driven Framework for Training-Free Referring Video Object Segmentation 

**Title (ZH)**: 解锁层次推理：一种基于大型语言模型的无需训练的视频对象分割框架 

**Authors**: Bingrui Zhao, Lin Yuanbo Wu, Xiangtian Fan, Deyin Liu, Lu Zhang, Ruyi He, Jialie Shen, Ximing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05751)  

**Abstract**: Referring Video Object Segmentation (RVOS) aims to segment an object of interest throughout a video based on a language description. The prominent challenge lies in aligning static text with dynamic visual content, particularly when objects exhibiting similar appearances with inconsistent motion and poses. However, current methods often rely on a holistic visual-language fusion that struggles with complex, compositional descriptions. In this paper, we propose \textbf{PARSE-VOS}, a novel, training-free framework powered by Large Language Models (LLMs), for a hierarchical, coarse-to-fine reasoning across text and video domains. Our approach begins by parsing the natural language query into structured semantic commands. Next, we introduce a spatio-temporal grounding module that generates all candidate trajectories for all potential target objects, guided by the parsed semantics. Finally, a hierarchical identification module select the correct target through a two-stage reasoning process: it first performs coarse-grained motion reasoning with an LLM to narrow down candidates; if ambiguity remains, a fine-grained pose verification stage is conditionally triggered to disambiguate. The final output is an accurate segmentation mask for the target object. \textbf{PARSE-VOS} achieved state-of-the-art performance on three major benchmarks: Ref-YouTube-VOS, Ref-DAVIS17, and MeViS. 

**Abstract (ZH)**: 基于语言描述的视频对象分割（RVOS）旨在根据语言描述对视频中的目标对象进行分割。主要挑战在于将静态文本与动态视觉内容对齐，特别是对于具有不一致运动和姿态的相似外观对象。然而，当前方法常常依赖整体的视觉-语言融合，这在处理复杂的组合性描述时存在困难。本文提出了一种新的、无需训练的框架PARSE-VOS，该框架由大型语言模型（LLMs）驱动，用于跨文本和视频领域进行层次化的粗细粒度推理。该方法首先将自然语言查询解析为结构化的语义命令。接着引入时空 grounding 模块生成所有潜在目标对象的所有候选轨迹，受解析语义的引导。最后，通过两阶段推理过程实现层次化的识别模块：首先使用LLM进行粗粒度运动推理以缩小候选人；如果仍有歧义，则有条件地触发细粒度姿态验证阶段进行消歧。最终输出是目标对象的准确分割掩模。PARSE-VOS在三个主要基准（Ref-YouTube-VOS、Ref-DAVIS17和MeViS）上取得了最先进的性能。 

---
# Reasoning Introduces New Poisoning Attacks Yet Makes Them More Complicated 

**Title (ZH)**: 推理引入了新的中毒攻击但使它们更加复杂 

**Authors**: Hanna Foerster, Ilia Shumailov, Yiren Zhao, Harsh Chaudhari, Jamie Hayes, Robert Mullins, Yarin Gal  

**Link**: [PDF](https://arxiv.org/pdf/2509.05739)  

**Abstract**: Early research into data poisoning attacks against Large Language Models (LLMs) demonstrated the ease with which backdoors could be injected. More recent LLMs add step-by-step reasoning, expanding the attack surface to include the intermediate chain-of-thought (CoT) and its inherent trait of decomposing problems into subproblems. Using these vectors for more stealthy poisoning, we introduce ``decomposed reasoning poison'', in which the attacker modifies only the reasoning path, leaving prompts and final answers clean, and splits the trigger across multiple, individually harmless components.
Fascinatingly, while it remains possible to inject these decomposed poisons, reliably activating them to change final answers (rather than just the CoT) is surprisingly difficult. This difficulty arises because the models can often recover from backdoors that are activated within their thought processes. Ultimately, it appears that an emergent form of backdoor robustness is originating from the reasoning capabilities of these advanced LLMs, as well as from the architectural separation between reasoning and final answer generation. 

**Abstract (ZH)**: 早期对大型语言模型（LLMs）的数据投毒攻击研究显示了后门注入的简便性。更近期的LLMs增加了逐步推理功能，扩展了攻击表面，不仅包括中间推理链（CoT），还包括其将问题分解为子问题的内在特性。利用这些向量进行更隐蔽的投毒，我们引入了“分解推理投毒”概念，在此过程中，攻击者仅修改推理路径，而保留提示和最终答案的清洁性，并将触发器分散在多个单独无害的组件中。

令人惊讶的是，虽然可以注入这些分解投毒，但可靠地激活它们以改变最终答案（而不仅仅是CoT）是出奇地困难。这种困难源于模型在其思维过程中激活后门时能够进行恢复。最终，似乎从这些先进LLMs的推理能力和推理与最终答案生成之间的架构分离中涌现出一种新的后门稳健性形式。 

---
# Llama-GENBA-10B: A Trilingual Large Language Model for German, English and Bavarian 

**Title (ZH)**: Llama-GENBA-10B：一种用于德语、英语和巴伐利亚语的三语大型语言模型 

**Authors**: Michael Hoffmann, Jophin John, Stefan Schweter, Gokul Ramakrishnan, Hoi-Fong Mak, Alice Zhang, Dmitry Gaynullin, Nicolay J. Hammer  

**Link**: [PDF](https://arxiv.org/pdf/2509.05668)  

**Abstract**: We present Llama-GENBA-10B, a trilingual foundation model addressing English-centric bias in large language models. Built on Llama 3.1-8B and scaled to 10B parameters, Llama-GENBA-10B is continuously pretrained on 164B tokens (82B English, 82B German, and 80M Bavarian), balancing resources while preventing English dominance. Targeted at the German NLP community, the model also promotes Bavarian as a low-resource language. Development tackled four challenges: (1) curating a multilingual corpus despite Bavarian scarcity, (2) creating a unified tokenizer for English, German, and Bavarian, (3) optimizing architecture and language-ratio hyperparameters for cross-lingual transfer, and (4) establishing the first standardized trilingual evaluation suite by translating German benchmarks into Bavarian. Evaluations show that Llama-GENBA-10B achieves strong cross-lingual performance, with the fine-tuned variant surpassing Apertus-8B-2509 and gemma-2-9b in Bavarian and establishing itself as the best model in its class for this language, while also outperforming EuroLLM in English and matching its results in German. Training on the Cerebras CS-2 demonstrated efficient large-scale multilingual pretraining with documented energy use, offering a blueprint for inclusive foundation models that integrate low-resource languages. 

**Abstract (ZH)**: 我们提出Llama-GENBA-10B，这是一种针对大型语言模型中以英语为中心偏见的三语基础模型。 

---
# LM-Searcher: Cross-domain Neural Architecture Search with LLMs via Unified Numerical Encoding 

**Title (ZH)**: LM-Searcher：通过统一数值编码利用LLM进行跨域神经架构搜索 

**Authors**: Yuxuan Hu, Jihao Liu, Ke Wang, Jinliang Zhen, Weikang Shi, Manyuan Zhang, Qi Dou, Rui Liu, Aojun Zhou, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05657)  

**Abstract**: Recent progress in Large Language Models (LLMs) has opened new avenues for solving complex optimization problems, including Neural Architecture Search (NAS). However, existing LLM-driven NAS approaches rely heavily on prompt engineering and domain-specific tuning, limiting their practicality and scalability across diverse tasks. In this work, we propose LM-Searcher, a novel framework that leverages LLMs for cross-domain neural architecture optimization without the need for extensive domain-specific adaptation. Central to our approach is NCode, a universal numerical string representation for neural architectures, which enables cross-domain architecture encoding and search. We also reformulate the NAS problem as a ranking task, training LLMs to select high-performing architectures from candidate pools using instruction-tuning samples derived from a novel pruning-based subspace sampling strategy. Our curated dataset, encompassing a wide range of architecture-performance pairs, encourages robust and transferable learning. Comprehensive experiments demonstrate that LM-Searcher achieves competitive performance in both in-domain (e.g., CNNs for image classification) and out-of-domain (e.g., LoRA configurations for segmentation and generation) tasks, establishing a new paradigm for flexible and generalizable LLM-based architecture search. The datasets and models will be released at this https URL. 

**Abstract (ZH)**: Recent progress in大型语言模型（LLMs）为解决复杂优化问题开辟了新途径，包括神经架构搜索（NAS）。然而，现有的LLM驱动的NAS方法高度依赖提示工程和领域特定调整，限制了其在多样化任务中的实用性和可扩展性。在本工作中，我们提出LM-Searcher，这是一种新的框架，利用LLMs进行跨领域的神经架构优化，而不需进行大量领域特定的适配。我们方法的核心是NCode，一种通用的神经架构数值字符串表示法，使跨领域的架构编码和搜索成为可能。我们还将NAS问题重新表述为排序任务，通过源自新型基于剪枝的子空间采样策略的指令调优样本训练LLMs，从候选池中选择高性能的架构。我们精心构建的数据集涵盖了广泛的架构-性能对，促进了稳健且可迁移的学习。综合实验表明，LM-Searcher在领域内（如用于图像分类的CNN）和跨领域（如用于分割和生成任务的LoRA配置）任务中均能达到竞争力的性能，确立了基于LLM的架构搜索的灵活且通用的新范式。相关数据集和模型将在此网站发布。 

---
# From Joy to Fear: A Benchmark of Emotion Estimation in Pop Song Lyrics 

**Title (ZH)**: 从喜悦到恐惧：流行歌曲歌词情感估计基准 

**Authors**: Shay Dahary, Avi Edana, Alexander Apartsin, Yehudit Aperstein  

**Link**: [PDF](https://arxiv.org/pdf/2509.05617)  

**Abstract**: The emotional content of song lyrics plays a pivotal role in shaping listener experiences and influencing musical preferences. This paper investigates the task of multi-label emotional attribution of song lyrics by predicting six emotional intensity scores corresponding to six fundamental emotions. A manually labeled dataset is constructed using a mean opinion score (MOS) approach, which aggregates annotations from multiple human raters to ensure reliable ground-truth labels. Leveraging this dataset, we conduct a comprehensive evaluation of several publicly available large language models (LLMs) under zero-shot scenarios. Additionally, we fine-tune a BERT-based model specifically for predicting multi-label emotion scores. Experimental results reveal the relative strengths and limitations of zero-shot and fine-tuned models in capturing the nuanced emotional content of lyrics. Our findings highlight the potential of LLMs for emotion recognition in creative texts, providing insights into model selection strategies for emotion-based music information retrieval applications. The labeled dataset is available at this https URL. 

**Abstract (ZH)**: 歌曲歌词中的情感内容在塑造听众体验和影响音乐偏好方面发挥着关键作用。本文研究了通过预测六种基本情感对应的情感强度得分来进行歌词多标签情感归类的任务。采用平均意见分（MOS）方法构建了一个手动标注的数据集，该方法汇集了多名人工评分者的意见以确保可靠的真实标签。利用该数据集，在零样本场景下对几种大型语言模型（LLMs）进行了全面评估。此外，我们对一个基于BERT的模型进行了微调，以预测多标签情感得分。实验结果揭示了零样本和微调模型在捕捉歌词细腻情感内容方面相对的优势与限制。我们的研究突显了LLMs在创意文本中情感识别的潜在应用，并提供了情感音乐信息检索应用中模型选择策略的见解。标注数据集可通过以下链接访问：this https URL。 

---
# Cross-Service Threat Intelligence in LLM Services using Privacy-Preserving Fingerprints 

**Title (ZH)**: 在LLM服务中使用隐私保护指纹进行跨服务威胁情报分享 

**Authors**: Waris Gill, Natalie Isak, Matthew Dressman  

**Link**: [PDF](https://arxiv.org/pdf/2509.05608)  

**Abstract**: The widespread deployment of LLMs across enterprise services has created a critical security blind spot. Organizations operate multiple LLM services handling billions of queries daily, yet regulatory compliance boundaries prevent these services from sharing threat intelligence about prompt injection attacks, the top security risk for LLMs. When an attack is detected in one service, the same threat may persist undetected in others for months, as privacy regulations prohibit sharing user prompts across compliance boundaries.
We present BinaryShield, the first privacy-preserving threat intelligence system that enables secure sharing of attack fingerprints across compliance boundaries. BinaryShield transforms suspicious prompts through a unique pipeline combining PII redaction, semantic embedding, binary quantization, and randomized response mechanism to potentially generate non-invertible fingerprints that preserve attack patterns while providing privacy. Our evaluations demonstrate that BinaryShield achieves an F1-score of 0.94, significantly outperforming SimHash (0.77), the privacy-preserving baseline, while achieving 64x storage reduction and 38x faster similarity search compared to dense embeddings. 

**Abstract (ZH)**: 企业服务中LLM的广泛部署创造了重要的安全盲区。组织运营着多个处理数十亿查询的LLM服务，但由于监管合规限制，这些服务无法共享关于提示注入攻击的威胁情报，这是LLMs最大的安全风险。当在一个服务中检测到攻击时，相同的威胁在其他服务中可能会持续数月未被检测到，因为隐私法规禁止跨合规边界共享用户提示。

我们提出了BinaryShield，这是一种首创的隐私保护威胁情报系统，能够安全地跨合规边界共享攻击指纹。BinaryShield通过一个独特的流水线，结合PII篡改、语义嵌入、二进制量化和随机响应机制，生成潜在不可逆的指纹，同时保留攻击模式并提供隐私保护。我们的评估结果显示，BinaryShield实现了0.94的F1分数，显著优于隐私保护基线SimHash（0.77），同时实现了64倍的存储减少和38倍更快的相似性搜索速度。 

---
# Icon$^{2}$: Aligning Large Language Models Using Self-Synthetic Preference Data via Inherent Regulation 

**Title (ZH)**: Icon$^{2}$: 通过内在调节使用自我合成偏好数据对大型语言模型进行对齐 

**Authors**: Qiyuan Chen, Hongsen Huang, Qian Shao, Jiahe Chen, Jintai Chen, Hongxia Xu, Renjie Hua, Ren Chuan, Jian Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05605)  

**Abstract**: Large Language Models (LLMs) require high quality preference datasets to align with human preferences. However, conventional methods for constructing such datasets face significant challenges: reliance on pre-collected instructions often leads to distribution mismatches with target models, while the need for sampling multiple stochastic responses introduces substantial computational overhead. In this work, we explore a paradigm shift by leveraging inherent regulation of LLMs' representation space for efficient and tailored preference dataset construction, named Icon$^{2}$. Specifically, it first extracts layer-wise direction vectors to encode sophisticated human preferences and then uses these vectors to filter self-synthesized instructions based on their inherent consistency. During decoding, bidirectional inherent control is applied to steer token representations, enabling the precise generation of response pairs with clear alignment distinctions. Experimental results demonstrate significant improvements in both alignment and efficiency. Llama3-8B and Qwen2-7B achieve an average win rate improvement of 13.89% on AlpacaEval 2.0 and 13.45% on Arena-Hard, while reducing computational costs by up to 48.1%. 

**Abstract (ZH)**: 大规模语言模型（LLMs）需要高质量的偏好数据集来与人类偏好对齐。然而，传统的方法在构建此类数据集时面临着重大挑战：依赖预先收集的指令往往会导致与目标模型的分布不匹配，而需要采样多个随机响应则引入了显著的计算开销。在本工作中，我们通过利用LLMs表示空间中的固有调节机制，探索了一种范式转变，称为Icon$^{2}$，以实现高效和定制化的偏好数据集构建。具体而言，它首先提取层wise方向向量以编码复杂的用户偏好，然后使用这些向量基于固有的一致性筛选自我合成的指令。在解码过程中，应用双向固有控制以引导标记表示，从而实现具有明确对齐差异的响应对的精确生成。实验结果表明，在对齐和效率方面均实现了显著改进。Llama3-8B和Qwen2-7B在AlpacaEval 2.0和Arena-Hard上的平均胜率分别提高了13.89%和13.45%，同时计算成本降低了48.1%。 

---
# Using Contrastive Learning to Improve Two-Way Reasoning in Large Language Models: The Obfuscation Task as a Case Study 

**Title (ZH)**: 使用对比学习提高大规模语言模型双向推理能力：混淆任务案例研究 

**Authors**: Serge Lionel Nikiema, Jordan Samhi, Micheline Bénédicte Moumoula, Albérick Euraste Djiré, Abdoul Kader Kaboré, Jacques Klein, Tegawendé F. Bissyandé  

**Link**: [PDF](https://arxiv.org/pdf/2509.05553)  

**Abstract**: This research addresses a fundamental question in AI: whether large language models truly understand concepts or simply recognize patterns. The authors propose bidirectional reasoning,the ability to apply transformations in both directions without being explicitly trained on the reverse direction, as a test for genuine understanding. They argue that true comprehension should naturally allow reversibility. For example, a model that can change a variable name like userIndex to i should also be able to infer that i represents a user index without reverse training. The researchers tested current language models and discovered what they term cognitive specialization: when models are fine-tuned on forward tasks, their performance on those tasks improves, but their ability to reason bidirectionally becomes significantly worse. To address this issue, they developed Contrastive Fine-Tuning (CFT), which trains models using three types of examples: positive examples that maintain semantic meaning, negative examples with different semantics, and forward-direction obfuscation examples. This approach aims to develop deeper understanding rather than surface-level pattern recognition and allows reverse capabilities to develop naturally without explicit reverse training. Their experiments demonstrated that CFT successfully achieved bidirectional reasoning, enabling strong reverse performance while maintaining forward task capabilities. The authors conclude that bidirectional reasoning serves both as a theoretical framework for assessing genuine understanding and as a practical training approach for developing more capable AI systems. 

**Abstract (ZH)**: 这项研究探讨了AI中的一个基础问题：大型语言模型是真正理解概念还是仅仅识别模式。作者提出双向推理能力——能够在不经过反向专门训练的情况下双向应用变换——作为真正理解的测试。他们认为真正的理解应该自然地具有可逆性。例如，能够将变量名userIndex改为i的模型，也应该能够推断i表示用户索引，而无需反向训练。研究人员测试了当前的语言模型，并发现他们所谓的认知专业化现象：当模型在正向任务上进行微调时，这些任务的性能会提高，但其双向推理能力会显著下降。为了解决这一问题，他们开发了对比微调（Contrastive Fine-Tuning, CFT），使用三类例子进行训练：保持语义意义的正向例子、具有不同语义的负向例子以及正向方向的混淆例子。这种方法旨在发展更深层次的理解，而非表面模式识别，并允许反向能力自然发展，而无需明确的反向训练。他们的实验表明，CFT成功实现了双向推理，在保持正向任务能力的同时，提高了反向任务的性能。作者得出结论，双向推理既作为一个评估真正理解的理论框架，也作为一个开发更强大AI系统的实际训练方法。 

---
# Combining TSL and LLM to Automate REST API Testing: A Comparative Study 

**Title (ZH)**: 结合TSL和LLM自动实现REST API测试：一项比较研究 

**Authors**: Thiago Barradas, Aline Paes, Vânia de Oliveira Neves  

**Link**: [PDF](https://arxiv.org/pdf/2509.05540)  

**Abstract**: The effective execution of tests for REST APIs remains a considerable challenge for development teams, driven by the inherent complexity of distributed systems, the multitude of possible scenarios, and the limited time available for test design. Exhaustive testing of all input combinations is impractical, often resulting in undetected failures, high manual effort, and limited test coverage. To address these issues, we introduce RestTSLLM, an approach that uses Test Specification Language (TSL) in conjunction with Large Language Models (LLMs) to automate the generation of test cases for REST APIs. The approach targets two core challenges: the creation of test scenarios and the definition of appropriate input data. The proposed solution integrates prompt engineering techniques with an automated pipeline to evaluate various LLMs on their ability to generate tests from OpenAPI specifications. The evaluation focused on metrics such as success rate, test coverage, and mutation score, enabling a systematic comparison of model performance. The results indicate that the best-performing LLMs - Claude 3.5 Sonnet (Anthropic), Deepseek R1 (Deepseek), Qwen 2.5 32b (Alibaba), and Sabia 3 (Maritaca) - consistently produced robust and contextually coherent REST API tests. Among them, Claude 3.5 Sonnet outperformed all other models across every metric, emerging in this study as the most suitable model for this task. These findings highlight the potential of LLMs to automate the generation of tests based on API specifications. 

**Abstract (ZH)**: 基于测试规范语言和大型语言模型的REST APIs自动测试用例生成方法 

---
# The Token Tax: Systematic Bias in Multilingual Tokenization 

**Title (ZH)**: 令牌税：多语言分词中的系统偏差 

**Authors**: Jessica M. Lundin, Ada Zhang, Nihal Karim, Hamza Louzan, Victor Wei, David Adelani, Cody Carroll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05486)  

**Abstract**: Tokenization inefficiency imposes structural disadvantages on morphologically complex, low-resource languages, inflating compute resources and depressing accuracy. We evaluate 10 large language models (LLMs) on AfriMMLU (9,000 MCQA items; 5 subjects; 16 African languages) and show that fertility (tokens/word) reliably predicts accuracy. Higher fertility consistently predicts lower accuracy across all models and subjects. We further find that reasoning models (DeepSeek, o1) consistently outperform non-reasoning peers across high and low resource languages in the AfriMMLU dataset, narrowing accuracy gaps observed in prior generations. Finally, translating token inflation to economics, a doubling in tokens results in quadrupled training cost and time, underscoring the token tax faced by many languages. These results motivate morphologically aware tokenization, fair pricing, and multilingual benchmarks for equitable natural language processing (NLP). 

**Abstract (ZH)**: 分词效率低下对形态复杂、资源匮乏的语言造成结构上的不利影响，增加计算资源并降低准确性。我们在AfriMMLU（9000个多项选择题；5个科目；16种非洲语言）上评估了10种大规模语言模型（LLMs），并发现分词 fertility（每个词的分词数量）可靠地预测了准确性。更高的分词 fertility 在所有模型和科目中一致地预测了更低的准确性。此外，我们发现推理模型（DeepSeek, o1）在AfriMMLU数据集中的一系列高资源和低资源语言中始终优于非推理模型，缩小了前几代模型中观察到的准确性差距。最后，将分词膨胀转化为经济学概念，分词数量翻倍会导致训练成本和时间增加四倍，强调了许多语言所面临的分词税。这些结果促使我们关注形态学意识的分词、公平定价以及多语言基准测试，以实现公平的自然语言处理（NLP）。 

---
# Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models 

**Title (ZH)**: Behind the Mask: 大型语言模型中隐匿式 Jailbreak 的基准测试 

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita  

**Link**: [PDF](https://arxiv.org/pdf/2509.05471)  

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越容易受到一种名为迷魂劫持的高级 adversarial prompting 影响。这种方法通过看似无害的语言嵌入恶意意图，以规避现有的安全机制。与显性的攻击不同，这些细微的提示利用了上下文的模糊性和语言的灵活性，对当前的防御系统构成重大挑战。本文探讨了迷魂劫持提示的构建及其影响，强调了其欺骗性特点和传统基于关键词的检测方法的局限性。我们引入了一个新的基准数据集，迷魂劫持提示集，包含500个精选示例（400个有害和100个无害提示），旨在严格测试LLM的安全协议。此外，我们提出了一种多维度评估框架，从七个维度衡量有害性：安全意识、技术可行性、实施保护措施、潜在危害性、教育价值、内容质量和合规评分。我们的研究发现，尽管模型在面对良性输入时表现出高度的安全性和内容质量，但在遭遇迷魂劫持尝试时，其性能和安全性显著下降。这一差异突显了LLM普遍存在的一种脆弱性，强调了迫切需要更细腻和适应性强的安全策略，以确保LLM在实际应用中的负责任和稳健部署。 

---
# Neural Breadcrumbs: Membership Inference Attacks on LLMs Through Hidden State and Attention Pattern Analysis 

**Title (ZH)**: 神经面包屑：通过隐藏状态和注意力模式分析对大模型进行成员推理攻击 

**Authors**: Disha Makhija, Manoj Ghuhan Arivazhagan, Vinayshekhar Bannihatti Kumar, Rashmi Gangadharaiah  

**Link**: [PDF](https://arxiv.org/pdf/2509.05449)  

**Abstract**: Membership inference attacks (MIAs) reveal whether specific data was used to train machine learning models, serving as important tools for privacy auditing and compliance assessment. Recent studies have reported that MIAs perform only marginally better than random guessing against large language models, suggesting that modern pre-training approaches with massive datasets may be free from privacy leakage risks. Our work offers a complementary perspective to these findings by exploring how examining LLMs' internal representations, rather than just their outputs, may provide additional insights into potential membership inference signals. Our framework, \emph{memTrace}, follows what we call \enquote{neural breadcrumbs} extracting informative signals from transformer hidden states and attention patterns as they process candidate sequences. By analyzing layer-wise representation dynamics, attention distribution characteristics, and cross-layer transition patterns, we detect potential memorization fingerprints that traditional loss-based approaches may not capture. This approach yields strong membership detection across several model families achieving average AUC scores of 0.85 on popular MIA benchmarks. Our findings suggest that internal model behaviors can reveal aspects of training data exposure even when output-based signals appear protected, highlighting the need for further research into membership privacy and the development of more robust privacy-preserving training techniques for large language models. 

**Abstract (ZH)**: 基于会员推理攻击的内部表示分析：探索大型语言模型潜在的会员推理信号 

---
# Direct-Scoring NLG Evaluators Can Use Pairwise Comparisons Too 

**Title (ZH)**: 直接评分的自然语言生成评估器也可以使用成对比较 

**Authors**: Logan Lawrence, Ashton Williamson, Alexander Shelton  

**Link**: [PDF](https://arxiv.org/pdf/2509.05440)  

**Abstract**: As large-language models have been increasingly used as automatic raters for evaluating free-form content, including document summarization, dialog, and story generation, work has been dedicated to evaluating such models by measuring their correlations with human judgment. For \textit{sample-level} performance, methods which operate by using pairwise comparisons between machine-generated text perform well but often lack the ability to assign absolute scores to individual summaries, an ability crucial for use cases that require thresholding. In this work, we propose a direct-scoring method which uses synthetic summaries to act as pairwise machine rankings at test time. We show that our method performs comparably to state-of-the-art pairwise evaluators in terms of axis-averaged sample-level correlations on the SummEval (\textbf{+0.03}), TopicalChat (\textbf{-0.03}), and HANNA (\textbf{+0.05}) meta-evaluation benchmarks, and release the synthetic in-context summaries as data to facilitate future work. 

**Abstract (ZH)**: 随着大型语言模型在自动评估自由形式内容（包括文档摘要、对话和故事情节生成）方面被越来越广泛应用，已有工作致力于通过衡量这些模型与人类判断的相关性来评估这些模型。对于样本级性能，依赖成对比较机器生成文本的方法表现良好，但往往缺乏为单个摘要分配绝对分数的能力，这是需要阈值要求的应用场景的关键。在本工作中，我们提出了一种直接评分方法，在测试时使用合成摘要作为成对机器排名。我们的方法在SummEval（+0.03）、TopicalChat（-0.03）和HANNA（+0.05）元评估基准中，在轴平均样本级相关性方面与最新的成对评估器表现相当，并发布了合成上下文摘要数据以促进未来研究。 

---
# Authorship Without Writing: Large Language Models and the Senior Author Analogy 

**Title (ZH)**: 不写作的作者身份：大型语言模型与资深作者类比 

**Authors**: Clint Hurshman, Sebastian Porsdam Mann, Julian Savulescu, Brian D. Earp  

**Link**: [PDF](https://arxiv.org/pdf/2509.05390)  

**Abstract**: The use of large language models (LLMs) in bioethical, scientific, and medical writing remains controversial. While there is broad agreement in some circles that LLMs cannot count as authors, there is no consensus about whether and how humans using LLMs can count as authors. In many fields, authorship is distributed among large teams of researchers, some of whom, including paradigmatic senior authors who guide and determine the scope of a project and ultimately vouch for its integrity, may not write a single word. In this paper, we argue that LLM use (under specific conditions) is analogous to a form of senior authorship. On this view, the use of LLMs, even to generate complete drafts of research papers, can be considered a legitimate form of authorship according to the accepted criteria in many fields. We conclude that either such use should be recognized as legitimate, or current criteria for authorship require fundamental revision. AI use declaration: GPT-5 was used to help format Box 1. AI was not used for any other part of the preparation or writing of this manuscript. 

**Abstract (ZH)**: 大型语言模型在生物伦理、科学和医学写作中的应用仍存在争议。虽然在某些圈子里普遍认为大型语言模型不能被视为作者，但对于人类使用大型语言模型是否可以被视为作者以及如何被视为作者尚无共识。在许多领域中，作者身份在由大量研究人员组成的团队中分布，其中一些人，包括主导项目并确定其范围并在最终为其 integrity 负责的范式意义上的资深作者，可能不会撰写一个字。在本文中，我们认为在特定条件下使用大型语言模型类似于一种资深作者身份的形式。在这种观点下，即使使用大型语言模型生成研究论文的完整草稿，也可以被视为根据许多领域接受的标准的一种合法的作者身份。我们得出结论，要么这种使用应该被认定为合法，要么现行的作者身份标准需要根本性的修订。AI使用声明：GPT-5用于帮助格式化Box 1部分。本论文的其他部分未使用AI进行任何准备或撰写工作。 

---
# A Lightweight Framework for Trigger-Guided LoRA-Based Self-Adaptation in LLMs 

**Title (ZH)**: 一种基于触发器引导的LoRA自适应轻量级框架在大规模语言模型中的应用 

**Authors**: Jiacheng Wei, Faguo Wu, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05385)  

**Abstract**: Large language models are unable to continuously adapt and learn from new data during reasoning at inference time. To address this limitation, we propose that complex reasoning tasks be decomposed into atomic subtasks and introduce SAGE, a trigger-guided dynamic fine-tuning framework that enables adaptive updates during reasoning at inference time. SAGE consists of three key components: (1) a Trigger module that detects reasoning failures through multiple evaluation metrics in real time; (2) a Trigger Buffer module that clusters anomaly samples using a streaming clustering process with HDBSCAN, followed by stability checks and similarity-based merging; and (3) a Lora Store module that dynamically optimizes parameter updates with an adapter pool for knowledge retention. Evaluation results show that SAGE demonstrates excellent accuracy, robustness, and stability on the atomic reasoning subtask through dynamic knowledge updating during test time. 

**Abstract (ZH)**: 大型语言模型在推理时无法连续适应和从新数据中学习。为此，我们提出将复杂推理任务分解为原子子任务，并引入SAGE，一种触发器引导的动态微调框架，允许在推理时进行自适应更新。SAGE 包含三个关键组件：（1）一个触发器模块，通过实时多种评估指标检测推理失败；（2）一个触发器缓冲模块，使用基于流的聚类过程和HDBSCAN进行异常样本聚类，随后进行稳定性检查和基于相似性的合并；（3）一个Lora存储模块，使用适配器池动态优化参数更新以保留知识。评价结果表明，SAGE 在测试时通过动态知识更新展示了出色的准确度、鲁棒性和稳定性。 

---
# Between a Rock and a Hard Place: Exploiting Ethical Reasoning to Jailbreak LLMs 

**Title (ZH)**: 在岩石和硬 place 之间：利用伦理推理突破大语言模型 blockade 

**Authors**: Shei Pern Chua, Thai Zhen Leng, Teh Kai Jun, Xiao Li, Xiaolin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05367)  

**Abstract**: Large language models (LLMs) have undergone safety alignment efforts to mitigate harmful outputs. However, as LLMs become more sophisticated in reasoning, their intelligence may introduce new security risks. While traditional jailbreak attacks relied on singlestep attacks, multi-turn jailbreak strategies that adapt dynamically to context remain underexplored. In this work, we introduce TRIAL (Trolley-problem Reasoning for Interactive Attack Logic), a framework that leverages LLMs ethical reasoning to bypass their safeguards. TRIAL embeds adversarial goals within ethical dilemmas modeled on the trolley problem. TRIAL demonstrates high jailbreak success rates towards both open and close-source models. Our findings underscore a fundamental limitation in AI safety: as models gain advanced reasoning abilities, the nature of their alignment may inadvertently allow for more covert security vulnerabilities to be exploited. TRIAL raises an urgent need in reevaluating safety alignment oversight strategies, as current safeguards may prove insufficient against context-aware adversarial attack. 

**Abstract (ZH)**: 大型语言模型的安全对齐努力已经减少了有害输出的风险。然而，随着语言模型在推理方面变得越来越复杂，它们的智能可能引入新的安全风险。虽然传统的监狱突破攻击依赖于单步骤攻击，但适应性强的多轮监狱突破策略仍然未被充分探索。在本工作中，我们提出了TRIAL（Trolley-problem Reasoning for Interactive Attack Logic）框架，该框架利用大型语言模型的伦理推理来绕过其安全防护。TRIAL将对抗性目标嵌入到基于电车问题建模的伦理困境中。TRIAL展示了对开源和闭源模型高度成功的监狱突破成功率。我们的研究结果揭示了一个基本的安全限制：随着模型获得高级推理能力，它们的对齐方式可能会无意中允许更多的隐蔽安全漏洞被利用。TRIAL强调了一个迫切的需求，即重新评估安全对齐监督策略，因为当前的安全防护可能不足以应对具有情境意识的对抗性攻击。 

---
# AI-in-the-Loop: Privacy Preserving Real-Time Scam Detection and Conversational Scambaiting by Leveraging LLMs and Federated Learning 

**Title (ZH)**: AI在环中:通过利用LLM和联邦学习实现的隐私保护实时诈骗检测与对话式诈骗诱捕 

**Authors**: Ismail Hossain, Sai Puppala, Sajedul Talukder, Md Jahangir Alam  

**Link**: [PDF](https://arxiv.org/pdf/2509.05362)  

**Abstract**: Scams exploiting real-time social engineering -- such as phishing, impersonation, and phone fraud -- remain a persistent and evolving threat across digital platforms. Existing defenses are largely reactive, offering limited protection during active interactions. We propose a privacy-preserving, AI-in-the-loop framework that proactively detects and disrupts scam conversations in real time. The system combines instruction-tuned artificial intelligence with a safety-aware utility function that balances engagement with harm minimization, and employs federated learning to enable continual model updates without raw data sharing. Experimental evaluations show that the system produces fluent and engaging responses (perplexity as low as 22.3, engagement $\approx$0.80), while human studies confirm significant gains in realism, safety, and effectiveness over strong baselines. In federated settings, models trained with FedAvg sustain up to 30 rounds while preserving high engagement ($\approx$0.80), strong relevance ($\approx$0.74), and low PII leakage ($\leq$0.0085). Even with differential privacy, novelty and safety remain stable, indicating that robust privacy can be achieved without sacrificing performance. The evaluation of guard models (LlamaGuard, LlamaGuard2/3, MD-Judge) shows a straightforward pattern: stricter moderation settings reduce the chance of exposing personal information, but they also limit how much the model engages in conversation. In contrast, more relaxed settings allow longer and richer interactions, which improve scam detection, but at the cost of higher privacy risk. To our knowledge, this is the first framework to unify real-time scam-baiting, federated privacy preservation, and calibrated safety moderation into a proactive defense paradigm. 

**Abstract (ZH)**: 利用实时社会工程学实施的诈骗——例如 phishing、冒充和电话诈骗——仍然是数字平台上的一个持久且不断演变的威胁。现有的防护措施主要是反应性的，在活跃交互过程中提供的保护有限。我们提出了一种保护隐私、AI 集成循环的框架，能够实时主动检测和打断诈骗对话。该系统结合了指令调优的人工智能与兼顾安全的效用函数，平衡参与度与最小化危害，并采用联邦学习来实现无原始数据共享的持续模型更新。实验评估表明，该系统生成流畅且富有参与度的响应（困惑度低至 22.3，参与度 ≈ 0.80），而人类研究证实，与强基线相比，在现实主义、安全性和有效性方面有显著提升。在联邦环境中，使用 FedAvg 训练的模型最多可维持 30 轮更新，同时保持高水平的参与度（≈0.80）、较强的相关性（≈0.74）和低个人身份信息泄露（≤0.0085）。即使在差分隐私下，新颖性和安全性仍保持稳定，表明可以实现稳健的隐私保护而不牺牲性能。对于防护模型（LlamaGuard、LlamaGuard2/3、MD-Judge）的评估显示了一个简单的模式：更严格的审查设置减少了暴露个人信息的机会，但也限制了模型在对话中的参与度。相反，更宽松的设置允许更长和更丰富的互动，从而提高诈骗检测，但以更高的隐私风险为代价。据我们所知，这是第一个将实时诈骗诱饵、联邦隐私保护和校准的安全调节统一到主动防御范式中的框架。 

---
# ForensicsData: A Digital Forensics Dataset for Large Language Models 

**Title (ZH)**: ForensicsData：面向大型语言模型的数字取证数据集 

**Authors**: Youssef Chakir, Iyad Lahsen-Cherif  

**Link**: [PDF](https://arxiv.org/pdf/2509.05331)  

**Abstract**: The growing complexity of cyber incidents presents significant challenges for digital forensic investigators, especially in evidence collection and analysis. Public resources are still limited because of ethical, legal, and privacy concerns, even though realistic datasets are necessary to support research and tool developments. To address this gap, we introduce ForensicsData, an extensive Question-Context-Answer (Q-C-A) dataset sourced from actual malware analysis reports. It consists of more than 5,000 Q-C-A triplets. A unique workflow was used to create the dataset, which extracts structured data, uses large language models (LLMs) to transform it into Q-C-A format, and then uses a specialized evaluation process to confirm its quality. Among the models evaluated, Gemini 2 Flash demonstrated the best performance in aligning generated content with forensic terminology. ForensicsData aims to advance digital forensics by enabling reproducible experiments and fostering collaboration within the research community. 

**Abstract (ZH)**: 网络事件日益增大的复杂性给数字取证调查人员带来了巨大挑战，尤其是在证据收集和分析方面。由于伦理、法律和隐私方面的关切，公共资源仍然有限，尽管真实的数据集对于支持研究和工具开发是必要的。为解决这一问题，我们引入了ForensicsData，这是一个来源于实际恶意软件分析报告的全面问题-上下文-答案（Q-C-A）数据集，包含超过5,000个Q-C-A三元组。采用了一种独特的workflow创建数据集，该workflow提取结构化数据、使用大语言模型（LLMs）将其转换为Q-C-A格式，并通过专门的评估过程确认其质量。在评估的模型中，Gemini 2 Flash在生成内容与取证术语的对齐方面表现最佳。ForensicsData旨在通过促进可重复实验和研究社区内的合作来推动数字取证的发展。 

---
# Backdoor Samples Detection Based on Perturbation Discrepancy Consistency in Pre-trained Language Models 

**Title (ZH)**: 基于扰动不一致性在预训练语言模型中的后门样本检测 

**Authors**: Zuquan Peng, Jianming Fu, Lixin Zou, Li Zheng, Yanzhen Ren, Guojun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.05318)  

**Abstract**: The use of unvetted third-party and internet data renders pre-trained models susceptible to backdoor attacks. Detecting backdoor samples is critical to prevent backdoor activation during inference or injection during training. However, existing detection methods often require the defender to have access to the poisoned models, extra clean samples, or significant computational resources to detect backdoor samples, limiting their practicality. To address this limitation, we propose a backdoor sample detection method based on perturbatio\textbf{N} discr\textbf{E}pancy consis\textbf{T}ency \textbf{E}valuation (\NETE). This is a novel detection method that can be used both pre-training and post-training phases. In the detection process, it only requires an off-the-shelf pre-trained model to compute the log probability of samples and an automated function based on a mask-filling strategy to generate perturbations. Our method is based on the interesting phenomenon that the change in perturbation discrepancy for backdoor samples is smaller than that for clean samples. Based on this phenomenon, we use curvature to measure the discrepancy in log probabilities between different perturbed samples and input samples, thereby evaluating the consistency of the perturbation discrepancy to determine whether the input sample is a backdoor sample. Experiments conducted on four typical backdoor attacks and five types of large language model backdoor attacks demonstrate that our detection strategy outperforms existing zero-shot black-box detection methods. 

**Abstract (ZH)**: 未审核的第三方和互联网数据使得预训练模型容易受到后门攻击。基于扰动离散性一致性评估（\NETE）的后门样本检测方法对于防止推理或训练期间后门激活至关重要。现有的检测方法往往要求防御者拥有中毒模型、额外的干净样本或大量的计算资源来检测后门样本，这限制了它们的实际应用。为解决这一局限性，我们提出了一种基于扰动离散性一致性评估（\NETE）的后门样本检测方法。这是一种新型的检测方法，可以在预训练和后训练阶段使用。在检测过程中，该方法只需使用一个即用型的预训练模型计算样本的对数概率，并基于掩码填充策略的自动化函数生成扰动。我们的方法基于一个有趣的现象，即后门样本的扰动离散性变化小于干净样本的变化。基于这一现象，我们使用曲率来衡量不同扰动样本和输入样本之间对数概率的不一致性，并评估扰动离散性的一致性以确定输入样本是否为后门样本。实验表明，我们的检测策略在四种典型的后门攻击和五种大型语言模型后门攻击上优于现有的零样本黑盒检测方法。 

---
# Standard vs. Modular Sampling: Best Practices for Reliable LLM Unlearning 

**Title (ZH)**: 标准采样 vs. 模块化采样：可靠卸载LLM的最佳实践 

**Authors**: Praveen Bushipaka, Lucia Passaro, Tommaso Cucinotta  

**Link**: [PDF](https://arxiv.org/pdf/2509.05316)  

**Abstract**: A conventional LLM Unlearning setting consists of two subsets -"forget" and "retain", with the objectives of removing the undesired knowledge from the forget set while preserving the remaining knowledge from the retain. In privacy-focused unlearning research, a retain set is often further divided into neighbor sets, containing either directly or indirectly connected to the forget targets; and augmented by a general-knowledge set. A common practice in existing benchmarks is to employ only a single neighbor set, with general knowledge which fails to reflect the real-world data complexities and relationships. LLM Unlearning typically involves 1:1 sampling or cyclic iteration sampling. However, the efficacy and stability of these de facto standards have not been critically examined. In this study, we systematically evaluate these common practices. Our findings reveal that relying on a single neighbor set is suboptimal and that a standard sampling approach can obscure performance trade-offs. Based on this analysis, we propose and validate an initial set of best practices: (1) Incorporation of diverse neighbor sets to balance forget efficacy and model utility, (2) Standard 1:1 sampling methods are inefficient and yield poor results, (3) Our proposed Modular Entity-Level Unlearning (MELU) strategy as an alternative to cyclic sampling. We demonstrate that this modular approach, combined with robust algorithms, provides a clear and stable path towards effective unlearning. 

**Abstract (ZH)**: 一种传统的LLM去学习设置包括两个子集——“遗忘”和“保留”，目标是从遗忘集中移除不需要的知识，同时保留剩余的知识。在以隐私为中心的去学习研究中，保留集通常进一步分成邻居集，包含直接或间接与遗忘目标连接的知识；并通过一般知识集进行扩充。现有基准中的一种常见做法是仅使用一个邻居集，这无法反映现实世界数据的复杂性和关系。LLM去学习通常涉及一对一采样或循环迭代采样。然而，这两种现成标准的有效性和稳定性尚未受到严格检验。在此研究中，我们系统地评估了这些常见做法。我们的发现表明，依赖单一邻居集是次优的，且标准采样方法会掩盖性能权衡。基于此分析，我们提出并验证了一套初步的最佳实践：（1）包含多样化的邻居集以平衡遗忘效果和模型实用性；（2）标准的一对一采样方法效率低下且结果不佳；（3）我们提出的模块化实体级去学习（MELU）策略作为循环采样的替代方案。我们证明，这种模块化方法结合稳健的算法，为有效的去学习提供了一条清晰且稳定的道路。 

---
# Large Language Model Integration with Reinforcement Learning to Augment Decision-Making in Autonomous Cyber Operations 

**Title (ZH)**: 大型语言模型与强化学习集成以增强自主网络操作中的决策制定 

**Authors**: Konur Tholl, François Rivest, Mariam El Mezouar, Ranwa Al Mallah  

**Link**: [PDF](https://arxiv.org/pdf/2509.05311)  

**Abstract**: Reinforcement Learning (RL) has shown great potential for autonomous decision-making in the cybersecurity domain, enabling agents to learn through direct environment interaction. However, RL agents in Autonomous Cyber Operations (ACO) typically learn from scratch, requiring them to execute undesirable actions to learn their consequences. In this study, we integrate external knowledge in the form of a Large Language Model (LLM) pretrained on cybersecurity data that our RL agent can directly leverage to make informed decisions. By guiding initial training with an LLM, we improve baseline performance and reduce the need for exploratory actions with obviously negative outcomes. We evaluate our LLM-integrated approach in a simulated cybersecurity environment, and demonstrate that our guided agent achieves over 2x higher rewards during early training and converges to a favorable policy approximately 4,500 episodes faster than the baseline. 

**Abstract (ZH)**: 强化学习（RL）在网络安全自主决策领域展现了巨大潜力，使代理能够通过直接与环境交互来学习。然而，自主网络操作（ACO）中的RL代理通常需要从零开始学习，必须执行不 desirable 的行为来学习其后果。本研究中，我们通过一种预训练于网络安全数据的大规模语言模型（LLM）引入外部知识，让RL代理能够直接利用这些知识做出知情决策。通过使用LLM引导初始训练，我们提高了基线性能，并减少了执行显然具有负面后果的探索性行为的需要。我们在模拟的网络安全环境中评估了我们的LLM集成方法，并证明我们的引导代理在早期训练期间获得的奖励超过基线的2倍以上，并且在大约4,500个回合内收敛到有利策略的速度也快于基线。 

---
# ProtSAE: Disentangling and Interpreting Protein Language Models via Semantically-Guided Sparse Autoencoders 

**Title (ZH)**: ProtSAE: 通过语义导向的稀疏自编码器解耦与解释蛋白质语言模型 

**Authors**: Xiangyu Liu, Haodi Lei, Yi Liu, Yang Liu, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05309)  

**Abstract**: Sparse Autoencoder (SAE) has emerged as a powerful tool for mechanistic interpretability of large language models. Recent works apply SAE to protein language models (PLMs), aiming to extract and analyze biologically meaningful features from their latent spaces. However, SAE suffers from semantic entanglement, where individual neurons often mix multiple nonlinear concepts, making it difficult to reliably interpret or manipulate model behaviors. In this paper, we propose a semantically-guided SAE, called ProtSAE. Unlike existing SAE which requires annotation datasets to filter and interpret activations, we guide semantic disentanglement during training using both annotation datasets and domain knowledge to mitigate the effects of entangled attributes. We design interpretability experiments showing that ProtSAE learns more biologically relevant and interpretable hidden features compared to previous methods. Performance analyses further demonstrate that ProtSAE maintains high reconstruction fidelity while achieving better results in interpretable probing. We also show the potential of ProtSAE in steering PLMs for downstream generation tasks. 

**Abstract (ZH)**: Sparse 自编码器（SAE）已成为大型语言模型机制可解释性的强大工具。近期研究表明，SAE 可应用于蛋白质语言模型（PLMs），旨在从其潜在空间中提取和分析生物上有意义的特征。然而，SAE 存在语义纠缠问题，其中单个神经元常混杂多种非线性概念，使得模型行为的可靠解释和操控变得困难。在本文中，我们提出了一种基于语义指导的 SAE，称为 ProtSAE。与现有方法需要注释数据集来筛选和解释激活不同，我们在训练过程中利用注释数据集和领域知识来指导语义解纠缠，从而减轻纠缠属性的影响。通过解释性实验，我们表明 ProtSAE 能够学习到比以往方法更生物相关且易于解释的隐藏特征。性能分析进一步证实，ProtSAE 在保持高重建保真度的同时，在可解释探针任务上取得了更好的表现。我们还展示了 ProtSAE 在下游生成任务中引导 PLMs 的潜在应用。 

---
