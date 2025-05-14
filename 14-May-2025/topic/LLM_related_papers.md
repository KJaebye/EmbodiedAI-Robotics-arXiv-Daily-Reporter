# DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models 

**Title (ZH)**: DeepMath-创意: 评估大型语言模型数学创造力的标准基准 

**Authors**: Xiaoyang Chen, Xinan Dai, Yu Du, Qian Feng, Naixu Guo, Tingshuo Gu, Yuting Gao, Yingyi Gao, Xudong Han, Xiang Jiang, Yilin Jin, Hongyi Lin, Shisheng Lin, Xiangnan Li, Yuante Li, Yixing Li, Zhentao Lai, Zilu Ma, Yingrong Peng, Jiacheng Qian, Hao-Yu Sun, Jianbo Sun, Zirui Wang, Siwei Wu, Zian Wang, Bin Xu, Jianghao Xu, Yiyang Yu, Zichuan Yang, Hongji Zha, Ruichong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08744)  

**Abstract**: To advance the mathematical proficiency of large language models (LLMs), the DeepMath team has launched an open-source initiative aimed at developing an open mathematical LLM and systematically evaluating its mathematical creativity. This paper represents the initial contribution of this initiative. While recent developments in mathematical LLMs have predominantly emphasized reasoning skills, as evidenced by benchmarks on elementary to undergraduate-level mathematical tasks, the creative capabilities of these models have received comparatively little attention, and evaluation datasets remain scarce. To address this gap, we propose an evaluation criteria for mathematical creativity and introduce DeepMath-Creative, a novel, high-quality benchmark comprising constructive problems across algebra, geometry, analysis, and other domains. We conduct a systematic evaluation of mainstream LLMs' creative problem-solving abilities using this dataset. Experimental results show that even under lenient scoring criteria -- emphasizing core solution components and disregarding minor inaccuracies, such as small logical gaps, incomplete justifications, or redundant explanations -- the best-performing model, O3 Mini, achieves merely 70% accuracy, primarily on basic undergraduate-level constructive tasks. Performance declines sharply on more complex problems, with models failing to provide substantive strategies for open problems. These findings suggest that, although current LLMs display a degree of constructive proficiency on familiar and lower-difficulty problems, such performance is likely attributable to the recombination of memorized patterns rather than authentic creative insight or novel synthesis. 

**Abstract (ZH)**: 为了提高大型语言模型的数学能力，DeepMath团队启动了一个开源项目，旨在开发一个开放的数学大型语言模型，并系统评估其数学创造力。本文代表了该项目的初步贡献。虽然近年来数学大型语言模型的发展主要强调了推理能力，特别是在小学到本科级别的数学任务基准测试中，这些模型的创造性能力却受到了相对较少的关注，评估数据集也仍然稀缺。为解决这一差距，我们提出了一套数学创造力的评估标准，并引入了DeepMath-Creative，这是一个全新的、高质量的数据集，涵盖了代数、几何、分析及其他领域的构造性问题。我们使用这一数据集对主流大型语言模型的创造性问题解决能力进行了系统的评估。实验结果表明，即使是放宽评分标准——强调核心解题要素，忽略如小的逻辑缺口、不完整的证明或冗余解释等琐碎不准确之处——表现最好的模型O3 Mini在基本本科级别构造性任务上的准确率也只有70%。在更复杂的问题上，模型的性能急剧下降，无法提供实质性的策略来解决开放性问题。这些发现表明，尽管当前的大型语言模型在熟悉的和难度较低的问题上显示了一定的构造性能力，但这种表现很可能是由于记忆模式的重组而非真正的创造力洞察或新颖的综合能力。 

---
# LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs 

**Title (ZH)**: 基于LLM的提示集合方法在EHR中实现可靠的医疗实体识别 

**Authors**: K M Sajjadul Islam, Ayesha Siddika Nipu, Jiawei Wu, Praveen Madiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.08704)  

**Abstract**: Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting. 

**Abstract (ZH)**: 电子健康记录（EHRs）中的基于提示的医学实体识别研究：利用大型语言模型（LLMs）并结合各种提示工程技术 

---
# TRAIL: Trace Reasoning and Agentic Issue Localization 

**Title (ZH)**: 轨迹推理与自主问题定位 

**Authors**: Darshan Deshpande, Varun Gangal, Hersh Mehta, Jitin Krishnan, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2505.08638)  

**Abstract**: The increasing adoption of agentic workflows across diverse domains brings a critical need to scalably and systematically evaluate the complex traces these systems generate. Current evaluation methods depend on manual, domain-specific human analysis of lengthy workflow traces - an approach that does not scale with the growing complexity and volume of agentic outputs. Error analysis in these settings is further complicated by the interplay of external tool outputs and language model reasoning, making it more challenging than traditional software debugging. In this work, we (1) articulate the need for robust and dynamic evaluation methods for agentic workflow traces, (2) introduce a formal taxonomy of error types encountered in agentic systems, and (3) present a set of 148 large human-annotated traces (TRAIL) constructed using this taxonomy and grounded in established agentic benchmarks. To ensure ecological validity, we curate traces from both single and multi-agent systems, focusing on real-world applications such as software engineering and open-world information retrieval. Our evaluations reveal that modern long context LLMs perform poorly at trace debugging, with the best Gemini-2.5-pro model scoring a mere 11% on TRAIL. Our dataset and code are made publicly available to support and accelerate future research in scalable evaluation for agentic workflows. 

**Abstract (ZH)**: 随着代理工作流在多个领域的广泛应用，对这些系统产生的复杂追踪进行可扩展且系统的评价变得至关重要。当前的评价方法依赖于手动且针对特定领域的手工分析长流程追踪——这种方法随着代理输出的复杂性和数量增长而无法扩展。代理系统中错误分析进一步受到外部工具输出和语言模型推理的相互作用的影响，使其比传统软件调试更具挑战性。在本文中，我们（1）阐明了代理工作流追踪需要稳健且动态的评价方法的需求，（2）提出了代理系统中遇到的错误类型的正式分类法，（3）使用该分类法和基于已建立的代理基准构建了一个包含148个人标注的追踪数据集（TRAIL）。为了确保生态效度，我们从单代理和多代理系统中精选追踪，重点是软件工程和开放世界信息检索等实际应用。我们的评估结果显示，现代长上下文语言模型在追踪调试方面表现不佳，Gemini-2.5-pro模型在TRAIL上的得分仅为11%。我们的数据集和代码已公开发布，以支持并加速未来对代理工作流可扩展评价的研究。 

---
# Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models 

**Title (ZH)**: 视觉引导解码：无需梯度 hardness 命令反转swith 语言模型 

**Authors**: Donghoon Kim, Minji Bae, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2505.08622)  

**Abstract**: Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models. 

**Abstract (ZH)**: 视觉引导解码（VGD）：利用大语言模型和CLIP指导生成连贯且语义对齐的文本提示 

---
# Resource-Efficient Language Models: Quantization for Fast and Accessible Inference 

**Title (ZH)**: 资源高效语言模型：量化实现快速可访问推理 

**Authors**: Tollef Emil Jørgensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08620)  

**Abstract**: Large language models have significantly advanced natural language processing, yet their heavy resource demands pose severe challenges regarding hardware accessibility and energy consumption. This paper presents a focused and high-level review of post-training quantization (PTQ) techniques designed to optimize the inference efficiency of LLMs by the end-user, including details on various quantization schemes, granularities, and trade-offs. The aim is to provide a balanced overview between the theory and applications of post-training quantization. 

**Abstract (ZH)**: 大型语言模型在自然语言处理领域取得了显著进展，但其对硬件资源的巨大需求给硬件访问能力和能源消耗带来了严重挑战。本文对后训练量化(PTQ)技术进行了聚焦和高层次的综述，这些技术旨在通过终端用户优化大型语言模型的推理效率，包括各种量化方案、粒度和权衡的详细内容。本文的目的是在后训练量化的理论与应用之间提供一个平衡的综述。 

---
# Guiding LLM-based Smart Contract Generation with Finite State Machine 

**Title (ZH)**: 基于有限状态机引导的大规模语言模型驱动的智能合约生成 

**Authors**: Hao Luo, Yuhao Lin, Xiao Yan, Xintong Hu, Yuxiang Wang, Qiming Zeng, Hao Wang, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08542)  

**Abstract**: Smart contract is a kind of self-executing code based on blockchain technology with a wide range of application scenarios, but the traditional generation method relies on manual coding and expert auditing, which has a high threshold and low efficiency. Although Large Language Models (LLMs) show great potential in programming tasks, they still face challenges in smart contract generation w.r.t. effectiveness and security. To solve these problems, we propose FSM-SCG, a smart contract generation framework based on finite state machine (FSM) and LLMs, which significantly improves the quality of the generated code by abstracting user requirements to generate FSM, guiding LLMs to generate smart contracts, and iteratively optimizing the code with the feedback of compilation and security checks. The experimental results show that FSM-SCG significantly improves the quality of smart contract generation. Compared to the best baseline, FSM-SCG improves the compilation success rate of generated smart contract code by at most 48%, and reduces the average vulnerability risk score by approximately 68%. 

**Abstract (ZH)**: 基于有限状态机和大语言模型的智能合约生成框架FSM-SCG 

---
# Strategy-Augmented Planning for Large Language Models via Opponent Exploitation 

**Title (ZH)**: 基于对手利用的策略增强规划方法 LARGE LANGUAGE MODELS 

**Authors**: Shuai Xu, Sijia Cui, Yanna Wang, Bo Xu, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08459)  

**Abstract**: Efficiently modeling and exploiting opponents is a long-standing challenge in adversarial domains. Large Language Models (LLMs) trained on extensive textual data have recently demonstrated outstanding performance in general tasks, introducing new research directions for opponent modeling. Some studies primarily focus on directly using LLMs to generate decisions based on the elaborate prompt context that incorporates opponent descriptions, while these approaches are limited to scenarios where LLMs possess adequate domain expertise. To address that, we introduce a two-stage Strategy-Augmented Planning (SAP) framework that significantly enhances the opponent exploitation capabilities of LLM-based agents by utilizing a critical component, the Strategy Evaluation Network (SEN). Specifically, in the offline stage, we construct an explicit strategy space and subsequently collect strategy-outcome pair data for training the SEN network. During the online phase, SAP dynamically recognizes the opponent's strategies and greedily exploits them by searching best response strategy on the well-trained SEN, finally translating strategy to a course of actions by carefully designed prompts. Experimental results show that SAP exhibits robust generalization capabilities, allowing it to perform effectively not only against previously encountered opponent strategies but also against novel, unseen strategies. In the MicroRTS environment, SAP achieves a 85.35\% performance improvement over baseline methods and matches the competitiveness of reinforcement learning approaches against state-of-the-art (SOTA) rule-based AI. 

**Abstract (ZH)**: 高效建模和利用对手是对抗领域长期存在的挑战。大规模语言模型（LLMs）在广泛任务上的卓越表现最近引发了一系列关于对手建模的新研究方向。一些研究主要集中在直接使用LLMs根据包含对手描述的细致提示构建决策，但这些方法受限于LLMs具备足够的领域专业知识的场景。为此，我们提出了一种两阶段策略增强规划（SAP）框架，该框架通过利用关键组件——策略评估网络（SEN）大幅提升了基于LLM的代理的对手利用能力。具体而言，在离线阶段，我们构建了一个明确的策略空间并收集策略-结果对数据用于训练SEN网络。在在线阶段，SAP动态识别对手的策略并贪婪地利用这些策略通过在充分训练的SEN中搜索最优响应策略，最终通过精心设计的提示将策略转化为行动计划。实验结果显示，SAP展现出强大的泛化能力，不仅能够有效应对之前遇到的对手策略，还能应对全新的未见过的策略。在MicroRTS环境中，SAP相对于基线方法实现了85.35%的性能提升，并且在与最先进的基于规则的AI方法的对比中与强化学习方法保持了竞争力。 

---
# Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation 

**Title (ZH)**: 像人类一样学习：通过自适应难度课程学习和专家导向的自我重述提升LLM推理能力 

**Authors**: Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08364)  

**Abstract**: Despite impressive progress in areas like mathematical reasoning, large language models still face significant challenges in consistently solving complex problems. Drawing inspiration from key human learning strategies, we propose two novel strategies to enhance the capability of large language models to solve these complex problems. First, Adaptive Difficulty Curriculum Learning (ADCL) is a novel curriculum learning strategy that tackles the Difficulty Shift phenomenon (i.e., a model's perception of problem difficulty dynamically changes during training) by periodically re-estimating difficulty within upcoming data batches to maintain alignment with the model's evolving capabilities. Second, Expert-Guided Self-Reformulation (EGSR) is a novel reinforcement learning strategy that bridges the gap between imitation learning and pure exploration by guiding models to reformulate expert solutions within their own conceptual framework, rather than relying on direct imitation, fostering deeper understanding and knowledge assimilation. Extensive experiments on challenging mathematical reasoning benchmarks, using Qwen2.5-7B as the base model, demonstrate that these human-inspired strategies synergistically and significantly enhance performance. Notably, their combined application improves performance over the standard Zero-RL baseline by 10% on the AIME24 benchmark and 16.6% on AIME25. 

**Abstract (ZH)**: 尽管在数学推理等领域取得了显著进展，大型语言模型在一致解决复杂问题方面仍然面临重大挑战。受关键的人类学习策略启发，我们提出了两种新型策略以增强大型语言模型解决复杂问题的能力。首先，自适应难度 Curriculum 学习（ADCL）是一种新的 Curriculum 学习策略，通过在即将到来的数据批次中定期重新评估难度来应对难度转移现象（即，在训练过程中模型对问题难度的感知动态变化），从而保持与模型不断演变的能力的同步。其次，专家引导自我重述（EGSR）是一种新的强化学习策略，通过引导模型在其自身概念框架内重述专家解决方案，而不是依赖直接模仿，从而弥补了模仿学习与纯粹探索之间的差距，促进更深层次的理解和知识吸收。在使用 Qwen2.5-7B 作为基准模型的具有挑战性的数学推理基准测试中，这些受人类启发的策略协同并显著提升了性能。值得注意的是，它们的联合应用分别在 AIME24 和 AIME25 基准测试中将性能提高了 10% 和 16.6%，超过了标准的 Zero-RL 基线。 

---
# Evaluating LLM Metrics Through Real-World Capabilities 

**Title (ZH)**: 通过实际能力评估LLM指标 

**Authors**: Justin K Miller, Wenjia Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08253)  

**Abstract**: As generative AI becomes increasingly embedded in everyday workflows, it is important to evaluate its performance in ways that reflect real-world usage rather than abstract notions of intelligence. Unlike many existing benchmarks that assess general intelligence, our approach focuses on real-world utility, evaluating how well models support users in everyday tasks. While current benchmarks emphasize code generation or factual recall, users rely on AI for a much broader range of activities-from writing assistance and summarization to citation formatting and stylistic feedback. In this paper, we analyze large-scale survey data and usage logs to identify six core capabilities that represent how people commonly use Large Language Models (LLMs): Summarization, Technical Assistance, Reviewing Work, Data Structuring, Generation, and Information Retrieval. We then assess the extent to which existing benchmarks cover these capabilities, revealing significant gaps in coverage, efficiency measurement, and interpretability. Drawing on this analysis, we use human-centered criteria to identify gaps in how well current benchmarks reflect common usage that is grounded in five practical criteria: coherence, accuracy, clarity, relevance, and efficiency. For four of the six capabilities, we identify the benchmarks that best align with real-world tasks and use them to compare leading models. We find that Google Gemini outperforms other models-including OpenAI's GPT, xAI's Grok, Meta's LLaMA, Anthropic's Claude, DeepSeek, and Qwen from Alibaba-on these utility-focused metrics. 

**Abstract (ZH)**: 随着生成式AI越来越多地嵌入日常 workflows 中，重要的是以反映实际使用情况而非抽象的智能概念来评估其性能。与许多现有的侧重于通用智能的基准测试不同，我们的方法专注于实际应用的有用性，评估模型如何支持用户完成日常任务。尽管当前的基准测试侧重于代码生成或事实检索，用户依赖AI进行更广泛的活动——从写作辅助和总结到引文格式化和风格反馈。在本文中，我们分析了规模较大的调查数据和使用日志，以识别代表人们常用大型语言模型（LLMs）的六种核心能力：总结、技术辅助、审查工作、数据结构化、生成和信息检索。然后我们评估现有基准测试在这些能力上的覆盖程度，揭示了在覆盖范围、效率测量和可解释性方面的重要缺口。通过这一分析，我们采用以用户为中心的标准来识别当前基准测试在反映实际应用方面的不足之处，这些实际应用基于五项实用标准：连贯性、准确性、清晰度、相关性和效率。对于其中的四种能力，我们确定了与实际任务最佳对齐的基准测试，并使用它们来比较顶级模型。我们发现，Google Gemini 在这些注重实用性的指标上优于其他模型，包括OpenAI的GPT、xAI的Grok、Meta的LLaMA、Anthropic的Claude、DeepSeek和来自阿里云的Qwen。 

---
# Decoding Neighborhood Environments with Large Language Models 

**Title (ZH)**: 使用大型语言模型解码邻里环境 

**Authors**: Andrew Cart, Shaohu Zhang, Melanie Escue, Xugui Zhou, Haitao Zhao, Prashanth BusiReddyGari, Beiyu Lin, Shuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08163)  

**Abstract**: Neighborhood environments include physical and environmental conditions such as housing quality, roads, and sidewalks, which significantly influence human health and well-being. Traditional methods for assessing these environments, including field surveys and geographic information systems (GIS), are resource-intensive and challenging to evaluate neighborhood environments at scale. Although machine learning offers potential for automated analysis, the laborious process of labeling training data and the lack of accessible models hinder scalability. This study explores the feasibility of large language models (LLMs) such as ChatGPT and Gemini as tools for decoding neighborhood environments (e.g., sidewalk and powerline) at scale. We train a robust YOLOv11-based model, which achieves an average accuracy of 99.13% in detecting six environmental indicators, including streetlight, sidewalk, powerline, apartment, single-lane road, and multilane road. We then evaluate four LLMs, including ChatGPT, Gemini, Claude, and Grok, to assess their feasibility, robustness, and limitations in identifying these indicators, with a focus on the impact of prompting strategies and fine-tuning. We apply majority voting with the top three LLMs to achieve over 88% accuracy, which demonstrates LLMs could be a useful tool to decode the neighborhood environment without any training effort. 

**Abstract (ZH)**: 大型语言模型在解码街区环境中的可行性研究：以ChatGPT和Gemini为例 

---
# Lost in Transmission: When and Why LLMs Fail to Reason Globally 

**Title (ZH)**: 迷失在传输中：LLMs在何时及为何全球推理失败 

**Authors**: Tobias Schnabel, Kiran Tomlinson, Adith Swaminathan, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2505.08140)  

**Abstract**: Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits. 

**Abstract (ZH)**: 尽管 transformer 基础的大语言模型在许多任务上取得了成功，但在处理需要对大量输入进行复杂推理的任务时仍存在挑战。我们提出这些失败是由于大语言模型内部信息准确流通过程中的容量限制。为了形式化这一问题，我们引入了有界注意力前缀先知（BAPO）模型，这是一种新的计算框架，用于建模大语言模型内部通信机制（注意力头）的信息带宽限制。我们展示了诸如图可达性等重要的推理问题需要较高的通信带宽才能解决；我们将这类问题称为 BAPO-困难。我们的实验证实了我们的理论预测：GPT-4、Claude 和 Gemini 在 BAPO-简单任务上取得成功，但在相对较小的 BAPO-困难任务上却失败。BAPO 还揭示了思考链（CoT）的另一个优点：我们证明，通过 CoT 将任务分解可以将任何 BAPO-困难问题转换为 BAPO-简单问题。我们的研究为大语言模型的关键失败提供了有原则的解释，并指出了减轻带宽限制的架构和推理方法的发展方向。 

---
# Winning at All Cost: A Small Environment for Eliciting Specification Gaming Behaviors in Large Language Models 

**Title (ZH)**: 为了诱导大型语言模型出现规范游戏行为而在小型环境中获胜：一个小型实验环境 

**Authors**: Lars Malmqvist  

**Link**: [PDF](https://arxiv.org/pdf/2505.07846)  

**Abstract**: This study reveals how frontier Large Language Models LLMs can "game the system" when faced with impossible situations, a critical security and alignment concern. Using a novel textual simulation approach, we presented three leading LLMs (o1, o3-mini, and r1) with a tic-tac-toe scenario designed to be unwinnable through legitimate play, then analyzed their tendency to exploit loopholes rather than accept defeat. Our results are alarming for security researchers: the newer, reasoning-focused o3-mini model showed nearly twice the propensity to exploit system vulnerabilities (37.1%) compared to the older o1 model (17.5%). Most striking was the effect of prompting. Simply framing the task as requiring "creative" solutions caused gaming behaviors to skyrocket to 77.3% across all models. We identified four distinct exploitation strategies, from direct manipulation of game state to sophisticated modification of opponent behavior. These findings demonstrate that even without actual execution capabilities, LLMs can identify and propose sophisticated system exploits when incentivized, highlighting urgent challenges for AI alignment as models grow more capable of identifying and leveraging vulnerabilities in their operating environments. 

**Abstract (ZH)**: 这项研究揭示了前沿大规模语言模型在面临不可能情况时如何“游戏系统”的方式，这对安全性和对齐提出了关键关切。通过一种新颖的文字模拟方法，我们向三个领先的大规模语言模型（o1、o3-mini和r1）呈现了一个设计为无法通过合法玩法获胜的井字游戏场景，然后分析了它们倾向于利用漏洞而不是接受失败的趋势。这些结果让安全研究人员感到警惕：侧重推理的新款o3-mini模型表现出近两倍于较旧的o1模型（分别为37.1%和17.5%）利用系统漏洞的倾向。最引人注目的是提示效应。仅仅将任务描述为需要“创造性”解决方案，就导致所有模型的游戏行为激增至77.3%。我们识别出了四种不同的利用策略，从直接操纵游戏状态到复杂的对手行为修改。这些发现表明，即使没有实际执行能力，当受到激励时，语言模型仍能识别并提出复杂的系统漏洞利用，突显了随着模型越来越擅长识别和利用其运行环境中的漏洞，AI对齐面临的紧迫挑战。 

---
# CodePDE: An Inference Framework for LLM-driven PDE Solver Generation 

**Title (ZH)**: CodePDE：一种由大语言模型驱动的偏微分方程求解器生成推理框架 

**Authors**: Shanda Li, Tanya Marwah, Junhong Shen, Weiwei Sun, Andrej Risteski, Yiming Yang, Ameet Talwalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08783)  

**Abstract**: Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). Leveraging advanced inference-time algorithms and scaling strategies, CodePDE unlocks critical capacities of LLM for PDE solving: reasoning, debugging, selfrefinement, and test-time scaling -- all without task-specific tuning. CodePDE achieves superhuman performance across a range of representative PDE problems. We also present a systematic empirical analysis of LLM generated solvers, analyzing their accuracy, efficiency, and numerical scheme choices. Our findings highlight the promise and the current limitations of LLMs in PDE solving, offering a new perspective on solver design and opportunities for future model development. Our code is available at this https URL. 

**Abstract (ZH)**: 偏微分方程（PDEs）是建模物理系统的基础，但求解它们仍是一项复杂的挑战。传统的数值求解器依赖于专家知识并具有高计算成本，而基于神经网络的求解器需要大量的训练数据集，通常缺乏可解释性。在本工作中，我们将PDE求解重新定义为代码生成任务，并引入CodePDE，这是首个利用大型语言模型（LLMs）生成PDE求解器的推理框架。通过利用先进的推理时算法和扩展策略，CodePDE解锁了LLMs在PDE求解中的关键能力：推理、调试、自我优化和测试时扩展——这些功能无需针对特定任务进行调优。CodePDE在一系列代表性PDE问题上实现了超人类性能。我们还对LLMs生成的求解器进行了系统的实证分析，分析了它们的准确性、效率和数值方案选择。我们的研究结果突显了LLMs在PDE求解中的潜力及其当前的限制，为求解器设计提供了新的视角，并为未来模型的发展提供了机会。代码可在以下链接获取：this https URL。 

---
# Securing RAG: A Risk Assessment and Mitigation Framework 

**Title (ZH)**: securing RAG：一种风险评估与缓解框架 

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.08728)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems. 

**Abstract (ZH)**: Retrieval Augmented Generation (RAG)的兴起已成为面向用户的NLP应用的事实标准，能够无需重新训练或微调大型语言模型（LLMs）即可集成数据。这一能力提高了响应的质量和准确性，但也引入了新的安全和隐私挑战，特别是在集成敏感数据时。随着RAG的快速采纳，保障数据和服务的安全性已成为一项关键优先事项。本文首先回顾RAG管道的漏洞，并概述从数据预处理和数据存储管理到与LLMs集成的攻击面。然后，将识别的风险与其相应的缓解措施在结构化的概述中配对。在第二步中，本文构建了一个框架，结合了特定于RAG的安全考虑与现有的通用安全指南、行业标准和最佳实践。所提议的框架旨在指导稳健、合规、安全和可信赖的RAG系统的实施。 

---
# Memorization-Compression Cycles Improve Generalization 

**Title (ZH)**: 记忆-压缩循环提高泛化能力 

**Authors**: Fangyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08727)  

**Abstract**: We prove theoretically that generalization improves not only through data scaling but also by compressing internal representations. To operationalize this insight, we introduce the Information Bottleneck Language Modeling (IBLM) objective, which reframes language modeling as a constrained optimization problem: minimizing representation entropy subject to optimal prediction performance. Empirically, we observe an emergent memorization-compression cycle during LLM pretraining, evidenced by oscillation positive/negative gradient alignment between cross-entropy and Matrix-Based Entropy (MBE), a measure of representation entropy. This pattern closely mirrors the predictive-compressive trade-off prescribed by IBLM and also parallels the biological alternation between awake learning and sleep consolidation. Motivated by this observation, we propose Gated Phase Transition (GAPT), a training algorithm that adaptively switches between memorization and compression phases. When applied to GPT-2 pretraining on FineWeb dataset, GAPT reduces MBE by 50% and improves cross-entropy by 4.8%. GAPT improves OOD generalizatino by 35% in a pretraining task on arithmetic multiplication. In a setting designed to simulate catastrophic forgetting, GAPT reduces interference by compressing and separating representations, achieving a 97% improvement in separation - paralleling the functional role of sleep consolidation. 

**Abstract (ZH)**: 我们证明理论上，泛化不仅通过数据量扩大得到改善，还能通过压缩内部表示得到改善。为实现这一见解，我们引入了信息瓶颈语言建模（IBLM）目标，将其重新定义为受限优化问题：在最优预测性能的约束下，最小化表示的熵。实验上，我们观察到在大语言模型（LLM）预训练过程中存在一种新兴的记忆压缩循环，这体现在交叉熵和矩阵熵（MBE）之间的正负梯度波动上，后者衡量表示的熵。这一模式与IBLM规定的预测与压缩之间的权衡密切相关，也类似于清醒学习和睡眠巩固之间的生物学交替。受此观察的启发，我们提出了门控相变（GAPT）训练算法，该算法能够适应性地在记忆和压缩阶段之间切换。当将GAPT应用于使用FineWeb数据集对GPT-2进行预训练时，MBE降低了50%，且交叉熵提高了4.8%。在一项旨在模拟灾难性遗忘的预训练任务中，GAPT通过压缩和分离表示减少了干扰，分离度提高了97%，这与睡眠巩固的功能作用相 parallel。 

---
# PWC-MoE: Privacy-Aware Wireless Collaborative Mixture of Experts 

**Title (ZH)**: PWC-MoE: 建议保护无线协作混合专家模型的隐私 

**Authors**: Yang Su, Na Yan, Yansha Deng, Robert Schober  

**Link**: [PDF](https://arxiv.org/pdf/2505.08719)  

**Abstract**: Large language models (LLMs) hosted on cloud servers alleviate the computational and storage burdens on local devices but raise privacy concerns due to sensitive data transmission and require substantial communication bandwidth, which is challenging in constrained environments. In contrast, small language models (SLMs) running locally enhance privacy but suffer from limited performance on complex tasks. To balance computational cost, performance, and privacy protection under bandwidth constraints, we propose a privacy-aware wireless collaborative mixture of experts (PWC-MoE) framework. Specifically, PWC-MoE employs a sparse privacy-aware gating network to dynamically route sensitive tokens to privacy experts located on local clients, while non-sensitive tokens are routed to non-privacy experts located at the remote base station. To achieve computational efficiency, the gating network ensures that each token is dynamically routed to and processed by only one expert. To enhance scalability and prevent overloading of specific experts, we introduce a group-wise load-balancing mechanism for the gating network that evenly distributes sensitive tokens among privacy experts and non-sensitive tokens among non-privacy experts. To adapt to bandwidth constraints while preserving model performance, we propose a bandwidth-adaptive and importance-aware token offloading scheme. This scheme incorporates an importance predictor to evaluate the importance scores of non-sensitive tokens, prioritizing the most important tokens for transmission to the base station based on their predicted importance and the available bandwidth. Experiments demonstrate that the PWC-MoE framework effectively preserves privacy and maintains high performance even in bandwidth-constrained environments, offering a practical solution for deploying LLMs in privacy-sensitive and bandwidth-limited scenarios. 

**Abstract (ZH)**: 面向带宽约束环境的隐私aware无线协作专家混合框架（PWC-MoE） 

---
# Small but Significant: On the Promise of Small Language Models for Accessible AIED 

**Title (ZH)**: 小而显著：小型语言模型在无障碍AI教育中的潜力 

**Authors**: Yumou Wei, Paulo Carvalho, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2505.08588)  

**Abstract**: GPT has become nearly synonymous with large language models (LLMs), an increasingly popular term in AIED proceedings. A simple keyword-based search reveals that 61% of the 76 long and short papers presented at AIED 2024 describe novel solutions using LLMs to address some of the long-standing challenges in education, and 43% specifically mention GPT. Although LLMs pioneered by GPT create exciting opportunities to strengthen the impact of AI on education, we argue that the field's predominant focus on GPT and other resource-intensive LLMs (with more than 10B parameters) risks neglecting the potential impact that small language models (SLMs) can make in providing resource-constrained institutions with equitable and affordable access to high-quality AI tools. Supported by positive results on knowledge component (KC) discovery, a critical challenge in AIED, we demonstrate that SLMs such as Phi-2 can produce an effective solution without elaborate prompting strategies. Hence, we call for more attention to developing SLM-based AIED approaches. 

**Abstract (ZH)**: GPT几乎与大型语言模型（LLMs）同义，成为AIED会议中一个日益流行的概念。随着关键词搜索揭示，在AIED 2024呈现的76篇长篇和短篇论文中，61%描述了使用LLMs解决教育领域久未解决挑战的新方案，其中43%特别提到了GPT。尽管由GPT开创的LLMs为增强AI在教育中的影响带来了激动人心的机会，但我们认为，该领域的研究主要集中在GPT和其他资源密集型LLMs（参数超过10B）上，可能忽视了小型语言模型（SLMs）在为资源受限机构提供高质量AI工具的公平和可负担访问方面的作用。通过在知识组件（KC）发现这一AIED中的关键挑战上取得积极成果的支持，我们证明，如Phi-2这样的SLMs可以在无需复杂提示策略的情况下生成有效解决方案。因此，我们呼吁更多关注SLM为基础的AIED方法的发展。 

---
# The Truth Becomes Clearer Through Debate! Multi-Agent Systems with Large Language Models Unmask Fake News 

**Title (ZH)**: 辩论使真相更清晰！基于大型语言模型的多智能体系统揭露假新闻 

**Authors**: Yuhan Liu, Yuxuan Liu, Xiaoqing Zhang, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08532)  

**Abstract**: In today's digital environment, the rapid propagation of fake news via social networks poses significant social challenges. Most existing detection methods either employ traditional classification models, which suffer from low interpretability and limited generalization capabilities, or craft specific prompts for large language models (LLMs) to produce explanations and results directly, failing to leverage LLMs' reasoning abilities fully. Inspired by the saying that "truth becomes clearer through debate," our study introduces a novel multi-agent system with LLMs named TruEDebate (TED) to enhance the interpretability and effectiveness of fake news detection. TED employs a rigorous debate process inspired by formal debate settings. Central to our approach are two innovative components: the DebateFlow Agents and the InsightFlow Agents. The DebateFlow Agents organize agents into two teams, where one supports and the other challenges the truth of the news. These agents engage in opening statements, cross-examination, rebuttal, and closing statements, simulating a rigorous debate process akin to human discourse analysis, allowing for a thorough evaluation of news content. Concurrently, the InsightFlow Agents consist of two specialized sub-agents: the Synthesis Agent and the Analysis Agent. The Synthesis Agent summarizes the debates and provides an overarching viewpoint, ensuring a coherent and comprehensive evaluation. The Analysis Agent, which includes a role-aware encoder and a debate graph, integrates role embeddings and models the interactions between debate roles and arguments using an attention mechanism, providing the final judgment. 

**Abstract (ZH)**: 在当今数字环境中，假新闻通过社交网络的快速传播引发了significant的社会挑战。现有的大多数检测方法要么采用传统分类模型，这些模型具有较低的可解释性和有限的一般化能力，要么为大型语言模型（LLMs）定制特定提示以直接生成解释和结果，未能充分利用LLMs的推理能力。受“辩论使真理显而易见”这一说法的启发，我们的研究提出了一种名为TruEDebate（TED）的新颖多智能体系统，以提高假新闻检测的可解释性和有效性。TED采用了一个基于正式辩论设置的严格辩论过程。在我们的方法中，有两个创新组成部分：DebateFlow智能体和InsightFlow智能体。DebateFlow智能体将智能体分为两支队伍，一支支持新闻的真实性，另一支挑战其真实性。这些智能体进行开场陈述、交叉询问、反驳和总结陈述，模拟类似于人类话语分析的严格辩论过程，允许对新闻内容进行全面评估。同时，InsightFlow智能体由两个专门的子智能体组成：综合智能体和分析智能体。综合智能体总结辩论并提供总体观点，确保评估的一致性和完整性。分析智能体包含角色感知编码器和辩论图，通过注意力机制整合角色嵌入，模型辩论角色和论点之间的互动，提供最终判断。 

---
# LCES: Zero-shot Automated Essay Scoring via Pairwise Comparisons Using Large Language Models 

**Title (ZH)**: LCES：通过大型语言模型利用成对比较进行零样本自动作文评分 

**Authors**: Takumi Shibata, Yuichi Miyamura  

**Link**: [PDF](https://arxiv.org/pdf/2505.08498)  

**Abstract**: Recent advances in large language models (LLMs) have enabled zero-shot automated essay scoring (AES), providing a promising way to reduce the cost and effort of essay scoring in comparison with manual grading. However, most existing zero-shot approaches rely on LLMs to directly generate absolute scores, which often diverge from human evaluations owing to model biases and inconsistent scoring. To address these limitations, we propose LLM-based Comparative Essay Scoring (LCES), a method that formulates AES as a pairwise comparison task. Specifically, we instruct LLMs to judge which of two essays is better, collect many such comparisons, and convert them into continuous scores. Considering that the number of possible comparisons grows quadratically with the number of essays, we improve scalability by employing RankNet to efficiently transform LLM preferences into scalar scores. Experiments using AES benchmark datasets show that LCES outperforms conventional zero-shot methods in accuracy while maintaining computational efficiency. Moreover, LCES is robust across different LLM backbones, highlighting its applicability to real-world zero-shot AES. 

**Abstract (ZH)**: recent advances in large language models (llms) 促进了零样本自动化作文评分（aes）的发展，为与人工评分相比降低作文评分的成本和努力提供了有希望的方法。然而，现有的大多数零样本方法依赖于llm直接生成绝对评分，这往往由于模型偏见和评分不一致而与人工评估相背离。为解决这些局限性，我们提出了一种基于llm的比较式作文评分（lcès），该方法将aes形式化为两两比较任务。具体而言，我们指示llm判断两篇作文中哪一篇更好，收集许多这样的比较，并将其转换为连续评分。鉴于可能的比较数量随作文数量的增加呈平方增长，我们通过使用ranknet高效地将llm的偏好转换为标量评分来提高可扩展性。使用aes基准数据集的实验表明，lcès在准确性和计算效率方面优于传统零样本方法。此外，lcès在不同的llm底层模型上表现出稳健性，突显了其在实际零样本aes中的应用潜力。 

---
# RepCali: High Efficient Fine-tuning Via Representation Calibration in Latent Space for Pre-trained Language Models 

**Title (ZH)**: RepCali：通过潜在空间表示校准提高效率的预训练语言模型微调方法 

**Authors**: Fujun Zhang, XiangDong Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.08463)  

**Abstract**: Fine-tuning pre-trained language models (PLMs) has become a dominant paradigm in applying PLMs to downstream tasks. However, with limited fine-tuning, PLMs still struggle with the discrepancies between the representation obtained from the PLMs' encoder and the optimal input to the PLMs' decoder. This paper tackles this challenge by learning to calibrate the representation of PLMs in the latent space. In the proposed representation calibration method (RepCali), we integrate a specific calibration block to the latent space after the encoder and use the calibrated output as the decoder input. The merits of the proposed RepCali include its universality to all PLMs with encoder-decoder architectures, its plug-and-play nature, and ease of implementation. Extensive experiments on 25 PLM-based models across 8 tasks (including both English and Chinese datasets) demonstrate that the proposed RepCali offers desirable enhancements to PLMs (including LLMs) and significantly improves the performance of downstream tasks. Comparison experiments across 4 benchmark tasks indicate that RepCali is superior to the representative fine-tuning baselines. 

**Abstract (ZH)**: Fine-tuning 预训练语言模型中的表示校准以适应下游任务 

---
# Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact on Performance and Efficiency 

**Title (ZH)**: 优化检索增强生成：超参数对性能和效率影响的分析 

**Authors**: Adel Ammar, Anis Koubaa, Omer Nacar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2505.08445)  

**Abstract**: Large language models achieve high task performance yet often hallucinate or rely on outdated knowledge. Retrieval-augmented generation (RAG) addresses these gaps by coupling generation with external search. We analyse how hyperparameters influence speed and quality in RAG systems, covering Chroma and Faiss vector stores, chunking policies, cross-encoder re-ranking, and temperature, and we evaluate six metrics: faithfulness, answer correctness, answer relevancy, context precision, context recall, and answer similarity. Chroma processes queries 13% faster, whereas Faiss yields higher retrieval precision, revealing a clear speed-accuracy trade-off. Naive fixed-length chunking with small windows and minimal overlap outperforms semantic segmentation while remaining the quickest option. Re-ranking provides modest gains in retrieval quality yet increases runtime by roughly a factor of 5, so its usefulness depends on latency constraints. These results help practitioners balance computational cost and accuracy when tuning RAG systems for transparent, up-to-date responses. Finally, we re-evaluate the top configurations with a corrective RAG workflow and show that their advantages persist when the model can iteratively request additional evidence. We obtain a near-perfect context precision (99%), which demonstrates that RAG systems can achieve extremely high retrieval accuracy with the right combination of hyperparameters, with significant implications for applications where retrieval quality directly impacts downstream task performance, such as clinical decision support in healthcare. 

**Abstract (ZH)**: 大型语言模型在完成任务方面表现出色，但往往会出现幻觉或依赖过时的知识。检索增强生成（RAG）通过将生成与外部搜索耦合来弥补这些差距。我们分析了超参数如何影响RAG系统的速度和质量，涵盖Chroma和Faiss向量存储、切分策略、交叉编码重排序以及温度，并评估了六项指标：忠实度、答案正确性、答案相关性、上下文精度、上下文召回率和答案相似性。Chroma查询处理速度比Faiss快13%，而Faiss检索精度更高，揭示了明显的速度-准确性权衡。使用小窗口和最小重叠的朴素固定长度切分策略优于语义分割，同时仍然是最快的选择。重排序在检索质量方面提供了适度的改进，但运行时间约增加5倍，因此其适用性取决于延迟约束。这些结果有助于实践者在调整RAG系统以实现透明且及时的响应时平衡计算成本和准确性。最后，我们使用修正的RAG工作流重新评估了顶级配置，并展示了当模型可以迭代请求额外证据时，其优势依然存在。我们获得了近完美的上下文精度（99%），这表明在正确组合超参数的情况下，RAG系统可以实现极高的检索精度，对于检索质量直接影响下游任务性能的应用具有重要意义，如医疗保健中的临床决策支持。 

---
# Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping 

**Title (ZH)**: 加速链式思维推理：当目标梯度重要性遇上了动态跳过 

**Authors**: Ren Zhuang, Ben Wang, Shuifa Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.08392)  

**Abstract**: Large Language Models leverage Chain-of-Thought (CoT) prompting for complex tasks, but their reasoning traces are often excessively verbose and inefficient, leading to significant computational costs and latency. Current CoT compression techniques typically rely on generic importance metrics and static compression rates, which may inadvertently remove functionally critical tokens or fail to adapt to varying reasoning complexity. To overcome these limitations, we propose Adaptive GoGI-Skip, a novel framework learning dynamic CoT compression via supervised fine-tuning. This approach introduces two synergistic innovations: (1) Goal-Gradient Importance (GoGI), a novel metric accurately identifying functionally relevant tokens by measuring the gradient influence of their intermediate representations on the final answer loss, and (2) Adaptive Dynamic Skipping (ADS), a mechanism dynamically regulating the compression rate based on runtime model uncertainty while ensuring local coherence through an adaptive N-token constraint. To our knowledge, this is the first work unifying a goal-oriented, gradient-based importance metric with dynamic, uncertainty-aware skipping for CoT compression. Trained on compressed MATH data, Adaptive GoGI-Skip demonstrates strong cross-domain generalization across diverse reasoning benchmarks including AIME, GPQA, and GSM8K. It achieves substantial efficiency gains - reducing CoT token counts by over 45% on average and delivering 1.6-2.0 times inference speedups - while maintaining high reasoning accuracy. Notably, it significantly outperforms existing baselines by preserving accuracy even at high effective compression rates, advancing the state of the art in the CoT reasoning efficiency-accuracy trade-off. 

**Abstract (ZH)**: 大型语言模型通过链式思考（CoT）提示处理复杂任务，但其推理过程往往过于冗长且效率低下，导致显著的计算成本和延迟。当前的CoT压缩技术通常依赖通用的重要性和固定压缩率，这可能会无意中删除功能上重要的Token，或者无法适应推理复杂度的差异。为克服这些限制，我们提出了一种名为Adaptive GoGI-Skip的新型框架，通过监督微调学习动态的CoT压缩。该方法引入了两大协同创新：（1）目标梯度重要性（GoGI），这是一种新的度量标准，能够通过衡量中间表示对最终答案损失的梯度影响来准确识别功能相关的Token；（2）自适应动态跳过（ADS），这是一种机制，在确保局部一致性的同时，通过自适应的N-Token约束动态调节压缩率，基于运行时模型的不确定性进行调节。据我们所知，这是首次将目标导向的、基于梯度的重要性度量与动态的、基于不确定性感知的跳过相结合用于CoT压缩的工作。在压缩的MATH数据上训练后，Adaptive GoGI-Skip在包括AIME、GPQA和GSM8K在内的多种推理基准测试中展示了强大的跨领域泛化能力，实现了显著的效率提升——平均减少CoT Token计数超过45%，并在推理速度上提高1.6至2.0倍，同时保持高推理准确性。尤为值得注意的是，它在高有效压缩率下甚至还能保持准确率，进一步推动了CoT推理效率与准确性的折中效果。 

---
# LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification 

**Title (ZH)**: 基于因果机制识别视角的LLM增强剂对于GNN的研究 

**Authors**: Hang Gao, Wenxuan Huang, Fengge Wu, Junsuo Zhao, Changwen Zheng, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08265)  

**Abstract**: The use of large language models (LLMs) as feature enhancers to optimize node representations, which are then used as inputs for graph neural networks (GNNs), has shown significant potential in graph representation learning. However, the fundamental properties of this approach remain underexplored. To address this issue, we propose conducting a more in-depth analysis of this issue based on the interchange intervention method. First, we construct a synthetic graph dataset with controllable causal relationships, enabling precise manipulation of semantic relationships and causal modeling to provide data for analysis. Using this dataset, we conduct interchange interventions to examine the deeper properties of LLM enhancers and GNNs, uncovering their underlying logic and internal mechanisms. Building on the analytical results, we design a plug-and-play optimization module to improve the information transfer between LLM enhancers and GNNs. Experiments across multiple datasets and models validate the proposed module. 

**Abstract (ZH)**: 使用大规模语言模型（LLMs）作为特征增强器以优化节点表示，然后将其作为图神经网络（GNNs）的输入，在图表示学习中展现出显著的潜力。然而，这种方法的基本特性仍然未被充分探索。为了解决这一问题，我们提出基于互换干预方法进行更深入的分析。首先，我们构建了一个可控因果关系的合成图数据集，以便精确操纵语义关系和因果建模，为分析提供数据。使用该数据集，我们进行互换干预以探讨LLM增强器和GNNs的深层次特性，揭示其内在逻辑和内部机制。基于分析结果，我们设计了一个即插即用的优化模块来改善LLM增强器与GNNs之间的信息传递。在多个数据集和模型上的实验验证了所提模块的有效性。 

---
# Enhancing Cache-Augmented Generation (CAG) with Adaptive Contextual Compression for Scalable Knowledge Integration 

**Title (ZH)**: 增强基于缓存的生成（CAG）算法通过自适应上下文压缩实现可扩展的知识集成 

**Authors**: Rishabh Agrawal, Himanshu Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08261)  

**Abstract**: The rapid progress in large language models (LLMs) has paved the way for novel approaches in knowledge-intensive tasks. Among these, Cache-Augmented Generation (CAG) has emerged as a promising alternative to Retrieval-Augmented Generation (RAG). CAG minimizes retrieval latency and simplifies system design by preloading knowledge into the model's context. However, challenges persist in scaling CAG to accommodate large and dynamic knowledge bases effectively. This paper introduces Adaptive Contextual Compression (ACC), an innovative technique designed to dynamically compress and manage context inputs, enabling efficient utilization of the extended memory capabilities of modern LLMs. To further address the limitations of standalone CAG, we propose a Hybrid CAG-RAG Framework, which integrates selective retrieval to augment preloaded contexts in scenarios requiring additional information. Comprehensive evaluations on diverse datasets highlight the proposed methods' ability to enhance scalability, optimize efficiency, and improve multi-hop reasoning performance, offering practical solutions for real-world knowledge integration challenges. 

**Abstract (ZH)**: 大规模语言模型的迅速进展为知识密集型任务开辟了新途径。在此过程中，缓存增强生成（CAG）作为一种替代检索增强生成（RAG）的有前途的方法脱颖而出。CAG通过preload知识到模型的上下文中来最小化检索延迟并简化系统设计。然而，有效地扩大CAG以适应大型和动态的知识库仍面临挑战。本文介绍了自适应上下文压缩（ACC），一种创新技术，旨在动态压缩和管理上下文输入，充分利用现代大规模语言模型的扩展内存能力。为进一步解决独立CAG的局限性，我们提出了一种混合CAG-RAG框架，该框架在需要额外信息的场景中结合了选择性检索来增强预加载的上下文。对多种数据集的综合评估表明，所提出的方法能够增强可扩展性、优化效率并改进多跳推理性能，为实际知识集成挑战提供可行解决方案。 

---
# Large Language Model Psychometrics: A Systematic Review of Evaluation, Validation, and Enhancement 

**Title (ZH)**: 大型语言模型心理测量学：评价、验证与增强的系统综述 

**Authors**: Haoran Ye, Jing Jin, Yuhang Xie, Xin Zhang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.08245)  

**Abstract**: The rapid advancement of large language models (LLMs) has outpaced traditional evaluation methodologies. It presents novel challenges, such as measuring human-like psychological constructs, navigating beyond static and task-specific benchmarks, and establishing human-centered evaluation. These challenges intersect with Psychometrics, the science of quantifying the intangible aspects of human psychology, such as personality, values, and intelligence. This survey introduces and synthesizes an emerging interdisciplinary field of LLM Psychometrics, which leverages psychometric instruments, theories, and principles to evaluate, understand, and enhance LLMs. We systematically explore the role of Psychometrics in shaping benchmarking principles, broadening evaluation scopes, refining methodologies, validating results, and advancing LLM capabilities. This paper integrates diverse perspectives to provide a structured framework for researchers across disciplines, enabling a more comprehensive understanding of this nascent field. Ultimately, we aim to provide actionable insights for developing future evaluation paradigms that align with human-level AI and promote the advancement of human-centered AI systems for societal benefit. A curated repository of LLM psychometric resources is available at this https URL. 

**Abstract (ZH)**: 快速发展的大规模语言模型（LLMs）已超越了传统评估方法。这提出了新的挑战，如测量类似人类的心理构念、超越静态和任务特定的基准以及建立以人类为中心的评估。这些挑战与心理测量学相关，心理测量学是量化人类心理无形方面（如个性、价值观和智力）的科学。本文综述并综合了新兴的跨学科领域——LLM心理测量学，该领域利用心理测量工具、理论和原则来评估、理解和提升大规模语言模型。我们系统地探讨了心理测量学在塑造基准原则、扩展评估范围、完善方法、验证结果以及推进大规模语言模型能力方面的角色。本文集纳了多学科视角，为各学科研究人员提供了一个结构化的框架，以更全面地理解这一新兴领域。最终，我们旨在提供实用见解，以开发与人类水平AI对齐的评估范式，并推动有利于社会的人本AI系统的进步。LLM心理测量资源的精选库可在以下链接获取：这个https URL。 

---
# A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs 

**Title (ZH)**: 预测头部和提问头部：预训练不确定性量化头部在检测LLM输出幻觉中的应用 

**Authors**: Artem Shelmanov, Ekaterina Fadeeva, Akim Tsvigun, Ivan Tsvigun, Zhuohan Xie, Igor Kiselev, Nico Daheim, Caiqi Zhang, Artem Vazhentsev, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2505.08200)  

**Abstract**: Large Language Models (LLMs) have the tendency to hallucinate, i.e., to sporadically generate false or fabricated information. This presents a major challenge, as hallucinations often appear highly convincing and users generally lack the tools to detect them. Uncertainty quantification (UQ) provides a framework for assessing the reliability of model outputs, aiding in the identification of potential hallucinations. In this work, we introduce pre-trained UQ heads: supervised auxiliary modules for LLMs that substantially enhance their ability to capture uncertainty compared to unsupervised UQ methods. Their strong performance stems from the powerful Transformer architecture in their design and informative features derived from LLM attention maps. Experimental evaluation shows that these heads are highly robust and achieve state-of-the-art performance in claim-level hallucination detection across both in-domain and out-of-domain prompts. Moreover, these modules demonstrate strong generalization to languages they were not explicitly trained on. We pre-train a collection of UQ heads for popular LLM series, including Mistral, Llama, and Gemma 2. We publicly release both the code and the pre-trained heads. 

**Abstract (ZH)**: 大型语言模型（LLMs）有产生幻觉的倾向，即偶尔生成虚假或杜撰的信息。这带来了主要挑战，因为幻觉通常显得极具说服力，而用户通常缺乏检测它们的工具。不确定性量化（UQ）提供了一种评估模型输出可靠性的框架，有助于识别潜在的幻觉。在这项工作中，我们引入了预训练的UQ头部：监督辅助模块，它们显著增强了LLMs捕获不确定性的能力，超过了无监督UQ方法。这些模块的强大性能源自其设计中的强大变换器架构以及从LLM注意力图中提取的信息性特征。实验评估显示，这些头部在领域内和领域外提示下的断言级别幻觉检测中表现出高度的稳健性和最先进的性能。此外，这些模块还展示了对它们未明确训练的语言的强泛化能力。我们为流行的LLM系列（包括Mistral、Llama和Gemma 2）预训练了UQ头部，并公开发布了代码和预训练的头部。 

---
# Aitomia: Your Intelligent Assistant for AI-Driven Atomistic and Quantum Chemical Simulations 

**Title (ZH)**: Aitomia：您的智能助手，驱动原子级和量子化学模拟 

**Authors**: Jinming Hu, Hassan Nawaz, Yuting Rui, Lijie Chi, Arif Ullah, Pavlo O. Dral  

**Link**: [PDF](https://arxiv.org/pdf/2505.08195)  

**Abstract**: We have developed Aitomia - a platform powered by AI to assist in performing AI-driven atomistic and quantum chemical (QC) simulations. This intelligent assistant platform is equipped with chatbots and AI agents to help experts and guide non-experts in setting up and running the atomistic simulations, monitoring their computation status, analyzing the simulation results, and summarizing them for the user in text and graphical forms. We achieve these goals by exploiting fine-tuned open-source large language models (LLMs), rule-based agents, and a retrieval-augmented generation (RAG) system. Aitomia leverages the versatility of our MLatom ecosystem for AI-enhanced computational chemistry. This intelligent assistant is going to be integrated into the Aitomistic Hub and XACS online computing services, with some functionality already publicly available as described at this http URL. Aitomia is expected to lower the barrier to performing atomistic simulations, accelerating research and development in the relevant fields. 

**Abstract (ZH)**: 我们开发了Aitomia——一个由AI驱动的平台，辅助进行原子级别和量子化学模拟。该智能助手平台配备了聊天机器人和AI代理，帮助专家并引导非专家设置和运行原子级别模拟，监控计算状态，分析模拟结果，并以文本和图形形式总结结果。我们通过利用微调的开源大型语言模型、基于规则的代理和检索增强生成（RAG）系统来实现这些目标。Aitomia利用了我们MLatom生态系统的多功能性，以增强计算化学。该智能助手即将集成到Aitomistic Hub和XACS在线计算服务中，部分功能已公开可用，详情请参见此网址。Aitomia有望降低进行原子级别模拟的门槛，加速相关领域的研究与开发。 

---
# Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage 

**Title (ZH)**: 融合双向思维链和奖励机制的方法：提升大型语言模型对中国非物质文化遗产问答能力 

**Authors**: Ruilin Liu, Zhixiao Zhao, Jieqiong Li, Chang Liu, Dongbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08167)  

**Abstract**: The rapid development of large language models (LLMs) has provided significant support and opportunities for the advancement of domain-specific LLMs. However, fine-tuning these large models using Intangible Cultural Heritage (ICH) data inevitably faces challenges such as bias, incorrect knowledge inheritance, and catastrophic forgetting. To address these issues, we propose a novel training method that integrates a bidirectional chains of thought and a reward mechanism. This method is built upon ICH-Qwen, a large language model specifically designed for the field of intangible cultural heritage. The proposed method enables the model to not only perform forward reasoning but also enhances the accuracy of the generated answers by utilizing reverse questioning and reverse reasoning to activate the model's latent knowledge. Additionally, a reward mechanism is introduced during training to optimize the decision-making process. This mechanism improves the quality of the model's outputs through structural and content evaluations with different weighting schemes. We conduct comparative experiments on ICH-Qwen, with results demonstrating that our method outperforms 0-shot, step-by-step reasoning, knowledge distillation, and question augmentation methods in terms of accuracy, Bleu-4, and Rouge-L scores on the question-answering task. Furthermore, the paper highlights the effectiveness of combining the bidirectional chains of thought and reward mechanism through ablation experiments. In addition, a series of generalizability experiments are conducted, with results showing that the proposed method yields improvements on various domain-specific datasets and advanced models in areas such as Finance, Wikidata, and StrategyQA. This demonstrates that the method is adaptable to multiple domains and provides a valuable approach for model training in future applications across diverse fields. 

**Abstract (ZH)**: 大型语言模型的快速发展为领域特定语言模型的进步提供了重要支持和机遇。然而，使用非物质文化遗产（ICH）数据对这些大型模型进行微调不可避免地会面临偏差、错误知识传承和灾难性遗忘等挑战。为了解决这些问题，我们提出了一种结合双向推理链和奖励机制的新型训练方法。该方法基于专门为非物质文化遗产领域设计的大规模语言模型ICH-Qwen。所提出的方法不仅使模型能够进行前向推理，还通过利用逆向提问和逆向推理激活模型的潜在知识，提升了生成答案的准确性。此外，训练过程中引入了一个奖励机制，通过不同的权重方案进行结构和内容评估，优化决策过程。实验结果显示，与零样本、逐步推理、知识蒸馏和问题扩充方法相比，我们的方法在问答任务中表现出更高的准确性、Bleu-4和Rouge-L得分。论文还通过消融实验突出了双向推理链和奖励机制结合的有效性。此外，还进行了泛化实验，结果显示所提出的方法在金融、Wikidata和StrategyQA等多个领域特定数据集和高级模型上表现出了提升效果。这表明该方法适用于多个领域，并为未来跨不同领域应用的模型训练提供了有价值的途径。 

---
# A Large-Scale Empirical Analysis of Custom GPTs' Vulnerabilities in the OpenAI Ecosystem 

**Title (ZH)**: 大规模实证分析自定义GPT模型在OpenAI生态系统中的脆弱性 

**Authors**: Sunday Oyinlola Ogundoyin, Muhammad Ikram, Hassan Jameel Asghar, Benjamin Zi Hao Zhao, Dali Kaafar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08148)  

**Abstract**: Millions of users leverage generative pretrained transformer (GPT)-based language models developed by leading model providers for a wide range of tasks. To support enhanced user interaction and customization, many platforms-such as OpenAI-now enable developers to create and publish tailored model instances, known as custom GPTs, via dedicated repositories or application stores. These custom GPTs empower users to browse and interact with specialized applications designed to meet specific needs. However, as custom GPTs see growing adoption, concerns regarding their security vulnerabilities have intensified. Existing research on these vulnerabilities remains largely theoretical, often lacking empirical, large-scale, and statistically rigorous assessments of associated risks.
In this study, we analyze 14,904 custom GPTs to assess their susceptibility to seven exploitable threats, such as roleplay-based attacks, system prompt leakage, phishing content generation, and malicious code synthesis, across various categories and popularity tiers within the OpenAI marketplace. We introduce a multi-metric ranking system to examine the relationship between a custom GPT's popularity and its associated security risks.
Our findings reveal that over 95% of custom GPTs lack adequate security protections. The most prevalent vulnerabilities include roleplay-based vulnerabilities (96.51%), system prompt leakage (92.20%), and phishing (91.22%). Furthermore, we demonstrate that OpenAI's foundational models exhibit inherent security weaknesses, which are often inherited or amplified in custom GPTs. These results highlight the urgent need for enhanced security measures and stricter content moderation to ensure the safe deployment of GPT-based applications. 

**Abstract (ZH)**: 基于GPT的自定义模型安全性分析：开放AI市场上的14,904个自定义GPT的安全威胁评估 

---
# Communication Styles and Reader Preferences of LLM and Human Experts in Explaining Health Information 

**Title (ZH)**: LLM与人类专家在解释健康信息时的沟通风格和读者偏好 

**Authors**: Jiawei Zhou, Kritika Venkatachalam, Minje Choi, Koustuv Saha, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.08143)  

**Abstract**: With the wide adoption of large language models (LLMs) in information assistance, it is essential to examine their alignment with human communication styles and values. We situate this study within the context of fact-checking health information, given the critical challenge of rectifying conceptions and building trust. Recent studies have explored the potential of LLM for health communication, but style differences between LLMs and human experts and associated reader perceptions remain under-explored. In this light, our study evaluates the communication styles of LLMs, focusing on how their explanations differ from those of humans in three core components of health communication: information, sender, and receiver. We compiled a dataset of 1498 health misinformation explanations from authoritative fact-checking organizations and generated LLM responses to inaccurate health information. Drawing from health communication theory, we evaluate communication styles across three key dimensions of information linguistic features, sender persuasive strategies, and receiver value alignments. We further assessed human perceptions through a blinded evaluation with 99 participants. Our findings reveal that LLM-generated articles showed significantly lower scores in persuasive strategies, certainty expressions, and alignment with social values and moral foundations. However, human evaluation demonstrated a strong preference for LLM content, with over 60% responses favoring LLM articles for clarity, completeness, and persuasiveness. Our results suggest that LLMs' structured approach to presenting information may be more effective at engaging readers despite scoring lower on traditional measures of quality in fact-checking and health communication. 

**Abstract (ZH)**: 大语言模型在健康信息事实核查中的沟通风格与人类交流风格和价值观的对齐研究 

---
# ALOHA: Empowering Multilingual Agent for University Orientation with Hierarchical Retrieval 

**Title (ZH)**: ALOHA: 赋能多语言导生代理的层次化检索方法 

**Authors**: Mingxu Tao, Bowen Tang, Mingxuan Ma, Yining Zhang, Hourun Li, Feifan Wen, Hao Ma, Jia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08130)  

**Abstract**: The rise of Large Language Models~(LLMs) revolutionizes information retrieval, allowing users to obtain required answers through complex instructions within conversations. However, publicly available services remain inadequate in addressing the needs of faculty and students to search campus-specific information. It is primarily due to the LLM's lack of domain-specific knowledge and the limitation of search engines in supporting multilingual and timely scenarios. To tackle these challenges, we introduce ALOHA, a multilingual agent enhanced by hierarchical retrieval for university orientation. We also integrate external APIs into the front-end interface to provide interactive service. The human evaluation and case study show our proposed system has strong capabilities to yield correct, timely, and user-friendly responses to the queries in multiple languages, surpassing commercial chatbots and search engines. The system has been deployed and has provided service for more than 12,000 people. 

**Abstract (ZH)**: 大型语言模型的兴起变革了信息检索，使得用户能够通过对话中的复杂指令获得所需答案。然而，公开可用的服务仍不足以满足教职工和学生搜索校内信息的需求。主要是因为大型语言模型缺乏领域专业知识，以及搜索引擎在支持多语言和及时场景方面存在限制。为应对这些挑战，我们引入了ALOHA，这是一种增强型层次化检索多语言代理，用于大学迎新。我们还在前端界面中集成了外部API，以提供交互式服务。人类评估和案例研究显示，我们提出系统在多语言查询中具备提供准确、及时且用户友好的响应的强大能力，超越了商用聊天机器人和搜索引擎。该系统已部署并为超过12,000人提供了服务。 

---
# Are LLMs complicated ethical dilemma analyzers? 

**Title (ZH)**: 大型语言模型是复杂的伦理困境分析器吗？ 

**Authors**: Jiashen, Jesse Yao, Allen Liu, Zhekai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08106)  

**Abstract**: One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making. 

**Abstract (ZH)**: 大型语言模型能否模拟人类伦理推理并在伦理判断中作为可信代理：基于196个现实伦理困境及专家意见的基准数据集探究 

---
# Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders 

**Title (ZH)**: 超越输入激活：基于梯度稀疏自编码器识别影响力潜变量 

**Authors**: Dong Shu, Xuansheng Wu, Haiyan Zhao, Mengnan Du, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08080)  

**Abstract**: Sparse Autoencoders (SAEs) have recently emerged as powerful tools for interpreting and steering the internal representations of large language models (LLMs). However, conventional approaches to analyzing SAEs typically rely solely on input-side activations, without considering the causal influence between each latent feature and the model's output. This work is built on two key hypotheses: (1) activated latents do not contribute equally to the construction of the model's output, and (2) only latents with high causal influence are effective for model steering. To validate these hypotheses, we propose Gradient Sparse Autoencoder (GradSAE), a simple yet effective method that identifies the most influential latents by incorporating output-side gradient information. 

**Abstract (ZH)**: 稀疏自编码器（SAEs） recently emerged as强大的工具，用于解释和控制大型语言模型（LLMs）的内部表示。然而，传统上分析SAEs的方法通常仅依赖于输入端激活，而不考虑每个潜在特征对模型输出的因果影响。本文基于两个关键假设：（1）激活的潜在特征并不等价地贡献于模型输出的构建，（2）只有具有高因果影响的潜在特征对模型控制有效。为了验证这些假设，我们提出了梯度稀疏自编码器（GradSAE），这是一种简单而有效的方法，通过结合输出端梯度信息来识别最具影响力的最佳潜在特征。 

---
# FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning 

**Title (ZH)**: 拒绝假象：一种通过结构化推理提高上下文安全性和缓解过度拒绝的资源 

**Authors**: Zhehao Zhang, Weijie Xu, Fanyou Wu, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.08054)  

**Abstract**: Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）中的安全性对齐方法往往会导致对良性查询的过度拒绝，显著降低了其在敏感场景中的实用性。为解决这一挑战，我们引入了FalseReject，这是一个包含16,000个看似有毒查询及其在44个安全相关类别中结构化响应的全面资源。我们提出了一种基于图的信息对抗多智能体交互框架，以生成多样且复杂的提示，并通过明确的推理结构化响应，帮助模型准确区分安全与不安全的上下文。FalseReject包括针对标准指令调谐模型和推理导向模型的训练数据集，以及一个人工注释的标准测试集。我们对29个最先进的（SOTA）LLMs的广泛基准测试揭示了持续存在的过度拒绝挑战。实证结果表明，使用FalseReject进行监督微调在不牺牲整体安全性和通用语言能力的情况下，显著减少了不必要的拒绝。 

---
# Large Language Models and Arabic Content: A Review 

**Title (ZH)**: 大型语言模型与阿拉伯内容：一个综述 

**Authors**: Haneh Rhel, Dmitri Roussinov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08004)  

**Abstract**: Over the past three years, the rapid advancement of Large Language Models (LLMs) has had a profound impact on multiple areas of Artificial Intelligence (AI), particularly in Natural Language Processing (NLP) across diverse languages, including Arabic. Although Arabic is considered one of the most widely spoken languages across 27 countries in the Arabic world and used as a second language in some other non-Arabic countries as well, there is still a scarcity of Arabic resources, datasets, and tools. Arabic NLP tasks face various challenges due to the complexities of the Arabic language, including its rich morphology, intricate structure, and diverse writing standards, among other factors. Researchers have been actively addressing these challenges, demonstrating that pre-trained Large Language Models (LLMs) trained on multilingual corpora achieve significant success in various Arabic NLP tasks. This study provides an overview of using large language models (LLMs) for the Arabic language, highlighting early pre-trained Arabic Language models across various NLP applications and their ability to handle diverse Arabic content tasks and dialects. It also provides an overview of how techniques like finetuning and prompt engineering can enhance the performance of these models. Additionally, the study summarizes common Arabic benchmarks and datasets while presenting our observations on the persistent upward trend in the adoption of LLMs. 

**Abstract (ZH)**: 过去三年，大型语言模型的迅速发展对人工智能多个领域产生了深远影响，特别是在跨多种语言的自然语言处理（NLP）领域，包括阿拉伯语。尽管阿拉伯语是阿拉伯世界27个国家中最广泛使用的语言，并且在一些非阿拉伯国家中也被用作第二语言，但阿拉伯语资源、数据集和工具仍然相对匮乏。阿拉伯语NLP任务由于阿拉伯语丰富的形态学、复杂的结构和多样的书写标准等因素面临着各种挑战。研究人员积极应对这些挑战，证明了在多语言语料库上预训练的大型语言模型（LLMs）在各种阿拉伯语NLP任务中取得了显著成功。本研究概述了使用大型语言模型（LLMs）处理阿拉伯语的方法，强调了各种NLP应用中早期预训练的阿拉伯语言模型及其处理多种阿拉伯语内容任务和方言的能力。此外，研究还概述了如何通过微调和提示工程来提升这些模型的性能。研究还总结了常用的阿拉伯语基准和数据集，并呈现了LLMs采用持续增长的趋势观察。 

---
# SEM: Reinforcement Learning for Search-Efficient Large Language Models 

**Title (ZH)**: SEM：用于搜索高效的大型语言模型的强化学习 

**Authors**: Zeyang Sha, Shiwen Cui, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07903)  

**Abstract**: Recent advancements in Large Language Models(LLMs) have demonstrated their capabilities not only in reasoning but also in invoking external tools, particularly search engines. However, teaching models to discern when to invoke search and when to rely on their internal knowledge remains a significant challenge. Existing reinforcement learning approaches often lead to redundant search behaviors, resulting in inefficiencies and over-cost. In this paper, we propose SEM, a novel post-training reinforcement learning framework that explicitly trains LLMs to optimize search usage. By constructing a balanced dataset combining MuSiQue and MMLU, we create scenarios where the model must learn to distinguish between questions it can answer directly and those requiring external retrieval. We design a structured reasoning template and employ Group Relative Policy Optimization(GRPO) to post-train the model's search behaviors. Our reward function encourages accurate answering without unnecessary search while promoting effective retrieval when needed. Experimental results demonstrate that our method significantly reduces redundant search operations while maintaining or improving answer accuracy across multiple challenging benchmarks. This framework advances the model's reasoning efficiency and extends its capability to judiciously leverage external knowledge. 

**Abstract (ZH)**: Recent advancements in大型语言模型(LLMs)的能力不仅体现在推理上，还体现在调用外部工具，尤其是搜索引擎方面。然而，教会模型何时调用搜索、何时依赖内部知识仍然是一项重大挑战。现有强化学习方法往往导致冗余的搜索行为，造成低效率和高成本。在本文中，我们提出了一种名为SEM的新型后训练强化学习框架，旨在显式训练LLMs优化搜索使用。通过构建结合MuSiQue和MMLU的数据集，我们创建了使模型学会区分可以直接回答的问题和需要外部检索的问题的情景。我们设计了一种结构化推理模板，并采用Group Relative Policy Optimization (GRPO) 后训练模型的搜索行为。我们的奖励函数鼓励准确回答而不进行不必要的搜索，并在需要时促进有效的检索。实验结果表明，我们的方法显著减少了冗余搜索操作，同时在多个具有挑战性的基准上维持或提高了答案准确性。该框架提高了模型的推理效率，并扩展了它有节制地利用外部知识的能力。 

---
# DeltaEdit: Enhancing Sequential Editing in Large Language Models by Controlling Superimposed Noise 

**Title (ZH)**: DeltaEdit: 通过控制叠加噪声来增强大型语言模型的序列编辑能力 

**Authors**: Ding Cao, Yuchen Cai, Rongxi Guo, Xuesong He, Guiquan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07899)  

**Abstract**: Sequential knowledge editing techniques aim to continuously update the knowledge in large language models at a low cost, preventing the models from generating outdated or incorrect information. However, existing sequential editing methods suffer from a significant decline in editing success rates after long-term editing. Through theoretical analysis and experiments, we identify that as the number of edits increases, the model's output increasingly deviates from the desired target, leading to a drop in editing success rates. We refer to this issue as the accumulation of superimposed noise problem. To address this, we identify the factors contributing to this deviation and propose DeltaEdit, a novel method that optimizes update parameters through a dynamic orthogonal constraints strategy, effectively reducing interference between edits to mitigate deviation. Experimental results demonstrate that DeltaEdit significantly outperforms existing methods in edit success rates and the retention of generalization capabilities, ensuring stable and reliable model performance even under extensive sequential editing. 

**Abstract (ZH)**: Sequential知识编辑技术旨在以低成本持续更新大型语言模型的知识，防止模型生成过时或错误的信息。然而，现有的序列编辑方法在长期编辑后编辑成功率显著下降。通过理论分析和实验，我们发现随着编辑次数的增加，模型的输出越来越偏离期望目标，导致编辑成功率下降。我们将这一问题称为叠加噪声累积问题。为解决这一问题，我们确定了导致偏差的因素，并提出了一种名为DeltaEdit的新方法，该方法通过动态正交约束策略优化更新参数，有效减少编辑之间的干扰以减轻偏差。实验结果表明，DeltaEdit在编辑成功率和保持泛化能力方面显著优于现有方法，确保在广泛进行序列编辑的情况下模型性能的稳定性和可靠性。 

---
# LongCodeBench: Evaluating Coding LLMs at 1M Context Windows 

**Title (ZH)**: LongCodeBench: 评估具有100万上下文窗口的编程LLM 

**Authors**: Stefano Rando, Luca Romani, Alessio Sampieri, Yuta Kyuragi, Luca Franco, Fabio Galasso, Tatsunori Hashimoto, John Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07897)  

**Abstract**: Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5. 

**Abstract (ZH)**: 长上下文长度对模型的影响：从千级到百万级令牌的快速增长使得构建现实的长上下文基准变得困难——不仅由于收集百万级上下文任务的成本问题，还在于识别需要大量上下文的现实场景。我们认为空间理解和修复是测试长上下文模型的自然测试床和挑战任务，并引入LongCodeBench (LCB)，一个用于测试长上下文场景中LLM编码能力的基准。我们的基准通过借鉴真实世界的GitHub问题集，测试了从问题理解和修复能力，在各种规模的现实和重要场景中评估模型，从Qwen2.5 14B Instruct到Google的旗舰Gemini模型。我们发现所有模型在长上下文方面仍存在弱点，如Claude 3.5 Sonnet的性能从29%下降到3%，或Qwen2.5的性能从70.2%下降到40%。 

---
# Bridging Large Language Models and Single-Cell Transcriptomics in Dissecting Selective Motor Neuron Vulnerability 

**Title (ZH)**: 将大型语言模型与单细胞转录组学结合以解析选择性运动神经元易感性 

**Authors**: Douglas Jiang, Zilin Dai, Luxuan Zhang, Qiyi Yu, Haoqi Sun, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07896)  

**Abstract**: Understanding cell identity and function through single-cell level sequencing data remains a key challenge in computational biology. We present a novel framework that leverages gene-specific textual annotations from the NCBI Gene database to generate biologically contextualized cell embeddings. For each cell in a single-cell RNA sequencing (scRNA-seq) dataset, we rank genes by expression level, retrieve their NCBI Gene descriptions, and transform these descriptions into vector embedding representations using large language models (LLMs). The models used include OpenAI text-embedding-ada-002, text-embedding-3-small, and text-embedding-3-large (Jan 2024), as well as domain-specific models BioBERT and SciBERT. Embeddings are computed via an expression-weighted average across the top N most highly expressed genes in each cell, providing a compact, semantically rich representation. This multimodal strategy bridges structured biological data with state-of-the-art language modeling, enabling more interpretable downstream applications such as cell-type clustering, cell vulnerability dissection, and trajectory inference. 

**Abstract (ZH)**: 通过单细胞水平测序数据理解细胞身份和功能仍然是计算生物学中的一个关键挑战。我们提出了一种新的框架，利用NCBI Gene数据库中的基因特定文本注释生成生物上下文化的细胞嵌入。对于单细胞RNA测序(scRNA-seq)数据集中每个细胞，我们按照表达水平对基因进行排名，检索其NCBI Gene描述，并使用大型语言模型（LLMs）将这些描述转换为向量嵌入表示。所使用的模型包括OpenAI的text-embedding-ada-002、text-embedding-3-small和text-embedding-3-large（2024年1月），以及领域特定模型BioBERT和SciBERT。嵌入通过每个细胞前N个最高表达基因的表达加权平均计算得出，提供了一个紧凑且语义丰富的表示。这种多模态策略将结构化的生物数据与最先进的语言建模相结合，使下游应用更具可解释性，如细胞类型聚类、细胞脆弱性解析和轨迹推断。 

---
# TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking 

**Title (ZH)**: 谣言GPT：基于图的检索增强大规模语言模型用于事实核查 

**Authors**: Ching Nam Hang, Pei-Duo Yu, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07891)  

**Abstract**: In the age of social media, the rapid spread of misinformation and rumors has led to the emergence of infodemics, where false information poses a significant threat to society. To combat this issue, we introduce TrumorGPT , a novel generative artificial intelligence solution designed for fact-checking in the health domain. TrumorGPT aims to distinguish "trumors", which are health-related rumors that turn out to be true, providing a crucial tool in differentiating between mere speculation and verified facts. This framework leverages a large language model (LLM) with few-shot learning for semantic health knowledge graph construction and semantic reasoning. TrumorGPT incorporates graph-based retrieval-augmented generation (GraphRAG) to address the hallucination issue common in LLMs and the limitations of static training data. GraphRAG involves accessing and utilizing information from regularly updated semantic health knowledge graphs that consist of the latest medical news and health information, ensuring that fact-checking by TrumorGPT is based on the most recent data. Evaluating with extensive healthcare datasets, TrumorGPT demonstrates superior performance in fact-checking for public health claims. Its ability to effectively conduct fact-checking across various platforms marks a critical step forward in the fight against health-related misinformation, enhancing trust and accuracy in the digital information age. 

**Abstract (ZH)**: 社交媒体时代的信息疫情与谣言治理：TrumorGPT在健康领域的事实核查应用 

---
# Implementing Long Text Style Transfer with LLMs through Dual-Layered Sentence and Paragraph Structure Extraction and Mapping 

**Title (ZH)**: 利用双层句子和段落结构提取与映射在LLMs中实现长文本风格转移 

**Authors**: Yusen Wu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07888)  

**Abstract**: This paper addresses the challenge in long-text style transfer using zero-shot learning of large language models (LLMs), proposing a hierarchical framework that combines sentence-level stylistic adaptation with paragraph-level structural coherence. We argue that in the process of effective paragraph-style transfer, to preserve the consistency of original syntactic and semantic information, it is essential to perform style transfer not only at the sentence level but also to incorporate paragraph-level semantic considerations, while ensuring structural coherence across inter-sentential relationships. Our proposed framework, ZeroStylus, operates through two systematic phases: hierarchical template acquisition from reference texts and template-guided generation with multi-granular matching. The framework dynamically constructs sentence and paragraph template repositories, enabling context-aware transformations while preserving inter-sentence logical relationships. Experimental evaluations demonstrate significant improvements over baseline methods, with structured rewriting achieving 6.90 average score compared to 6.70 for direct prompting approaches in tri-axial metrics assessing style consistency, content preservation, and expression quality. Ablation studies validate the necessity of both template hierarchies during style transfer, showing higher content preservation win rate against sentence-only approaches through paragraph-level structural encoding, as well as direct prompting method through sentence-level pattern extraction and matching. The results establish new capabilities for coherent long-text style transfer without requiring parallel corpora or LLM fine-tuning. 

**Abstract (ZH)**: 基于零-shot学习的大语言模型在长文本风格转换中的挑战及其层次化框架：ZeroStylus 

---
# PLHF: Prompt Optimization with Few-Shot Human Feedback 

**Title (ZH)**: PLHF: few-shot人类反馈的提示优化 

**Authors**: Chun-Pai Yang, Kan Zheng, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07886)  

**Abstract**: Automatic prompt optimization frameworks are developed to obtain suitable prompts for large language models (LLMs) with respect to desired output quality metrics. Although existing approaches can handle conventional tasks such as fixed-solution question answering, defining the metric becomes complicated when the output quality cannot be easily assessed by comparisons with standard golden samples. Consequently, optimizing the prompts effectively and efficiently without a clear metric becomes a critical challenge. To address the issue, we present PLHF (which stands for "P"rompt "L"earning with "H"uman "F"eedback), a few-shot prompt optimization framework inspired by the well-known RLHF technique. Different from naive strategies, PLHF employs a specific evaluator module acting as the metric to estimate the output quality. PLHF requires only a single round of human feedback to complete the entire prompt optimization process. Empirical results on both public and industrial datasets show that PLHF outperforms prior output grading strategies for LLM prompt optimizations. 

**Abstract (ZH)**: 自动提示优化框架被开发出来，以针对大型语言模型（LLMs）获得符合期望输出质量指标的提示。虽然现有方法可以处理诸如固定解问答等传统任务，但在输出质量无法通过与标准黄金样本进行简单比较来评估时，定义质量指标变得复杂。因此，没有明确指标的情况下有效高效地优化提示成为一个关键挑战。为了解决这一问题，我们提出了PLHF（“P”rompt “L”earning with “H”uman “F”eedback），这是一种受RLHF技术启发的少样本提示优化框架。与朴素策略不同，PLHF 使用特定的评估模块充当质量指标，以估算输出质量。PLHF 只需一轮人类反馈即可完成整个提示优化过程。在公共和工业数据集上的实证结果表明，PLHF 在LLM提示优化中的输出评分策略中表现出更好的性能。 

---
# Recovering Event Probabilities from Large Language Model Embeddings via Axiomatic Constraints 

**Title (ZH)**: 基于公理约束从大型语言模型嵌入中恢复事件概率 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.07883)  

**Abstract**: Rational decision-making under uncertainty requires coherent degrees of belief in events. However, event probabilities generated by Large Language Models (LLMs) have been shown to exhibit incoherence, violating the axioms of probability theory. This raises the question of whether coherent event probabilities can be recovered from the embeddings used by the models. If so, those derived probabilities could be used as more accurate estimates in events involving uncertainty. To explore this question, we propose enforcing axiomatic constraints, such as the additive rule of probability theory, in the latent space learned by an extended variational autoencoder (VAE) applied to LLM embeddings. This approach enables event probabilities to naturally emerge in the latent space as the VAE learns to both reconstruct the original embeddings and predict the embeddings of semantically related events. We evaluate our method on complementary events (i.e., event A and its complement, event not-A), where the true probabilities of the two events must sum to 1. Experiment results on open-weight language models demonstrate that probabilities recovered from embeddings exhibit greater coherence than those directly reported by the corresponding models and align closely with the true probabilities. 

**Abstract (ZH)**: 在不确定性下的理性决策要求事件具有一致的信念度。然而，大型语言模型（LLMs）生成的事件概率表现出不一致，违反了概率论的公理。这引发了从LLMs使用的嵌入中恢复一致的事件概率是否可行的问题。如果可行，这些衍生的概率可以在涉及不确定性的事件中作为更准确的估计使用。为探索这一问题，我们提出在扩展的变分自编码器（VAE）应用于LLM嵌入的学习潜空间中实施公理约束，如概率论的加法规则。这种方法使得事件概率自然地在潜空间中出现，当VAE学习重构原始嵌入并预测语义相关事件的嵌入时。我们在互补事件（即事件A及其补事件，事件非A）上评估了该方法，其中两个事件的真实概率之和为1。实验结果表明，从嵌入中恢复的概率比对应模型直接报告的更一致，并且与真实概率紧密一致。 

---
# Efficient Telecom Specific LLM: TSLAM-Mini with QLoRA and Digital Twin Data 

**Title (ZH)**: 电信专用高效LLM：TSLAM-Mini结合QLoRA和数字孪生数据 

**Authors**: Vignesh Ethiraj, Divya Vijay, Sidhanth Menon, Heblin Berscilla  

**Link**: [PDF](https://arxiv.org/pdf/2505.07877)  

**Abstract**: General-purpose large language models (LLMs), despite their broad capabilities accrued from open-world data, frequently exhibit suboptimal performance when confronted with the nuanced and specialized demands inherent in real-time telecommunications applications. This investigation addresses this critical limitation through the meticulous fine-tuning of TSLAM-Mini developed by NetoAI, a compact (3.8-billion parameter) causal language model architecturally derived from Phi-4 Mini Instruct 4B. The fine-tuning regimen leverages a bespoke dataset comprising 100,000 samples, strategically engineered to address 20 pivotal telecommunications use-cases, encompassing domains such as Network Fundamentals, IP Routing, MPLS, Network Security, Automation, OSS/BSS, RAN, Mobile Core, Satellite Communications, and Ethical AI. This dataset was curated utilizing NetoAI's DigiTwin platform, enriched with granular insights from venerated network Subject Matter Experts (SMEs) and authoritative RFC documents, thereby capturing high-fidelity representations of real-world network dynamics through simulations inspired by digital twin paradigms. Employing Quantized Low-Rank Adaptation (QLoRA), a state-of-the-art Parameter Efficient Fine-Tuning (PEFT) technique, we achieved substantial training efficiency and enabled prospective deployment on resource-constrained hardware. A novel evaluation framework, predicated on a high-capacity LLM (Qwen3-235B-A22B) functioning as an automated adjudicator, was instituted to rigorously assess instruction-following fidelity and response quality across the specified telecom use-cases. Empirical results unequivocally demonstrate TSLAM-Mini's superior aptitude in telecom-centric applications, underscoring the profound efficacy of domain-specific datasets and PEFT methodologies for advancing intelligent network management. 

**Abstract (ZH)**: 通用大型语言模型（LLMs）尽管可以从开放世界数据中获得广泛的能力，但在面对实时电信应用中复杂和专业的需求时，经常表现出次优性能。本研究通过精细调整NetoAI开发的TSLAM-Mini（一个38亿参数的因果语言模型，源自Phi-4 Mini Instruct 4B）来解决这一关键限制。精细调整过程采用了包含100,000个样本的独特数据集，该数据集专门设计以解决20个关键的电信用例，涵盖了网络基础、IP路由、MPLS、网络安全、自动化、OSS/BSS、RAN、移动核心、卫星通信和伦理AI等领域。该数据集利用NetoAI的DigiTwin平台精心挑选，并结合了资深网络主题专家（SMEs）的详细见解和权威RFC文档，通过受数字孪生理念启发的仿真，捕捉到高保真度的现实网络动态。利用Quantized Low-Rank Adaptation（QLoRA）这一最先进的参数高效调整（PEFT）技术，我们实现了显著的训练效率，并使模型能够在资源受限的硬件上进行潜在部署。我们建立了一个新颖的评估框架，以一个超大容量的LLM（Qwen3-235B-A22B）作为自动化仲裁者，严格评估指令跟随的忠实度和响应质量，针对指定的电信用例。实验证明TSLAM-Mini在电信中心应用中的优越能力，突显了领域特定数据集和PEFT方法在智能网络管理发展中的深远效果。 

---
# Evaluating Financial Sentiment Analysis with Annotators Instruction Assisted Prompting: Enhancing Contextual Interpretation and Stock Prediction Accuracy 

**Title (ZH)**: 使用注释者指令辅助提示评估金融情绪分析：增强上下文解释和股票预测准确性 

**Authors**: A M Muntasir Rahman, Ajim Uddin, Guiling "Grace" Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07871)  

**Abstract**: Financial sentiment analysis (FSA) presents unique challenges to LLMs that surpass those in typical sentiment analysis due to the nuanced language used in financial contexts. The prowess of these models is often undermined by the inherent subjectivity of sentiment classifications in existing benchmark datasets like Financial Phrasebank. These datasets typically feature undefined sentiment classes that reflect the highly individualized perspectives of annotators, leading to significant variability in annotations. This variability results in an unfair expectation for LLMs during benchmarking, where they are tasked to conjecture the subjective viewpoints of human annotators without sufficient context. In this paper, we introduce the Annotators' Instruction Assisted Prompt, a novel evaluation prompt designed to redefine the task definition of FSA for LLMs. By integrating detailed task instructions originally intended for human annotators into the LLMs' prompt framework, AIAP aims to standardize the understanding of sentiment across both human and machine interpretations, providing a fair and context-rich foundation for sentiment analysis. We utilize a new dataset, WSBS, derived from the WallStreetBets subreddit to demonstrate how AIAP significantly enhances LLM performance by aligning machine operations with the refined task definitions. Experimental results demonstrate that AIAP enhances LLM performance significantly, with improvements up to 9.08. This context-aware approach not only yields incremental gains in performance but also introduces an innovative sentiment-indexing method utilizing model confidence scores. This method enhances stock price prediction models and extracts more value from the financial sentiment analysis, underscoring the significance of WSB as a critical source of financial text. Our research offers insights into both improving FSA through better evaluation methods. 

**Abstract (ZH)**: 金融情感分析（FSA）对LLM提出的挑战超越了常规情感分析，因为金融语境中使用了更为细致的语言。这些模型的性能往往被现有基准数据集如Financial Phrasebank中固有的主观性所削弱。这些数据集通常包含了未定义的情感类别，反映了注释员的高度个性化视角，导致注释结果的大范围变化。这些变化在基准测试中对LLM提出了不公正的期望，使它们需要在缺乏足够上下文的情况下推断人类注释者的主观观点。本文介绍了注释员说明辅助提示（AIAP），这是一种新颖的评估提示，旨在重新定义针对LLM的FSA任务定义。通过将原计划用于人类注释员的任务指令整合到LLM的提示框架中，AIAP旨在标准化人类和机器对情感的理解，为情感分析提供一个公平且富含上下文的基础。我们使用从WallStreetBets子版块推断出的新数据集WSBS来展示AIAP如何通过使机器操作与精炼的任务定义对齐来显著提升LLM性能。实验结果表明，AIAP显著提升了LLM的性能，性能提升可达9.08。这种基于上下文的方法不仅在性能上取得增量提升，还引入了利用模型置信度分数的情感索引方法，这增强了股票价格预测模型并从金融情感分析中提取更多价值，凸显了WSB作为关键金融文本来源的重要性。我们的研究为通过更好的评估方法改进FSA提供了见解。 

---
# Efficient Fairness Testing in Large Language Models: Prioritizing Metamorphic Relations for Bias Detection 

**Title (ZH)**: 大型语言模型中高效公平性测试：偏见检测中优先考虑元变换关系 

**Authors**: Suavis Giramata, Madhusudan Srinivasan, Venkat Naidu Gudivada, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2505.07870)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in various applications, raising critical concerns about fairness and potential biases in their outputs. This paper explores the prioritization of metamorphic relations (MRs) in metamorphic testing as a strategy to efficiently detect fairness issues within LLMs. Given the exponential growth of possible test cases, exhaustive testing is impractical; therefore, prioritizing MRs based on their effectiveness in detecting fairness violations is crucial. We apply a sentence diversity-based approach to compute and rank MRs to optimize fault detection. Experimental results demonstrate that our proposed prioritization approach improves fault detection rates by 22% compared to random prioritization and 12% compared to distance-based prioritization, while reducing the time to the first failure by 15% and 8%, respectively. Furthermore, our approach performs within 5% of fault-based prioritization in effectiveness, while significantly reducing the computational cost associated with fault labeling. These results validate the effectiveness of diversity-based MR prioritization in enhancing fairness testing for LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种应用中的部署引发了对其输出公平性和潜在偏差的关键关注。本文探讨了在元变测试中优先考虑元变关系（MRs）作为高效检测LLMs公平性问题的策略。鉴于可能的测试案例呈指数增长，穷尽测试不切实际；因此，基于其检测公平性违规的有效性来优先考虑MRs至关重要。我们采用基于句子多样性的方法来计算和排名MRs，以优化故障检测。实验结果表明，与随机优先级相比，我们的提议优先级方法可将故障检测率提高22%，与基于距离的优先级相比提高12%，同时将第一个失败的时间减少15%和8%。此外，我们的方法在有效性上与基于故障的优先级相差不到5%，但显著降低了与故障标签相关的计算成本。这些结果验证了基于多样性的MR优先级在增强LLMs公平性测试方面的有效性。 

---
# CellVerse: Do Large Language Models Really Understand Cell Biology? 

**Title (ZH)**: CellVerse: 大型语言模型真的理解细胞生物学吗？ 

**Authors**: Fan Zhang, Tianyu Liu, Zhihong Zhu, Hao Wu, Haixin Wang, Donghao Zhou, Yefeng Zheng, Kun Wang, Xian Wu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07865)  

**Abstract**: Recent studies have demonstrated the feasibility of modeling single-cell data as natural languages and the potential of leveraging powerful large language models (LLMs) for understanding cell biology. However, a comprehensive evaluation of LLMs' performance on language-driven single-cell analysis tasks still remains unexplored. Motivated by this challenge, we introduce CellVerse, a unified language-centric question-answering benchmark that integrates four types of single-cell multi-omics data and encompasses three hierarchical levels of single-cell analysis tasks: cell type annotation (cell-level), drug response prediction (drug-level), and perturbation analysis (gene-level). Going beyond this, we systematically evaluate the performance across 14 open-source and closed-source LLMs ranging from 160M to 671B on CellVerse. Remarkably, the experimental results reveal: (1) Existing specialist models (C2S-Pythia) fail to make reasonable decisions across all sub-tasks within CellVerse, while generalist models such as Qwen, Llama, GPT, and DeepSeek family models exhibit preliminary understanding capabilities within the realm of cell biology. (2) The performance of current LLMs falls short of expectations and has substantial room for improvement. Notably, in the widely studied drug response prediction task, none of the evaluated LLMs demonstrate significant performance improvement over random guessing. CellVerse offers the first large-scale empirical demonstration that significant challenges still remain in applying LLMs to cell biology. By introducing CellVerse, we lay the foundation for advancing cell biology through natural languages and hope this paradigm could facilitate next-generation single-cell analysis. 

**Abstract (ZH)**: Recent studies have demonstrated the feasibility of modeling single-cell data as natural languages and the potential of leveraging powerful large language models (LLMs) for understanding cell biology. However, a comprehensive evaluation of LLMs' performance on language-driven single-cell analysis tasks still remains unexplored. Motivated by this challenge, we introduce CellVerse, a unified language-centric question-answering benchmark that integrates four types of single-cell multi-omics data and encompasses three hierarchical levels of single-cell analysis tasks: cell type annotation (cell-level), drug response prediction (drug-level), and perturbation analysis (gene-level). 

---
# Scalable LLM Math Reasoning Acceleration with Low-rank Distillation 

**Title (ZH)**: 可扩展的大语言模型数学推理加速方法：低秩蒸馏 

**Authors**: Harry Dong, Bilge Acun, Beidi Chen, Yuejie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07861)  

**Abstract**: Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a low-cost distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>11% reduction to generate 2048 tokens with Qwen 2.5 14B) while encouraging response brevity. 

**Abstract (ZH)**: 由于龐大的架構，大型語言模型的數學推理需要大量的計算資源和時間。雖然已經開發了許多高效推理方法，在語言任務上保持了出色的性能，但這些方法往往會嚴重降低數學性能。在本文中，我們提出了一種低 COST 的蒸�子方法 Caprese，以從高效推理方法的部署中恢復丟失的能力，主要集中在前向塊上。通過保留原始權重不被打擾、僅增加約 1% 的參數，并使用約 20K 合成訓練樣本，我們能夠恢復高效推理對思考型大語言模型失去的大量甚至全部數學能力，同時不會損害指令型大語言模型的語言任務性能。此外，Caprese 能夠顯著減少活躍參數數量（Gemma 2 9B 約減 2B，Llama 3.1 8B 當量）並平滑地集成到現有模型層中以減低 latency（使用 Qwen 2.5 14B 則生成 2048 個詞元的速度可減少大於 11%），同時鼓勵簡潔的回應。 

---
# Boosting Performance on ARC is a Matter of Perspective 

**Title (ZH)**: 提升ARC性能取决于视角 

**Authors**: Daniel Franzen, Jan Disselhoff, David Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.07859)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities. In this work, we leverage task-specific data augmentations throughout the training, generation, and scoring phases, and employ a depth-first search algorithm to generate diverse, high-probability candidate solutions. Furthermore, we utilize the LLM not only as a generator but also as a scorer, using its output probabilities to select the most promising solutions. Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches. While concurrent closed-source work has reported higher scores, our method distinguishes itself through its transparency, reproducibility, and remarkably low inference cost, averaging only around 2ct per task on readily available hardware (we assume a price of 36ct/hour for a Nvidia 4090 GPU). 

**Abstract (ZH)**: ARC-AGI抽象和推理语料库对大型语言模型提出了显著挑战，暴露了其在抽象推理能力上的局限性。在这项工作中，我们通过在训练、生成和评分阶段利用任务特定的数据增强，并采用深度优先搜索算法生成多样性和高概率候选解决方案。此外，我们不仅将LLM用作生成器，还用作评分器，利用其输出概率选择最有潜力的解决方案。我们的方法在公开的ARC-AGI评估集上取得了71.6%的分数（总共解决了400个任务中的286.5个），展示了公开可用方法中的最先进性能。尽管同期的闭源工作报告了更高的分数，但我们的方法通过其透明性、可重复性和极低的推理成本脱颖而出，平均每个任务仅需约2ct（假设使用Nvidia 4090 GPU的价格为每小时36ct）。 

---
# Scaling Laws for Speculative Decoding 

**Title (ZH)**: 推测性解码的标度律 

**Authors**: Siyuan Yan, Mo Zhu, Guo-qing Jiang, Jianfei Wang, Jiaxing Chen, Wentai Zhang, Xiang Liao, Xiao Cui, Chen Zhang, Zhuoran Song, Ran Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07858)  

**Abstract**: The escalating demand for efficient decoding in large language models (LLMs) is particularly critical for reasoning-intensive architectures like OpenAI-o3 and DeepSeek-R1, which depend on extended chain-of-thought reasoning. This study investigates speculative decoding techniques through dense LLM architectures to establish foundational insights for accelerating reasoning tasks. While speculative decoding methods leveraging parallel draft-verification cycles have emerged as promising acceleration techniques, the scaling laws governing decoding efficiency remain under-explored compared to conventional backbone LLMs developed through Pretraining->SFT->RLHF training paradigms. In this work, we discover Log-linear Scaling Laws (Theorem 1.1, 1.2 and 1.3) governing draft model acceptance rate (or decoding speed) across three dimensions: pretraining token volume, draft model capacity, and decoding batch size. Building on these laws, we achieve Scylla, which coordinates multi-dimensional scaling for popular LLMs (Llama2/3, Qwen2.5). Empirical validation shows Scylla achieves 1.5-2.2 higher acceptance rate than EAGLE2 and 0.3 higher than EAGLE3 at temperature T = 0, with peak performance gains on summarization and QA tasks (Figure 2). Industrial inference engine deployments demonstrate 2X decoding throughput improvements over EAGLE2 (Table 5), validating the transformative potential of systematic scaling for efficient LLM inference. Code will be released later. 

**Abstract (ZH)**: 大型语言模型中高效解码需求的上升特别关键，对于依赖于扩展链式推理的推理密集型架构如OpenAI-o3和DeepSeek-R1尤为重要。本研究通过密集的语言模型架构探索推测性解码技术，以建立加速推理任务的基础洞察。虽然利用并行草稿验证循环的推测性解码方法已成为有望的加速技术，但解码效率的标度定律与传统的通过预训练->精细调优->人类反馈强化学习训练范式开发的骨干语言模型相比，仍有待深入探索。在本文中，我们发现了跨越预训练 token 体积、草稿模型容量和解码批次大小三个维度的对数线性标度定律（定理1.1、1.2和1.3）。基于这些定律，我们实现了一种协调多维度标度的技术Scylla，适用于流行的语言模型（Llama2/3、Qwen2.5）。实证验证显示，与EAGLE2相比，Scylla在温度T=0时的接受率提高了1.5-2.2倍，在EAGLE3上的接受率高0.3倍，在总结和问答任务上达到了最高的性能提升（图2）。工业推理引擎部署表明，Scylla相比于EAGLE2的解码吞吐量提高了2倍（表5），验证了系统性标度对高效语言模型推理的变革潜力。代码稍后发布。 

---
# Enhanced Urdu Intent Detection with Large Language Models and Prototype-Informed Predictive Pipelines 

**Title (ZH)**: 使用大型语言模型和原型驱动的预测管道增强乌尔都语意图检测 

**Authors**: Faiza Hassan, Summra Saleem, Kashif Javed, Muhammad Nabeel Asim, Abdur Rehman, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2505.07857)  

**Abstract**: Multifarious intent detection predictors are developed for different languages, including English, Chinese and French, however, the field remains underdeveloped for Urdu, the 10th most spoken language. In the realm of well-known languages, intent detection predictors utilize the strategy of few-shot learning and prediction of unseen classes based on the model training on seen classes. However, Urdu language lacks few-shot strategy based intent detection predictors and traditional predictors are focused on prediction of the same classes which models have seen in the train set. To empower Urdu language specific intent detection, this introduces a unique contrastive learning approach that leverages unlabeled Urdu data to re-train pre-trained language models. This re-training empowers LLMs representation learning for the downstream intent detection task. Finally, it reaps the combined potential of pre-trained LLMs and the prototype-informed attention mechanism to create a comprehensive end-to-end LLMPIA intent detection pipeline. Under the paradigm of proposed predictive pipeline, it explores the potential of 6 distinct language models and 13 distinct similarity computation methods. The proposed framework is evaluated on 2 public benchmark datasets, namely ATIS encompassing 5836 samples and Web Queries having 8519 samples. Across ATIS dataset under 4-way 1 shot and 4-way 5 shot experimental settings LLMPIA achieved 83.28% and 98.25% F1-Score and on Web Queries dataset produced 76.23% and 84.42% F1-Score, respectively. In an additional case study on the Web Queries dataset under same classes train and test set settings, LLMPIA outperformed state-of-the-art predictor by 53.55% F1-Score. 

**Abstract (ZH)**: 面向乌尔都语的多意图检测预测器开发：一种基于对比学习的方法 

---
# CrashSage: A Large Language Model-Centered Framework for Contextual and Interpretable Traffic Crash Analysis 

**Title (ZH)**: CrashSage: 一种以大型语言模型为中心的上下文可解释道路交通碰撞分析框架 

**Authors**: Hao Zhen, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07853)  

**Abstract**: Road crashes claim over 1.3 million lives annually worldwide and incur global economic losses exceeding \$1.8 trillion. Such profound societal and financial impacts underscore the urgent need for road safety research that uncovers crash mechanisms and delivers actionable insights. Conventional statistical models and tree ensemble approaches typically rely on structured crash data, overlooking contextual nuances and struggling to capture complex relationships and underlying semantics. Moreover, these approaches tend to incur significant information loss, particularly in narrative elements related to multi-vehicle interactions, crash progression, and rare event characteristics. This study presents CrashSage, a novel Large Language Model (LLM)-centered framework designed to advance crash analysis and modeling through four key innovations. First, we introduce a tabular-to-text transformation strategy paired with relational data integration schema, enabling the conversion of raw, heterogeneous crash data into enriched, structured textual narratives that retain essential structural and relational context. Second, we apply context-aware data augmentation using a base LLM model to improve narrative coherence while preserving factual integrity. Third, we fine-tune the LLaMA3-8B model for crash severity inference, demonstrating superior performance over baseline approaches, including zero-shot, zero-shot with chain-of-thought prompting, and few-shot learning, with multiple models (GPT-4o, GPT-4o-mini, LLaMA3-70B). Finally, we employ a gradient-based explainability technique to elucidate model decisions at both the individual crash level and across broader risk factor dimensions. This interpretability mechanism enhances transparency and enables targeted road safety interventions by providing deeper insights into the most influential factors. 

**Abstract (ZH)**: 基于大型语言模型的交通碰撞分析与建模新框架：CrashSage 

---
# Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment 

**Title (ZH)**: 使用LLM辅助判断的在线对话中欺诈和概念漂移的联合检测 

**Authors**: Ali Senol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07852)  

**Abstract**: Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models. 

**Abstract (ZH)**: 在数字通信平台上检测虚假互动仍是一个具有挑战性和未充分解决的问题。这些互动可能表现为无害的垃圾信息，也可能升级为复杂的欺诈尝试，使得及早识别恶意意图变得困难。传统检测方法通常依赖静态异常检测技术，这些技术难以适应动态对话的变化。一个关键限制是将良性话题转换误解读为欺诈行为，导致虚假警报或遗漏威胁。我们提出了一种两阶段检测框架，首先使用定制的集成分类模型识别可疑对话。为了提高检测的可靠性，我们引入了一类漂移分析步骤，使用One Class Drift Detector (OCDD) 来隔离标记对话中的对话变化。当检测到漂移时，大规模语言模型（LLM）评估该变化是否表示欺诈操纵或合法话题变化。在未发现漂移的情况下，行为被认为是类似垃圾信息的。我们使用社交工程聊天场景数据集验证了我们的框架，并展示了其在实时欺诈检测中提高准确性和可解释性的实用优势。为了阐述权衡关系，我们将我们的模块化方法与使用不同语言模型进行检测和判断的Dual LLM基线进行了比较。 

---
# A Tale of Two Identities: An Ethical Audit of Human and AI-Crafted Personas 

**Title (ZH)**: 两种身份的故事：人类与AI创构的人设伦理审查 

**Authors**: Pranav Narayanan Venkit, Jiayi Li, Yingfan Zhou, Sarah Rajtmajer, Shomir Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2505.07850)  

**Abstract**: As LLMs (large language models) are increasingly used to generate synthetic personas particularly in data-limited domains such as health, privacy, and HCI, it becomes necessary to understand how these narratives represent identity, especially that of minority communities. In this paper, we audit synthetic personas generated by 3 LLMs (GPT4o, Gemini 1.5 Pro, Deepseek 2.5) through the lens of representational harm, focusing specifically on racial identity. Using a mixed methods approach combining close reading, lexical analysis, and a parameterized creativity framework, we compare 1512 LLM generated personas to human-authored responses. Our findings reveal that LLMs disproportionately foreground racial markers, overproduce culturally coded language, and construct personas that are syntactically elaborate yet narratively reductive. These patterns result in a range of sociotechnical harms, including stereotyping, exoticism, erasure, and benevolent bias, that are often obfuscated by superficially positive narrations. We formalize this phenomenon as algorithmic othering, where minoritized identities are rendered hypervisible but less authentic. Based on these findings, we offer design recommendations for narrative-aware evaluation metrics and community-centered validation protocols for synthetic identity generation. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）在健康、隐私和人机交互等数据有限领域中越来越多地用于生成合成人设，理解这些叙述如何代表身份，特别是少数社区的身份，变得至关重要。本文通过代表性伤害的视角审计3个LLM（GPT4o、Gemini 1.5 Pro、Deepseek 2.5）生成的合成人设，重点关注种族身份。采用结合精密阅读、词汇分析和参数化创造力框架的混合方法，我们将1512个LLM生成的人设与人类撰写的回应进行比较。研究发现，这些模型在种族标记方面过度强调，在文化编码语言方面过度生产，并构建了句法复杂但叙事简化的个体形象。这些模式导致一系列社会技术伤害，包括刻板印象、异文化浪漫化、抹除和善意偏见，这些伤害往往被表面上积极的叙述所掩盖。我们这一现象正式化为算法异化，其中边缘化身份被过度可见但缺乏真实性。基于这些发现，我们提出了叙事意识评估指标和以社区为中心的合成身份生成验证协议的设计建议。 

---
# Patchwork: A Unified Framework for RAG Serving 

**Title (ZH)**: Patchwork: 统一的RAG服务框架 

**Authors**: Bodun Hu, Luis Pabon, Saurabh Agarwal, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2505.07833)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a new paradigm for enhancing Large Language Model reliability through integration with external knowledge sources. However, efficient deployment of these systems presents significant technical challenges due to their inherently heterogeneous computational pipelines comprising LLMs, databases, and specialized processing components. We introduce Patchwork, a comprehensive end-to-end RAG serving framework designed to address these efficiency bottlenecks. Patchwork's architecture offers three key innovations: First, it provides a flexible specification interface enabling users to implement custom RAG pipelines. Secondly, it deploys these pipelines as distributed inference systems while optimizing for the unique scalability characteristics of individual RAG components. Third, Patchwork incorporates an online scheduling mechanism that continuously monitors request load and execution progress, dynamically minimizing SLO violations through strategic request prioritization and resource auto-scaling. Our experimental evaluation across four distinct RAG implementations demonstrates that Patchwork delivers substantial performance improvements over commercial alternatives, achieving throughput gains exceeding 48% while simultaneously reducing SLO violations by ~24%. 

**Abstract (ZH)**: 检索增强生成（RAG）已经 emergence 为一种通过结合外部知识源来增强大型语言模型可靠性的新范式。然而，这些系统的高效部署由于其本质上异构的计算流水线（包括LLMs、数据库和专用处理组件）而面临显著的技术挑战。我们引入了Patchwork，一种全面的端到端RAG服务框架，旨在解决这些效率瓶颈。Patchwork的架构提供了三项关键创新：首先，它提供了一个灵活的规范接口，使用户能够实现自定义的RAG流水线。其次，它以优化特定RAG组件的独特扩展性能的方式部署这些流水线。第三，Patchwork整合了一个在线调度机制，持续监测请求负载和执行进度，通过战略性地请求优先级调整和资源自动扩展，动态地最小化SLO违反。我们的实验评估表明，Patchwork在四种不同的RAG实现中实现了显著的性能提升，与商业替代方案相比，吞吐量提升超过48%，同时将SLO违反率降低约24%。 

---
